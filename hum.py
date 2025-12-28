import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import os
from torch import autocast
from peft import LoraConfig, get_peft_model
from utils import print_rank0
from transformers import AutoModel, Qwen2Model, AutoModelForCausalLM
import collections
import pickle


class HUM(nn.Module):
    """
    HUM Recommendation Model.
    """
    
    def __init__(self, args, tokenizer):
        """Initialize HUM model."""
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        
        # Load Base LLM
        if 'Qwen' in args.backbone:
            self.llm = AutoModelForCausalLM.from_pretrained(
                args.root_path + args.backbone,
                torch_dtype="bfloat16",
            )
            self.llm.resize_token_embeddings(len(tokenizer))
        

        # LoRA Configuration
        if args.lora:
            print_rank0("Initialize LoRA From Scratch!", self.args.rank)
            from peft import TaskType
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                bias="none",
                inference_mode=False, 
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.llm = get_peft_model(self.llm, config)
            self.trainable2float()

        self.item_embs = None

    def trainable2float(self):
        """Convert trainable parameters to float32 for LoRA."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print_rank0(f"Trainable Parameter: {name}", self.args.rank)
                param.data = param.data.float()

    def get_embedding(self, input_ids, attention_mask):
        """Extract embedding from LLM's last hidden state."""
        llm_output = self.llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            output_attentions=True
        )

        if 'Qwen' == self.args.backbone[:4]:
            return self.gather_indexes(llm_output.hidden_states[-1], attention_mask.sum(dim=-1) - 1)
        else:
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)

    def forward(self, inputs):
        """
        Forward pass for training.
        """
        device = next(self.parameters()).device
        kl_loss = torch.tensor(0.0, device=device)
        generative_loss = torch.tensor(0.0, device=device)
        raw_loss_per_sample = generative_loss
        seq_batch = inputs['sequence_input_ids'].size(0)
        
        # 1. Extract Embeddings from LLM
        item_embs = self.get_embedding(
            inputs['item_input_ids'], 
            inputs['item_attention_mask']
        ).view(seq_batch, self.args.train_nega_count + 1, -1)
        
        user_emb = self.get_embedding(
            input_ids=inputs['sequence_input_ids'], 
            attention_mask=inputs['sequence_attention_mask']
        )
        
        # 3. Contrastive Loss Computation
        with autocast(device_type='cuda', enabled=False):
            user_f = user_emb.unsqueeze(-1).float()
            scores = torch.bmm(item_embs.float(), user_f).squeeze(-1)
            
            loss_contrast = F.cross_entropy(
                scores / self.args.similarity_temperature,
                inputs['target_position'],
                reduction='none'
            )
        
        tail_loss = torch.tensor(0.0, device=device)
        
        return [loss_contrast, generative_loss, kl_loss, tail_loss, raw_loss_per_sample]

    def valid_step(self, inputs):
        """Validation/testing forward pass."""
        user_emb = self.get_embedding(
            input_ids=inputs['sequence_input_ids'], 
            attention_mask=inputs['sequence_attention_mask']
        )
        
        store_cls = user_emb
        
        user_emb_f = user_emb.float().unsqueeze(-1)
        negative_items = inputs['negative_items']
        total_len_negative = negative_items.size(1)
        max_batch_size = 128
        
        scores_list = []
        for start_idx in range(0, negative_items.size(1), max_batch_size):
            batch_negative_items = negative_items[:, start_idx:start_idx + max_batch_size]
            batch_item_embs = self.item_embs[batch_negative_items].to(user_emb_f.device)
            
            with autocast(device_type='cuda', enabled=False):
                batch_item_embs = batch_item_embs.float()
                batch_scores = torch.bmm(batch_item_embs, user_emb_f).squeeze(-1) / math.sqrt(total_len_negative)
                scores_list.append(batch_scores)
        
        scores = torch.cat(scores_list, dim=-1).to(user_emb_f.device)
        torch.cuda.empty_cache()

        return scores, "", inputs['target_position'], store_cls

    @torch.no_grad()
    def generate_embs_hum(self, item_tokens):
        """Generate enhanced item embeddings for validation/testing."""
        if hasattr(self, 'item_embs') and self.item_embs is not None:
            del self.item_embs
        torch.cuda.empty_cache()
        print_rank0(f"GPU:{self.args.rank} Generating Enhanced Item Embeddings (Base)", self.args.rank)
        
        item_ids = item_tokens['item_ids']
        item_attn = item_tokens['item_attn']
        device = next(self.parameters()).device

        item_embs_all = []
        batch_size = 512
        
        if self.args.rank == 0:
            iterator = tqdm(range(0, item_ids.size()[0], batch_size), desc='Generate base enhanced embs')
        else:
            iterator = range(0, item_ids.size()[0], batch_size)
        
        for start_idx in iterator:
            batch_item_ids = item_ids[start_idx: start_idx + batch_size].to(device)
            batch_item_attn = item_attn[start_idx: start_idx + batch_size].to(device)
            
            batch_item_embs = self.get_embedding(
                input_ids=batch_item_ids, 
                attention_mask=batch_item_attn
            )
            
            item_embs_all.append(batch_item_embs.detach().cpu())  
            torch.cuda.empty_cache()
        
        ph_emb = torch.full((1, item_embs_all[0].size()[-1]), 0.0)
        item_embs_all.append(ph_emb.detach())
        self.item_embs = torch.cat(item_embs_all, dim=0).to(device)
        
        assert self.item_embs.size()[0] == item_ids.size()[0] + 1
        print_rank0(f"Enhanced item_emb shape = {self.item_embs.shape}", self.args.rank)
        
        # Optional: save to file
        # torch.save(self.item_embs, f'others/item_embs_base.pt')
    

    def gather_indexes(self, output, gather_index):
        """Gather embeddings at specific positions."""
        gather_index -= 1
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
