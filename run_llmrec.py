import os
import pickle
import torch.multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataloader import get_dataloader
from utils import LossMeter
from trainer_base import TrainerBase
from param import parse_args
from llmrec import LLMRec
import math
import time
from torch.cuda.amp import GradScaler
from torch import autocast
from metrics import cal_recall, cal_ndcg
from utils import amazon_dataset2fullname
from transformers import GenerationConfig
from tqdm import tqdm
from utils import print_rank0
import collections
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, PreTrainedTokenizerFast, AutoModel,AutoTokenizer, Qwen2Tokenizer
import wandb
import os
import json
from pathlib import Path
from HUM_Trainer import Trainer

os.environ['WANDB_MODE'] = 'disabled'


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 
def compute_mmd(x, y, sigma=1.0):
    if len(x) == 0 or len(y) == 0:
        return torch.tensor(0.0, device=x.device)
    def gaussian_kernel(a, b):
        dists = (a.unsqueeze(1) - b.unsqueeze(0)) ** 2
        return torch.exp(-dists / (2 * sigma ** 2))
    K_xx = gaussian_kernel(x, x).mean()
    K_yy = gaussian_kernel(y, y).mean()
    K_xy = gaussian_kernel(x, y).mean()
    return K_xx + K_yy - 2 * K_xy

def main_worker(local_rank, args):
    args.gpu = local_rank
    torch.cuda.set_device(args.gpu)

    args.single_domain_iid = pickle.load(open(f"./dataset/{args.dataset}/single_domain_iid.pkl", 'rb'))
    value_to_domain_index = {}
    for domain, items in args.single_domain_iid.items():
        for idx, item in enumerate(items):
            value_to_domain_index[item] = (domain, list(args.single_domain_iid.keys()).index(domain))
    args.value_to_domain_index = value_to_domain_index

    print_rank0(f'Process Launching at GPU {args.gpu}')

    if args.distributed:
        args.dist_backend = "nccl"
        dist.init_process_group(
            backend=args.dist_backend,
            init_method="env://",  # 用 torchrun 启动时推荐使用 env://
        )

    print_rank0(f'Building train loader at GPU {args.gpu}')
    if 'Qwen' in args.backbone:

        tokenizer = AutoTokenizer.from_pretrained(
            args.root_path + args.backbone,
            model_max_length=2048,
            padding_side="left",
        )
        tokenized_domain = {domain: tokenizer(domain,
                                        # max_length=item_max_tokens,
                                        padding=False,
                                        add_special_tokens=False,
                                        return_tensors=None,
                                        )['input_ids'] for domain in list(args.single_domain_iid.keys())}
        
        if args.prompt_selection == 0:
            text = 'Compress the following description about the user or item into the last token:'

        tokenized_text = tokenizer(text,
                                        # max_length=item_max_tokens,
                                        padding=False,
                                        add_special_tokens=False,
                                        return_tensors=None,
                                        )['input_ids']
        print(tokenized_text)
        special_tokens_dict = {'additional_special_tokens': ['<|emb|>', '<|thought|>']}

        tokenizer.add_special_tokens(special_tokens_dict)
        token_id_dict = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in special_tokens_dict['additional_special_tokens']
        }
        tokenizer.token_id_dict = token_id_dict
    train_loader, valid_loader, test_loader = get_dataloader(args, tokenizer, tokenized_text,tokenized_text, token_id_dict)

    trainer = Trainer(args, tokenizer, train_loader, valid_loader, test_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    args.dataset = amazon_dataset2fullname[args.dataset] if args.dataset in amazon_dataset2fullname.keys() else args.dataset
    args.dataset = args.dataset + f'-{args.ratio}-{args.user_k}-{args.item_k}'


    gpu_count = args.num_gpus
    # torchrun 自动设置这些环境变量
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    # args.world_size = 2
    args.gpu = args.local_rank
    print(args.gpu)

    if args.rank == 0:
        print("============runner run with args=================")
        print(args)

    if args.distributed:
        gpu_count = torch.cuda.device_count()
        main_worker(args.rank, args)
        # mp.spawn(main_worker, (args,), nprocs=gpu_count, join=True)
    else:
        main_worker(0, args)