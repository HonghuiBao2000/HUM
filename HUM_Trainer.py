
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
from hum import HUM
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
from typing import Dict, List, Any, Optional

# Import HUM Logger (integrated with EvaluationLogger)
try:
    from hum_logger import init_logger, get_global_logger, EvaluationLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

@dataclass
class DomainTrackingState:
    keys: List[str]
    weights: torch.Tensor
    loss_list: torch.Tensor
    domain_losses: Dict[str, torch.Tensor]
    domain_counts: Dict[str, int]
    cur_domain_losses: Dict[str, torch.Tensor]
    cur_domain_counts: Dict[str, int]
    accumulated_losses: List

    @classmethod
    def initialize(cls, domain_keys, device):
        return cls(
            keys=domain_keys,
            weights=torch.ones(len(domain_keys), device=device),
            loss_list=torch.zeros(len(domain_keys), device=device),
            domain_losses=collections.defaultdict(float),
            domain_counts=collections.defaultdict(int),
            cur_domain_losses=collections.defaultdict(float),
            cur_domain_counts=collections.defaultdict(int),
            accumulated_losses=[]
        )

    def record_batch(self, target_domains, losses):
        self.accumulated_losses.extend(zip(target_domains, losses))
        for domain, loss in self.accumulated_losses:
            self.domain_losses[domain] += loss
            self.domain_counts[domain] += 1
            self.cur_domain_losses[domain] += loss
            self.cur_domain_counts[domain] += 1

    def build_cur_step_tensor(self):
        cur_step_tensor = torch.zeros(len(self.keys), device=self.weights.device)
        for idx, domain in enumerate(self.keys):
            if domain in self.domain_losses:
                avg_loss = self.cur_domain_losses[domain] / (self.cur_domain_counts[domain] + 1e-16)
                cur_step_tensor[idx] = avg_loss
        return cur_step_tensor

    def maybe_update_weights(self, step_idx, eta, dro_temperature=1.0):
        if step_idx % 50 != 49:
            return
        for idx, domain in enumerate(self.keys):
            if domain in self.domain_losses:
                avg_loss = self.domain_losses[domain] / (self.domain_counts[domain] + 1e-16)
                self.loss_list[idx] = avg_loss
        update_factor = eta * (self.loss_list.detach() / dro_temperature)
        self.weights = self.weights * torch.exp(update_factor)
        self.weights = self.weights / torch.sum(self.weights)
        self.domain_losses.clear()
        self.domain_counts.clear()
        self.loss_list.zero_()
        torch.cuda.empty_cache()

    def weighted_loss(self, cur_step_tensor):
        return torch.sum(self.weights.detach() * cur_step_tensor)

    def reset_after_step(self):
        self.cur_domain_losses.clear()
        self.cur_domain_counts.clear()
        self.accumulated_losses.clear()



@dataclass
class EvaluationState:
    predict_scores: List[Any]
    labels: List[torch.Tensor]
    example_indices: List[torch.Tensor]
    generated_text: Dict[int, Any]
    seq_cls: List[torch.Tensor]
    domains: List[torch.Tensor]

    @classmethod
    def create(cls):
        return cls(
            predict_scores=[],
            labels=[],
            example_indices=[],
            generated_text=collections.defaultdict(int),
            seq_cls=[],
            domains=[]
        )

    def append_batch(self, scores, seq_cls, batch_data):
        self.predict_scores.append(scores)
        self.labels.append(batch_data['target_position'])
        self.example_indices.append(batch_data['example_index'])
        if 'target_domain' in batch_data:
            self.domains.append(batch_data['target_domain'].detach())
        self.seq_cls.append(seq_cls.detach())

    def stacked_labels(self):
        return torch.cat(self.labels, dim=0) if self.labels else torch.tensor([])

    def stacked_example_indices(self):
        return torch.cat(self.example_indices, dim=0) if self.example_indices else torch.tensor([])


class Trainer(TrainerBase):
    def __init__(self, args, tokenizer, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        # Initialize HUM Logger
        if LOGGER_AVAILABLE:
            self.hum_logger = init_logger(rank=args.rank, log_dir='logs')
            self.eval_logger = EvaluationLogger(rank=args.rank, log_file=f'logs/evaluation_results_rank{args.rank}.log')
        else:
            self.hum_logger = None
            self.eval_logger = None

        self.model = HUM(args, tokenizer)
        self.tokenizer = tokenizer

        # GPU Options
        print_rank0(f'Model Launching at GPU {self.args.gpu}', self.args.rank)
        self.model = self.model.to(args.gpu)
        self.device = next(self.model.parameters()).device
        
        # Load item/domain info if path exists
        try:
            self.id_to_domain = pickle.load(open(f'{args.data_path}/{args.dataset}/inverted_iid.pkl', 'rb'))
            self.domain_to_id = pickle.load(open(f'{args.data_path}/{args.dataset}/single_domain_iid.pkl', 'rb'))
        except Exception:
            self.id_to_domain = {}
            self.domain_to_id = {}

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True)
        if args.load:
            self.load(args.load)

        self.loss_names = ['cross_entropy_loss', 'generate_loss', 'Kl_loss', 'Tail_loss', 'total_loss']
        self.best_valid_result = 0
        self.early_stop_step = 0
        self.print_trainable_parameters()
        self.start_epoch = getattr(args, 'start_epoch', 0)

        self._step_counter = 0

        # Record Trainer initialization complete
        if self.hum_logger:
            self.hum_logger.log_system_info(f"HUM Trainer initialized on GPU {self.args.gpu}")


    def train(self):
        if self.args.distributed:
            dist.barrier()

        global_step = 1
        scaler = GradScaler()
        result = {'exit': False}
        domain_keys = list(self.domain_to_id.keys())

        for epoch_idx in range(self.args.epoch):
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch_idx)
            epoch = epoch_idx + self.start_epoch if self.start_epoch != 0 else epoch_idx
            epoch_start_time = time.time()
            
            self.model.train()
            loss_meters = [LossMeter(100) for _ in range(len(self.loss_names))]
            loader_length = len(self.train_loader)
            logger_batch = (loader_length // 100) + 1
            eta = 0.001
            domain_state = DomainTrackingState.initialize(domain_keys, self.device)
            id_to_domain = self.id_to_domain  

            for step_i, batch in enumerate(tqdm(self.train_loader, desc='Training')):
                self.transfer_device(batch)
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.fp16):
                    losses = self.model(batch)

                # losses: [loss_contrast (none), generative_loss, kl_loss, tail_loss, raw_loss_per_sample]
                primary_losses = losses[0]
                
                target_ids = batch['target_iid'].tolist()
                target_domains = [id_to_domain[tid] for tid in target_ids] 
                domain_state.record_batch(target_domains, primary_losses)

                cur_step_domain_loss_list = None
                if (step_i + 1) % self.args.gradient_accumulation_steps == 0:
                    global_step += 1
                    cur_step_domain_loss_list = domain_state.build_cur_step_tensor()

                domain_state.maybe_update_weights(step_i, eta, self.args.dro_temperature)

                if cur_step_domain_loss_list is not None:
                    loss_weightsum = domain_state.weighted_loss(cur_step_domain_loss_list)
                    total_loss = loss_weightsum + losses[1] + losses[2] + losses[3]
                    
                    self._perform_optimization_step(scaler, total_loss)
                    domain_state.reset_after_step()

                # Update loss meters for logging
                if 'total_loss' in locals():
                    self._update_loss_meters(loss_meters, [primary_losses.mean(), losses[1], losses[2], losses[3], total_loss])
                else:
                    self._update_loss_meters(loss_meters, [primary_losses.mean(), losses[1], losses[2], losses[3], torch.tensor(0.0)])

                if step_i % self.args.gradient_accumulation_steps == 0:
                    self._log_training_step(epoch, step_i, global_step, loader_length,
                                            loss_meters, epoch_start_time, logger_batch)

                if (self.args.save_by_step != 0 and
                        global_step % self.args.save_by_step == 0 and
                        step_i % self.args.gradient_accumulation_steps == 0):
                    result = self._validate_and_maybe_save(global_step)

            if self.args.distributed:
                dist.barrier()

            if self.args.save_by_step == 0:
                result = self.valid_epoch(epoch)
                if result['save'] and self.args.rank == 0:
                    self.save(self.args.output.format('valid_best'))

            if self.args.distributed:
                dist.barrier()

            if result['exit']:
                self.load(self.args.output.format('valid_best'))
                self.valid_epoch('Test', mode='test')
                return
            
    def _update_loss_meters(self, loss_meters, losses):
        for i, _ in enumerate(losses):
            if i == 0 or i == len(losses) - 1:
                value = losses[i].mean().item()
            else:
                value = losses[i].item()
            loss_meters[i].update(value)

    def _log_training_step(self, epoch, step_i, global_step, loader_length,
                           loss_meters, epoch_start_time, logger_batch):
        remain_year, remain_min, remain_sec = self.remain_time(epoch_start_time, step_i, loader_length)
        log_str = (f"Global Step:{global_step} | Train Epoch {epoch} | Step:{step_i} / {loader_length} | "
                   f"Remain Time:{remain_year}:{remain_min}:{remain_sec} | ")
        for i in range(len(loss_meters)):
            log_str += f'{self.loss_names[i]}:{loss_meters[i].val:.3f} | '
        if (step_i % logger_batch) == 0:
            if self.hum_logger:
                metrics = {self.loss_names[i]: loss_meters[i].val for i in range(len(loss_meters))}
                self.hum_logger.log_training_step(epoch, step_i, loader_length,
                                                   loss_meters[0].val, **metrics)
            else:
                print_rank0(log_str, self.args.rank)

    def _perform_optimization_step(self, scaler, loss):
        scaler.scale(loss).backward()
        with autocast(device_type='cuda', dtype=torch.float32, enabled=self.args.fp16):
            if self.args.clip_grad_norm > 0:
                scaler.unscale_(optimizer=self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
        scaler.step(self.optim)
        scaler.update()
        self.optim.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _validate_and_maybe_save(self, global_step):
        result = self.valid_epoch(global_step)
        torch.cuda.empty_cache()
        if result['save']:
            if self.hum_logger:
                self.hum_logger.log_model_save(self.args.output.format("valid_best"), global_step)
            else:
                print_rank0(f"Save model at global step at {global_step}", self.args.rank)
            self.save(self.args.output.format("valid_best"))
        self.model.train()
        if self.args.distributed:
            dist.barrier()
        return result
    
    def _generate_item_embeddings(self, dataloader):
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.fp16):
            item_tokens = dataloader.dataset.get_items_tokens()
            if self.args.distributed:
                self.model.module.generate_embs_hum(item_tokens)
            else:
                self.model.generate_embs_hum(item_tokens)

    def _collect_validation_batches(self, dataloader, logger_batch):
        state = EvaluationState.create()
        loader_length = len(dataloader)
        for batch_idx, batch_data in enumerate(dataloader):
            self.transfer_device(batch_data)
            if (batch_idx % logger_batch) == 0:
                if self.hum_logger:
                    self.hum_logger.log_system_info(
                        f"Local Rank{self.args.rank}-Evaluation:{batch_idx}/{loader_length}")
                else:
                    print_rank0(f"Local Rank{self.args.rank}-Evaluation:{batch_idx}/{loader_length}",
                                self.args.rank)
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.fp16):
                if self.args.distributed:
                    scores, generated_text, _, seq_cls = self.model.module.valid_step(batch_data)
                else:
                    scores, generated_text, _, seq_cls = self.model.valid_step(batch_data)
            state.append_batch(scores, seq_cls, batch_data)
            torch.cuda.empty_cache()
        label = state.stacked_labels()
        example_index = state.stacked_example_indices()
        return state, label, example_index

    def _contains_negative(self, tensor_list):
        if len(tensor_list) == 0:
            return False
        for tensor in tensor_list:
            if (tensor < 0).any():
                return True
        return False

    def _pad_predict_scores(self, predict_score, example_index):
        if len(predict_score) == 0:
            return predict_score
        max_dim1_size = max(tensor.size(1) for tensor in predict_score)
        predict_score = [
            torch.cat([tensor.cpu(),
                       torch.zeros(tensor.size(0), max_dim1_size - tensor.size(1), device='cpu')], dim=1)
            if tensor.size(1) < max_dim1_size else tensor[:, :max_dim1_size].cpu()
            for tensor in predict_score
        ]
        return torch.cat(predict_score, dim=0).to(example_index.device)

    def _handle_test_mode(self, predict_score, label, example_index, dataloader, eval_state):
        rank = self.args.rank

        def gather_tensor_list(tensor_list):
            if not tensor_list: return torch.tensor([], device=self.device)
            local_tensor = torch.cat(tensor_list, dim=0)
            local_size = torch.tensor([local_tensor.size(0)], device=local_tensor.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(all_sizes, local_size)
            max_size = max(int(size.item()) for size in all_sizes)
            padded = torch.zeros(max_size, *local_tensor.shape[1:], dtype=local_tensor.dtype,
                                 device=local_tensor.device)
            padded[:local_tensor.size(0)] = local_tensor
            gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, padded)
            result = []
            for g, size in zip(gathered, all_sizes):
                result.append(g[:size])
            return torch.cat(result, dim=0)

        if self.args.distributed:
            dist.barrier()
            gathered_seq_cls = gather_tensor_list(eval_state.seq_cls)
            gathered_domains = gather_tensor_list(eval_state.domains)

            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = os.path.basename(self.args.output).split("-", 1)[-1].replace(".pth", "")
            save_path = os.path.join(f"others/user_rep/{time_str}_{suffix}_evaluation_outputs.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if rank == 0:
                torch.save({
                    "seq_cls": gathered_seq_cls,
                    "domain": gathered_domains
                }, save_path)

        if self.args.flag_special_test is None:
            payload = {
                'predict_score': predict_score.cpu(),
                'label': label.cpu(),
                'example_index': example_index.cpu()
            }
            torch.save(payload, f'predict_result_rank{rank}.pt')

        if self.args.distributed:
            dist.barrier()

        all_predict_score = []
        all_label = []
        all_example_index = []

        for gpu_rank in range(self.args.num_gpus):
            data = torch.load(f'predict_result_rank{gpu_rank}.pt')
            all_predict_score.append(data['predict_score'])
            all_label.append(data['label'])
            all_example_index.append(data['example_index'])

        predict_score, label = self.clean_dist_duplicate(
            all_predict_score, all_label, all_example_index)

        self._run_test_reporting(dataloader, predict_score, label)
        return predict_score, label

    def _run_test_reporting(self, dataloader, predict_score, label):
        rank = self.args.rank
        predict_score_for_analysis = predict_score

        if self.eval_logger:
            self.eval_logger.log_section_header("MAIN RESULTS")
            id2domain = self.build_id2domain_test(dataloader)
            self.eval_logger.evaluate_by_domain(predict_score_for_analysis, label, id2domain, "Main")
            self.eval_logger.log_section_footer()

            self.eval_logger.log_summary_table()
            self.eval_logger.log_detailed_summary_table()
        else:
            print_rank0("Overall Results", rank)
            id2domain = self.build_id2domain_test(dataloader)
            self.evaluate_by_domain(predict_score, label, id2domain, rank)

    def _handle_valid_mode(self, predict_score, label, example_index):
        if self.args.distributed:
            all_predict_score = [torch.zeros_like(predict_score) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_predict_score, predict_score.contiguous())

            all_label = [torch.zeros_like(label) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_label, label.contiguous())
            all_example_index = [torch.zeros_like(example_index) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_example_index, example_index.contiguous())
            predict_score, label = self.clean_dist_duplicate(all_predict_score, all_label, all_example_index)
        return predict_score, label

    def _finalize_validation(self, epoch, predict_score, label):
        final_score = predict_score
        recall = cal_recall(label.cpu(), final_score.cpu(), [1, 2, 5, 10])
        recall_1 = recall[-1]
        if self.hum_logger:
            self.hum_logger.log_system_info(f'Overall Recall:{recall}')
        else:
            print_rank0(f'Overall Recall:{recall}', self.args.rank)

        if self.hum_logger:
            validation_results = {
                "Recall@1": recall[0],
                "Recall@2": recall[1],
                "Recall@5": recall[2],
                "Recall@10": recall[3]
            }
            self.hum_logger.log_validation_end(epoch, validation_results)

        if recall_1 > self.best_valid_result:
            self.early_stop_step = 0
            self.best_valid_result = recall_1
        else:
            self.early_stop_step += 1
        torch.cuda.empty_cache()

        if self.hum_logger:
            self.hum_logger.log_system_info(
                f"Epoch: {epoch}, cuda: {self.args.rank}, early_stop_step: {self.early_stop_step}")
        else:
            print(f"Epoch: {epoch}, cuda: {self.args.rank}, self.early_stop_step: {self.early_stop_step}")

        if self.early_stop_step > self.args.early_stop_step_num:
            return {'save': False, 'exit': True, 'result': [recall_1]}
        elif self.early_stop_step > 0:
            return {'save': False, 'exit': False, 'result': [recall_1]}
        else:
            return {'save': True, 'exit': False, 'result': [recall_1]}
            
    def build_id2domain_test(self, dataloader, filter_fn=None):
        """
        Build a mapping from domain to list of sample indices (id2domain_test).
        If filter_fn is provided, include only samples where filter_fn(idx, sample) returns True.

        Args:
            dataloader: DataLoader containing dataset with .data attribute.
            filter_fn: callable(idx, sample) -> bool, optional filter function.

        Returns:
            id2domain_test: defaultdict(list) mapping each domain to indices.
        """
        id2domain_test = collections.defaultdict(list)
        for idx, sample in enumerate(dataloader.dataset.data):
            if filter_fn is None or filter_fn(idx, sample):
                domain = sample[-1]
                id2domain_test[domain].append(idx)
        return id2domain_test

    def evaluate_by_domain(self, predict_score, label, id2domain_test, rank=0):
        """
        Perform domain-wise evaluation on the passed id2domain_test.
        """

        for domain_key, indices in id2domain_test.items():
            domain_scores = predict_score[indices].cpu()
            domain_label = label[indices].cpu()

            recall = cal_recall(domain_label, domain_scores, [1, 5, 10, 20])
            ndcg = cal_ndcg(domain_label, domain_scores, [1, 5, 10, 20])

            # Use logger to record domain results
            if self.hum_logger:
                self.hum_logger.log_domain_results(domain_key, len(domain_label), {
                    "Recall@5": recall[1],
                    "Recall@10": recall[2], 
                    "NDCG@5": ndcg[1],
                    "NDCG@10": ndcg[2]
                })
            else:
                print_rank0(f"-----------------{domain_key}-----------------", rank)
                if self.hum_logger:
                    self.hum_logger.log_data_info(f"There are {len(domain_label)} samples in the domain of {domain_key}")
                else:
                    print(f"There are {len(domain_label)} samples in the domain of {domain_key}")
                print_rank0(f"Recall@5:{recall[1]} ---- Recall@10:{recall[2]}", rank)
                print_rank0(f"NDCG@5:{ndcg[1]} ---- NDCG@10:{ndcg[2]}", rank)


    @torch.no_grad()
    def valid_epoch(self, epoch, mode='valid'):
        dataloader = self.val_loader if mode == 'valid' else self.test_loader
        self.model.eval()
        
        if self.hum_logger:
            self.hum_logger.log_validation_start(epoch)

        self._generate_item_embeddings(dataloader)
       
        loader_length = len(dataloader)
        logger_batch = (loader_length // 10) + 1

        eval_state, label, example_index = self._collect_validation_batches(dataloader, logger_batch)
        predict_score = eval_state.predict_scores

        _ = self._contains_negative(predict_score)
        torch.cuda.empty_cache()

        predict_score = self._pad_predict_scores(predict_score, example_index)

        if mode == 'test':
            predict_score, label = self._handle_test_mode(
                predict_score, label, example_index, dataloader, eval_state)
        else:
            predict_score, label = self._handle_valid_mode(
                predict_score, label, example_index)
        
        return self._finalize_validation(epoch, predict_score, label)

    def clean_dist_duplicate(self, all_predict_score, all_label, all_example_index):
        # Original single score path
        all_predict_score = torch.concat(all_predict_score, dim=0).cpu()
        all_label = torch.concat(all_label, dim=0).cpu()

        predict_score = torch.zeros_like(all_predict_score)
        label = torch.zeros_like(all_label)
        example_index = torch.concat(all_example_index, dim=0).cpu()

        predict_score[example_index] = all_predict_score
        label[example_index] = all_label
        exp_cnt = max(example_index) + 1

        return predict_score[:exp_cnt], label[:exp_cnt]


    def transfer_device(self, data):
        device = next(self.model.parameters()).device
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

    def save(self, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        saved_parameters = {}
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for param_name, param in model_generator:
            if param.requires_grad:
                saved_parameters[param_name] = param
        torch.save(saved_parameters, path)
        
        # Record model save
        if self.hum_logger:
            self.hum_logger.log_model_save(path, getattr(self, 'current_epoch', 0))

    def load(self, path, loc=None):
        weights = torch.load(path, map_location=next(self.model.parameters()).device)
        if self.args.distributed:
            load_result = self.model.module.load_state_dict(weights, strict=False)
            if self.hum_logger:
                self.hum_logger.log_model_load(path)
            else:
                print_rank0(load_result, self.args.rank)
        else:
            load_result = self.model.load_state_dict(weights, strict=False)
            if self.hum_logger:
                self.hum_logger.log_model_load(path)
            else:
                print_rank0(load_result, self.args.rank)



    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for _, param in model_generator:
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if self.hum_logger:
            self.hum_logger.log_model_info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        else:
            print_rank0(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}", self.args.rank
            )