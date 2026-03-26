import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_optimizations():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.75"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, timeout=timedelta(hours=3))

def cleanup_distributed():
    dist.destroy_process_group()

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_batch = {k: v.to(self.device, non_blocking=True) 
                               for k, v in self.next_batch.items()}
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            self.preload()
        return batch

def get_calibration_dataloader(tokenizer, dataset_path, num_samples=128, seq_len=512, batch_size=4):
    logger.info(f"Loading calibration dataset from {dataset_path}")
    
    if os.path.isdir(dataset_path):
        data_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.json')]
        dataset = load_dataset('json', data_files=data_files, split='train')
    else:
        dataset = load_dataset(dataset_path, split='train')
    
    text_field = 'text'
    for field in ['text', 'content', 'sentence', 'paragraph']:
        if field in dataset.features:
            text_field = field
            break
    
    initial_samples = min(num_samples * 20, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(initial_samples))
    
    def tokenize_function(examples):
        return tokenizer(examples[text_field])
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=list(dataset.features),
        desc="Tokenizing"
    )
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc="Grouping texts"
    )
    
    if len(lm_dataset) > num_samples:
        lm_dataset = lm_dataset.select(range(num_samples))
        
    logger.info(f"Final calibration dataset size: {len(lm_dataset)} samples")

    return DataLoader(
        lm_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

class FinalPrunedMoE(nn.Module):
    def __init__(self, config, gate, experts, shared_experts):
        super().__init__()
        self.config = config
        self.gate = gate
        self.experts = experts
        self.shared_experts = shared_experts
        
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = len(experts)
        self.routed_scaling_factor = config.routed_scaling_factor
        
        if self.num_experts_per_tok > self.n_routed_experts:
            self.num_experts_per_tok = self.n_routed_experts

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            if isinstance(shared_output, torch.Tensor) and shared_output.dim() == 3:
                shared_output = shared_output.view(-1, hidden_dim)

        if hidden_states.dtype != self.gate.weight.dtype:
            router_input = hidden_states.to(self.gate.weight.dtype)
        else:
            router_input = hidden_states
            
        router_logits = self.gate(router_input)
        if router_logits.dim() == 3:
            router_logits = router_logits.view(-1, self.n_routed_experts)

        if self.config.scoring_func == 'softmax':
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        else:
            routing_weights = router_logits

        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.config.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_routed_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.n_routed_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
            
            current_state = hidden_states_flat[None, top_x.tolist()].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x.tolist(), idx.tolist(), None]
            current_hidden_states = current_hidden_states * self.routed_scaling_factor
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        if shared_output is not None:
            if shared_output.dim() == 2:
                shared_output = shared_output.view(batch_size, sequence_length, hidden_dim)
            final_hidden_states = final_hidden_states + shared_output
            
        return final_hidden_states

class ESNAccumulator:
    def __init__(self, num_experts, hidden_dim, device):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.sum_xxT = [torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device='cpu') for _ in range(num_experts)]
        self.sum_x = [torch.zeros(hidden_dim, dtype=torch.float64, device='cpu') for _ in range(num_experts)]
        self.expert_energy = torch.zeros(num_experts, dtype=torch.float64, device='cpu')
        
        self.counts = [0 for _ in range(num_experts)]

    def update(self, expert_idx, hidden_states):
        if hidden_states.shape[0] == 0: return
        h = hidden_states.detach().to(torch.float64).cpu()
        
        energy_sum = torch.norm(h, p=2, dim=-1).sum()
        self.expert_energy[expert_idx] += energy_sum
        
        h_norm = F.normalize(h, p=2, dim=-1)
        
        term = torch.matmul(h_norm.t(), h_norm)
        self.sum_xxT[expert_idx] += term
        
        self.sum_x[expert_idx] += h_norm.sum(dim=0)
        self.counts[expert_idx] += hidden_states.shape[0]

    def compute_scores(self, pca_rank, beta=0.0):
        logger.info(f"Computing Hybrid Scores (PCA Rank={pca_rank}, Beta={beta})")
        
        expert_bases = {}
        valid_indices = []
        
        for i in range(self.num_experts):
            if self.counts[i] < pca_rank: continue 
            
            N = self.counts[i]
            expected_xxT = self.sum_xxT[i] / N
            mean_vec = self.sum_x[i] / N
            true_cov = expected_xxT - torch.outer(mean_vec, mean_vec)
            
            try:
                vals, vecs = torch.linalg.eigh(true_cov)
            except RuntimeError:
                continue
            
            basis = vecs[:, -pca_rank:] 
            expert_bases[i] = basis.to(self.device, dtype=torch.float32)
            valid_indices.append(i)
            
        esn_raw_scores = {}
        
        for e_target in tqdm(valid_indices, desc="Calculating Subspace Novelty"):
            U_target = expert_bases[e_target]
            others = [expert_bases[j] for j in valid_indices if j != e_target]
            if not others:
                esn_raw_scores[e_target] = 1.0
                continue
            
            U_others = torch.cat(others, dim=1)
            Q_others, _ = torch.linalg.qr(U_others, mode='reduced')
            
            projection = torch.matmul(Q_others.t(), U_target)
            energy = torch.norm(projection, p='fro') ** 2
            
            P_le = energy.item() / pca_rank
            P_le = min(max(P_le, 0.0), 1.0)
            esn_raw_scores[e_target] = 1.0 - P_le
            
        final_esn_list = []
        final_energy_list = []
        
        total_energies = self.expert_energy.numpy()
        
        for i in range(self.num_experts):
            s_esn = esn_raw_scores.get(i, 0.0)
            final_esn_list.append(s_esn)
            final_energy_list.append(total_energies[i])
            
        rank_esn = np.argsort(np.argsort(np.array(final_esn_list)))
        rank_energy = np.argsort(np.argsort(np.array(final_energy_list)))
        
        norm_rank_esn = rank_esn / (self.num_experts - 1 + 1e-6)
        norm_rank_energy = rank_energy / (self.num_experts - 1 + 1e-6)
        
        final_scores = []
        for i in range(self.num_experts):
            hybrid_score = (1 - beta) * norm_rank_esn[i] + beta * norm_rank_energy[i]
            final_scores.append(hybrid_score)
                
        return np.array(final_scores)

def stage2_esn_pruning(model, tokenizer, retention_budget, moe_layers, args, device):
    logger.info("Stage 2: Weighted Expert Subspace Novelty (ESN) Pruning")
    
    dataloader = get_calibration_dataloader(
        tokenizer, args.dataset_path, 
        num_samples=args.num_calib_samples, 
        batch_size=args.batch_size
    )
    
    def get_weighted_moe_hook(accumulator, config):
        def hook(module, input, output):
            try:
                x = input[0].detach()
                
                if x.dim() == 2: x_3d = x.unsqueeze(1)
                else: x_3d = x

                gate_input = x_3d.to(module.gate.weight.dtype)
                
                with torch.no_grad():
                    topk_indices, topk_weights, _ = module.gate(gate_input)
                
                x_flat = x.view(-1, x.shape[-1])
                topk_indices = topk_indices.view(-1, topk_indices.shape[-1])
                topk_weights = topk_weights.view(-1, topk_weights.shape[-1])
                
                topk_weights = topk_weights.to(x.dtype)
                
                unique_experts = torch.unique(topk_indices)
                for e_idx in unique_experts:
                    e_idx = e_idx.item()
                    mask = (topk_indices == e_idx)
                    row_indices, col_indices = torch.where(mask)
                    
                    if row_indices.shape[0] == 0: continue
                    
                    expert_input = x_flat[row_indices]
                    weights = topk_weights[row_indices, col_indices].unsqueeze(1)
                    expert_layer = module.experts[e_idx]
                    
                    with torch.no_grad():
                        h_raw = expert_layer(expert_input)
                        h_weighted = h_raw * weights
                    
                    accumulator.update(e_idx, h_weighted)
                    
            except Exception as e:
                logger.error(f"Error in hook: {e}")
                raise e
        return hook

    total_layers = len(moe_layers)
    for i, layer_idx in enumerate(moe_layers):
        logger.info(f"[{i+1}/{total_layers}] Processing Layer {layer_idx}")
        
        num_experts = model.config.n_routed_experts
        acc = ESNAccumulator(num_experts, model.config.hidden_size, device)
        
        layer = model.model.layers[layer_idx].mlp
        handle = layer.register_forward_hook(get_weighted_moe_hook(acc, model.config))
        
        prefetcher = DataPrefetcher(dataloader, device)
        batch = prefetcher.next()
        model.eval()
        
        with torch.no_grad(), autocast():
            pbar = tqdm(total=len(dataloader), desc=f"Calibrating Layer {layer_idx}", leave=False)
            while batch is not None:
                if 'labels' in batch: batch.pop('labels')
                model(**batch)
                batch = prefetcher.next()
                pbar.update(1)
            pbar.close()
        
        handle.remove()
        
        target_count = retention_budget[layer_idx]
        
        scores = acc.compute_scores(pca_rank=args.pca_rank, beta=args.beta)
        
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = sorted(sorted_indices[:target_count].tolist())
        
        logger.info(f"Pruning Layer {layer_idx}: Keeping {len(keep_indices)} experts")
        
        original_mlp = model.model.layers[layer_idx].mlp
        old_gate_weight = original_mlp.gate.weight.data
        
        new_gate = nn.Linear(model.config.hidden_size, len(keep_indices), bias=False, device=device, dtype=old_gate_weight.dtype)
        new_gate.weight.data = old_gate_weight[keep_indices].clone()
        
        new_experts = nn.ModuleList([original_mlp.experts[i] for i in keep_indices])
        
        final_moe = FinalPrunedMoE(model.config, new_gate, new_experts, original_mlp.shared_experts)
        model.model.layers[layer_idx].mlp = final_moe
        
        del acc
        torch.cuda.empty_cache()
        import gc
        gc.collect() 
        
    logger.info("Weighted ESN Pruning Finished")

class RankAccumulator:
    def __init__(self, hidden_dim):
        self.n_samples = 0
        self.sum_x = torch.zeros(hidden_dim, dtype=torch.float64)
        self.sum_xxT = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64)
    
    def update(self, x):
        x = x.to(torch.float64)
        self.n_samples += x.shape[0]
        self.sum_x += x.sum(dim=0).cpu()
        batch_xxT = torch.matmul(x.t(), x)
        self.sum_xxT += batch_xxT.cpu()

    def compute_effective_rank(self):
        if self.n_samples == 0: return 0.0
        mean = self.sum_x / self.n_samples
        expected_xxT = self.sum_xxT / self.n_samples
        covariance_matrix = expected_xxT - torch.outer(mean, mean)
        tr_C = torch.trace(covariance_matrix)
        tr_C2 = torch.sum(covariance_matrix ** 2)
        if tr_C2 == 0: return 0.0
        r_l = (tr_C ** 2) / tr_C2
        return r_l.item()

def stage1_rank_based_budget_allocation(model, tokenizer, args, device):
    logger.info(f"Stage 1: Budget Allocation (Alpha={args.alpha})")
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    max_limit = model.config.n_routed_experts
    top_k = model.config.num_experts_per_tok
    min_limit = max(args.min_experts_per_layer, top_k)
    
    first_k_dense = model.config.first_k_dense_replace
    moe_layers = list(range(first_k_dense, num_layers))
    
    accumulators = {l: RankAccumulator(hidden_dim) for l in moe_layers}
    
    def get_rank_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple): h = output[0]
            else: h = output
            h = h.detach()
            if h.dim() == 3: h = h.view(-1, h.shape[-1])
            accumulators[layer_idx].update(h)
        return hook

    hooks = []
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'mlp'):
            h = layer.mlp.register_forward_hook(get_rank_hook(layer_idx))
            hooks.append(h)
            
    dataloader = get_calibration_dataloader(tokenizer, args.dataset_path, num_samples=args.num_calib_samples, batch_size=args.batch_size)
    model.eval()
    prefetcher = DataPrefetcher(dataloader, device)
    batch = prefetcher.next()
    
    with torch.no_grad(), autocast():
        pbar = tqdm(desc="Rank Calibration", total=len(dataloader))
        while batch is not None:
            if 'labels' in batch: batch.pop('labels')
            model(**batch)
            batch = prefetcher.next()
            pbar.update(1)
        pbar.close()
    
    for h in hooks: h.remove()
    
    layer_scores = {}
    for layer_idx in moe_layers:
        r_l = accumulators[layer_idx].compute_effective_rank()
        s_l = r_l ** args.alpha
        layer_scores[layer_idx] = s_l
        logger.info(f"Layer {layer_idx}: Rank={r_l:.2f}, Score={s_l:.2f}")
    
    total_moe_experts = len(moe_layers) * max_limit
    target_global_budget = int(total_moe_experts * args.global_retention_ratio)
    
    min_total_required = len(moe_layers) * min_limit
    
    if target_global_budget < min_total_required:
        target_global_budget = min_total_required
        
    retention_budget = {l: min_limit for l in moe_layers}
    current_total = sum(retention_budget.values())
    remaining_budget = target_global_budget - current_total
    
    score_values = np.array([layer_scores[l] for l in moe_layers])
    total_score = score_values.sum()
    norm_scores = score_values / total_score if total_score > 0 else np.ones_like(score_values)/len(score_values)
    
    if remaining_budget > 0:
        for i, layer_idx in enumerate(moe_layers):
            extra = int(norm_scores[i] * remaining_budget)
            space = max_limit - retention_budget[layer_idx]
            retention_budget[layer_idx] += min(extra, space)
            
    diff = target_global_budget - sum(retention_budget.values())
    sorted_indices = np.argsort(score_values)
    
    while diff > 0:
        added = False
        for i in range(len(moe_layers) - 1, -1, -1):
            idx = sorted_indices[i]
            l = moe_layers[idx]
            if retention_budget[l] < max_limit:
                retention_budget[l] += 1
                diff -= 1
                added = True
                if diff == 0: break
        if not added: break
        
    while diff < 0:
        removed = False
        for i in range(len(moe_layers)):
            idx = sorted_indices[i]
            l = moe_layers[idx]
            if retention_budget[l] > min_limit:
                retention_budget[l] -= 1
                diff += 1
                removed = True
                if diff == 0: break
        if not removed: break
        
    return retention_budget, moe_layers

def worker_main(rank, world_size, args):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        logger.info(f"Starting Pruning with {world_size} GPUs")
        logger.info(f"Ratio: {args.global_retention_ratio}, Alpha: {args.alpha}, Beta: {args.beta}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
        low_cpu_mem_usage=True,
    )
    
    retention_budget = None
    moe_layers = None
    
    if rank == 0:
        retention_budget, moe_layers = stage1_rank_based_budget_allocation(model, tokenizer, args, device)
    
    budget_list = [retention_budget]
    layers_list = [moe_layers]
    dist.broadcast_object_list(budget_list, src=0)
    dist.broadcast_object_list(layers_list, src=0)
    
    retention_budget = budget_list[0]
    moe_layers = layers_list[0]
    
    torch.cuda.empty_cache()
    dist.barrier()
    
    if rank == 0:
        stage2_esn_pruning(model, tokenizer, retention_budget, moe_layers, args, device)
        
        logger.info(f"Saving pruned model to {args.output_path}")
        avg_experts = int(sum(retention_budget.values()) / len(retention_budget))
        model.config.n_routed_experts = avg_experts
        
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        
        with open(os.path.join(args.output_path, 'pruning_config.json'), 'w') as f:
            json.dump({
                'budget': {k: int(v) for k,v in retention_budget.items()},
                'args': vars(args)
            }, f, indent=2)
            
    dist.barrier()
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--global_retention_ratio', type=float, default=0.5)
    parser.add_argument('--min_experts_per_layer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--pca_rank', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--num_calib_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=8)
    
    args = parser.parse_args()
    setup_optimizations()
    mp.spawn(worker_main, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)

if __name__ == '__main__':
    main()