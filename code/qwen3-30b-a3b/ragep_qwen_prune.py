import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import gc
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalPrunedQwenMoE(nn.Module):
    def __init__(self, config, gate, experts, shared_expert, shared_expert_gate=None):
        super().__init__()
        self.config = config
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = len(experts)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        batch_size, seq_len, hidden_dim = orig_shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            if self.shared_expert_gate is not None:
                gate_out = self.shared_expert_gate(hidden_states_flat)
                shared_output = shared_output * torch.sigmoid(gate_out)

        router_input = hidden_states.to(self.gate.weight.dtype)
        router_logits = self.gate(router_input)
        if router_logits.dim() == 3:
            router_logits = router_logits.view(-1, self.num_experts)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        
        topk_weights = topk_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0: continue
            
            current_state = hidden_states_flat[top_x]
            expert_out = self.experts[expert_idx](current_state)
            current_weights = topk_weights[top_x, idx, None]
            expert_out = expert_out * current_weights
            final_hidden_states.index_add_(0, top_x, expert_out.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        if shared_output is not None:
            shared_output = shared_output.view(batch_size, seq_len, hidden_dim)
            final_hidden_states = final_hidden_states + shared_output
            
        return final_hidden_states

class RankAccumulator:
    def __init__(self, hidden_dim, device):
        self.device = device
        self.n_samples = 0
        self.sum_x = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
        self.sum_xxT = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device=device)
    
    def update(self, x):
        x = x.to(dtype=torch.float64) 
        self.n_samples += x.shape[0]
        self.sum_x += x.sum(dim=0)
        batch_xxT = torch.matmul(x.t(), x)
        self.sum_xxT += batch_xxT

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

class ESNAccumulator:
    def __init__(self, num_experts, hidden_dim, device):
        self.num_experts = num_experts
        self.device = device
        self.sum_xxT = [torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device=device) for _ in range(num_experts)]
        self.sum_x = [torch.zeros(hidden_dim, dtype=torch.float64, device=device) for _ in range(num_experts)]
        self.expert_energy = torch.zeros(num_experts, dtype=torch.float64, device=device)
        self.counts = [0 for _ in range(num_experts)]

    def update(self, expert_idx, hidden_states):
        if hidden_states.shape[0] == 0: return
        h = hidden_states.detach().to(dtype=torch.float64) 
        
        energy_sum = torch.norm(h, p=2, dim=-1).sum()
        self.expert_energy[expert_idx] += energy_sum
        
        h_norm = F.normalize(h, p=2, dim=-1)
        term = torch.matmul(h_norm.t(), h_norm)
        self.sum_xxT[expert_idx] += term
        self.sum_x[expert_idx] += h_norm.sum(dim=0)
        self.counts[expert_idx] += hidden_states.shape[0]

    def compute_scores(self, pca_rank, beta=0.0):
        logger.info(f"Computing scores on {self.device}")
        expert_bases = {}
        valid_indices = []
        
        for i in range(self.num_experts):
            actual_count = self.counts[i]
            if actual_count < 2: continue
            
            current_rank = min(pca_rank, actual_count - 1)
            
            N = actual_count
            expected_xxT = self.sum_xxT[i] / N
            mean_vec = self.sum_x[i] / N
            true_cov = expected_xxT - torch.outer(mean_vec, mean_vec)
            
            try:
                vals, vecs = torch.linalg.eigh(true_cov)
                basis = vecs[:, -current_rank:]
                expert_bases[i] = basis.to(dtype=torch.float32)
                valid_indices.append(i)
            except RuntimeError:
                continue
            
        esn_raw_scores = {}
        if len(valid_indices) > 0:
            for e_target in tqdm(valid_indices, desc="Calculating Spectral Novelty", leave=False):
                U_target = expert_bases[e_target]
                others = [expert_bases[j] for j in valid_indices if j != e_target]
                if not others:
                    esn_raw_scores[e_target] = 1.0
                    continue
                
                U_others = torch.cat(others, dim=1)
                Q_others, _ = torch.linalg.qr(U_others, mode='reduced')
                projection = torch.matmul(Q_others.t(), U_target)
                energy = torch.norm(projection, p='fro') ** 2
                
                target_rank = U_target.shape[1]
                P_le = energy.item() / target_rank
                P_le = min(max(P_le, 0.0), 1.0)
                esn_raw_scores[e_target] = 1.0 - P_le
        
        final_scores = []
        total_energies = self.expert_energy.cpu().numpy()
        esn_list = []
        energy_list = []
        
        for i in range(self.num_experts):
            esn_list.append(esn_raw_scores.get(i, 0.0))
            energy_list.append(total_energies[i])
            
        rank_esn = np.argsort(np.argsort(np.array(esn_list)))
        rank_energy = np.argsort(np.argsort(np.array(energy_list)))
        
        norm_rank_esn = rank_esn / (self.num_experts - 1 + 1e-6)
        norm_rank_energy = rank_energy / (self.num_experts - 1 + 1e-6)
        
        for i in range(self.num_experts):
            score = (1 - beta) * norm_rank_esn[i] + beta * norm_rank_energy[i]
            final_scores.append(score)
            
        return np.array(final_scores)

def get_calibration_dataloader(tokenizer, dataset_path, num_blocks=512, block_size=2048, batch_size=4):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading raw dataset from {dataset_path}")
    if os.path.isdir(dataset_path):
        data_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.json')]
        dataset = load_dataset('json', data_files=data_files, split='train')
    else:
        dataset = load_dataset(dataset_path, split='train')

    initial_cnt = min(len(dataset), num_blocks * 50) 
    dataset = dataset.shuffle(seed=42).select(range(initial_cnt))
    
    text_column_name = "text"
    if "content" in dataset.column_names: text_column_name = "content"
    
    logger.info("Tokenizing and Packing")
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    def group_texts(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_datasets.map(group_texts, batched=True)

    if len(lm_dataset) > num_blocks:
        lm_dataset = lm_dataset.select(range(num_blocks))
        
    logger.info(f"Packed Dataset: {len(lm_dataset)} blocks of length {block_size}")
    
    return DataLoader(
        lm_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=default_data_collator,
        pin_memory=True
    )

def stage1_budget_allocation(model, dataloader, moe_layers, args):
    logger.info(f"Stage 1: Budget Allocation (Alpha={args.alpha})")
    
    accumulators = {}
    hooks = []
    
    hidden_dim = model.config.hidden_size
    
    for layer_idx in moe_layers:
        mlp = model.model.layers[layer_idx].mlp
        device = mlp.gate.weight.device
        accumulators[layer_idx] = RankAccumulator(hidden_dim, device)
        
        def get_hook(l_idx):
            def hook(module, input, output):
                x = input[0].detach()
                x = x.view(-1, x.shape[-1])
                accumulators[l_idx].update(x)
            return hook
            
        hooks.append(mlp.register_forward_hook(get_hook(layer_idx)))
        
    pbar = tqdm(dataloader, desc="Stage 1 Calibration")
    for batch in pbar:
        input_ids = batch['input_ids'].to(model.device)
        with torch.no_grad():
            model(input_ids)
            
    for h in hooks: h.remove()
    
    layer_scores = {}
    for layer_idx in moe_layers:
        r_l = accumulators[layer_idx].compute_effective_rank()
        s_l = r_l ** args.alpha
        layer_scores[layer_idx] = s_l
        logger.info(f"Layer {layer_idx}: Rank={r_l:.2f}, Score={s_l:.2f}")
        
    del accumulators
    torch.cuda.empty_cache()
    
    total_experts_per_layer = model.config.num_experts
    total_moe_experts = len(moe_layers) * total_experts_per_layer
    target_global_budget = int(total_moe_experts * args.retention_ratio)
    
    top_k = model.config.num_experts_per_tok
    min_limit = max(args.min_experts_per_layer, top_k)
    
    retention_budget = {l: min_limit for l in moe_layers}
    current_total = sum(retention_budget.values())
    remaining_budget = target_global_budget - current_total
    
    score_values = np.array([layer_scores[l] for l in moe_layers])
    total_score = score_values.sum()
    norm_scores = score_values / total_score if total_score > 0 else np.ones_like(score_values)/len(score_values)
    
    if remaining_budget > 0:
        for i, layer_idx in enumerate(moe_layers):
            extra = int(norm_scores[i] * remaining_budget)
            space = total_experts_per_layer - retention_budget[layer_idx]
            retention_budget[layer_idx] += min(extra, space)
            
    diff = target_global_budget - sum(retention_budget.values())
    sorted_indices = np.argsort(score_values)
    
    while diff > 0:
        added = False
        for i in range(len(moe_layers) - 1, -1, -1):
            idx = sorted_indices[i]
            l = moe_layers[idx]
            if retention_budget[l] < total_experts_per_layer:
                retention_budget[l] += 1
                diff -= 1
                added = True
                if diff == 0: break
        if not added: break
        
    return retention_budget

def get_weighted_moe_hook(accumulator, config):
    def hook(module, input, output):
        try:
            x = input[0].detach()
            gate_in = x.to(module.gate.weight.dtype)
            logits = module.gate(gate_in)
            probs = F.softmax(logits, dim=-1)
            
            topk_weights, topk_indices = torch.topk(probs, config.num_experts_per_tok, dim=-1)
            
            if getattr(config, 'norm_topk_prob', True):
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
            
            x_flat = x.view(-1, x.shape[-1])
            indices_flat = topk_indices.view(-1, topk_indices.shape[-1])
            weights_flat = topk_weights.view(-1, topk_weights.shape[-1]).to(x.dtype)
            
            unique_experts = torch.unique(indices_flat)
            for e_idx in unique_experts:
                e_idx = e_idx.item()
                mask = (indices_flat == e_idx)
                row_idx, col_idx = torch.where(mask)
                if row_idx.shape[0] == 0: continue
                
                selected_x = x_flat[row_idx]
                selected_w = weights_flat[row_idx, col_idx].unsqueeze(1)
                expert_out = module.experts[e_idx](selected_x)
                weighted_out = expert_out * selected_w
                accumulator.update(e_idx, weighted_out)
        except Exception as e:
            raise e
    return hook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--retention_ratio', type=float, default=0.75)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--min_experts_per_layer', type=int, default=4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--pca_rank', type=int, default=16)
    parser.add_argument('--num_calib_samples', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    dataloader = get_calibration_dataloader(
        tokenizer, 
        args.dataset_path, 
        num_blocks=args.num_calib_samples, 
        block_size=args.block_size, 
        batch_size=args.batch_size
    )
    
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_layers.append(i)
    
    logger.info(f"Found {len(moe_layers)} MoE layers")
    
    retention_budget = stage1_budget_allocation(model, dataloader, moe_layers, args)
    
    total_experts = model.config.num_experts
    
    for layer_idx in moe_layers:
        target_k = retention_budget[layer_idx]
        logger.info(f"Processing Layer {layer_idx} (Keep {target_k}/{total_experts})")
        
        layer_module = model.model.layers[layer_idx]
        mlp = layer_module.mlp
        current_device = mlp.gate.weight.device
        
        acc = ESNAccumulator(total_experts, model.config.hidden_size, current_device)
        handle = mlp.register_forward_hook(get_weighted_moe_hook(acc, model.config))
        
        pbar = tqdm(dataloader, desc="Stage 2 Calibration", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(model.device)
            with torch.no_grad():
                model(input_ids)
        
        handle.remove()
        
        scores = acc.compute_scores(pca_rank=args.pca_rank, beta=args.beta)
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = sorted(sorted_indices[:target_k].tolist())
        
        old_gate_weight = mlp.gate.weight
        new_gate = nn.Linear(
            model.config.hidden_size,
            len(keep_indices),
            bias=False,
            device=current_device,
            dtype=old_gate_weight.dtype
        )
        new_gate.weight.data = old_gate_weight.data[keep_indices].clone()
        new_experts = nn.ModuleList([mlp.experts[i] for i in keep_indices])
        
        shared_expert = getattr(mlp, 'shared_expert', None)
        shared_expert_gate = getattr(mlp, 'shared_expert_gate', None)
        if shared_expert is None and hasattr(mlp, 'shared_experts'):
             shared_expert = mlp.shared_experts
        
        pruned_mlp = FinalPrunedQwenMoE(
            model.config,
            new_gate,
            new_experts,
            shared_expert,
            shared_expert_gate
        )
        
        model.model.layers[layer_idx].mlp = pruned_mlp
        
        del acc
        del mlp
        torch.cuda.empty_cache()
        gc.collect()
        
    logger.info(f"Saving pruned model to {args.output_path}")
    avg_experts = int(sum(retention_budget.values()) / len(retention_budget))
    model.config.num_experts = avg_experts
    
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    with open(os.path.join(args.output_path, 'pruning_config.json'), 'w') as f:
        json.dump({
            'args': vars(args),
            'budget': {str(k): int(v) for k, v in retention_budget.items()}
        }, f, indent=2)

if __name__ == "__main__":
    main()