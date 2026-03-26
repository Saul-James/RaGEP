import os
import itertools
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_calibration_dataloader(tokenizer, dataset_path, num_samples=128, seq_len=2048, batch_size=4):
    logger.info(f"Loading dataset from {dataset_path}")
    
    if os.path.isdir(dataset_path):
        data_files = [
            os.path.join(dataset_path, f) 
            for f in os.listdir(dataset_path) 
            if f.endswith('.json') or f.endswith('.jsonl')
        ]
        dataset = load_dataset('json', data_files=data_files, split='train')
    else:
        try:
            dataset = load_dataset(dataset_path, split='train', trust_remote_code=True)
        except:
            dataset = load_dataset('json', data_files=dataset_path, split='train')
            
    text_field = 'text'
    for field in ['text', 'content', 'sentence', 'paragraph', 'body']:
        if field in dataset.features:
            text_field = field
            break
            
    dataset = dataset.shuffle(seed=42)
    
    def tokenize_function(examples):
        return tokenizer(examples[text_field])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.features)
    )

    def group_texts(examples):
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    
    if len(lm_dataset) > num_samples:
        lm_dataset = lm_dataset.select(range(num_samples))
    
    logger.info(f"Prepared {len(lm_dataset)} sequences.")
    
    return DataLoader(
        lm_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4
    )