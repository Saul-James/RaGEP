# RaGEP: Rank-aware Geometric Expert Pruning for Mixture-of-Experts Language Models

This repository contains the pruning code used for the ICML 2026 submission:

`RaGEP: Rank-aware Geometric Expert Pruning for Mixture-of-Experts Language Models`

The release currently focuses on the pruning stage. It includes implementations for three MoE model families:

- Mixtral-8x7B
- DeepSeek-V2-Lite
- Qwen3-30B-A3B

## What the code does

The method is implemented as a two-stage pruning pipeline.

1. Stage 1 allocates a layer-wise expert budget using a rank-based score computed from calibration activations.
2. Stage 2 prunes experts inside each MoE layer using weighted expert subspace novelty (ESN), optionally mixed with expert energy.

Each pruning script saves:

- the pruned model weights
- the tokenizer
- `pruning_config.json` with all runtime arguments and the retained expert budget

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── check_environment.py
├── run_pruning.py
├── code
│   ├── Mixtral 8x7B
│   │   ├── ragep_mixtral_prune.py
│   │   └── run_mixtral_ragep.sh
│   ├── deepseek-V2-lite
│   │   ├── ragep_deepseek_prune.py
│   │   └── run_deepseek_ragep.sh
│   ├── qwen3-30b-a3b
│   │   ├── ragep_qwen_prune.py
│   │   └── run_ragep.sh
│   └── dataset
│       ├── c4.txt
│       └── data_utils.py

```

## Environment

The code assumes:

- Python 3.10 or newer
- PyTorch with CUDA support
- enough GPU memory to load the target base model
- internet access or a local cache for Hugging Face model weights and datasets

Suggested setup:

```bash
conda create -n ragep python=3.10 -y
conda activate ragep
pip install -r requirements.txt
```

Before launching a pruning job, run:

```bash
python check_environment.py
```

If you want to validate a local dataset pointer or local model path:

```bash
python check_environment.py \
  --dataset_path code/dataset/c4.txt \
  --model_path /path/to/local/model
```

## Input formats

### Model path

`--model_path` can be either:

- a local directory containing the pretrained model
- a Hugging Face model identifier

### Dataset path

`--dataset_path` can be:

- a Hugging Face dataset name
- a local JSON or JSONL file
- a local directory containing JSON or JSONL files
- a text pointer file

The new launcher supports pointer files. If a `.txt` file is provided, the first non-empty line is treated as the real dataset path or dataset name.

Important: `code/dataset/c4.txt` in this release is only a record of the calibration shard name used during our experiments. It is not a distributed dataset file by itself. For an actual run, replace it with:

- a real local JSON or JSONL path
- a directory containing calibration shards
- your own pointer file
- a Hugging Face dataset name that is directly loadable by `datasets.load_dataset(...)`

The pruning scripts expect one text-like field in the calibration dataset. They automatically look for the first matching field among:

- `text`
- `content`
- `sentence`
- `paragraph`
- `body` for the shared dataset utility

## Quick start

Use the unified launcher from the repository root.

### 1. Mixtral

```bash
python run_pruning.py mixtral \
  --model_path /path/to/mixtral_model \
  --dataset_path /path/to/calibration_dataset \
  --output_path outputs/mixtral_pruned \
  --global_retention_ratio 0.5 \
  --alpha 1.0 \
  --beta 0.5 \
  --pca_rank 32 \
  --min_experts_per_layer 2 \
  --num_calib_samples 64 \
  --batch_size 4
```

### 2. DeepSeek-V2-Lite

```bash
python run_pruning.py deepseek \
  --model_path /path/to/deepseek_model \
  --dataset_path /path/to/calibration_dataset \
  --output_path outputs/deepseek_pruned \
  --global_retention_ratio 0.5 \
  --alpha 1.0 \
  --beta 0.8 \
  --pca_rank 32 \
  --min_experts_per_layer 1 \
  --num_calib_samples 64 \
  --batch_size 4 \
  --num_gpus 1
```

### 3. Qwen3-30B-A3B

```bash
python run_pruning.py qwen \
  --model_path /path/to/qwen_model \
  --dataset_path /path/to/calibration_dataset \
  --output_path outputs/qwen_pruned \
  --retention_ratio 0.75 \
  --alpha 1.0 \
  --beta 1.0 \
  --pca_rank 16 \
  --min_experts_per_layer 4 \
  --num_calib_samples 64 \
  --block_size 2048 \
  --batch_size 4
```

## Shell launchers

For convenience, each model directory also includes a shell launcher:

- `code/Mixtral 8x7B/run_mixtral_ragep.sh`
- `code/deepseek-V2-lite/run_deepseek_ragep.sh`
- `code/qwen3-30b-a3b/run_ragep.sh`

These scripts now:

- use safe shell settings
- expose reasonable defaults for pruning hyperparameters
- require `MODEL_PATH` and `DATASET_PATH` to be set explicitly
- call the unified `run_pruning.py` entry point

Example:

```bash
MODEL_PATH=/path/to/mixtral_model \
DATASET_PATH=/path/to/calibration_dataset \
DRY_RUN=1 \
bash "code/Mixtral 8x7B/run_mixtral_ragep.sh"
```

## Dry-run mode

To verify command resolution without loading any model:

```bash
python run_pruning.py mixtral \
  --model_path /path/to/model \
  --dataset_path /path/to/calibration_dataset \
  --output_path outputs/test \
  --dry_run
```

This is useful for reviewers because it checks the entry point, argument parsing, and dataset pointer resolution without requiring a multi-GPU server.

## Output

After a successful run, `--output_path` will contain:

- the pruned model checkpoint
- tokenizer files
- `pruning_config.json`

`pruning_config.json` stores:

- the pruning hyperparameters
- the per-layer retained expert counts

## Notes by model family

### Mixtral

- script: `code/Mixtral 8x7B/ragep_mixtral_prune.py`
- uses `--global_retention_ratio`
- default `pca_rank` in the script is `32`

### DeepSeek-V2-Lite

- script: `code/deepseek-V2-lite/ragep_deepseek_prune.py`
- uses PyTorch distributed spawn
- `--num_gpus` should match the number of visible GPUs
- running with `--num_gpus 1` is supported for a basic single-node launch

### Qwen3-30B-A3B

- script: `code/qwen3-30b-a3b/ragep_qwen_prune.py`
- uses `--retention_ratio` instead of `--global_retention_ratio`
- exposes an additional `--block_size` argument for calibration packing

## Common failure modes

### 1. Missing Python packages

Symptom:

- `ModuleNotFoundError` for `transformers`, `datasets`, `accelerate`, or `sentencepiece`

Fix:

```bash
pip install -r requirements.txt
```

### 2. Reviewer runs the shell script without editing placeholders

This was a problem in the original release. The shell scripts have now been rewritten so they no longer contain empty parameter assignments.

The minimum required variable is:

```bash
MODEL_PATH=/path/to/model
```

### 3. Dataset path points to a text file

The unified launcher supports this now, but the pointer file must resolve to a real dataset path or a dataset name that `load_dataset(...)` can open.

### 4. Out-of-memory errors

Typical fixes:

- reduce `--batch_size`
- reduce `--num_calib_samples`
- reduce `--block_size` for Qwen
- use fewer visible GPUs only if the model still fits

### 5. Hugging Face remote-code models

The model loading code uses `trust_remote_code=True`. If a model family has additional upstream requirements, install those dependencies in the same environment.

## Minimal reviewer checklist

For a reviewer or rebuttal response, the following sequence is enough to confirm the release is structured correctly:

```bash
pip install -r requirements.txt
python check_environment.py
python run_pruning.py mixtral --model_path /path/to/model --dataset_path /path/to/calibration_dataset --output_path outputs/test --dry_run
```
