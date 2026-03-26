#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/outputs/mixtral_pruned}"

RETENTION_RATIO="${RETENTION_RATIO:-0.5}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.5}"
PCA_RANK="${PCA_RANK:-32}"
MIN_EXPERTS="${MIN_EXPERTS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Please set MODEL_PATH to a local model path or Hugging Face model id." >&2
    exit 1
fi

if [[ -z "$DATASET_PATH" ]]; then
    echo "Please set DATASET_PATH to a dataset name, JSON/JSONL file, or directory." >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
    EXTRA_ARGS+=(--dry_run)
fi

python "$ROOT_DIR/run_pruning.py" mixtral \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --global_retention_ratio "$RETENTION_RATIO" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --pca_rank "$PCA_RANK" \
    --min_experts_per_layer "$MIN_EXPERTS" \
    --num_calib_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"
