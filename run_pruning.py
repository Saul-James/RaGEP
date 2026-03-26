#!/usr/bin/env python3
"""Unified entry point for the RaGEP pruning scripts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_MAP = {
    "mixtral": REPO_ROOT / "code" / "Mixtral 8x7B" / "ragep_mixtral_prune.py",
    "deepseek": REPO_ROOT / "code" / "deepseek-V2-lite" / "ragep_deepseek_prune.py",
    "qwen": REPO_ROOT / "code" / "qwen3-30b-a3b" / "ragep_qwen_prune.py",
}


def resolve_dataset_path(dataset_path: str) -> str:
    """Allow a .txt file to point to the real dataset path or HF dataset name."""
    candidate = Path(dataset_path)
    if not candidate.is_file() or candidate.suffix.lower() != ".txt":
        return dataset_path

    for line in candidate.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        pointed = Path(value)
        if not pointed.is_absolute():
            pointed = (candidate.parent / pointed).resolve()
        return str(pointed) if pointed.exists() else value

    raise ValueError(f"No usable dataset entry found in pointer file: {dataset_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", required=True, help="Local model path or Hugging Face model id.")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Dataset name, local JSON/JSONL path, local directory, or a .txt pointer file.",
    )
    parser.add_argument("--output_path", required=True, help="Directory used to save the pruned model.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Rank-aware budget allocation exponent.")
    parser.add_argument("--beta", type=float, default=0.5, help="Balance between ESN novelty and energy.")
    parser.add_argument("--pca_rank", type=int, default=32, help="Subspace rank used in Stage 2.")
    parser.add_argument("--num_calib_samples", type=int, default=64, help="Number of calibration sequences.")
    parser.add_argument("--batch_size", type=int, default=4, help="Calibration batch size.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without launching the pruning job.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the RaGEP pruning code used in the paper."
    )
    subparsers = parser.add_subparsers(dest="model_family", required=True)

    mixtral = subparsers.add_parser("mixtral", help="Launch Mixtral-8x7B pruning.")
    add_common_arguments(mixtral)
    mixtral.add_argument("--global_retention_ratio", type=float, default=0.5)
    mixtral.add_argument("--min_experts_per_layer", type=int, default=2)

    deepseek = subparsers.add_parser("deepseek", help="Launch DeepSeek-V2-Lite pruning.")
    add_common_arguments(deepseek)
    deepseek.set_defaults(beta=0.8, batch_size=4)
    deepseek.add_argument("--global_retention_ratio", type=float, default=0.5)
    deepseek.add_argument("--min_experts_per_layer", type=int, default=1)
    deepseek.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs passed to mp.spawn.")

    qwen = subparsers.add_parser("qwen", help="Launch Qwen3-30B-A3B pruning.")
    add_common_arguments(qwen)
    qwen.set_defaults(beta=1.0, pca_rank=16)
    qwen.add_argument("--retention_ratio", type=float, default=0.75)
    qwen.add_argument("--min_experts_per_layer", type=int, default=4)
    qwen.add_argument("--block_size", type=int, default=2048)

    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    script = SCRIPT_MAP[args.model_family]
    dataset_path = resolve_dataset_path(args.dataset_path)

    cmd = [
        sys.executable,
        str(script),
        "--model_path",
        args.model_path,
        "--dataset_path",
        dataset_path,
        "--output_path",
        args.output_path,
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--pca_rank",
        str(args.pca_rank),
        "--num_calib_samples",
        str(args.num_calib_samples),
        "--batch_size",
        str(args.batch_size),
    ]

    if args.model_family in {"mixtral", "deepseek"}:
        cmd.extend(
            [
                "--global_retention_ratio",
                str(args.global_retention_ratio),
                "--min_experts_per_layer",
                str(args.min_experts_per_layer),
            ]
        )

    if args.model_family == "deepseek":
        cmd.extend(["--num_gpus", str(args.num_gpus)])

    if args.model_family == "qwen":
        cmd.extend(
            [
                "--retention_ratio",
                str(args.retention_ratio),
                "--min_experts_per_layer",
                str(args.min_experts_per_layer),
                "--block_size",
                str(args.block_size),
            ]
        )

    return cmd


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = build_command(args)

    print("Resolved command:", flush=True)
    print(" ".join(shlex.quote(part) for part in command), flush=True)

    if args.dry_run:
        return 0

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
