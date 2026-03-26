#!/usr/bin/env python3
"""Basic environment and path checks for the RaGEP code release."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = [
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "tqdm",
]
OPTIONAL_MODULES = [
    "accelerate",
    "sentencepiece",
    "safetensors",
]


def resolve_dataset_pointer(value: str) -> str:
    path = Path(value)
    if not path.is_file() or path.suffix.lower() != ".txt":
        return value

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return value


def load_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local environment for the RaGEP release.")
    parser.add_argument("--dataset_path", help="Optional dataset path or pointer file to validate.")
    parser.add_argument("--model_path", help="Optional local model path to validate.")
    args = parser.parse_args()

    print(f"Repository root: {REPO_ROOT}")
    print(f"Python version: {sys.version.split()[0]}")

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            print(f"[OK] required module `{module_name}` version {load_version(module_name)}")
        except Exception as exc:
            missing.append((module_name, str(exc)))

    for module_name in OPTIONAL_MODULES:
        try:
            print(f"[OK] optional module `{module_name}` version {load_version(module_name)}")
        except Exception:
            print(f"[WARN] optional module `{module_name}` is not installed")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"  - GPU {idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"[WARN] Unable to query CUDA details: {exc}")

    if args.dataset_path:
        resolved = resolve_dataset_pointer(args.dataset_path)
        exists = Path(resolved).exists()
        print(f"Dataset input: {args.dataset_path}")
        print(f"Resolved dataset target: {resolved}")
        print(f"Resolved dataset exists locally: {exists}")

    if args.model_path:
        model_path = Path(args.model_path)
        print(f"Model path exists locally: {model_path.exists()}")

    if missing:
        print("\nMissing required modules:")
        for module_name, exc in missing:
            print(f"  - {module_name}: {exc}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
