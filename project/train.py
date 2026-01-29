"""
train.py — one command, two backends:

- Mac Apple Silicon: uses MLX-LM LoRA (calls `python -m mlx_lm.lora`)
- Linux/Windows with CUDA: uses Transformers + PEFT LoRA (native training)

Dataset format (both):
  data_dir/train.jsonl and data_dir/valid.jsonl
  each line: {"text": "..."}

Examples:
  python train.py --backend auto --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --data_dir data/mlx/training

  python train.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --data_dir data/mlx/training --output_dir adapters_hf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def read_jsonl_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = (obj.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def run_mlx_lora(args: argparse.Namespace) -> None:
    # We call mlx-lm as a subprocess because it’s the cleanest “works out of the box”.
    # It expects --data pointing to a folder with train.jsonl/valid.jsonl.
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--data",
        str(args.data_dir),
        "--train",
        "--fine-tune-type",
        "lora",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--adapter-path",
        str(args.output_dir),
    ]

    print("▶ Using MLX-LM backend (Apple Silicon).")
    print("▶ Command:\n ", " ".join(cmd))
    subprocess.check_call(cmd)


# ---------------- HF backend ----------------


@dataclass
class TrainConfigHF:
    model: str
    data_dir: Path
    output_dir: Path
    max_seq_len: int
    per_device_batch_size: int
    grad_accum: int
    lr: float
    epochs: int
    fp16: bool
    bf16: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float


class JsonlTextDataset:
    """Minimal dataset reading {"text": ...} JSONL."""

    def __init__(self, jsonl_path: Path):
        self.texts = read_jsonl_texts(jsonl_path)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx]}


def run_hf_peft_lora(args: argparse.Namespace) -> None:
    print("▶ Using Hugging Face + PEFT backend (CUDA).")

    # Imports here so Mac users without torch/transformers don’t crash
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model

    cfg = TrainConfigHF(
        model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        fp16=not args.bf16,
        bf16=args.bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    train_path = cfg.data_dir / "train.jsonl"
    valid_path = cfg.data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {valid_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    if tokenizer.pad_token is None:
        # common for causal LMs
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    # LoRA config: keeps it simple and broadly compatible
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora)

    train_ds = JsonlTextDataset(train_path)
    valid_ds = JsonlTextDataset(valid_path)

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Batch may be list-like in Trainer map; keep it simple:
        text = batch["text"]
        out = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )
        return out

    # Trainer expects datasets that return tokenized dicts.
    # We’ll wrap by tokenizing in __getitem__ to avoid adding extra deps.
    class TokenizedWrapper:
        def __init__(self, raw_ds):
            self.raw_ds = raw_ds

        def __len__(self):
            return len(self.raw_ds)

        def __getitem__(self, idx):
            item = self.raw_ds[idx]
            return tokenize_batch(item)

    token_train = TokenizedWrapper(train_ds)
    token_valid = TokenizedWrapper(valid_ds)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=cfg.bf16 and torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_train,
        eval_dataset=token_valid,
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA adapters
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    print(f"✅ HF LoRA adapters saved to: {cfg.output_dir}")


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        choices=["auto", "mlx", "hf"],
        default="auto",
        help="auto: MLX on Apple Silicon else HF on CUDA; or force mlx/hf",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model ID. For MLX: mlx-community/...-4bit. For HF: a transformers model id.",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Folder containing train.jsonl and valid.jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("adapters"),
        help="Where to save adapters",
    )

    # Common knobs
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iters", type=int, default=200, help="MLX only")
    p.add_argument("--epochs", type=int, default=1, help="HF only")
    p.add_argument("--lr", type=float, default=2e-4, help="HF only")
    p.add_argument("--max_seq_len", type=int, default=1024, help="HF only")
    p.add_argument("--grad_accum", type=int, default=8, help="HF only")

    # LoRA knobs (HF only; MLX uses its defaults)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true", help="HF only: use bf16 if supported")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Basic dataset check
    if not (args.data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"Missing {(args.data_dir / 'train.jsonl')}")
    if not (args.data_dir / "valid.jsonl").exists():
        raise FileNotFoundError(f"Missing {(args.data_dir / 'valid.jsonl')}")

    backend = args.backend
    if backend == "auto":
        if is_apple_silicon():
            backend = "mlx"
        elif has_cuda():
            backend = "hf"
        else:
            raise RuntimeError(
                "auto backend could not find Apple Silicon (mlx) or CUDA (hf).\n"
                "Use --backend hf with a CUDA machine, or run MLX on an Apple Silicon Mac."
            )

    if backend == "mlx":
        run_mlx_lora(args)
    elif backend == "hf":
        if not has_cuda():
            raise RuntimeError("HF backend selected but CUDA is not available.")
        run_hf_peft_lora(args)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
