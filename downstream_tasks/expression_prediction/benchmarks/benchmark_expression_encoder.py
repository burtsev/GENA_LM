#!/usr/bin/env python3
"""Measure legacy and four-CLS expression encoders in isolated GPU processes.

The benchmark uses FP32 parameters/AdamW states under BF16 autocast. Optimizer
state initialization and warmup happen before CUDA peak counters are reset.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import ModernBertConfig, ModernBertModel

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from downstream_tasks.expression_prediction.expression_profile_encoder import (
    ExpressionModernBertModel,
    build_four_cls_panel,
    cls_positions,
)


NEW_CONFIG = REPO_ROOT / "downstream_tasks/expression_prediction/configs/expression_profile_encoder_4cls"
DEFAULT_LEGACY_CONFIG = REPO_ROOT.parent / "pert_data/michael/models/moderngena-tf-bins50"


def make_model(kind: str, legacy_config: Path, gradient_checkpointing: bool):
    config_path = legacy_config if kind == "legacy" else NEW_CONFIG
    config = ModernBertConfig.from_pretrained(config_path)
    config._attn_implementation = "flash_attention_2"
    model = ModernBertModel(config) if kind == "legacy" else ExpressionModernBertModel(config)
    if kind == "redesign":
        model.tf_adapter = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def make_inputs(kind: str, batch_size: int, gene_count: int, device: torch.device):
    generator = torch.Generator().manual_seed(7)
    genes = torch.randint(7, 57, (batch_size, gene_count), generator=generator)
    if kind == "legacy":
        ids = torch.cat(
            [torch.full((batch_size, 1), 4), genes, torch.zeros((batch_size, 1), dtype=torch.long)], dim=1
        )
    else:
        ids = torch.stack([build_four_cls_panel(row, bos_token_id=4) for row in genes])
    return ids.to(device), torch.ones_like(ids, device=device)


def train_step(model, optimizer, ids, mask, kind: str) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if kind == "legacy":
            readout = hidden[:, -1]
        else:
            readout = hidden[:, list(cls_positions(ids.shape[1] - 5))].reshape(ids.shape[0], -1)
            readout = model.tf_adapter(readout)
        loss = readout.float().square().mean()
    loss.backward()
    optimizer.step()


def measure(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    model = make_model(args.model, Path(args.legacy_config), args.gradient_checkpointing).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ids, mask = make_inputs(args.model, args.batch_size, args.gene_count, device)

    # First update creates AdamW moment tensors. Further updates warm kernels/caches.
    train_step(model, optimizer, ids, mask, args.model)
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, ids, mask, args.model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    elapsed_ms = []
    for _ in range(args.measure_steps):
        start = time.perf_counter()
        train_step(model, optimizer, ids, mask, args.model)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "gene_count": args.gene_count,
        "sequence_length": ids.shape[1],
        "mixed_precision": "bfloat16",
        "parameter_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "optimizer": "AdamW",
        "gradient_checkpointing": args.gradient_checkpointing,
        "flash_attention_2": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "step_time_ms_mean": statistics.mean(elapsed_ms),
        "step_time_ms_median": statistics.median(elapsed_ms),
        "warmup_steps_after_optimizer_init": args.warmup_steps,
        "measured_steps": args.measure_steps,
    }


def run_isolated(args, kind: str) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        kind,
        "--legacy-config",
        args.legacy_config,
        "--batch-size",
        str(args.batch_size),
        "--gene-count",
        str(args.gene_count),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("both", "legacy", "redesign"), default="both")
    parser.add_argument("--legacy-config", default=str(DEFAULT_LEGACY_CONFIG))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gene-count", type=int, default=24830)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model == "both":
        print(json.dumps({kind: run_isolated(args, kind) for kind in ("legacy", "redesign")}, indent=2))
    else:
        print(json.dumps(measure(args), sort_keys=True))


if __name__ == "__main__":
    main()
