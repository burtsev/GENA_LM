#!/usr/bin/env python3
"""Measure real end-to-end final_tf throughput across physical batch sizes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from downstream_tasks.expression_prediction.benchmarks.train_existing_profiles_pilot import (
    batch_from_samples,
    instantiate_split,
)
from downstream_tasks.expression_prediction.expression_model_final_tf import ExpressionCounts
from downstream_tasks.expression_prediction.run_expression_finetuning_final_tf import (
    build_expression_panel_bank,
)


def make_batch(dataset, start, batch_size, device):
    return batch_from_samples(
        [dataset[(start + offset) % len(dataset)] for offset in range(batch_size)], device
    )


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch)
    output.loss.backward()
    optimizer.step()
    return output.loss.detach().float().item()


@torch.no_grad()
def eval_step(model, batch):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(**batch).loss.detach().float().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=7)
    parser.add_argument("--memory-limit-fraction", type=float, default=0.92)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ.setdefault("GENALM_HOME", str(REPO_ROOT.parent))
    os.environ.setdefault("PERT_DATA_ROOT", str(args.assets))
    os.environ.setdefault("PERT_GENOME", str(args.assets / "genomes/GRCh38.primary_assembly.genome.fa"))
    torch.manual_seed(45)
    torch.cuda.manual_seed_all(45)
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory

    config_dir = REPO_ROOT / "downstream_tasks/expression_prediction/configs"
    with initialize_config_dir(str(config_dir), version_base=None):
        config = compose(config_name="final_tf")
    train_dataset = instantiate_split(config, "train")
    valid_dataset = instantiate_split(config, "valid")
    expression_panel_bank = build_expression_panel_bank([train_dataset, valid_dataset])
    model_config = OmegaConf.to_container(config.model_kwargs, resolve=True)
    model_config.pop("_target_", None)
    model = ExpressionCounts(**instantiate(OmegaConf.create(model_config)))
    model.set_expression_panel_bank(expression_panel_bank)
    checkpoint = torch.load(args.assets / "models/all_datasets_2/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )

    results = []
    cursor = 0
    for batch_size in args.batch_sizes:
        try:
            torch.cuda.empty_cache()
            batch = make_batch(train_dataset, cursor, batch_size, device)
            for _ in range(args.warmups):
                train_step(model, optimizer, batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            losses = []
            for _ in range(args.measurements):
                started = time.perf_counter()
                losses.append(train_step(model, optimizer, batch))
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()

            model.eval()
            valid_batch = make_batch(valid_dataset, cursor, batch_size, device)
            for _ in range(args.warmups):
                eval_step(model, valid_batch)
            torch.cuda.synchronize()
            eval_times = []
            for _ in range(args.measurements):
                started = time.perf_counter()
                eval_step(model, valid_batch)
                torch.cuda.synchronize()
                eval_times.append(time.perf_counter() - started)
            model.train()

            median_step = statistics.median(times)
            median_eval = statistics.median(eval_times)
            result = {
                "batch_size_items": batch_size,
                "expression_panels_per_step": batch_size * 14,
                "median_train_step_seconds": median_step,
                "train_items_per_second": batch_size / median_step,
                "train_expression_panels_per_second": batch_size * 14 / median_step,
                "median_eval_step_seconds": median_eval,
                "eval_items_per_second": batch_size / median_eval,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "allocated_fraction_of_device": peak_allocated / total_memory,
                "loss_median": statistics.median(losses),
                "status": "ok",
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            cursor += batch_size
            del batch, valid_batch
            torch.cuda.empty_cache()
            if peak_allocated / total_memory >= args.memory_limit_fraction:
                break
        except torch.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            model.train()
            torch.cuda.empty_cache()
            result = {"batch_size_items": batch_size, "status": "oom", "error": str(error)}
            results.append(result)
            print(json.dumps(result), flush=True)
            break

    successful = [result for result in results if result["status"] == "ok"]
    best = max(successful, key=lambda result: result["train_items_per_second"])
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "device_memory_bytes": total_memory,
        "mixed_precision": "bfloat16 autocast",
        "parameter_dtype": "float32",
        "optimizer": "AdamW with initialized states",
        "warmups_per_batch_size": args.warmups,
        "measurements_per_batch_size": args.measurements,
        "train_dataset_items": len(train_dataset),
        "valid_dataset_items": len(valid_dataset),
        "results": results,
        "fastest_batch_size_items": best["batch_size_items"],
        "fastest_train_items_per_second": best["train_items_per_second"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output), "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
