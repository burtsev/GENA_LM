#!/usr/bin/env python3
"""Checkpointed GPU pilot using real profiles and promoter sequences from final_tf."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from downstream_tasks.expression_prediction.expression_model_final_tf import ExpressionCounts
from downstream_tasks.expression_prediction.run_expression_finetuning_final_tf import (
    build_expression_panel_bank,
)


def instantiate_split(config, split: str):
    shared = OmegaConf.to_container(config.shared_dataset_params, resolve=True)
    dataset = OmegaConf.to_container(config[f"{split}_dataset_human"], resolve=True)
    dataset.update(shared)
    return instantiate(OmegaConf.create(dataset))


def batch_from_samples(samples, device):
    result = {}
    variable_length = {
        "input_ids": 0,
        "attention_mask": 0,
        "labels": 0.0,
        "labels_mask": 0,
    }
    max_length = max(sample["input_ids"].shape[1] for sample in samples)
    for key, pad_value in variable_length.items():
        values = []
        for sample in samples:
            value = sample[key]
            pad_length = max_length - value.shape[1]
            if pad_length:
                pad_shape = list(value.shape)
                pad_shape[1] = pad_length
                value = torch.cat([value, value.new_full(pad_shape, pad_value)], dim=1)
            values.append(value)
        result[key] = torch.stack(values).to(device)
    for key in ("tf_panel_indices", "tf_target_positions", "dataset_flag"):
        result[key] = torch.stack([sample[key] for sample in samples]).to(device)
    return result


def batch_from_sample(sample, device):
    return batch_from_samples([sample], device)


def grad_norm(module):
    norms = [parameter.grad.detach().float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)).item()


@torch.no_grad()
def validation_loss(model, dataset, sample_count, device):
    if sample_count <= 0:
        return None
    stride = max(1, len(dataset) // sample_count)
    indices = [(index * stride) % len(dataset) for index in range(sample_count)]
    was_training = model.training
    model.eval()
    losses = []
    for sample_index in indices:
        batch = batch_from_sample(dataset[sample_index], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses.append(model(**batch).loss.detach().float().item())
    model.train(was_training)
    return {"mean": sum(losses) / len(losses), "losses": losses, "sample_indices": indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--validation-samples", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-profile pilot")

    os.environ.setdefault("GENALM_HOME", str(REPO_ROOT.parent))
    os.environ.setdefault("PERT_DATA_ROOT", str(args.assets))
    os.environ.setdefault("PERT_GENOME", str(args.assets / "genomes/GRCh38.primary_assembly.genome.fa"))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    legacy_checkpoint = torch.load(args.assets / "models/all_datasets_2/pytorch_model.bin", map_location="cpu")
    load_result = model.load_state_dict(legacy_checkpoint, strict=False)
    del legacy_checkpoint
    model.tf.config.expression_encoding_signature = train_dataset.expression_encoding_signature

    device = torch.device("cuda")
    model.to(device)
    model.train()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or not all(name.startswith(("tf.", "tf_adapter.")) for name in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {sorted(trainable)[:10]}")
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=1e-4)
    validation_before = validation_loss(model, valid_dataset, args.validation_samples, device)

    # Spread samples across genes rather than repeatedly fitting one cached sample.
    stride = max(1, len(train_dataset) // args.steps)
    sample_indices = [(index * stride) % len(train_dataset) for index in range(args.steps)]
    losses = []
    times_ms = []
    encoder_grad_norms = []
    adapter_grad_norms = []
    target_genes = []
    real_profile_counts = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for step, sample_index in enumerate(sample_indices, start=1):
        sample = train_dataset[sample_index]
        batch = batch_from_sample(sample, device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)
        output.loss.backward()
        encoder_grad_norms.append(grad_norm(model.tf))
        adapter_grad_norms.append(grad_norm(model.tf_adapter))
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(output.loss.detach().float().item())
        times_ms.append((time.perf_counter() - started) * 1000.0)
        target_genes.append(sample["gene_id"][0] if sample["gene_id"] else None)
        real_profile_counts.append(len(sample["selected_keys"]))
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step {step}/{args.steps}: loss={losses[-1]:.6f}, time_ms={times_ms[-1]:.2f}", flush=True)

    validation_after = validation_loss(model, valid_dataset, args.validation_samples, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_checkpoint = args.output_dir / "pytorch_model.bin"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, full_checkpoint)
    torch.save(optimizer.state_dict(), args.output_dir / "optimizer.pt")
    encoder_dir = args.output_dir / "expression_encoder"
    model.tf.save_pretrained(encoder_dir)
    bundle_dir = args.assets / "models/moderngena-tf-4cls"
    for name in (
        "panel_genes.txt",
        "bin_edges.npy",
        "expression_encoding.json",
        "profile_provenance.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        shutil.copy2(bundle_dir / name, encoder_dir / name)

    report = {
        "scope": "short real-data optimization pilot; not a convergence run",
        "gpu": torch.cuda.get_device_name(device),
        "steps": args.steps,
        "sample_indices": sample_indices,
        "target_genes": target_genes,
        "real_profiles_per_step": real_profile_counts,
        "gene_count": len(train_dataset.tf_genes),
        "expression_tokens": len(train_dataset.tf_genes) + 5,
        "quarter_sizes": train_dataset.expression_encoding_metadata["cls_placement"]["quarter_sizes"],
        "cls_positions": train_dataset.expression_encoding_metadata["cls_placement"]["cls_positions"],
        "encoding_signature": train_dataset.expression_encoding_signature,
        "mixed_precision": "bfloat16 autocast",
        "parameter_dtype": str(next(model.tf.parameters()).dtype).replace("torch.", ""),
        "encoder_parameters": sum(parameter.numel() for parameter in model.tf.parameters()),
        "adapter_parameters": sum(parameter.numel() for parameter in model.tf_adapter.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "losses": losses,
        "step_times_ms": times_ms,
        "encoder_grad_norms": encoder_grad_norms,
        "adapter_grad_norms": adapter_grad_norms,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "missing_adapter_tensors_from_initial_checkpoint": sum(
            key.startswith("tf_adapter.") for key in load_result.missing_keys
        ),
        "missing_frozen_tensors_from_initial_checkpoint": sum(
            not key.startswith(("tf.", "tf_adapter.")) for key in load_result.missing_keys
        ),
        "checkpoint": str(full_checkpoint),
        "optimizer_checkpoint": str(args.output_dir / "optimizer.pt"),
        "matching_expression_encoder": str(encoder_dir),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
