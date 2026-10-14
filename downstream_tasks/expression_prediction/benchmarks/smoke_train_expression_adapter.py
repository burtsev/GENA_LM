#!/usr/bin/env python3
"""Synthetic H100 smoke training through the real frozen downstream predictor.

This validates wiring and optimization isolation. It is not a replacement for
training on the missing canonical 24,830-gene final_tf dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from downstream_tasks.expression_prediction.expression_model_final_tf import ExpressionCounts
from downstream_tasks.expression_prediction.expression_profile_encoder import build_four_cls_panel


DEFAULT_ASSETS = REPO_ROOT.parent / "pert_data/michael"
DEFAULT_CONFIG = REPO_ROOT / "downstream_tasks/expression_prediction/configs/expression_profile_encoder_4cls"


def changed(before, parameter):
    return not torch.equal(before, parameter.detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default=str(DEFAULT_ASSETS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dna-length", type=int, default=32)
    parser.add_argument("--gene-count", type=int, default=24830)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-predictor smoke training")
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    assets = Path(args.assets)
    model = ExpressionCounts(
        config={},
        hf=True,
        hf_model_name=str(assets / "models/modernbert_large"),
        hf_model_name_decoder=str(assets / "models/decoder"),
        tf_config_path=str(DEFAULT_CONFIG),
        tf_model_path=None,
        tf_parameter_dtype="float32",
        expression_encoder_only=True,
        bert_trainable_layers=0,
    )
    checkpoint_path = assets / "models/all_datasets_2/pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(checkpoint, strict=False)
    del checkpoint

    device = torch.device("cuda")
    model.to(device)
    model.train()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=1e-4)

    frozen_samples = {
        "bert_embedding": model.bert.embeddings.tok_embeddings.weight,
        "decoder_attention": model.decoder.layers[0].attn.Wqkv.weight,
        "dna_layernorm": model.dna_ln.weight,
        "tf_layernorm": model.tf_ln.weight,
        "prediction_head": model.classifier.weight,
    }
    before_frozen = {name: parameter.detach().cpu().clone() for name, parameter in frozen_samples.items()}
    before_trainable = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}

    generator = torch.Generator().manual_seed(23)
    dna_ids = torch.randint(7, model.bert.config.vocab_size, (1, 1, args.dna_length), generator=generator)
    dna_ids[..., 0] = 0
    dna_ids = dna_ids.to(device)
    dna_mask = torch.ones_like(dna_ids)
    gene_ids = torch.randint(7, 57, (args.gene_count,), generator=generator)
    tf_ids = build_four_cls_panel(gene_ids, bos_token_id=4).view(1, 1, -1).to(device)
    tf_mask = torch.ones_like(tf_ids)
    labels = torch.zeros(1, 1, args.dna_length, 1, device=device)
    labels_mask = torch.zeros_like(labels, dtype=torch.bool)
    labels_mask[..., 0, 0] = True
    dataset_flag = torch.ones(1, 1, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    times_ms = []
    encoder_grad_norms = []
    adapter_grad_norms = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=dna_ids,
                attention_mask=dna_mask,
                labels=labels,
                labels_mask=labels_mask,
                tf_ids=tf_ids,
                tf_attention_mask=tf_mask,
                dataset_flag=dataset_flag,
            )
        output.loss.backward()
        encoder_grad_norms.append(
            torch.linalg.vector_norm(
                torch.stack([parameter.grad.detach().float().norm() for parameter in model.tf.parameters()])
            ).item()
        )
        adapter_grad_norms.append(
            torch.linalg.vector_norm(
                torch.stack([parameter.grad.detach().float().norm() for parameter in model.tf_adapter.parameters()])
            ).item()
        )
        optimizer.step()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
        losses.append(output.loss.detach().float().item())

    changed_trainable = [
        name for name, parameter in trainable.items() if changed(before_trainable[name], parameter)
    ]
    changed_frozen = [
        name for name, parameter in frozen_samples.items() if changed(before_frozen[name], parameter)
    ]
    report = {
        "scope": "synthetic wiring smoke test; not biological training",
        "gpu": torch.cuda.get_device_name(device),
        "steps": args.steps,
        "dna_length": args.dna_length,
        "gene_count": args.gene_count,
        "expression_tokens": tf_ids.shape[-1],
        "mixed_precision": "bfloat16",
        "parameter_dtype": str(next(model.tf.parameters()).dtype).replace("torch.", ""),
        "encoder_parameters": sum(parameter.numel() for parameter in model.tf.parameters()),
        "adapter_parameters": sum(parameter.numel() for parameter in model.tf_adapter.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "optimizer_parameter_names_valid": all(
            name.startswith(("tf.", "tf_adapter.")) for name in trainable
        ),
        "branch_modes": {"tf": model.tf.training, "tf_adapter": model.tf_adapter.training},
        "frozen_modes": {
            "bert": model.bert.training,
            "decoder": model.decoder.training,
            "dna_ln": model.dna_ln.training,
            "tf_ln": model.tf_ln.training,
            "classifier": model.classifier.training,
        },
        "losses": losses,
        "step_times_ms": times_ms,
        "encoder_grad_norms": encoder_grad_norms,
        "adapter_grad_norms": adapter_grad_norms,
        "changed_trainable_tensor_count": len(changed_trainable),
        "changed_only_expression_branch": all(
            name.startswith(("tf.", "tf_adapter.")) for name in changed_trainable
        ),
        "changed_frozen_samples": changed_frozen,
        "adapter_keys_in_state_dict": sorted(
            key for key in model.state_dict() if key.startswith("tf_adapter.")
        ),
        "missing_adapter_tensors_from_legacy_checkpoint": sum(
            key.startswith("tf_adapter.") for key in load_result.missing_keys
        ),
        "missing_frozen_tensors_from_legacy_checkpoint": sum(
            not key.startswith(("tf.", "tf_adapter.")) for key in load_result.missing_keys
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
