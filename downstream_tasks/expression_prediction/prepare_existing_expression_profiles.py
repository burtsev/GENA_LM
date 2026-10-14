"""Build the canonical four-CLS encoding bundle for downloaded expression profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from downstream_tasks.expression_prediction.expression_profile_encoder import save_encoding_bundle


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--panel-genes", type=Path, required=True)
    parser.add_argument("--bin-edges", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    genes = [line.strip() for line in args.panel_genes.read_text().splitlines() if line.strip()]
    if len(genes) != len(set(genes)):
        raise ValueError(f"Canonical panel has duplicated gene ids: {args.panel_genes}")
    canonical = set(genes)
    schema_counts: dict[str, int] = {}
    profile_count = 0
    for path in sorted(args.profiles_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if len(header) != len(set(header)):
            raise ValueError(f"Profile has duplicated gene ids: {path}")
        outside = [gene for gene in header if gene not in canonical]
        if outside:
            raise ValueError(f"{path} contains gene outside the canonical panel: {outside[0]}")
        schema_hash = hashlib.sha256(("\n".join(header) + "\n").encode()).hexdigest()
        schema_counts[schema_hash] = schema_counts.get(schema_hash, 0) + 1
        profile_count += 1

    if profile_count == 0:
        raise ValueError(f"No CSV profiles found in {args.profiles_dir}")
    edges = np.load(args.bin_edges)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    metadata = save_encoding_bundle(args.output_dir, genes, edges, tokenizer)
    provenance = {
        "profile_count": profile_count,
        "raw_schema_counts": schema_counts,
        "panel_genes_source_sha256": file_sha256(args.panel_genes),
        "bin_edges_source_sha256": file_sha256(args.bin_edges),
        "historical_expression_checkpoint_metadata": "unavailable; new encoder initializes from scratch",
        "encoding_signature_inputs": metadata,
    }
    (args.output_dir / "profile_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **provenance}, indent=2))


if __name__ == "__main__":
    main()
