"""Shared four-CLS expression-profile encoding and ModernBERT implementation.

The expression encoder deliberately has its own model class. Transformers 4.55.2
derives local/global attention from ``layer_id % global_attn_every_n_layers``;
that would make a two-layer model global -> local. The classes below read an
explicit schedule from the serialized config before rotary embeddings are built,
while preserving the real block indices (and therefore layer-0 normalization).
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBertEncoderLayer,
    ModernBertMLP,
    ModernBertModel,
    ModernBertRotaryEmbedding,
    ModernBertUnpaddedRotaryEmbedding,
)


EXPRESSION_ENCODING_VERSION = 1
EXPRESSION_CLS_COUNT = 4
EXPRESSION_CLS_TOKEN_ID = 0
EXPRESSION_ATTENTION_SCHEDULE = ("local", "global")
EXPRESSION_CLS_STRATEGY = "bos_then_four_contiguous_gene_quarters_each_followed_by_cls"

EXPECTED_SPECIAL_TOKEN_IDS = {
    "cls": 0,
    "unk": 1,
    "pad": 2,
    "mask": 3,
    "bos": 4,
    "sep": 5,
    "eos": 6,
}
EXPECTED_BIN_TOKEN_IDS = tuple(range(7, 57))

EXPECTED_EXPRESSION_CONFIG = {
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "hidden_activation": "relu",
    "local_attention": 16,
    "global_attn_every_n_layers": 2,
    "global_rope_theta": 10000.0,
    "local_rope_theta": 10000.0,
    "max_position_embeddings": 24835,
    "vocab_size": 57,
    "attention_dropout": 0.1,
    "embedding_dropout": 0.1,
    "mlp_dropout": 0.1,
    "attention_bias": False,
    "mlp_bias": False,
    "norm_bias": False,
    "norm_eps": 0.00001,
    "initializer_range": 0.02,
    "initializer_cutoff_factor": 2.0,
}


def quarter_sizes(n_genes: int, n_quarters: int = EXPRESSION_CLS_COUNT) -> Tuple[int, ...]:
    """Split genes contiguously and put remainder genes in earlier quarters."""
    if n_genes < 0:
        raise ValueError(f"n_genes must be non-negative, got {n_genes}")
    if n_quarters <= 0:
        raise ValueError(f"n_quarters must be positive, got {n_quarters}")
    base, remainder = divmod(n_genes, n_quarters)
    return tuple(base + (index < remainder) for index in range(n_quarters))


def cls_positions(n_genes: int) -> Tuple[int, ...]:
    """Return zero-based positions of the four distributed CLS tokens."""
    positions = []
    position = 1  # [BOS]
    for size in quarter_sizes(n_genes):
        position += size
        positions.append(position)
        position += 1  # [CLS]
    return tuple(positions)


def gene_token_positions(n_genes: int) -> np.ndarray:
    """Map each canonical gene index to its token position after CLS insertion."""
    positions = np.empty(n_genes, dtype=np.int64)
    gene_start = 0
    token_start = 1
    for size in quarter_sizes(n_genes):
        gene_end = gene_start + size
        positions[gene_start:gene_end] = np.arange(token_start, token_start + size)
        gene_start = gene_end
        token_start += size + 1
    return positions


def build_four_cls_panel(
    gene_token_ids: Union[np.ndarray, torch.Tensor],
    bos_token_id: int,
    cls_token_id: int = EXPRESSION_CLS_TOKEN_ID,
) -> Union[np.ndarray, torch.Tensor]:
    """Build ``[BOS] + quarter + [CLS]`` repeated four times."""
    if gene_token_ids.ndim != 1:
        raise ValueError(f"gene_token_ids must be rank 1, got {gene_token_ids.shape}")
    sizes = quarter_sizes(len(gene_token_ids))
    if isinstance(gene_token_ids, torch.Tensor):
        pieces = [gene_token_ids.new_tensor([bos_token_id])]
        offset = 0
        for size in sizes:
            pieces.extend((gene_token_ids[offset : offset + size], gene_token_ids.new_tensor([cls_token_id])))
            offset += size
        return torch.cat(pieces)

    values = np.asarray(gene_token_ids)
    pieces = [np.asarray([bos_token_id], dtype=values.dtype)]
    offset = 0
    for size in sizes:
        pieces.extend((values[offset : offset + size], np.asarray([cls_token_id], dtype=values.dtype)))
        offset += size
    return np.concatenate(pieces)


def mask_target_gene(panel: torch.Tensor, gene_index: int, mask_token_id: int) -> torch.Tensor:
    """Mask one gene by canonical index without confusing inserted CLS positions."""
    if panel.ndim not in (1, 2):
        raise ValueError(f"panel must have rank 1 or 2, got {tuple(panel.shape)}")
    n_genes = panel.shape[-1] - 1 - EXPRESSION_CLS_COUNT
    if not 0 <= gene_index < n_genes:
        raise IndexError(f"gene_index {gene_index} is outside [0, {n_genes})")
    token_position = int(gene_token_positions(n_genes)[gene_index])
    panel[..., token_position] = mask_token_id
    return panel


def gather_four_cls_states(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cls_token_id: int = EXPRESSION_CLS_TOKEN_ID,
    validate: bool = True,
) -> torch.Tensor:
    """Gather the four fixed-position CLS states and concatenate them.

    Expression panels are validated when their canonical bank is constructed.  The
    training hot path can therefore use ``validate=False`` and avoid reducing a CUDA
    tensor to a Python boolean on every forward pass.
    """
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected hidden_states [batch, length, hidden] and ids [batch, length]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError(
            f"shape mismatch: hidden={tuple(hidden_states.shape)}, ids={tuple(input_ids.shape)}"
        )
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"shape mismatch: ids={tuple(input_ids.shape)}, mask={tuple(attention_mask.shape)}"
        )

    if validate:
        selected = input_ids.eq(cls_token_id)
        if attention_mask is not None:
            selected = selected & attention_mask.bool()
        counts = selected.sum(dim=1)
        if not bool(counts.eq(EXPRESSION_CLS_COUNT).all()):
            raise ValueError(
                "Each expression panel must contain exactly four unmasked CLS tokens; "
                f"got {counts.tolist()}"
            )
        batch, _, hidden = hidden_states.shape
        return hidden_states[selected].view(batch, EXPRESSION_CLS_COUNT, hidden).flatten(start_dim=1)

    n_genes = input_ids.shape[1] - 1 - EXPRESSION_CLS_COUNT
    positions = torch.as_tensor(cls_positions(n_genes), device=input_ids.device)
    selected = hidden_states.index_select(1, positions)
    return selected.flatten(start_dim=1)


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_gene_order(observed: Sequence[str], canonical: Sequence[str], source: str = "dataset") -> None:
    """Reject reordered, missing, or extra genes with a useful first mismatch."""
    if list(observed) == list(canonical):
        return
    mismatch = next(
        (
            index,
            observed[index] if index < len(observed) else "<missing>",
            canonical[index] if index < len(canonical) else "<missing>",
        )
        for index in range(max(len(observed), len(canonical)))
        if index >= len(observed) or index >= len(canonical) or observed[index] != canonical[index]
    )
    raise ValueError(
        f"{source} gene order is incompatible with the expression checkpoint: "
        f"observed {len(observed)} genes, expected {len(canonical)}; first mismatch at "
        f"index {mismatch[0]}: observed={mismatch[1]!r}, expected={mismatch[2]!r}"
    )


def tokenizer_definition(tokenizer) -> dict:
    vocab = tokenizer.get_vocab()
    special_ids = {
        name: getattr(tokenizer, f"{name}_token_id") for name in EXPECTED_SPECIAL_TOKEN_IDS
    }
    bin_ids = []
    while f"[BIN_{len(bin_ids)}]" in vocab:
        bin_ids.append(vocab[f"[BIN_{len(bin_ids)}]"])
    return {"special_token_ids": special_ids, "bin_token_ids": bin_ids, "vocab_size": len(vocab)}


def validate_tokenizer_definition(tokenizer) -> dict:
    definition = tokenizer_definition(tokenizer)
    if definition["special_token_ids"] != EXPECTED_SPECIAL_TOKEN_IDS:
        raise ValueError(
            "Expression tokenizer special-token IDs changed: "
            f"observed {definition['special_token_ids']}, expected {EXPECTED_SPECIAL_TOKEN_IDS}"
        )
    if tuple(definition["bin_token_ids"]) != EXPECTED_BIN_TOKEN_IDS:
        raise ValueError(
            "Expression tokenizer must preserve 50 ordered bin tokens with IDs 7..56; "
            f"got {definition['bin_token_ids']}"
        )
    if definition["vocab_size"] != 57:
        raise ValueError(f"Expression tokenizer vocabulary must contain 57 tokens, got {definition['vocab_size']}")
    return definition


def build_encoding_metadata(
    genes: Sequence[str],
    bin_edges: np.ndarray,
    tokenizer,
    attention_schedule: Sequence[str] = EXPRESSION_ATTENTION_SCHEDULE,
) -> dict:
    tokenizer_info = validate_tokenizer_definition(tokenizer)
    bin_edges = np.asarray(bin_edges)
    if bin_edges.shape != (49,):
        raise ValueError(f"50 expression-bin tokens require 49 bin edges, got {bin_edges.shape}")
    sizes = quarter_sizes(len(genes))
    return {
        "format_version": EXPRESSION_ENCODING_VERSION,
        "gene_count": len(genes),
        "gene_order_sha256": _sha256_lines(genes),
        "bin_edges_sha256": _sha256_array(np.asarray(bin_edges)),
        "tokenizer": tokenizer_info,
        "cls_placement": {
            "strategy": EXPRESSION_CLS_STRATEGY,
            "cls_token_id": EXPRESSION_CLS_TOKEN_ID,
            "quarter_sizes": list(sizes),
            "cls_positions": list(cls_positions(len(genes))),
            "total_tokens": len(genes) + 1 + EXPRESSION_CLS_COUNT,
        },
        "attention_schedule": list(attention_schedule),
        "local_attention": {"left": 8, "right": 8, "includes_self": True},
        "profile_alignment": {
            "strategy": "gene_id_to_canonical_order",
            "missing_measurement_token": "[UNK]",
        },
    }


def validate_encoding_metadata(
    metadata: Mapping,
    genes: Sequence[str],
    bin_edges: np.ndarray,
    tokenizer,
) -> dict:
    """Validate all encoding inputs against historical checkpoint metadata."""
    expected = build_encoding_metadata(genes, bin_edges, tokenizer)
    if dict(metadata) != expected:
        differing = sorted(key for key in set(metadata) | set(expected) if metadata.get(key) != expected.get(key))
        raise ValueError(
            "Expression encoding metadata does not match the checkpoint; differing fields: "
            + ", ".join(differing)
        )
    return expected


def load_encoding_bundle(
    metadata_path: Union[str, Path],
    panel_genes_path: Union[str, Path],
    bin_edges_path: Union[str, Path],
    tokenizer,
) -> Tuple[List[str], np.ndarray, dict]:
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expression checkpoint metadata is missing: {metadata_path}. Historical gene order, bins, "
            "CLS placement, and attention schedule cannot be assumed."
        )
    genes_path = Path(panel_genes_path)
    edges_path = Path(bin_edges_path)
    if not genes_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError(
            f"Expression checkpoint must include canonical genes and bin edges; got {genes_path} and {edges_path}"
        )
    genes = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(genes)) != len(genes):
        raise ValueError(f"Canonical expression gene list contains duplicates: {genes_path}")
    edges = np.load(edges_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_encoding_metadata(metadata, genes, edges, tokenizer)
    return genes, edges, metadata


def save_encoding_bundle(
    output_dir: Union[str, Path],
    genes: Sequence[str],
    bin_edges: np.ndarray,
    tokenizer,
) -> dict:
    """Persist the canonical inputs required to reproduce expression tokenization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_encoding_metadata(genes, bin_edges, tokenizer)
    (output_dir / "panel_genes.txt").write_text(
        "".join(f"{gene}\n" for gene in genes), encoding="utf-8"
    )
    np.save(output_dir / "bin_edges.npy", np.asarray(bin_edges))
    (output_dir / "expression_encoding.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tokenizer.save_pretrained(output_dir)
    return metadata


def encoding_signature(metadata: Mapping) -> str:
    payload = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_expression_config(config: ModernBertConfig, require_exact_architecture: bool = True) -> None:
    schedule = tuple(getattr(config, "expression_attention_schedule", ()))
    if schedule != EXPRESSION_ATTENTION_SCHEDULE:
        raise ValueError(
            "Expression encoder config is missing the required serialized local -> global schedule: "
            f"got {schedule or '<missing>'}, expected {EXPRESSION_ATTENTION_SCHEDULE}"
        )
    if len(schedule) != config.num_hidden_layers:
        raise ValueError(
            f"attention schedule has {len(schedule)} entries for {config.num_hidden_layers} layers"
        )
    if require_exact_architecture:
        differing = {
            key: (getattr(config, key, None), value)
            for key, value in EXPECTED_EXPRESSION_CONFIG.items()
            if getattr(config, key, None) != value
        }
        if differing:
            raise ValueError(f"Expression encoder architecture differs from the required redesign: {differing}")


class ExpressionModernBertAttention(ModernBertAttention):
    """ModernBERT attention whose local/global choice is explicitly serialized."""

    def __init__(self, config: ModernBertConfig, layer_id: int):
        nn.Module.__init__(self)
        validate_expression_config(config, require_exact_architecture=False)
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden size {config.hidden_size} is not divisible by {config.num_attention_heads} heads"
            )
        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        attention_kind = config.expression_attention_schedule[layer_id]
        if attention_kind == "local":
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
            rope_theta = config.local_rope_theta or config.global_rope_theta
            max_position_embeddings = config.local_attention
        elif attention_kind == "global":
            self.local_attention = (-1, -1)
            rope_theta = config.global_rope_theta
            max_position_embeddings = config.max_position_embeddings
        else:
            raise ValueError(f"Unknown attention kind {attention_kind!r} at layer {layer_id}")

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            config_copy = copy.deepcopy(config)
            config_copy.rope_theta = rope_theta
            self.rotary_emb = ModernBertRotaryEmbedding(config=config_copy)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else nn.Identity()
        self.pruned_heads = set()


class ExpressionModernBertEncoderLayer(ModernBertEncoderLayer):
    def __init__(self, config: ModernBertConfig, layer_id: int):
        # Initialize GradientCheckpointingLayer without constructing upstream attention.
        super(ModernBertEncoderLayer, self).__init__()
        self.config = config
        # Preserve upstream block indices: layer 0 alone has no pre-attention norm.
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        )
        self.attn = ExpressionModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)


class ExpressionModernBertModel(ModernBertModel):
    """ModernBERT with a checkpointed expression-specific attention schedule."""

    config_class = ModernBertConfig

    def __init__(self, config: ModernBertConfig):
        validate_expression_config(config, require_exact_architecture=False)
        # Call the pre-trained-model base without constructing upstream layers.
        super(ModernBertModel, self).__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ExpressionModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()


def has_model_weights(path: Union[str, Path]) -> bool:
    root = Path(path)
    return any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")) or any(
        root.glob("model-*.safetensors")
    )


def load_expression_encoder(
    config_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    attn_implementation: str = "flash_attention_2",
    parameter_dtype: torch.dtype = torch.float32,
) -> ExpressionModernBertModel:
    """Initialize from scratch or load only an architecture-compatible checkpoint."""
    source = Path(checkpoint_path) if checkpoint_path else Path(config_path)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Expression encoder config is missing: {source / 'config.json'}")
    config = ModernBertConfig.from_pretrained(source)
    config._attn_implementation = attn_implementation
    try:
        validate_expression_config(config)
    except ValueError as exc:
        if checkpoint_path is not None:
            raise ValueError(
                f"Expression checkpoint {source} is incompatible with the four-CLS redesign. "
                "Legacy 1,024-wide weights cannot initialize the required 256-wide encoder."
            ) from exc
        raise

    if checkpoint_path is not None:
        if not has_model_weights(source):
            raise FileNotFoundError(f"Expression checkpoint has no model weights: {source}")
        artifact_paths = {
            name: source / name
            for name in ("panel_genes.txt", "bin_edges.npy", "expression_encoding.json")
        }
        missing_artifacts = [str(path) for path in artifact_paths.values() if not path.is_file()]
        if missing_artifacts:
            raise FileNotFoundError(
                "Matching expression checkpoints must include their canonical encoding artifacts; "
                f"missing {missing_artifacts}"
            )
        with artifact_paths["expression_encoding.json"].open("r", encoding="utf-8") as handle:
            checkpoint_metadata = json.load(handle)
        model = ExpressionModernBertModel.from_pretrained(
            source,
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=parameter_dtype,
        )
        model.config.expression_encoding_signature = encoding_signature(checkpoint_metadata)
    else:
        warnings.warn(
            "No matching expression checkpoint was configured; initializing the new 256-wide encoder from scratch.",
            UserWarning,
        )
        model = ExpressionModernBertModel(config).to(dtype=parameter_dtype)
    model.config.parameter_dtype = str(parameter_dtype).replace("torch.", "")
    return model


def remove_legacy_expression_weights(
    state_dict: Mapping[str, torch.Tensor],
    expected_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Drop all legacy ``tf.*`` weights while retaining a matching new encoder.

    Both the historical and redesigned model use the public ``tf`` attribute. A
    mismatched embedding width identifies the historical checkpoint; once found,
    no legacy expression weight is reused, even if an incidental tensor shape
    happens to match.
    """
    expression_keys = [key for key in state_dict if key.startswith("tf.")]
    if not expression_keys:
        return dict(state_dict)
    if expected_state_dict is not None:
        incompatible = [
            key
            for key in expression_keys
            if key not in expected_state_dict or state_dict[key].shape != expected_state_dict[key].shape
        ]
        if not incompatible:
            return dict(state_dict)
    warnings.warn(
        "Ignoring legacy 1,024-wide expression-encoder weights under 'tf.*'. The redesigned 256-wide "
        "expression encoder must start from scratch or a matching checkpoint. "
        "Compatible DNA, decoder, normalization, and prediction-head weights will still be loaded.",
        UserWarning,
    )
    return {key: value for key, value in state_dict.items() if not key.startswith("tf.")}
