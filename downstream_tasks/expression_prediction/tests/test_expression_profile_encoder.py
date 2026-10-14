import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import ModernBertConfig

from downstream_tasks.expression_prediction.expression_profile_encoder import (
    EXPRESSION_ATTENTION_SCHEDULE,
    ExpressionModernBertModel,
    build_encoding_metadata,
    build_four_cls_panel,
    cls_positions,
    gather_four_cls_states,
    gene_token_positions,
    load_expression_encoder,
    mask_target_gene,
    quarter_sizes,
    remove_legacy_expression_weights,
    validate_encoding_metadata,
    validate_expression_config,
    validate_gene_order,
)
from downstream_tasks.expression_prediction.expression_model_final_tf import ExpressionCounts


class StubTokenizer:
    cls_token_id = 0
    unk_token_id = 1
    pad_token_id = 2
    mask_token_id = 3
    bos_token_id = 4
    sep_token_id = 5
    eos_token_id = 6

    def get_vocab(self):
        vocab = {
            "[CLS]": 0,
            "[UNK]": 1,
            "[PAD]": 2,
            "[MASK]": 3,
            "[BOS]": 4,
            "[SEP]": 5,
            "[EOS]": 6,
        }
        vocab.update({f"[BIN_{index}]": index + 7 for index in range(50)})
        return vocab


def small_config() -> ModernBertConfig:
    config = ModernBertConfig(
        vocab_size=57,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_activation="relu",
        local_attention=4,
        global_attn_every_n_layers=2,
        global_rope_theta=10000.0,
        local_rope_theta=10000.0,
        max_position_embeddings=32,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        norm_eps=1e-5,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        reference_compile=False,
        pad_token_id=2,
        bos_token_id=4,
        eos_token_id=6,
    )
    config.expression_attention_schedule = list(EXPRESSION_ATTENTION_SCHEDULE)
    config.expression_cls_count = 4
    config.expression_cls_token_id = 0
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize(
    ("n_genes", "sizes"),
    [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 0)), (10, (3, 3, 2, 2)), (24830, (6208, 6208, 6207, 6207))],
)
def test_quarters_put_remainders_first(n_genes, sizes):
    assert quarter_sizes(n_genes) == sizes


def test_requested_full_panel_cls_positions():
    assert cls_positions(24830) == (6209, 12418, 18626, 24834)
    assert quarter_sizes(24759) == (6190, 6190, 6190, 6189)
    assert cls_positions(24759) == (6191, 12382, 18573, 24763)
    assert cls_positions(10) == (4, 8, 11, 14)


def test_gene_positions_and_target_mask_follow_inserted_cls_tokens():
    genes = torch.arange(7, 17)
    panel = build_four_cls_panel(genes, bos_token_id=4)
    assert panel.tolist() == [4, 7, 8, 9, 0, 10, 11, 12, 0, 13, 14, 0, 15, 16, 0]
    assert gene_token_positions(10).tolist() == [1, 2, 3, 5, 6, 7, 9, 10, 12, 13]
    original_cls = torch.where(panel == 0)[0].tolist()
    mask_target_gene(panel, gene_index=6, mask_token_id=3)
    assert panel[9].item() == 3
    assert torch.where(panel == 0)[0].tolist() == original_cls


def test_unknown_panel_and_right_padding_keep_four_unmasked_cls_tokens():
    unknown = build_four_cls_panel(torch.full((5,), 1), bos_token_id=4)
    padded = torch.cat([unknown, torch.full((3,), 2)])
    mask = torch.cat([torch.ones_like(unknown), torch.zeros(3, dtype=torch.long)])
    hidden = torch.arange(padded.numel() * 3, dtype=torch.float32).view(1, -1, 3)
    gathered = gather_four_cls_states(hidden, padded.unsqueeze(0), mask.unsqueeze(0))
    expected = hidden[0, torch.tensor(cls_positions(5))].reshape(1, 12)
    torch.testing.assert_close(gathered, expected)


def test_cls_gather_is_ordered_and_gradients_reach_every_readout():
    panel = build_four_cls_panel(torch.arange(7, 15), bos_token_id=4).unsqueeze(0)
    hidden = torch.randn(1, panel.shape[1], 256, requires_grad=True)
    hidden.retain_grad()
    gathered = gather_four_cls_states(hidden, panel, torch.ones_like(panel))
    assert gathered.shape == (1, 1024)
    chunks = gathered.view(1, 4, 256)
    loss = sum((index + 1) * chunks[:, index].sum() for index in range(4))
    loss.backward()
    for position in cls_positions(8):
        assert hidden.grad[0, position].abs().sum() > 0


def test_cls_gather_rejects_missing_or_masked_cls():
    panel = build_four_cls_panel(torch.arange(7, 11), bos_token_id=4).unsqueeze(0)
    mask = torch.ones_like(panel)
    mask[0, cls_positions(4)[1]] = 0
    with pytest.raises(ValueError, match="exactly four unmasked CLS"):
        gather_four_cls_states(torch.randn(1, panel.shape[1], 8), panel, mask)


def test_attention_schedule_preserves_indices_and_layer_zero_norm():
    model = ExpressionModernBertModel(small_config())
    assert isinstance(model.layers[0].attn_norm, nn.Identity)
    assert isinstance(model.layers[1].attn_norm, nn.LayerNorm)
    assert model.layers[0].attn.layer_id == 0
    assert model.layers[1].attn.layer_id == 1
    assert model.layers[0].attn.local_attention == (2, 2)
    assert model.layers[1].attn.local_attention == (-1, -1)


def test_required_config_has_exact_width_window_dtype_and_parameter_count():
    config_dir = Path(__file__).parents[1] / "configs/expression_profile_encoder_4cls"
    config = ModernBertConfig.from_pretrained(config_dir)
    config._attn_implementation = "eager"
    validate_expression_config(config)
    model = ExpressionModernBertModel(config)
    assert model.layers[0].attn.local_attention == (8, 8)
    assert model.layers[1].attn.local_attention == (-1, -1)
    assert config.torch_dtype == torch.bfloat16
    assert config.parameter_dtype == "float32"
    assert sum(parameter.numel() for parameter in model.parameters()) == 933_376


def test_schedule_and_weights_survive_save_reload(tmp_path):
    model = ExpressionModernBertModel(small_config())
    model.save_pretrained(tmp_path)
    reloaded = ExpressionModernBertModel.from_pretrained(tmp_path, attn_implementation="eager")
    assert tuple(reloaded.config.expression_attention_schedule) == EXPRESSION_ATTENTION_SCHEDULE
    assert reloaded.layers[0].attn.local_attention == (2, 2)
    assert reloaded.layers[1].attn.local_attention == (-1, -1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key])


def test_incompatible_legacy_config_and_weights_are_explicit(tmp_path):
    legacy = ModernBertConfig(
        vocab_size=57,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
    )
    legacy.expression_attention_schedule = ["local", "global"]
    legacy.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Legacy 1,024-wide weights cannot initialize"):
        load_expression_encoder(
            tmp_path,
            checkpoint_path=tmp_path,
            attn_implementation="eager",
        )

    state = {"tf.embeddings.weight": torch.ones(1), "bert.keep": torch.ones(1)}
    with pytest.warns(UserWarning, match="legacy 1,024-wide"):
        filtered = remove_legacy_expression_weights(state)
    assert list(filtered) == ["bert.keep"]


def test_gene_order_and_encoding_metadata_mismatch_are_rejected():
    with pytest.raises(ValueError, match="first mismatch at index 1"):
        validate_gene_order(["A", "C", "B"], ["A", "B", "C"], source="validation")

    tokenizer = StubTokenizer()
    genes = ["A", "B", "C"]
    edges = np.linspace(0.0, 1.0, 49, dtype=np.float32)
    metadata = build_encoding_metadata(genes, edges, tokenizer)
    assert validate_encoding_metadata(metadata, genes, edges, tokenizer) == metadata
    changed = json.loads(json.dumps(metadata))
    changed["gene_order_sha256"] = "historical-value-is-unknown"
    with pytest.raises(ValueError, match="gene_order_sha256"):
        validate_encoding_metadata(changed, genes, edges, tokenizer)


def test_small_cpu_forward_backward_integration():
    model = ExpressionModernBertModel(small_config())
    panel = build_four_cls_panel(torch.arange(7, 17), bos_token_id=4).unsqueeze(0)
    output = model(input_ids=panel, attention_mask=torch.ones_like(panel)).last_hidden_state
    output.retain_grad()
    conditioning = gather_four_cls_states(output, panel, torch.ones_like(panel))
    assert conditioning.shape == (1, 64)
    conditioning.square().mean().backward()
    for position in cls_positions(10):
        assert output.grad[0, position].abs().sum() > 0
    assert model.embeddings.tok_embeddings.weight.grad is not None
    assert model.layers[0].attn.Wqkv.weight.grad is not None
    assert model.layers[1].attn.Wqkv.weight.grad is not None


class FakeDnaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(57, 1024)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids), attentions=None)


class FakePredictionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1024))

    def forward(self, inputs_embeds, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=inputs_embeds * self.scale)


def expression_only_model():
    model = ExpressionCounts.__new__(ExpressionCounts)
    nn.Module.__init__(model)
    config_dir = Path(__file__).parents[1] / "configs/expression_profile_encoder_4cls"
    config = ModernBertConfig.from_pretrained(config_dir)
    config._attn_implementation = "eager"
    model.config = SimpleNamespace(use_return_dict=True)
    model.bert = FakeDnaEncoder()
    model.tf = ExpressionModernBertModel(config)
    model.tf_adapter = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024))
    model.dna_ln = nn.LayerNorm(1024)
    model.tf_ln = nn.LayerNorm(1024)
    model.register_buffer("tf_panel_bank", torch.empty((0, 0), dtype=torch.uint8), persistent=False)
    model.tf_mask_token_id = 3
    model.decoder = FakePredictionTransformer()
    model.classifier = nn.Linear(1024, 1)
    model.activation = nn.Identity()
    model.loss_fct = nn.MSELoss(reduction="none")
    model.weight = 1
    model.use_deviation_loss = False
    model.use_multinomial_loss = False
    model.weight_deviation_loss = 1.0
    model.weight_multinomial_loss = 1.0
    model._frozen_bert_modules = []
    model.expression_encoder_only = True
    model._configure_expression_encoder_only()
    return model


def test_resident_panel_bank_masks_targets_after_gather_and_is_not_checkpointed():
    model = expression_only_model().eval()
    panel = build_four_cls_panel(torch.arange(7, 17), bos_token_id=4)
    bank = torch.stack([panel]).to(torch.uint8)
    model.set_expression_panel_bank(bank)
    target_positions = torch.tensor([[1, 6]], dtype=torch.int32)
    captured = {}

    def capture_ids(_module, inputs):
        captured["ids"] = inputs[0].detach().clone()

    handle = model.tf.embeddings.tok_embeddings.register_forward_pre_hook(capture_ids)
    try:
        output = model(
            input_ids=torch.randint(0, 57, (1, 2, 4)),
            attention_mask=torch.ones(1, 2, 4, dtype=torch.long),
            dataset_flag=torch.ones(1, 2),
            tf_panel_indices=torch.zeros(1, 2, dtype=torch.int32),
            tf_target_positions=target_positions,
        )
    finally:
        handle.remove()

    assert output.logits.shape == (2, 1, 1)
    assert captured["ids"][0, 1].item() == 3
    assert captured["ids"][1, 6].item() == 3
    torch.testing.assert_close(model.tf_panel_bank.cpu(), bank)
    assert "tf_panel_bank" not in model.state_dict()


def test_expression_only_optimizer_updates_encoder_and_both_adapter_layers():
    torch.manual_seed(19)
    model = expression_only_model()
    model.train()
    assert model.tf.training and model.tf_adapter.training
    assert not model.bert.training
    assert not model.decoder.training
    assert not model.dna_ln.training
    assert not model.tf_ln.training
    assert not model.classifier.training

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(name.startswith(("tf.", "tf_adapter.")) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 3_032_576
    assert sum(parameter.numel() for parameter in model.tf_adapter.parameters()) == 2_099_200
    assert "tf_adapter.0.weight" in model.state_dict()
    assert "tf_adapter.2.weight" in model.state_dict()

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
    panel = build_four_cls_panel(torch.arange(7, 17), bos_token_id=4)
    tf_ids = torch.stack([panel, panel]).view(1, 2, -1)
    input_ids = torch.randint(0, 57, (1, 2, 4))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.zeros(1, 2, 4, 1)
    labels_mask = torch.ones_like(labels, dtype=torch.bool)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        labels_mask=labels_mask,
        tf_ids=tf_ids,
        tf_attention_mask=torch.ones_like(tf_ids),
        dataset_flag=torch.ones(1, 2),
    )
    output.loss.backward()

    assert model.tf_adapter[0].weight.grad.abs().sum() > 0
    assert model.tf_adapter[2].weight.grad.abs().sum() > 0
    assert model.tf.embeddings.tok_embeddings.weight.grad.abs().sum() > 0
    assert model.tf.layers[0].attn.Wqkv.weight.grad.abs().sum() > 0
    assert model.tf.layers[1].attn.Wqkv.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.bert.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.dna_ln.parameters())
    assert all(parameter.grad is None for parameter in model.tf_ln.parameters())
    assert all(parameter.grad is None for parameter in model.classifier.parameters())

    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    assert changed
    assert all(name.startswith(("tf.", "tf_adapter.")) for name in changed)
    assert any(name.startswith("tf.") for name in changed)
    assert "tf_adapter.0.weight" in changed
    assert "tf_adapter.2.weight" in changed


def test_checkpoint_reports_new_adapter_separately_from_frozen_missing_weights():
    model = expression_only_model()
    checkpoint = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("tf_adapter.") and name != "classifier.weight"
    }
    with pytest.warns(UserWarning) as captured:
        with pytest.raises(RuntimeError, match="frozen predictor components"):
            model.load_state_dict(checkpoint, strict=False)
    messages = [str(item.message) for item in captured]
    assert any("projection adapter from scratch" in message for message in messages)


def test_historical_conditioning_layernorm_is_remapped():
    model = expression_only_model()
    checkpoint = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("tf_adapter.") and not name.startswith("tf_ln.")
    }
    checkpoint["desc_ln.weight"] = torch.full_like(model.tf_ln.weight, 2.0)
    checkpoint["desc_ln.bias"] = torch.full_like(model.tf_ln.bias, 3.0)
    with pytest.warns(UserWarning, match="projection adapter from scratch"):
        result = model.load_state_dict(checkpoint, strict=False)
    assert "tf_ln.weight" not in result.missing_keys
    assert "tf_ln.bias" not in result.missing_keys
    torch.testing.assert_close(model.tf_ln.weight, torch.full_like(model.tf_ln.weight, 2.0))
    torch.testing.assert_close(model.tf_ln.bias, torch.full_like(model.tf_ln.bias, 3.0))
