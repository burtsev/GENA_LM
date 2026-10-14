import torch
import torch.nn as nn
import warnings
from transformers.modeling_outputs import TokenClassifierOutput
# from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig, ModernBertModel
from transformers.utils import cached_file
from transformers.utils import logging as hf_logging
from downstream_tasks.expression_prediction.expression_profile_encoder import (
    EXPRESSION_CLS_COUNT,
    gather_four_cls_states,
    load_expression_encoder,
    remove_legacy_expression_weights,
)
hf_logging.set_verbosity_warning()

@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    deviation_loss: Optional[torch.FloatTensor] = None
    multinomial_loss: Optional[torch.FloatTensor] = None
   
class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)
        
def cls_deviation_from_mean_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    n_keys: int,
    group_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Loss on matching deviations from the mean: deviations of logits from the group mean
    should match deviations of labels from the group mean (CLS only, mask=1 only).

    For each n_keys: group mean is computed only over positions with mask == 1.
    loss = MSE(logit_dev - label_dev) over masked positions in each group.

    logits: (B*N, L, 1)
    labels: (B*N, L, 1)
    labels_mask: (B*N, L, 1)
    n_keys: N — group size
    """
    B = logits.shape[0] // n_keys
    cls_logits = logits[:, 0:1, :]   # (B*N, 1, 1)
    cls_labels = labels[:, 0:1, :]   # (B*N, 1, 1)
    cls_mask = labels_mask[:, 0:1, :]  # (B*N, 1, 1)

    cls_logits = cls_logits.view(B, n_keys, 1, 1)   # (B, N, 1, 1)
    cls_labels = cls_labels.view(B, n_keys, 1, 1)   # (B, N, 1, 1)
    cls_mask = cls_mask.view(B, n_keys, 1, 1)       # (B, N, 1, 1)

    cnt = cls_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1, 1, 1)
    mean_logits = (cls_logits * cls_mask).sum(dim=1, keepdim=True) / cnt   # (B, 1, 1, 1)
    mean_labels = (cls_labels * cls_mask).sum(dim=1, keepdim=True) / cnt   # (B, 1, 1, 1)

    logit_dev = (cls_logits - mean_logits) * cls_mask   # (B, N, 1, 1)
    label_dev = (cls_labels - mean_labels) * cls_mask   # (B, N, 1, 1)
    dev_sq = (logit_dev - label_dev).pow(2)             # (B, N, 1, 1)

    loss_per_group = dev_sq.sum(dim=1) / cnt.squeeze(1)  # (B, 1, 1)
    group_has_mask = (cls_mask.sum(dim=1) > 0).view(B)  # (B,)
    if group_mask is not None:
        group_has_mask = group_has_mask & group_mask.to(cls_mask.device).bool().view(B)
    group_weights = group_has_mask.to(loss_per_group.dtype)
    return (loss_per_group.view(B) * group_weights).sum() / group_weights.sum().clamp_min(1.0)


import torch
import torch.nn.functional as F

def cls_multinomial_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    n_keys: int,
    min_target_max: float = 2.0,
    group_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multinomial (NLL) loss over each n_keys for CLS token only.
    In each group of n: masked log_softmax(logits) and NLL w.r.t. target distribution
    (labels normalized per group only over positions with mask == 1).
    Positions with mask 0 participate neither in target nor in softmax normalization.

    logits:       (B*N, L, 1)
    labels:       (B*N, L, 1)
    labels_mask:  (B*N, L, 1)
    n_keys:       N — group size
    """
    B = logits.shape[0] // n_keys

    # CLS-only (B*N, 1) -> (B*N,)
    cls_logits = logits[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_labels = labels[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_mask   = labels_mask[:, 0:1, :].squeeze(-1).squeeze(-1) # (B*N,)

    # (B, N)
    cls_logits = cls_logits.view(B, n_keys)
    cls_labels = cls_labels.view(B, n_keys)
    cls_mask   = cls_mask.view(B, n_keys)

    mask_bool = cls_mask > 0
    group_has_mask = mask_bool.any(dim=1)  # (B,)
    if group_mask is not None:
        group_has_mask = group_has_mask & group_mask.to(cls_mask.device).bool().view(B)
    target_max = (cls_labels * cls_mask).max(dim=1).values  # (B,)
    keep = group_has_mask & (target_max > min_target_max)

    eps = 1e-8
    w = cls_logits.masked_fill(~mask_bool, 0.0)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)

    p = w / w_sum                     
    log_probs = torch.log(p.clamp(min=eps))
    log_probs = log_probs.masked_fill(~mask_bool, 0.0)

    target = cls_labels * cls_mask
    target_sum = target.sum(dim=1, keepdim=True).clamp(min=eps)
    target = target / target_sum

    nll_per_group = -(target * log_probs).sum(dim=1)
    
    keep_weights = keep.to(nll_per_group.dtype)
    return (nll_per_group * keep_weights).sum() / keep_weights.sum().clamp_min(1.0)

class ExpressionCounts(nn.Module):
    """
    Expected shapes:
      - input_ids:      (B*N, L)
      - attention_mask: (B*N, L)
      - token_type_ids: (B*N, L) [optional]
      - tf_ids:            (B, N, T)  [BOS] + four gene quarters, each followed by [CLS]
      - tf_attention_mask: (B, N, T)  [optional] 1 - real token, 0 - padding
      - dataset_flag:   (B, N)   [in block of N elements either all 1 (INPUTS duplicates), or all 0 (DESC duplicates)]
      - labels:         (B*N, L, 1)
      - labels_mask:    (B*N, L, 1)
    """

    def __init__(
        self,
        config,
        hf_model_name_decoder,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        num_encoder_layers = 3,
        nhead = 8,
        weight = 1,
        hidden_ff = 1024,
        bert_cpt = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        tf_model_path: str = None,
        tf_config_path: str = None,
        tf_parameter_dtype: str = "float32",
        expression_encoder_only: bool = True,
        bert_trainable_layers: Optional[int] = None,
        bert_train_embeddings: bool = False,
        use_deviation_loss: bool = False,
        use_multinomial_loss: bool = False,
        weight_deviation_loss: float = 1.0,
        weight_multinomial_loss: float = 1.0,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        def _is_main_process():
            return (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            )

        updated_state_dict = None

        # 1) DNA model (GENA) 
        if hf:
            model_name = hf_model_name.lower()

            if "modern" in model_name:
                if _is_main_process():
                    print(f"Using ModernGENA from {hf_model_name}")
                self.bert, info = ModernBertModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    attention_dropout=dropout_prob,
                    embedding_dropout=dropout_prob,
                    mlp_dropout=dropout_prob,
                    output_loading_info=True,
                )
                config = self.bert.config
                if _is_main_process():
                    print("missing:", len(info["missing_keys"]), info["missing_keys"][:10])
                    print("unexpected:", len(info["unexpected_keys"]), info["unexpected_keys"][:10])
                    print("mismatched:", info.get("mismatched_keys", [])[:5])
                    print(
                        "bert dropouts:",
                        {
                            "attention_dropout": config.attention_dropout,
                            "embedding_dropout": config.embedding_dropout,
                            "mlp_dropout": config.mlp_dropout,
                        }
                    )

            elif "gena" in model_name:
                if _is_main_process():
                    print(f"Using GENA from {hf_model_name}")
                hf_config = BertConfig.from_pretrained(hf_model_name)
                self.bert = BertModel(hf_config, add_pooling_layer=False)
                weights_path = cached_file(hf_model_name, "pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location="cpu")
                updated_state_dict = {
                    k.replace("bert.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("bert.")
                }
                config = hf_config

            else:
                raise ValueError(
                    f"Unsupported hf_model_name: {hf_model_name}. "
                )
        else:
            self.bert = BertModel(config, add_pooling_layer=False)
            checkpoint = torch.load(bert_cpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            updated_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}
            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)

        if updated_state_dict is not None:
            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)
            if len(missing_k) != 0 and _is_main_process():
                print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
            if len(unexpected_k) != 0 and _is_main_process():
                print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")


        self.config = config
        self._frozen_bert_modules = self._freeze_bert(
            bert_trainable_layers, bert_train_embeddings, verbose=_is_main_process()
        )

        # 2) Expression panel encoder. A missing model path intentionally means scratch init;
        # the old 1,024-wide encoder is not shape-compatible with this 256-wide model.
        if tf_config_path is None:
            raise ValueError("tf_config_path must be provided")
        try:
            parameter_dtype = getattr(torch, tf_parameter_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown tf_parameter_dtype {tf_parameter_dtype!r}") from exc
        if _is_main_process():
            source = tf_model_path if tf_model_path is not None else f"scratch ({tf_config_path})"
            print(f"Using four-CLS expression panel encoder from {source}; parameter dtype={parameter_dtype}")
        self.tf = load_expression_encoder(
            config_path=tf_config_path,
            checkpoint_path=tf_model_path,
            attn_implementation="flash_attention_2",
            parameter_dtype=parameter_dtype,
        )

        # 3) Condition: concatenate four CLS states, normalize, and add to every DNA token.
        self.gen_hidden_size = config.hidden_size
        expression_width = EXPRESSION_CLS_COUNT * self.tf.config.hidden_size
        if expression_width != self.gen_hidden_size:
            raise ValueError(
                f"4 * expression hidden_size={expression_width} must match "
                f"DNA encoder hidden_size={self.gen_hidden_size}"
            )
        self.tf_adapter = nn.Sequential(
            nn.Linear(self.gen_hidden_size, self.gen_hidden_size),
            nn.GELU(),
            nn.Linear(self.gen_hidden_size, self.gen_hidden_size),
        )
        self.dna_ln = nn.LayerNorm(self.gen_hidden_size)
        self.tf_ln = nn.LayerNorm(self.gen_hidden_size)
        self.register_buffer(
            "tf_panel_bank", torch.empty((0, 0), dtype=torch.uint8), persistent=False
        )
        self.tf_mask_token_id = int(self.tf.config.mask_token_id)

        # 4) Decoder
        if _is_main_process():
            print(f"Using ModernBERT for dercoder from {hf_model_name_decoder}")
        self.decoder, info2 = ModernBertModel.from_pretrained(
                hf_model_name_decoder,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                attention_dropout=dropout_prob,
                embedding_dropout=dropout_prob,
                mlp_dropout=dropout_prob,
                output_loading_info=True
            )
        if _is_main_process():
            print("missing:", len(info2["missing_keys"]), info2["missing_keys"][:10])
            print("unexpected:", len(info2["unexpected_keys"]), info2["unexpected_keys"][:10])
            print("mismatched:", info2.get("mismatched_keys", [])[:5])
            print(
                "decoder dropouts:",
                {
                    "attention_dropout": self.decoder.config.attention_dropout,
                    "embedding_dropout": self.decoder.config.embedding_dropout,
                    "mlp_dropout": self.decoder.config.mlp_dropout,
                }
            )




        # 6) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight

        self.use_deviation_loss = use_deviation_loss
        self.use_multinomial_loss = use_multinomial_loss
        self.weight_deviation_loss = weight_deviation_loss
        self.weight_multinomial_loss = weight_multinomial_loss

        dtype = next(self.bert.parameters()).dtype
        device = next(self.bert.parameters()).device

        # 5) Classifier
        self.classifier = nn.Linear(self.decoder.config.hidden_size, 1, device=device, dtype=dtype)


        if hasattr(self.decoder, "embeddings") and hasattr(self.decoder.embeddings, "tok_embeddings"):
            self.decoder.embeddings.tok_embeddings.weight.requires_grad_(False)

        self.expression_encoder_only = expression_encoder_only
        if self.expression_encoder_only:
            self._configure_expression_encoder_only()

    @property
    def expression_encoder(self):
        """Backward-compatible accessor for code introduced with the four-CLS redesign."""
        return self.tf

    def set_expression_panel_bank(self, panel_bank: torch.Tensor) -> None:
        """Attach the validated canonical panel bank without serializing data in checkpoints."""
        if panel_bank.ndim != 2 or panel_bank.shape[0] == 0 or panel_bank.shape[1] == 0:
            raise ValueError(f"panel_bank must be a non-empty [profiles, tokens] tensor, got {panel_bank.shape}")
        if panel_bank.min().item() < 0 or panel_bank.max().item() >= self.tf.config.vocab_size:
            raise ValueError("Expression panel bank contains token IDs outside the encoder vocabulary")
        self.tf_panel_bank = panel_bank.detach().to(dtype=torch.uint8, device="cpu").contiguous()

    def _configure_expression_encoder_only(self):
        """Freeze the predictor and leave only the expression encoder and adapter trainable."""
        self.requires_grad_(False)
        self.tf.requires_grad_(True)
        self.tf_adapter.requires_grad_(True)

    def _freeze_bert(self, trainable_layers, train_embeddings, verbose=False):
        """Freeze the DNA encoder except its last `trainable_layers` layers.

        trainable_layers: None or < 0 -> the whole encoder is trainable (train_embeddings is ignored);
        0 -> every layer and the final norm are frozen; k -> the last k layers and the final norm are trainable.
        Returns the frozen submodules: they are kept in eval mode, see train().
        """
        if trainable_layers is None or trainable_layers < 0:
            if verbose:
                print("DNA encoder: fully trainable")
            return []

        if hasattr(self.bert, "layers"):        # ModernBERT
            layers = self.bert.layers
            final_norm = getattr(self.bert, "final_norm", None)
        elif hasattr(self.bert, "encoder"):     # BERT
            layers = self.bert.encoder.layer
            final_norm = None
        else:
            raise ValueError(f"Cannot find encoder layers in {type(self.bert).__name__}")

        n_layers = len(layers)
        if trainable_layers > n_layers:
            raise ValueError(f"bert_trainable_layers={trainable_layers} > number of layers {n_layers}")

        frozen = list(layers[: n_layers - trainable_layers])
        if trainable_layers == 0 and final_norm is not None:
            frozen.append(final_norm)
        if not train_embeddings:
            frozen.append(self.bert.embeddings)
        for module in frozen:
            module.requires_grad_(False)

        if verbose:
            n_trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.bert.parameters())
            print(
                f"DNA encoder: last {trainable_layers}/{n_layers} layers trainable, "
                f"embeddings trainable={train_embeddings}, "
                f"{n_trainable:.3e}/{n_total:.3e} parameters trainable"
            )
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self.expression_encoder_only:
            # Gradients still flow through these frozen modules to the conditioning
            # context, but their dropout and other training behavior remain disabled.
            for module in (self.bert, self.decoder, self.dna_ln, self.tf_ln, self.classifier):
                module.eval()
            if mode:
                self.tf.train()
                self.tf_adapter.train()
        else:
            # The trainer calls model.train() every step; frozen DNA layers must stay without dropout.
            for module in self._frozen_bert_modules:
                module.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Make legacy expression incompatibility explicit while preserving other weights."""
        state_dict = dict(state_dict)
        # all_datasets_2 used the same 1,024-wide conditioning LayerNorm for the
        # description branch. Preserve those compatible frozen weights.
        for suffix in ("weight", "bias"):
            current_key = f"tf_ln.{suffix}"
            historical_key = f"desc_ln.{suffix}"
            if current_key not in state_dict and historical_key in state_dict:
                if state_dict[historical_key].shape == self.state_dict()[current_key].shape:
                    state_dict[current_key] = state_dict[historical_key]
        state_dict = remove_legacy_expression_weights(state_dict, self.state_dict())
        incompatible = super().load_state_dict(state_dict, strict=strict, assign=assign)
        adapter_missing = [key for key in incompatible.missing_keys if key.startswith("tf_adapter.")]
        encoder_missing = [key for key in incompatible.missing_keys if key.startswith("tf.")]
        frozen_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("tf.", "tf_adapter."))
        ]
        if adapter_missing:
            warnings.warn(
                f"Initializing the new projection adapter from scratch; checkpoint omitted {len(adapter_missing)} "
                "tf_adapter tensors.",
                UserWarning,
            )
        if encoder_missing:
            warnings.warn(
                f"Initializing the redesigned expression encoder from scratch; checkpoint omitted "
                f"{len(encoder_missing)} tf tensors.",
                UserWarning,
            )
        if frozen_missing:
            raise RuntimeError(
                "Checkpoint is missing weights from frozen predictor components, which cannot be silently "
                "left randomly initialized in expression-only training: " + ", ".join(frozen_missing[:10])
            )
        return incompatible


    def forward(
        self,
        input_ids=None,              # (B*N, L)
        attention_mask=None,         # (B*N, L) or None
        labels_mask=None,            # (B*N, L, 1)
        labels=None,                 # (B*N, L, 1)
        return_dict=None,
        tf_ids=None,                 # (B, N, T)
        tf_attention_mask=None,      # (B, N, T) or None: 1 - real token, 0 - padding
        tf_panel_indices=None,       # (B, N), rows in the resident expression panel bank
        tf_target_positions=None,    # (B, N), token positions to replace by [MASK]
        dataset_flag=None,           # (B, N): 1 -> INPUTS duplicates; 0 -> DESC duplicates
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")
        
        B, N = dataset_flag.shape

        # 1) Reshape
        if input_ids is not None and input_ids.dim() == 3:          # (B, N, L) -> (B*N, L)
            if input_ids.shape[:2] != (B, N):
                raise ValueError(f"input_ids has shape {tuple(input_ids.shape)}, but dataset_flag is {(B, N)}")
            input_ids = input_ids.reshape(B * N, input_ids.shape[-1])

        if attention_mask is not None and attention_mask.dim() == 3: # (B, N, L) -> (B*N, L)
            attention_mask = attention_mask.reshape(B * N, attention_mask.shape[-1])

        # if token_type_ids is not None and token_type_ids.dim() == 3: # (B, N, L) -> (B*N, L)
        #     token_type_ids = token_type_ids.reshape(B * N, token_type_ids.shape[-1])

        if labels is not None and labels.dim() == 4:                 # (B, N, L, 1) -> (B*N, L, 1)
            labels = labels.reshape(B * N, labels.shape[-2], labels.shape[-1])

        if labels_mask is not None and labels_mask.dim() == 4:       # (B, N, L, 1) -> (B*N, L, 1)
            labels_mask = labels_mask.reshape(B * N, labels_mask.shape[-2], labels_mask.shape[-1])

        if tf_ids is None:
            if tf_panel_indices is None or tf_target_positions is None:
                raise ValueError(
                    "Provide tf_ids directly or provide both tf_panel_indices and tf_target_positions"
                )
            if tf_panel_indices.shape != (B, N) or tf_target_positions.shape != (B, N):
                raise ValueError(
                    "Panel indices and target positions must match dataset_flag shape "
                    f"{(B, N)}, got {tuple(tf_panel_indices.shape)} and {tuple(tf_target_positions.shape)}"
                )
        elif tf_ids.dim() == 3:                                           # (B, N, T) -> (B*N, T)
            tf_ids = tf_ids.reshape(B * N, tf_ids.shape[-1])

        if tf_attention_mask is not None and tf_attention_mask.dim() == 3:                        # (B, N, T) -> (B*N, T)
            tf_attention_mask = tf_attention_mask.reshape(B * N, tf_attention_mask.shape[-1])

        # 2) DNA model, remove duplicates
        src = input_ids 
        if src is None:
            raise ValueError("input_ids must be provided")
        device = src.device
        BxN, seq_len = src.shape[:2]
        B, N = dataset_flag.shape
        if B * N != BxN:
            raise ValueError(f"Batch mismatch: dataset_flag {tuple(dataset_flag.shape)} vs input_ids rows {BxN}")
        
        flag = dataset_flag.to(device).bool()     
        block_flag = flag[:, 0]                  
        idx_all = torch.arange(BxN, device=device)
        idx_grid = idx_all.view(B, N)            

        rep_inputs_idx = idx_grid[block_flag, 0]                         
        unique_inputs_idx_mode2 = idx_grid[~block_flag, :].reshape(-1)   
        idx_unique_inputs = torch.cat([unique_inputs_idx_mode2, rep_inputs_idx], dim=0)

        pos_in_compact = torch.full((BxN,), -1, dtype=torch.long, device=device)
        pos_in_compact[idx_unique_inputs] = torch.arange(idx_unique_inputs.numel(), device=device)

        map_inputs = torch.empty(BxN, dtype=torch.long, device=device)
        map_inputs[unique_inputs_idx_mode2] = pos_in_compact[unique_inputs_idx_mode2]
        if rep_inputs_idx.numel() > 0:
            rows_dup = idx_grid[block_flag, :].reshape(-1)
            rep_pos = pos_in_compact[rep_inputs_idx]                     
            map_inputs[rows_dup] = rep_pos.repeat_interleave(N)

        bert_outputs = self.bert(
                input_ids=input_ids[idx_unique_inputs],                         
                attention_mask=attention_mask[idx_unique_inputs],
                return_dict=True,
            )
        seq_compact = bert_outputs.last_hidden_state                    # (U_inp, L, H)
        if self.expression_encoder_only:
            seq_compact = self.dna_ln(seq_compact)
        sequence_output = seq_compact[map_inputs]                       # (B*N, L, H)

        # 3) Expression panel -> four CLS states in sequence order -> one 1,024-vector.
        if tf_ids is None:
            if self.tf_panel_bank.numel() == 0:
                raise RuntimeError("Expression panel bank has not been attached to the model")
            flat_indices = tf_panel_indices.reshape(-1).to(device=device, dtype=torch.long)
            target_positions = tf_target_positions.reshape(-1).to(device=device, dtype=torch.long)
            tf_ids = self.tf_panel_bank.index_select(0, flat_indices).clone()
            tf_ids.scatter_(1, target_positions[:, None], self.tf_mask_token_id)
            tf_ids = tf_ids.long()
        else:
            tf_ids = tf_ids.to(device=device, dtype=torch.long)
        if tf_attention_mask is not None:
            tf_attention_mask = tf_attention_mask.to(device)
        tf_hidden = self.tf(
            input_ids=tf_ids,
            attention_mask=tf_attention_mask,
            return_dict=True,
        ).last_hidden_state                                             # (B*N, T, 256)
        tf_output = gather_four_cls_states(
            tf_hidden, tf_ids, tf_attention_mask, validate=False
        )                                                               # (B*N, 1024)
        tf_output = self.tf_adapter(tf_output)                           # (B*N, 1024)

        if not self.expression_encoder_only:
            sequence_output = self.dna_ln(sequence_output)
        tf_output = self.tf_ln(tf_output.to(sequence_output.dtype))      # (B*N, H)

        tf_broadcast = tf_output[:, None, :] * attention_mask[:, :, None].to(sequence_output.dtype)
        sequence_output = sequence_output + tf_broadcast


        # 4) Decoder
        dec_out = self.decoder(
            inputs_embeds=sequence_output,        # (B*N, L, H)
            attention_mask=attention_mask,        # (B*N, L)
            return_dict=True,
        )
        decoder_output = dec_out.last_hidden_state  # (B*N, L, H)

        # final_tf supervises only the DNA CLS token; avoid materializing token-wide logits.
        logits = self.activation(self.classifier(decoder_output[:, :1, :]))  # (B*N, 1, 1)

        # 5) Loss
        loss = None
        labels_reshaped = labels_mask_reshaped = cls_loss = deviation_loss = multinomial_loss = other_loss = None

        

        if labels is not None:
            labels_reshaped = labels.to(logits.device)
            labels_mask_reshaped = labels_mask.to(logits.device) if labels_mask is not None else None

            cls_labels = labels_reshaped[:, 0:1, :]
            cls_mask = (
                labels_mask_reshaped[:, 0:1, :].to(logits.dtype)
                if labels_mask_reshaped is not None
                else torch.ones_like(logits)
            )
            unreduced_cls_loss = self.loss_fct(logits, cls_labels)
            cls_loss = (unreduced_cls_loss * cls_mask).sum() / cls_mask.sum().clamp_min(1.0)
            loss = cls_loss

            if labels_mask_reshaped is not None and self.use_deviation_loss:
                deviation_loss = cls_deviation_from_mean_loss(
                    logits, labels_reshaped, labels_mask_reshaped, n_keys=N, group_mask=block_flag
                )
                loss = loss + self.weight_deviation_loss * deviation_loss
            if labels_mask_reshaped is not None and self.use_multinomial_loss:
                multinomial_loss = cls_multinomial_loss(
                    logits, labels_reshaped, labels_mask_reshaped, n_keys=N, group_mask=block_flag
                )
                loss = loss + self.weight_multinomial_loss * multinomial_loss

        if not return_dict:
            return (loss, logits)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss,
            deviation_loss=deviation_loss,
            multinomial_loss=multinomial_loss,

        )
