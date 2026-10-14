# Expression adapter training report

Date: 2026-10-14

## Status

The expression branch is implemented and completed a checkpointed 1,000-step
real-data pilot through the frozen DNA encoder, prediction Transformer,
conditioning LayerNorms, and prediction head on an NVIDIA H100 80GB HBM3.

The run uses the 858 existing downloaded expression profiles. They cover three
raw measured-gene schemas, so each profile is explicitly aligned by Ensembl ID
to a persisted 24,759-gene canonical panel. Missing measurements become
`[UNK]`. The resulting sequence has 24,764 tokens, quarter sizes `6190, 6190,
6190, 6189`, and CLS positions `6191, 12382, 18573, 24763`. This is an explicit
data-dependent deviation from the requested 24,830-gene reference geometry;
the original reference geometry remains covered by tests.

## Trainable and frozen components

| Component | Parameters | Gradient | Runtime mode during training |
|---|---:|---|---|
| Four-CLS expression encoder | 933,376 | Enabled | Train |
| Projection adapter | 2,099,200 | Enabled | Train |
| **Trainable expression branch** | **3,032,576** | **Enabled** | **Train** |
| DNA encoder | 376,759,296 | Frozen | Eval |
| Prediction Transformer | 55,581,696 | Frozen | Eval |
| DNA and expression LayerNorms | 4,096 | Frozen | Eval |
| Prediction head | 1,025 | Frozen | Eval |

The optimizer receives only parameters named `tf.*` and `tf_adapter.*`.
Accelerate model and optimizer checkpoints therefore save the adapter weights
and their AdamW state automatically.

## Checkpoint initialization

The smoke run loaded the DNA encoder, prediction Transformer, DNA LayerNorm, and
prediction head from the downloaded `all_datasets_2/pytorch_model.bin`.
Its historical conditioning LayerNorm is named `desc_ln`; the compatible weights
are remapped to the frozen `tf_ln`. The new adapter's four tensors and the new
256-wide encoder's fourteen tensors were reported separately as intentional
scratch initialization. No frozen predictor tensors were missing after remapping.

Legacy 1,024-wide `tf.*` tensors are rejected as a unit if their shapes identify
the historical expression encoder. A genuinely missing frozen predictor tensor
now stops checkpoint loading instead of leaving a random frozen parameter.

## H100 real-profile pilot training

Command:

```bash
python -m downstream_tasks.expression_prediction.benchmarks.train_existing_profiles_pilot \
  --assets /mnt/promoter-pretrain/virtcell/pert_data/michael \
  --output-dir /mnt/promoter-pretrain/virtcell/runs/expression/final_tf-4cls-adapter-existing-profiles/pilot-1000steps \
  --steps 1000 --validation-samples 16
```

Each update used one real promoter target with up to 14 real expression
profiles, 1,024 DNA tokens, the complete 24,764-token expression panel, BF16
autocast, FP32 parameters, and AdamW. Samples were spread across ten genes.
Loss is not expected to be monotonic because each step targets a different gene
and profile group. Median training loss across all updates was 0.1181. On the
same 16 held-out samples, mean loss moved from 30.5543 to 1.5215 and median loss
from 0.7354 to 0.5082. The initial mean was dominated by one 463.17 outlier;
after training the largest held-out loss was 8.27. This is evidence that the
branch learns, but the small fixed validation sample is not a convergence claim.

Median steady-state update time at physical batch 1 was 237.46 ms.

Measured peak CUDA memory was 9,382,581,760 bytes allocated (8.74 GiB) and
9,527,361,536 bytes reserved (8.87 GiB). Both the encoder and adapter received
nonzero gradients. The initial checkpoint was missing the
four new adapter tensors and fourteen redesigned-encoder tensors, as intended,
and no frozen predictor tensors.

The loadable outputs are:

- full model: `/mnt/promoter-pretrain/virtcell/runs/expression/final_tf-4cls-adapter-existing-profiles/pilot-1000steps/pytorch_model.bin`;
- AdamW state: `optimizer.pt` in the same directory;
- matching standalone expression encoder and its complete encoding bundle:
  `expression_encoder/`;
- machine-readable measurements: `training_report.json`.

The full checkpoint is 1.7 GiB and contains `tf.*` and all four
`tf_adapter.*` tensors. The optimizer checkpoint is 24 MiB.

## Earlier synthetic wiring smoke

Command:

```bash
python downstream_tasks/expression_prediction/benchmarks/smoke_train_expression_adapter.py \
  --steps 3 --dna-length 32
```

The run used a full 24,835-token expression panel for 24,830 genes, BF16
autocast, FP32 parameters, AdamW, and the real frozen predictor checkpoints.
Synthetic DNA and targets were used only to validate training mechanics.

| Step | Loss | Encoder gradient norm | Adapter gradient norm | Time |
|---:|---:|---:|---:|---:|
| 1 | 13.140625 | 658.4515 | 926.8103 | 6610.77 ms¹ |
| 2 | 0.738525 | 48.1235 | 60.0448 | 42.97 ms |
| 3 | 0.347916 | 23.3637 | 31.1641 | 42.16 ms |

¹ The first step includes FlashAttention/kernel compilation and is not a steady-state timing.

Observed peak CUDA memory was 2,334,796,800 bytes allocated and 2,499,805,184
bytes reserved. All 18 trainable tensors changed. Sampled weights from the DNA
encoder, prediction Transformer, both LayerNorms, and prediction head were
bitwise unchanged. Both adapter Linear layers and both expression-attention
layers had nonzero gradients.

No checkpoint was retained from this synthetic run because its learned values
are not biologically meaningful. The report script verifies that all four
adapter tensors are present in `state_dict` and that normal downstream
checkpoint saving includes them.

## Encoder-branch benchmark

The isolated full-length comparison was rerun with the adapter included:

| Branch | Parameters | Peak allocated | Peak reserved | Median step |
|---|---:|---:|---:|---:|
| Legacy 1,024-wide encoder | 14,743,552 | 2,154.45 MiB | 2,472 MiB | 67.48 ms |
| Four-CLS encoder plus adapter | 3,032,576 | 584.61 MiB | 724 MiB | 17.66 ms |

This benchmark used batch size 1, 24,830 genes, BF16 autocast, FP32
parameters/AdamW states, FlashAttention 2, no gradient checkpointing, initialized
optimizer states, two warmups, and five measured updates.

## Automated validation

`tests/test_expression_profile_encoder.py` has 18 passing CPU tests. In addition
to the panel/schedule/serialization coverage, the adapter tests verify:

- exact adapter and trainable-branch parameter counts;
- frozen/eval versus trainable/train module modes;
- gradients through the frozen predictor into the encoder and both adapter layers;
- an optimizer update changes only the expression encoder and adapter;
- adapter tensors are included in model state;
- intentionally missing adapter weights are reported separately;
- compatible historical conditioning LayerNorm remapping;
- rejection of missing frozen predictor weights.

## Full-run status and remaining dependency

`configs/final_tf.yaml` and `finetune_expression_tf.sh` now point to the existing
profile mappings, genome, frozen predictor checkpoints, and generated canonical
bundle under `pert_data/michael`. The training split has 1,432,440 queued
samples over 23,874 target genes and 832 profiles; validation uses 26 held-out
H1 profiles. The archive has no independent test mapping, so validation is not
silently reused as test data.

The 1,000-step pilot is intentionally bounded. A statistically meaningful full
run remains to be scheduled. No expression-pretraining entry point exists in this
repository; standalone encoder pretraining still requires the external project
to use the same model class and encoding bundle.

## End-to-end batch-size benchmark and ETA

The real end-to-end path was measured with initialized AdamW state, BF16
autocast, FlashAttention 2, two warmups, and seven timed train and evaluation
steps per completed size. One dataset item contains 14 expression panels.

| Physical batch | Panels | Train step | Items/s | Peak allocated | Eval items/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 0.2362 s | 4.234 | 8.68 GiB | 16.97 |
| 2 | 28 | 0.4539 s | 4.407 | 15.60 GiB | 20.72 |
| 4 | 56 | 0.8604 s | 4.649 | 29.44 GiB | 23.45 |
| 6 | 84 | 1.3160 s | 4.559 | 43.28 GiB | 24.23 |
| **8** | **112** | **1.6947 s** | **4.721** | **57.12 GiB** | **24.66** |

Batch 8 is the fastest completed result and leaves about 22.0 GiB of device
memory outside peak allocations. The isolated batch-9 process terminated before
writing a result, so batch 10 was not attempted. The launcher now defaults to
physical batch 8 and the historical effective batch 80, which gives 10 gradient
accumulation microbatches on the available single H100.

At those settings, one optimizer update is estimated at 16.95 seconds. The
configured 350,000 updates require about **68.65 days** of training compute on
one H100. A complete 35,086-item validation pass takes about **23.7 minutes**.
With validation every 10,000 updates plus final validation, the total estimate
is **69.24 days**, excluding startup and checkpoint I/O. The previous interval
of 50 would have spent roughly 115 additional days validating, so validation
and checkpoint intervals are now 10,000.

Alternative budgets are easier to interpret:

- one full 1,432,440-item epoch at effective batch 80: about **3.53 days**
  including one validation pass;
- the same 350,000-item exposure as 350,000 batch-1 updates: about **21.0
  hours** including one validation pass (4,375 effective-batch-80 updates).

Machine-readable benchmark details are in
`runs/expression/final_tf-4cls-adapter-existing-profiles/batch_size_benchmark.json`.

## Training-path optimization update

The production path now keeps a deduplicated `uint8` expression-panel bank on
the GPU and transfers only panel indices plus target positions. Expression
attention masks are eliminated because every canonical panel is fixed-length.
Target masking happens after the device-side gather. Physical batch-8 DataLoader
iterations are accumulated sequentially in groups of ten instead of constructing
and transferring one 80-item tensor batch.

Training no longer copies full predictions, labels, and metadata to the CPU on
every microbatch. CUDA-synchronizing conditional loss and CLS checks were removed
from the model hot path; validation retains the strict checks and full prediction
collection. The TPM-only head and loss now operate only on DNA CLS, and the
frozen DNA LayerNorm runs before promoter representations are expanded across
profiles.

The updated path passed 22 CPU tests and a real-data H100 BF16
forward/backward/AdamW step. A warmed physical batch-8 remeasurement gave
1.6867 seconds (4.743 items/s), 57.16 GiB peak allocated, and 64.65 GiB peak
reserved. The earlier direct-model result was 1.6947 seconds (4.721 items/s);
the larger expected benefit is outside that microbenchmark, in avoided
DataLoader collation, host-to-device transfer, and training-metric synchronization.
