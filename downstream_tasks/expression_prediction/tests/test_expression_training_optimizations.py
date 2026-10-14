import contextlib
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from downstream_tasks.expression_prediction.expression_profile_encoder import build_four_cls_panel
from downstream_tasks.expression_prediction.run_expression_finetuning_final_tf import (
    GradientAccumulationLoader,
    TrainerWithTest,
    build_expression_panel_bank,
)


class FakePanelDataset:
    def __init__(self, panels, unknown):
        self.all_keys = list(panels)
        self.tf_ids = panels
        self.tf_unknown_panel = unknown
        self.configured = None

    def expression_panel_entries(self):
        return list(self.tf_ids.items())

    def configure_expression_panel_indices(self, key_to_index, unknown_index):
        self.configured = (dict(key_to_index), unknown_index)


def test_expression_panel_bank_is_uint8_deduplicated_and_indexed():
    first = build_four_cls_panel(torch.tensor([7, 8, 9, 10]), bos_token_id=4)
    second = build_four_cls_panel(torch.tensor([10, 9, 8, 7]), bos_token_id=4)
    unknown = build_four_cls_panel(torch.ones(4, dtype=torch.long), bos_token_id=4)
    train = FakePanelDataset({"a": first, "b": second}, unknown)
    valid = FakePanelDataset({"a-copy": first.clone()}, unknown.clone())

    bank = build_expression_panel_bank([train, valid])

    assert bank.dtype == torch.uint8
    assert bank.shape == (3, first.numel())
    assert train.configured[0]["a"] == valid.configured[0]["a-copy"]
    assert train.configured[1] == valid.configured[1]


def test_accumulation_loader_groups_physical_dataloader_iterations():
    physical = DataLoader(TensorDataset(torch.arange(10)), batch_size=2, shuffle=False)
    grouped = GradientAccumulationLoader(physical, accumulation_steps=3)

    groups = list(grouped)

    assert len(grouped) == 2
    assert [len(group) for group in groups] == [3, 2]
    assert torch.cat([batch[0] for group in groups for batch in group]).tolist() == list(range(10))


class FakeAccelerator:
    def backward(self, loss):
        # Match Accelerate's configured two-way loss scaling.
        (loss / 2).backward()

    def no_sync(self, _model):
        return contextlib.nullcontext()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return {"loss": (x.float() * self.weight).mean().square()}


def test_training_step_accumulates_without_collecting_full_predictions():
    trainer = TrainerWithTest.__new__(TrainerWithTest)
    trainer.model = TinyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.accelerator = FakeAccelerator()
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.device = torch.device("cpu")
    trainer.model_forward_args = {"x"}
    trainer.batch_transform_fn = None
    trainer.batch_metrics_fn = lambda _batch, output: {"loss": output["loss"]}
    keep_calls = []
    trainer.keep_for_metrics_fn = lambda _batch, _output: keep_calls.append(1) or {"prediction": [1.0]}
    trainer.metrics_fn = lambda data: data
    trainer.forward_kwargs = {}
    trainer.clip_grad = None
    trainer.global_grad_norms = []
    trainer._get_gradients_global_norm = lambda: 0.0
    trainer.lr_scheduler = None

    before = trainer.model.weight.detach().clone()
    metrics, retained = trainer.step(
        ({"x": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([3.0, 4.0])}),
        is_train_mode=True,
    )

    assert metrics["loss"] > 0
    assert retained == {}
    assert keep_calls == []
    assert not torch.equal(before, trainer.model.weight.detach())

    _, retained = trainer.step({"x": torch.tensor([1.0, 2.0])}, is_train_mode=False)
    assert keep_calls == [1]
    assert retained["prediction"] == [[1.0]]
