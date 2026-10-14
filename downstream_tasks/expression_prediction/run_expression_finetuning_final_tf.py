# stdlib
import json
import contextlib
import hashlib
import logging
import math
import os
import shutil
import time
from collections import defaultdict
from functools import partial
from itertools import chain, compress, islice
from pathlib import Path

# third-party
import torch
import numpy as np
import pandas as pd
import transformers
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# accelerate
import accelerate
import random
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list


# local
from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff, prepare_run
import lm_experiments_tools.optimizers as optimizers
from lm_experiments_tools import get_optimizer

from downstream_tasks.expression_prediction.expression_dataset_final import worker_init_fn
from downstream_tasks.expression_prediction.expression_profile_encoder import (
    EXPRESSION_CLS_COUNT,
    EXPRESSION_CLS_TOKEN_ID,
    cls_positions,
)
from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import score_predictions, mean_and_residuals_correlation
from downstream_tasks.expression_prediction.datasets.src.correlation_selected_cells import calculate_target_genes_metrics

def set_global_seed(seed: int):
    if seed is None:
        return
    print(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--experiment_config', type=str, help='path to the experiment config')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--save_predictions', action='store_true', help='save predictions to file')



def merge_default_params_with_dataset_config(dataset_config, default_params, logger):
    """
    Merge default parameters with dataset configuration.
    """
    def _merge_nested_dicts(target, source, path=""):
        for key, value in source.items():
            current_path = f"{path}.{key}" if path else key
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _merge_nested_dicts(target[key], value, current_path)
                else:
                    logger.warning(
                        f"Parameter '{current_path}' is specified in both default params and dataset config. "
                        f"Using dataset config value: {target[key]}"
                    )
            else:
                if isinstance(value, dict):
                    target[key] = value.copy()
                else:
                    target[key] = value
                logger.info(f"Added default parameter '{current_path}': {value}")

    merged_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    default_params_dict = OmegaConf.to_container(default_params, resolve=True) if hasattr(default_params, 'items') else default_params
    _merge_nested_dicts(merged_config, default_params_dict)
    return merged_config

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from accelerate.utils import broadcast_object_list


class TrainerWithTest(Trainer):
    """Evaluates the test split every time the valid split is evaluated, and logs the epoch.

    Only the valid metrics are returned, so optimize_metric and save_best keep working on valid.
    """

    def __init__(self, *args, test_dataloader=None, test_metrics_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataloader = test_dataloader
        self.test_metrics_fn = test_metrics_fn

    def step(self, batch, is_train_mode=True):
        """Process physical batches sequentially and synchronize once per optimizer update.

        The upstream trainer expects one materialized effective batch and slices it after
        moving every tensor to the GPU.  Training receives a tuple of physical DataLoader
        batches instead, so only one microbatch is transferred/resident at a time.
        """
        microbatches = list(batch) if isinstance(batch, (tuple, list)) else [batch]
        if len(microbatches) == 0:
            raise ValueError("An optimizer update needs at least one physical batch")

        if is_train_mode:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        metric_tensors = defaultdict(lambda: 0.0)
        metrics_data = defaultdict(list)
        n_microbatches = len(microbatches)

        with torch.set_grad_enabled(is_train_mode):
            for microbatch_index, microbatch in enumerate(microbatches):
                if self.batch_transform_fn:
                    microbatch = self.batch_transform_fn(microbatch)

                forward_batch = {}
                batch_sizes = []
                for key, value in microbatch.items():
                    if key in self.model_forward_args and isinstance(value, torch.Tensor):
                        value = value.to(self.device, non_blocking=True)
                        microbatch[key] = value
                        forward_batch[key] = value
                        batch_sizes.append(value.size(0))
                if not batch_sizes or any(size != batch_sizes[0] for size in batch_sizes):
                    raise RuntimeError(f"not all model inputs share a batch dimension: {batch_sizes}")

                is_last = microbatch_index == n_microbatches - 1
                sync_context = contextlib.nullcontext if (not is_train_mode or is_last) else self.accelerator.no_sync
                with sync_context(self.model):
                    outputs = self.model(**forward_batch, **self.forward_kwargs)
                    loss = outputs["loss"]

                    metrics = self.batch_metrics_fn(microbatch, outputs)
                    for key, value in metrics.items():
                        if isinstance(value, torch.Tensor):
                            metric_tensors[key] = metric_tensors[key] + value.detach() / n_microbatches
                        else:
                            metric_tensors[key] = metric_tensors[key] + value / n_microbatches

                    # Full predictions are required for validation metrics, but copying
                    # them every training microbatch serializes the GPU and is unnecessary.
                    if not is_train_mode and self.keep_for_metrics_fn and self.metrics_fn:
                        for key, value in self.keep_for_metrics_fn(microbatch, outputs).items():
                            metrics_data[key].append(value.detach().cpu() if isinstance(value, torch.Tensor) else value)

                    if is_train_mode:
                        # Accelerator divides by configured accumulation steps. Rescale a
                        # short final group so its gradients remain a true microbatch mean.
                        scale = self.args.gradient_accumulation_steps / n_microbatches
                        self.accelerator.backward(loss * scale)

        if is_train_mode:
            if self.clip_grad:
                global_grad_norm = self._clip_gradients()
            else:
                global_grad_norm = self._get_gradients_global_norm()
            self.global_grad_norms.append(global_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

        # One scalar synchronization per reported metric and optimizer/eval step, not
        # one synchronization for every microbatch and every conditional loss branch.
        batch_metrics = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in metric_tensors.items()
        }
        return batch_metrics, metrics_data

    def collect_metrics(self, split: str):
        if split != "train":
            return super().collect_metrics(split)
        keep_for_metrics_fn = self.keep_for_metrics_fn
        try:
            self.keep_for_metrics_fn = None
            return super().collect_metrics(split)
        finally:
            self.keep_for_metrics_fn = keep_for_metrics_fn

    def validate(self, dataloader, split='valid', write_tb=True):
        metrics = super().validate(dataloader, split=split, write_tb=write_tb)
        if split == 'valid' and self.test_dataloader is not None:
            valid_metrics_fn = self.metrics_fn
            self.metrics_fn = self.test_metrics_fn
            try:
                super().validate(self.test_dataloader, split='test', write_tb=write_tb)
            finally:
                self.metrics_fn = valid_metrics_fn
        if split == 'valid':
            self._log_epoch(write_tb)
        return metrics

    def _log_epoch(self, write_tb):
        """Epoch as a fractional number: the integer part is the epoch, the rest is its progress."""
        if not (write_tb and self.tb and self.accelerator.is_main_process):
            return
        n_samples = self.n_iter * self.global_batch_size
        dataset = getattr(self.train_dataloader, "dataset", None)
        epoch = n_samples / len(dataset) if dataset is not None and len(dataset) > 0 else self.n_epoch
        self.tb.add_scalar('epoch/iterations', epoch, self.n_iter)
        self.tb.add_scalar('epoch/samples', epoch, n_samples)
        self.tb.flush()


def find_last_checkpoint(model_path_root) -> Optional[str]:
    """The latest model_<iteration> checkpoint under model_path_root/<run>/, or None.

    Only checkpoints with a full accelerate state are taken, so the optimizer, the scheduler
    and the iteration can be restored. model_best is skipped on purpose: a killed run has to
    continue from its last iteration, not from its best metric.
    """
    root = Path(model_path_root)
    if not root.is_dir():
        return None
    checkpoints = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("model_*"):
            iteration = ckpt.name[len("model_"):]
            if iteration.isdigit() and (ckpt / "accelerate_state").is_dir() and (ckpt / "trainer.pckl").is_file():
                checkpoints.append((int(iteration), str(ckpt)))
    return max(checkpoints)[1] if checkpoints else None


def _collect_dataset_configs(experiment_config, prefix: str) -> List[Any]:
    """Берёт все ключи вида train_dataset*, valid_dataset*, test_dataset*."""
    return [v for k, v in experiment_config.items() if str(k).startswith(prefix)]


def merge_shared_params_and_check_n_keys(
    dataset_cfgs: List[Any],
    shared_dataset_params: Optional[Any],
    merge_fn,
    split: str,
    alogger,
) -> Tuple[List[Any], Optional[int]]:
    """
    Мёрджит shared params (значение в блоке датасета важнее shared) и проверяет n_keys.
    n_keys обязан быть задан явно (в shared_dataset_params или в блоке датасета) и одинаков
    внутри сплита: collate_fn складывает сэмплы разных датасетов в один батч.
    """
    out = [cfg.copy() for cfg in dataset_cfgs]
    if shared_dataset_params is not None:
        for i, cfg in enumerate(out):
            out[i] = merge_fn(cfg, shared_dataset_params, alogger)

    values = set()
    for cfg in out:
        v = cfg.get("n_keys", None)
        if v is None or int(v) <= 0:
            raise ValueError(f"n_keys must be set explicitly (> 0) for every {split} dataset, got {v}")
        values.add(int(v))
    if len(values) > 1:
        raise ValueError(f"All {split} datasets must share one n_keys, got {sorted(values)}")

    n_keys = values.pop() if values else None
    alogger.info(f"[n_keys] {split}: {n_keys}")
    return out, n_keys


def build_dataset_from_cfgs(dataset_cfgs: List[Any]) -> Tuple[Dataset, List[Dataset]]:
    """Instantiate -> Dataset or ConcatDataset."""
    datasets = [instantiate(cfg) for cfg in dataset_cfgs]
    if len(datasets) == 0:
        raise ValueError("No datasets after instantiate()")
    if len(datasets) == 1:
        return datasets[0], datasets
    return ConcatDataset(datasets), datasets


def validate_expression_encoding_consistency(datasets: List[Dataset]) -> str:
    """Require every training/evaluation dataset to use one checkpoint encoding."""
    missing = [type(dataset).__name__ for dataset in datasets if not hasattr(dataset, "expression_encoding_signature")]
    if missing:
        raise ValueError(f"Datasets do not expose expression encoding metadata: {missing}")
    signatures = {dataset.expression_encoding_signature for dataset in datasets}
    if len(signatures) != 1:
        raise ValueError(
            "Training, validation, and test datasets use inconsistent expression gene order, bins, "
            f"tokenizer, CLS placement, or attention schedule: {sorted(signatures)}"
        )
    return signatures.pop()


def persist_expression_encoding_artifacts(shared_params, model_path: Path, tf_tokenizer: str) -> None:
    """Keep the canonical encoding definition next to downstream checkpoints."""
    destination = model_path / "expression_encoding"
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("panel_genes_path", "bin_edges_path", "encoding_metadata_path"):
        source = Path(str(shared_params[key])).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Required expression encoding artifact is missing: {source}")
        shutil.copy2(source, destination / source.name)
    AutoTokenizer.from_pretrained(tf_tokenizer).save_pretrained(destination)


def build_loader(
    dataset: Dataset,
    accelerator,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    collate_fn,
    worker_init_fn,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler


class GradientAccumulationLoader:
    """Group physical DataLoader iterations into one optimizer-update unit."""

    def __init__(self, loader: DataLoader, accumulation_steps: int):
        if accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}")
        self.loader = loader
        self.accumulation_steps = accumulation_steps
        self.dataset = loader.dataset
        self.sampler = loader.sampler

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            group = tuple(islice(iterator, self.accumulation_steps))
            if not group:
                return
            yield group

    def __len__(self):
        return math.ceil(len(self.loader) / self.accumulation_steps)


def build_expression_panel_bank(datasets: List[Dataset]) -> torch.Tensor:
    """Deduplicate dataset panels and configure compact per-sample bank indices.

    The returned uint8 tensor is registered as a non-persistent model buffer.  With
    the 57-token vocabulary, all existing profiles fit in about 20 MiB on the GPU.
    """
    panels = []
    digest_to_indices = defaultdict(list)

    def add_panel(panel: torch.Tensor) -> int:
        panel = panel.detach().to(dtype=torch.uint8, device="cpu").contiguous()
        if panel.ndim != 1:
            raise ValueError(f"Expression panels must be rank one, got {tuple(panel.shape)}")
        expected_positions = cls_positions(panel.numel() - 1 - EXPRESSION_CLS_COUNT)
        observed_cls = panel.eq(EXPRESSION_CLS_TOKEN_ID).nonzero(as_tuple=False).flatten().tolist()
        if observed_cls != list(expected_positions):
            raise ValueError(
                f"Panel CLS positions {observed_cls} do not match canonical positions {expected_positions}"
            )
        digest = hashlib.sha256(panel.numpy().tobytes()).digest()
        for index in digest_to_indices[digest]:
            if torch.equal(panels[index], panel):
                return index
        index = len(panels)
        panels.append(panel)
        digest_to_indices[digest].append(index)
        return index

    for dataset in datasets:
        key_to_index = {
            key: add_panel(panel) for key, panel in dataset.expression_panel_entries()
        }
        unknown_index = add_panel(dataset.tf_unknown_panel)
        dataset.configure_expression_panel_indices(key_to_index, unknown_index)

    if not panels:
        raise ValueError("Cannot build an expression panel bank from an empty dataset list")
    lengths = {panel.numel() for panel in panels}
    if len(lengths) != 1:
        raise ValueError(f"Expression panels have inconsistent lengths: {sorted(lengths)}")
    return torch.stack(panels)

def main():
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    #  Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs]
    )
    #  mixed_precision="bf16",
    alogger = get_logger(__name__) 
   # alogger = get_logger('')  # accelerate logger (привязан к процессу)
    alogger.info(f'num processes: {accelerator.num_processes}')
    alogger.info(f'mixed precision: {accelerator.mixed_precision}')
    alogger.info(f'accelerator state: {accelerator.state}')

    prepare_run(args, alogger, logger_fmt, accelerator=accelerator)

    if accelerator.is_main_process:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = None
    obj = [timestamp]
    broadcast_object_list(obj)
    timestamp = obj[0]

    experiment_config_path = Path(args.experiment_config).expanduser().absolute()

    with initialize_config_dir(str(experiment_config_path.parents[0])):
        experiment_config = compose(config_name=experiment_config_path.name)

    if "args_params" in experiment_config:
        trainer_kwargs = instantiate(experiment_config["args_params"])
        for k, v in trainer_kwargs.items():
            if hasattr(args, k):
                alogger.warning(f"Setting attr {k}:{v} (overwritten by cfg)")
            else:
                alogger.info(f"Setting attr {k}:{v}")
            args.__setattr__(k, v)

    set_global_seed(args.seed)

    # A killed run continues by itself: the last checkpoint of the previous run is found,
    # its own directory is reused, so TensorBoard keeps writing into the same log.
    auto_resume_ckpt = None
    if args.resume is None:
        if accelerator.is_main_process:
            auto_resume_ckpt = find_last_checkpoint(args.model_path)
        obj = [auto_resume_ckpt]
        broadcast_object_list(obj)
        auto_resume_ckpt = obj[0]

    if args.resume is not None:
        alogger.warning(
            f"Resuming from cpt {args.resume}. This will overwrite reset_lr, reset_optimizer, reset_iteration and init_cpt options"
        )
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", args.model_path + "/" + args.resume)
        args.__setattr__("model_path", args.model_path + "/resume_" + args.resume + "/")
    elif auto_resume_ckpt is not None:
        alogger.warning(
            f"Continuing the interrupted run from {auto_resume_ckpt}: model, optimizer, scheduler "
            f"and iteration are restored, and the run keeps its own directory and TensorBoard log"
        )
        # a directory (not a .bin file) plus these three flags is what makes Trainer.load()
        # restore the full accelerate state instead of the weights only
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", auto_resume_ckpt)
        args.__setattr__("model_path", str(Path(auto_resume_ckpt).parent))

    if accelerator.is_main_process:
        if args.model_path is None:
            raise ValueError("Model path should not be None")

    if auto_resume_ckpt is None:
        args.model_path = os.path.join(args.model_path, timestamp)
    alogger.info(f"rank: {accelerator.process_index}, Model path: {args.model_path}")

    if accelerator.is_main_process and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        # on a continued run the files of the original run are kept as they are
        suffix = "" if auto_resume_ckpt is None else f".{timestamp}"
        args_dict = collect_run_configuration(args)
        json.dump(args_dict, open(model_path / f'config{suffix}.json', 'w'), indent=4)
        open(model_path / f'git{suffix}.diff', 'w').write(get_git_diff())
        # Сохраняем копию Hydra-конфига
        content = "\n".join(open(experiment_config_path).readlines())
        with open(Path(args.model_path) / f"experiment_config{suffix}.yaml", "w") as fout:
            fout.write(content)

    accelerator.wait_for_everyone()

    # Padding
    tokenizer = AutoTokenizer.from_pretrained(args.gen_tokenizer, trust_remote_code=True)
    if getattr(args, "tf_tokenizer", None) is None:
        raise ValueError("args_params.tf_tokenizer must be set in the experiment config")

    if accelerator.is_main_process:
        shared_for_artifacts = OmegaConf.to_container(
            experiment_config.get("shared_dataset_params", {}), resolve=True
        )
        persist_expression_encoding_artifacts(
            shared_for_artifacts, Path(args.model_path), args.tf_tokenizer
        )
    accelerator.wait_for_everyone()

    def _pad_1d(x: torch.Tensor, length: int, pad_value: int, pad_left: bool = False) -> torch.Tensor:
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        pad = x.new_full((pad_len,), pad_value)
        return torch.cat([pad, x], dim=0) if pad_left else torch.cat([x, pad], dim=0)


    def _pad_2d(x: torch.Tensor, max_len: int, pad_value, dim: int = 0) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)


    def _pad_3d(x: torch.Tensor, max_len: int, pad_value, dim: int = 1) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)

    def collate_fn(batch):
        pad_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'labels_mask']
        no_pad_keys = ['tpm', 'dataset_flag']
        special_keys = ['gene_id', 'selected_keys', 'dataset_description', 'name', 'chrom', 'reverse', 'start', 'end']

        pad_token_ids = {
            'input_ids': tokenizer.pad_token_id,
            'attention_mask': 0,
            'token_type_ids': 0,
            'labels': 0.0,
            'labels_mask': 0,
        }

        max_seq_len = max(sample['input_ids'].size(1) for sample in batch)

        batch_dict = {key: [] for key in pad_keys + no_pad_keys + special_keys}

        for sample in batch:
            for key in pad_keys:
                x = sample[key]
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:  # (n_keys, L)
                    x = _pad_2d(x, max_seq_len, pad_token_ids[key], dim=1)    # pad по L -> dim=1
                if key in ['labels', 'labels_mask']:                                                       
                    x = _pad_3d(x, max_seq_len, pad_token_ids[key], dim=1)
                batch_dict[key].append(x)

            for key in no_pad_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

            for key in special_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

        # stack
        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)  # (B, n_keys, Lmax) или (B, n_keys, Lmax, 1)

        for key in no_pad_keys:
            if len(batch_dict[key]) > 0:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

        batch_dict['tf_panel_indices'] = torch.stack(
            [sample['tf_panel_indices'] for sample in batch], dim=0
        )
        batch_dict['tf_target_positions'] = torch.stack(
            [sample['tf_target_positions'] for sample in batch], dim=0
        )

        return batch_dict


    # Data
    kwargs_workers = args.data_n_workers

    train_cfgs = _collect_dataset_configs(experiment_config, "train_dataset")
    valid_cfgs = _collect_dataset_configs(experiment_config, "valid_dataset")
    test_cfgs = _collect_dataset_configs(experiment_config, "test_dataset")
    shared_dataset_params = experiment_config.get("shared_dataset_params", None)

    if len(train_cfgs) == 0:
        raise ValueError("No training datasets found (no train_dataset* in config)")
    if len(test_cfgs) > 0 and len(valid_cfgs) == 0:
        raise ValueError("test_dataset* is evaluated together with validation, so valid_dataset* is required")

    train_cfgs, n_keys_global_train = merge_shared_params_and_check_n_keys(
        train_cfgs, shared_dataset_params, merge_default_params_with_dataset_config, "train", alogger
    )
    valid_cfgs, n_keys_global_valid = merge_shared_params_and_check_n_keys(
        valid_cfgs, shared_dataset_params, merge_default_params_with_dataset_config, "valid", alogger
    )
    test_cfgs, n_keys_global_test = merge_shared_params_and_check_n_keys(
        test_cfgs, shared_dataset_params, merge_default_params_with_dataset_config, "test", alogger
    )

    expanded_train_cfgs = []
    for cfg in train_cfgs:
        repeat = int(cfg.get("repeat_factor", 1))
        expanded_train_cfgs.extend([cfg.copy()] * repeat)

    train_cfgs = expanded_train_cfgs

    train_dataset, train_datasets_list = build_dataset_from_cfgs(train_cfgs)
    if accelerator.is_main_process:
        for i, ds in enumerate(train_datasets_list):
            alogger.info(f"train dataset {i}: {ds.describe() if hasattr(ds,'describe') else type(ds)}")
        alogger.info(f"total len(train_dataset)={len(train_dataset)} | n_keys_global={n_keys_global_train}")

    train_dataloader, train_sampler = build_loader(
        dataset=train_dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        drop_last=False,
        num_workers=kwargs_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    train_dataloader = GradientAccumulationLoader(
        train_dataloader, args.gradient_accumulation_steps
    )

    if len(valid_cfgs) > 0:
        valid_dataset, valid_datasets_list = build_dataset_from_cfgs(valid_cfgs)
        if accelerator.is_main_process:
            for i, ds in enumerate(valid_datasets_list):
                alogger.info(f"valid dataset {i}: {ds.describe() if hasattr(ds,'describe') else type(ds)}")
            alogger.info(f"total len(valid_dataset)={len(valid_dataset)} | n_keys_global={n_keys_global_valid}")

        valid_dataloader, valid_sampler = build_loader(
            dataset=valid_dataset,
            accelerator=accelerator,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=False,
            drop_last=False,
            num_workers=kwargs_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )

        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        valid_sampler = None
        if accelerator.is_main_process:
            alogger.info("No validation data is used.")

    if len(test_cfgs) > 0:
        test_dataset, test_datasets_list = build_dataset_from_cfgs(test_cfgs)
        if accelerator.is_main_process:
            for i, ds in enumerate(test_datasets_list):
                alogger.info(f"test dataset {i}: {ds.describe() if hasattr(ds,'describe') else type(ds)}")
            alogger.info(f"total len(test_dataset)={len(test_dataset)} | n_keys_global={n_keys_global_test}")

        test_dataloader, _ = build_loader(
            dataset=test_dataset,
            accelerator=accelerator,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=False,
            drop_last=False,
            num_workers=kwargs_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )
    else:
        test_dataloader = None
        test_datasets_list = []
        if accelerator.is_main_process:
            alogger.info("No test data is used.")

    encoding_signature = validate_expression_encoding_consistency(
        train_datasets_list + (valid_datasets_list if len(valid_cfgs) > 0 else []) + test_datasets_list
    )
    expression_panel_bank = build_expression_panel_bank(
        train_datasets_list + (valid_datasets_list if len(valid_cfgs) > 0 else []) + test_datasets_list
    )
    if accelerator.is_main_process:
        alogger.info(f"expression encoding signature: {encoding_signature}")
        alogger.info(
            f"expression panel bank: {tuple(expression_panel_bank.shape)}, "
            f"{expression_panel_bank.numel() / (1024 ** 2):.2f} MiB uint8"
        )

    if accelerator.is_main_process:
        alogger.info(
        f"GLOBAL: len(train_dataset)={len(train_dataset)} | n_keys_global={n_keys_global_train} "
    )


    # Model 

    if "model_kwargs" in experiment_config:
        model_kwargs = instantiate(experiment_config["model_kwargs"])
    else:
        model_kwargs = {}

    model_cls = get_cls_by_name(args.model_cls)
    if accelerator.is_main_process:
        alogger.info(f'Using model class: {model_cls}')
    model = model_cls(**model_kwargs)
    model.set_expression_panel_bank(expression_panel_bank)
    expression_encoder = getattr(model, "expression_encoder", None)
    checkpoint_encoding_signature = getattr(
        getattr(expression_encoder, "config", None),
        "expression_encoding_signature",
        None,
    )
    if checkpoint_encoding_signature is not None and checkpoint_encoding_signature != encoding_signature:
        raise ValueError(
            "The expression checkpoint encoding does not match the training/validation/test encoding: "
            f"checkpoint={checkpoint_encoding_signature}, datasets={encoding_signature}"
        )
    model_activation_fn = model.activation

    # Optimizer 
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if accelerator.is_main_process:
        alogger.info(f'Using optimizer class: {optimizer_cls}')

    # Frozen predictor parameters must not enter AdamW (weight decay would still move them).
    trainable_named_params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if getattr(model, "expression_encoder_only", False):
        unexpected_trainable = [
            name for name, _ in trainable_named_params if not name.startswith(("tf.", "tf_adapter."))
        ]
        if unexpected_trainable:
            raise RuntimeError(
                "Expression-only training exposed frozen predictor parameters: "
                + ", ".join(unexpected_trainable[:10])
            )
    trainable_params = [parameter for _, parameter in trainable_named_params]
    if accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        alogger.info(f"trainable parameters: {n_trainable:.3e} / {n_total:.3e}")
        alogger.info(
            "trainable expression branch: "
            f"encoder={sum(p.numel() for p in model.tf.parameters() if p.requires_grad):,}, "
            f"adapter={sum(p.numel() for p in model.tf_adapter.parameters() if p.requires_grad):,}"
        )

    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        optimizer = optimizer_cls(
            trainable_params, lr=args.lr,
            scale_parameter=args.scale_parameter,
            relative_step=args.relative_step,
            warmup_init=args.warmup_init,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optimizer_cls(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Batch transform
    def batch_transform_fn(batch):
        result = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels_mask': batch['labels_mask'],
            'dataset_flag': batch['dataset_flag'],
            'labels': batch['labels'],
            'gene_id': batch['gene_id'],
            'selected_keys': batch['selected_keys'],
            'dataset_description': batch['dataset_description'],
            'tf_panel_indices': batch['tf_panel_indices'],
            'tf_target_positions': batch['tf_target_positions'],
        }
        return result
    
    # Metrics
    def keep_for_metrics_fn(batch, output):
        logits = output["logits"].detach().cpu()
        labels = output["labels_reshaped"].detach().cpu()
        masks  = output["labels_mask_reshaped"].detach().cpu()

        y_true = labels[:, 0, 0]      
        y_pred = logits[:, 0, 0]     
        mask   = masks[:, 0, 0] > 0   

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        preds = y_pred.unsqueeze(1)   
        target = y_true.unsqueeze(1) 
        reduce_dims = (0, 1)

        data = {} 
        
        for k in ["cls_loss", "other_loss", "multinomial_loss", "deviation_loss"]:
            if k in output and output[k] is not None:
                data[k] = output[k].detach().cpu()

        # `gene_id` and `selected_keys` are already compacted in the dataset to contain
        # only entries with valid TPM targets. In contrast, `mask` lives in the full
        # B * n_keys space, so indexing compact metadata with absolute `mask_idx`
        # can go out of range. Keep metadata in the same compact order as y_true/y_pred.
        gene_id = list(chain.from_iterable(batch['gene_id']))
        keys_id = list(chain.from_iterable(batch['selected_keys']))
        dataset_description = [
            (descs[0] if isinstance(descs, (list, tuple)) and len(descs) > 0 else descs)
            for descs, gene_ids in zip(batch['dataset_description'], batch['gene_id'])
            for _ in range(len(gene_ids))
        ]
        data['tpm_true'] = y_true.tolist()
        data['tpm_preds'] = y_pred.tolist()
        data['gene_id'] = gene_id
        data['keys_id'] = keys_id
        data['dataset_description'] = dataset_description

        assert len(data['tpm_true']) == len(data['gene_id']) == len(data['keys_id']) == len(data['dataset_description']), \
            (
                "Mismatch in compacted metadata sizes: "
                f"tpm_true={len(data['tpm_true'])}, "
                f"gene_id={len(data['gene_id'])}, "
                f"keys_id={len(data['keys_id'])}, "
                f"dataset_description={len(data['dataset_description'])}"
            )

        return data

    def make_metrics_fn(model_path, save_predictions=False, file_prefix=""):
        def metrics_fn(data):
            metrics = {}
            for k in ["cls_loss", "other_loss", "multinomial_loss", "deviation_loss"]:
                if k in data and data[k] is not None:
                    metrics[k] = torch.mean(data[k]).item()
            tpm_true = data['tpm_true']
            tpm_preds = data['tpm_preds']
            gene_id = data['gene_id']
            keys_id = data['keys_id']
            dataset_description = data['dataset_description']

            assert len(tpm_true) == len(tpm_preds) == len(gene_id) == len(keys_id) == len(dataset_description), \
                f"Mismatch! tpm_true: {len(tpm_true)}, tpm_preds: {len(tpm_preds)}, gene_id: {len(gene_id)}, keys_id {len(keys_id)}, dataset_description {len(dataset_description)}"

            df = pd.DataFrame({
                'gene_id': gene_id,
                'cell_type': keys_id,
                'tpm_true': tpm_true,
                'tpm_pred': tpm_preds,
                'dataset_description': dataset_description,
            })
            if save_predictions and accelerator.is_main_process:
                df.to_csv(os.path.join(model_path, f"{file_prefix}labels.csv"))

            for dataset_desc in df['dataset_description'].unique():
                df_dataset = df[df['dataset_description'] == dataset_desc]

                df_pred = df_dataset.pivot_table(
                    index='gene_id',
                    columns='cell_type',
                    values='tpm_pred',
                    aggfunc='first'
                )
                df_true = df_dataset.pivot_table(
                    index='gene_id',
                    columns='cell_type',
                    values='tpm_true',
                    aggfunc='first'
                )
                if save_predictions and accelerator.is_main_process:
                    df_true.to_csv(os.path.join(model_path, f"{file_prefix}{dataset_desc}_true.csv"))
                    df_pred.to_csv(os.path.join(model_path, f"{file_prefix}{dataset_desc}_pred.csv"))

                if df_true.empty or df_pred.empty:
                    continue

                if not df_true.empty and not df_pred.empty:
                    gene_correlations = []
                    for gene in df_true.index:
                        gene_true = df_true.loc[gene]
                        gene_pred = df_pred.loc[gene]
                        mask = pd.notna(gene_true) & pd.notna(gene_pred)
                        gene_true = gene_true[mask]
                        gene_pred = gene_pred[mask]

                        # if len(gene_pred) > 1 and np.std(gene_pred) == 0:
                        #     alogger.error(f"dataset {dataset_desc} gene {gene} has all predicted values the same")
                        #     raise ValueError(f"All predicted values for {gene} are the same. Are you missing cell type descriptions?")

                        if len(gene_true) > 3 and np.std(gene_true) > 0:
                            try:
                                corr = np.corrcoef(gene_true, gene_pred)[0, 1]
                                if not np.isnan(corr):
                                    gene_correlations.append(corr)
                            except Exception:
                                continue

                    cell_correlations = []
                    for cell_type in df_true.columns:
                        cell_true = df_true[cell_type]
                        cell_pred = df_pred[cell_type]
                        if np.std(cell_pred.values) == 0 and len(cell_pred) > 1:
                            raise ValueError(f"All predicted values for {cell_type} are the same")

                        cell_true = cell_true[pd.notna(cell_true)]
                        cell_pred = cell_pred[pd.notna(cell_pred)]

                        if len(cell_true) > 3 and np.std(cell_true) != 0:
                            try:
                                corr = np.corrcoef(cell_true, cell_pred)[0, 1]
                                if not np.isnan(corr):
                                    cell_correlations.append(corr)
                            except Exception:
                                continue

                    if gene_correlations:
                        metrics[f'pearson_corr_cells_{dataset_desc}'] = float(np.mean(gene_correlations))
                    if cell_correlations:
                        metrics[f'pearson_corr_genes_{dataset_desc}'] = float(np.mean(cell_correlations))

                    ALLOWED = {
                    "Expression_dataset_v1_GRCh38_csv dataset",
                    "Expression_dataset_v1_mm10_CPM dataset",
                    }

                    if dataset_desc in ALLOWED:
                        # клетоспецифичность
                        df_true = df_true.reset_index()
                        df_pred = df_pred.reset_index()
                        score = score_predictions(
                            df_true,
                            df_pred,
                            experiment_config.selected_targets_path,
                            need_log=False,
                            logger=alogger
                        )
                        if score and score.get('deviation_r', None):
                            metrics[f'score_predictions_{dataset_desc}'] = score['deviation_r']
                        score2 = mean_and_residuals_correlation(df_true,
                            df_pred, need_log = False)
                        if isinstance(score2, dict):
                            for k, v in score2.items():
                                if isinstance(v, (np.floating, np.integer)):
                                    v = v.item()
                                metrics[f"mean_residual_{k}_{dataset_desc}"] = v
                    

            return metrics

        return metrics_fn

    metrics_fn = make_metrics_fn(args.model_path, save_predictions=args.save_predictions)
    test_metrics_fn = make_metrics_fn(args.model_path, save_predictions=args.save_predictions, file_prefix="test_")

    model, optimizer = accelerator.prepare(model, optimizer)

    # Trainer
    trainer = TrainerWithTest(
        args, accelerator, model, optimizer, train_dataloader,
        valid_dataloader=valid_dataloader,
        train_sampler=train_sampler,
        batch_transform_fn=batch_transform_fn,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn,
        test_dataloader=test_dataloader,
        test_metrics_fn=test_metrics_fn,
    )

    # Training
    accelerator.wait_for_everyone()
    trainer.train()
    accelerator.wait_for_everyone()

    # Post-training / validation / save
    if args.save_best:
        best_model_path = str(Path(args.model_path) / 'model_best.pth')
        if accelerator.is_main_process:
            alogger.info(f'Loading best saved model from {best_model_path}')
        trainer.load(best_model_path)

    if valid_dataloader is not None:
        if accelerator.is_main_process:
            alogger.info('Runnning validation on valid (and test, if set) data:')
        trainer.validate(valid_dataloader, write_tb=False)

    if accelerator.is_main_process:
        trainer.save_metrics(args.model_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
