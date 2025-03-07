import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import lightning as L
import json
import os
import sys
from datetime import datetime, timedelta
from dataloader.cc2CRPS_data import cc2DataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LambdaCallback
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingLR,
    LambdaLR,
)
import importlib

package = os.environ.get("MODEL_FAMILY", "pgu_ens")


def read_config(file_path):
    try:
        with open(file_path) as f:
            print(f"Loading config: {file_path}")
            config = json.load(f)
            config = config["config"]
            config = TrainingConfig(**config)
    except json.decoder.JSONDecodeError as e:
        print("Failed to decode json at {}".format(file_path))
        raise e
    return config


def create_directory_structure(base_directory):
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(f"{base_directory}/models", exist_ok=True)
    os.makedirs(f"{base_directory}/logs", exist_ok=True)
    os.makedirs(f"{base_directory}/figures", exist_ok=True)


def dynamic_import(items):
    for item in items:
        path_name = ".".join(item.split(".")[:-1])
        item_name = item.split(".")[-1]
        print("Importing {}".format(item))

        _module = importlib.import_module(path_name)
        globals()[item_name] = getattr(_module, item_name)


def string_to_type(type_str: str) -> type:
    assert "." in type_str
    module_name, _, type_name = type_str.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, type_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import type '{type_str}': {e}") from e
    return cls


def effective_parameters(num_devices, batch_size, lr, total_iterations):
    if num_devices == 1:
        # keep current batch size
        return batch_size, lr, total_iterations

    effective_bs = batch_size * num_devices
    effective_lr = lr * num_devices
    effective_total_iterations = total_iterations * num_devices

    return batch_size, effective_lr, effective_total_iterations


imports = [
    "common.util.calculate_wavelet_snr",
    "common.util.get_latest_run_dir",
    "common.util.get_next_run_number",
    "common.util.read_checkpoint",
    "common.util.get_rank",
]

dynamic_import(imports)

sys.path.append(os.path.abspath(package))

imports = [
    "util.roll_forecast",
    "callbacks.TrainDataPlotterCallback",
    "callbacks.DiagnosticCallback",
    "callbacks.LazyLoggerCallback",
    "config.get_config",
    "config.get_args",
    "config.TrainingConfig",
]

imports = [f"{package}.{x}" for x in imports]

dynamic_import(imports)
dynamic_import([f"{package}.cc2.cc2CRPS"])
model_class = string_to_type(f"{package}.cc2.cc2CRPS")
loss_fn = string_to_type(f"{package}.loss.loss_fn")


class cc2CRPSModel(model_class, L.LightningModule):
    def __init__(self, config):
        L.LightningModule.__init__(self)
        model_class.__init__(
            self,
            config,
        )
        self.loss_fn = loss_fn
        self.config = config

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = roll_forecast(
            self,
            data,
            forcing,
            self.config.rollout_length,
            loss_fn=self.loss_fn,
        )

        self.log("train_loss", loss["loss"], sync_dist=True)
        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def validation_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = roll_forecast(
            self,
            data,
            forcing,
            self.config.rollout_length,
            loss_fn=self.loss_fn,
        )

        self.log("val_loss", loss["loss"], sync_dist=True)
        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        T_max = it - config.warmup_iterations

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-7,
        )

        if args.only_config and config.current_iteration > 0:
            steps_after_warmup = config.current_iteration - config.warmup_iterations

            for _ in range(steps_after_warmup):
                cosine_scheduler.step()

            scheduler = cosine_scheduler
        else:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=config.warmup_iterations,
            )

            scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


rank = get_rank()
args = get_args()

if args.run_name is not None:
    latest_dir = get_latest_run_dir(f"runs/{args.run_name}")
    if not latest_dir:
        raise ValueError(f"No existing runs found with name: {args.run_name}")

    config = read_config(f"{latest_dir}/run-info.json")

    model = cc2CRPSModel(config)
    model = read_checkpoint(f"{latest_dir}/models", model)

    if args.only_config is False:
        config.apply_args(args)

    if args.run_name is not None:
        config.run_name = args.run_name

else:
    assert args.generate_run_name, "Must provide a run name or generate one"
    config = get_config()
    model = cc2CRPSModel(config)

new_run_number = get_next_run_number(f"runs/{config.run_name}")

config.run_dir = f"runs/{config.run_name}/{new_run_number}"
config.run_number = new_run_number

print(
    "Rank {} starting at {} using run directory {}".format(
        rank, datetime.now(), config.run_dir
    )
)

bs, lr, it = effective_parameters(
    config.num_devices, config.batch_size, config.learning_rate, config.num_iterations
)

cc2Data = cc2DataModule(
    zarr_path=config.data_path,
    batch_size=bs,
    n_x=config.history_length,
    n_y=config.rollout_length,
    limit_to=config.limit_data_to,
    apply_smoothing=config.apply_smoothing,
    input_resolution=config.input_resolution,
)

train_loader = cc2Data.train_dataloader()
val_loader = cc2Data.val_dataloader()

max_steps = it

if args.only_config:
    max_steps -= config.current_iteration

trainer = L.Trainer(
    max_steps=max_steps,
    precision=config.precision,
    accelerator="cuda",
    devices=config.num_devices,
    num_nodes=config.num_nodes,
    strategy=config.strategy,
    callbacks=[
        TrainDataPlotterCallback(train_loader, config),
        DiagnosticCallback(config),
        ModelCheckpoint(monitor="val_loss", dirpath=f"{config.run_dir}/models"),
        LambdaCallback(
            on_sanity_check_end=lambda trainer, pl_module: create_directory_structure(
                config.run_dir
            )
        ),
        LazyLoggerCallback(config),
    ],
    gradient_clip_val=1.0,
)

torch.set_float32_matmul_precision("high")

trainer.fit(model, train_loader, val_loader)

print("rank {} finished at {}".format(rank, datetime.now()))
