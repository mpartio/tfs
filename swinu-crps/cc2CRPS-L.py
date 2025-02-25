import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import lightning as L
import json
import os
from datetime import datetime, timedelta
from cc2CRPS import cc2CRPS
from crps import AlmostFairCRPSLoss
from util import calculate_wavelet_snr
from cc2CRPS_data import cc2DataModule
from cc2CRPS_callbacks import (
    TrainDataPlotterCallback,
    DiagnosticCallback,
    LazyLoggerCallback,
)
from cc2util import (
    roll_forecast,
    get_latest_run_dir,
    get_next_run_number,
    read_checkpoint
)
from lightning.pytorch.callbacks import ModelCheckpoint, LambdaCallback
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingLR,
    LambdaLR,
)
from dataclasses import asdict
from config import get_config, get_args, TrainingConfig


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


class cc2CRPSModel(cc2CRPS, L.LightningModule):
    def __init__(self, config):
        L.LightningModule.__init__(self)
        cc2CRPS.__init__(
            self,
            config,
        )
        self.crps_loss = AlmostFairCRPSLoss(alpha=0.95)
        self.config = config

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, tendencies, predictions = roll_forecast(
            self,
            x,
            y,
            config.rollout_length,
            loss_fn=self.crps_loss,
            num_members=self.config.num_members,
        )
        loss = loss.mean()
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss, "tendencies": tendencies, "predictions": predictions}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        loss, tendencies, predictions = roll_forecast(
            self,
            x,
            y,
            config.rollout_length,
            loss_fn=self.crps_loss,
            num_members=self.config.num_members,
        )
        loss = loss.mean()
        self.log("val_loss", loss, sync_dist=True)
        return {"loss": loss, "tendencies": tendencies, "predictions": predictions}

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        T_max = config.num_iterations - config.warmup_iterations

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
        config._run_name = args.run_name

else:
    config = get_config()
    model = cc2CRPSModel(config)

new_run_number = get_next_run_number(f"runs/{config.run_name}")

config.run_dir = f"runs/{config.run_name}/{new_run_number}"
config.run_number = new_run_number

print("Starting run at {}".format(datetime.now()))
print("Run directory: ", config.run_dir)

cc2Data = cc2DataModule(
    zarr_path=config.data_path,
    batch_size=config.batch_size * config.num_devices,
    n_x=config.history_length,
    n_y=config.rollout_length,
    limit_to=config.limit_data_to,
    apply_smoothing=config.apply_smoothing,
)

train_loader = cc2Data.train_dataloader()
val_loader = cc2Data.val_dataloader()

max_steps = config.num_iterations

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

print("Done at {}".format(datetime.now()))
