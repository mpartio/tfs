import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import lightning as L
import os
from glob import glob
from datetime import datetime, timedelta
from cc2CRPS import cc2CRPS
from crps import AlmostFairCRPSLoss
from util import calculate_wavelet_snr
from cc2CRPS_data import cc2DataModule, cc2ZarrModule
from cc2CRPS_callbacks import TrainDataPlotterCallback, DiagnosticCallback
from cc2util import (
    roll_forecast,
    get_latest_run_dir,
    get_next_run_number,
    gaussian_smooth,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from dataclasses import asdict
from config import get_config, get_args, TrainingConfig
from pytorch_lightning.loggers import CSVLogger


def read_checkpoint(file_path):
    try:
        # Find latest checkpoint
        checkpoints = glob(f"{file_path}/*.ckpt")
        assert checkpoints, "No model checkpoints found in directory {}".format(
            file_path
        )
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, weights_only=False)
        new_state_dict = {}
        state_dict = ckpt["state_dict"]

        for k, v in state_dict.items():
            new_k = k.replace("model.", "")
            new_state_dict[new_k] = v

        config = TrainingConfig(**ckpt["config"])
        model = cc2CRPSModel(config)
        model.load_state_dict(new_state_dict)

        return model, config

    except ValueError as e:
        print("Model checkpoint file not found from path: ", file_path)
        raise e


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

        if config.apply_smoothing:
            x = gaussian_smooth(x)
            y = gaussian_smooth(y)

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

        if config.apply_smoothing:
            x = gaussian_smooth(x)
            y = gaussian_smooth(y)

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

    def on_save_checkpoint(self, checkpoint):
        # Save config alongside model checkpoint
        checkpoint["config"] = asdict(self.config)

    def on_load_checkpoint(self, checkpoint):
        # Load config when restoring checkpoint
        if "config" in checkpoint:
            self.config = TrainingConfig(**checkpoint["config"])

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_iterations,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.num_iterations - config.warmup_iterations,
            eta_min=0,
        )
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        # scheduler = get_lr_schedule(
        #    optimizer, warmup_iterations=1000, total_iterations=self.max_iterations
        # )

        # return [optimizer], [scheduler]


args = get_args()

if args.run_name is not None:
    latest_dir = get_latest_run_dir(f"runs/{args.run_name}")
    if not latest_dir:
        raise ValueError(f"No existing runs found with name: {args.run_name}")

    model, config = read_checkpoint(f"{latest_dir}/models")
    config.apply_args(args)

else:
    config = get_config()
    model = cc2CRPSModel(config)

new_run_number = get_next_run_number(f"runs/{config.run_name}")

config.run_dir = f"runs/{config.run_name}/{new_run_number}"
config.run_number = new_run_number

print("Starting run at {}".format(datetime.now()))
print("Run directory: ", config.run_dir)

create_directory_structure(config.run_dir)

data_path = config.data_path

if data_path.endswith(".zarr"):
    cc2Data = cc2ZarrModule(
        zarr_path=data_path,
        batch_size=config.batch_size * config.num_devices,
        n_x=config.history_length,
        n_y=config.rollout_length,
        limit_to=config.limit_data_to,
    )
else:
    cc2Data = cc2DataModule(
        batch_size=config.batch_size * config.num_devices,
        n_x=config.history_length,
        n_y=config.rollout_length,
        dataset_size=config.data_path,
    )

train_loader = cc2Data.train_dataloader()
val_loader = cc2Data.val_dataloader()

trainer = L.Trainer(
    max_steps=config.num_iterations,
    precision=config.precision,
    accelerator="cuda",
    devices=config.num_devices,
    num_nodes=config.num_nodes,
    strategy=config.strategy,
    callbacks=[
        TrainDataPlotterCallback(train_loader, config),
        DiagnosticCallback(config),
        ModelCheckpoint(monitor="val_loss", dirpath=f"{config.run_dir}/models"),
    ],
    logger=CSVLogger(f"{config.run_dir}/logs"),
    gradient_clip_val=1.0,
)

torch.set_float32_matmul_precision("high")

trainer.fit(model, train_loader, val_loader)

print("Done at {}".format(datetime.now()))
