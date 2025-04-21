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
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingLR,
    LambdaLR,
)
from pgu.cc2 import cc2CRPS
from pgu.layers import get_padded_size
from pgu.util import roll_forecast
from pgu.loss import loss_fn
from common.util import (
    get_rank,
    get_next_run_number,
    string_to_type,
    find_latest_checkpoint_path,
)
import importlib


def dynamic_import(items):
    for item in items:
        path_name = ".".join(item.split(".")[:-1])
        item_name = item.split(".")[-1]
        rank_zero_info("Importing {}".format(item))

        _module = importlib.import_module(path_name)
        globals()[item_name] = getattr(_module, item_name)


imports = [
    "common.util.adapt_checkpoint_to_model",
]

dynamic_import(imports)


class cc2CRPSModel(L.LightningModule):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        prognostic_params: list[str, ...],
        forcing_params: list[str, ...],
        history_length: int = 2,
        hidden_dim: int = 96,
        patch_size: int = 4,
        encoder1_depth: int = 2,
        encoder2_depth: int = 2,
        decoder1_depth: int = 2,
        decoder2_depth: int = 2,
        window_size: int = 8,
        num_heads: int = 4,
        add_skip_connection: bool = False,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        learning_rate: float = 1e-3,
        warmup_iterations: int = 1000,
        weight_decay: float = 1e-3,
        rollout_length: int = 1,
        init_weights_from_ckpt: bool = False,
        adapt_ckpt_resolution: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Extract only model-specific parameters
        model_kwargs = {
            k: v
            for k, v in self.hparams.items()
            if k
            in [
                "input_resolution",
                "history_length",
                "hidden_dim",
                "patch_size",
                "encoder1_depth",
                "encoder2_depth",
                "decoder1_depth",
                "decoder2_depth",
                "window_size",
                "num_heads",
                "add_skip_connection",
                "mlp_ratio",
                "drop_rate",
                "attn_drop_rate",
                "prognostic_params",
                "forcing_params",
            ]
        }

        self.model = cc2CRPS(config=model_kwargs)
        self.loss_fn = loss_fn

        self.run_name = os.environ.get("CC2_RUN_NAME", None)
        self.run_number = int(os.environ.get("CC2_RUN_NUMBER", -1))
        self.run_dir = os.environ.get("CC2_RUN_DIR", None)

        assert self.run_name is not None, "CC2_RUN_NAME not set"

        if self.run_dir and self.hparams.init_weights_from_ckpt:
            prev_run_dir = (
                "/".join(self.run_dir.split("/")[:-1]) + "/" + str(self.run_number - 1)
            )
            ckpt_path = find_latest_checkpoint_path(prev_run_dir)

            rank_zero_info(f"Initializing weights from: {ckpt_path}")

            ckpt = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )  # Load to CPU first
            state_dict = ckpt["state_dict"]

            if self.hparams.adapt_ckpt_resolution:
                rank_zero_info("Adapting checkpoint resolution...")
                # You need the old/new size info here.
                # It might come from hparams, or you might need to load
                # the old config associated with the checkpoint.
                # This part requires careful handling of parameters.
                # Let's assume old/new size can be calculated or passed via hparams.
                old_patch_size = self.hparams.config.patch_size
                old_resolution = get_padded_size(
                    *self.hparams.config.input_resolution,
                    self.hparams.config.patch_size,
                )
                old_resolution = (
                    old_resolution[0] // old_patch_size,
                    old_resolution[1] // old_patch_size,
                )

                config.apply_args(args)
                new_resolution = get_padded_size(
                    *config.input_resolution, config.patch_size
                )
                new_resolution = (
                    new_resolution[0] // config.patch_size,
                    new_resolution[1] // config.patch_size,
                )

                old_size = calculate_old_size(...)  # Placeholder
                new_size = calculate_new_size(self.hparams.model_config)  # Placeholder

                # Use your existing adaptation function
                state_dict = adapt_checkpoint_to_model(
                    state_dict, self.model.state_dict(), old_size, new_size
                )

            # Load the (potentially adapted) state dict
            # strict=False allows missing/extra keys (e.g., different final layer)
            load_result = self.load_state_dict(state_dict, strict=False)
            print("Weight loading results", load_result)
        new_run_number = get_next_run_number(f"runs/{self.run_name}")

        rank = get_rank()

        if self.run_name is not None:
            print(
                "Rank {} starting at {} using run directory {}".format(
                    rank, datetime.now(), self.run_dir
                )
            )

        self.test_predictions = []
        self.test_truth = []

    def forward(self, *args, **kwargs):  # data, forcing, step):
        return self.model(*args, **kwargs)  # data, forcing, step)

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
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
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self.loss_fn,
        )

        self.log("val_loss", loss["loss"], sync_dist=True)
        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def test_step(self, batch, batch_idx):
        data, forcing = batch

        _, tendencies, predictions = roll_forecast(
            self,
            data,
            forcing,
            self.hparams.rollout_length,  # Access from hparams
            loss_fn=None,
        )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)
        predictions = torch.concatenate((analysis_time, predictions), dim=1)
        self.test_predictions.append(predictions)
        truth = torch.concatenate((analysis_time, data[1]), dim=1)
        self.test_truth.append(truth)

        return {
            "tendencies": tendencies,
            "predictions": predictions,
            "source": data[0],
            "truth": data[1],
        }

    def on_test_end(self):
        # Get the run directory from the checkpoint path
        output_dir = f"{self.run_dir}/test-output/"
        os.makedirs(output_dir, exist_ok=True)

        predictions = torch.concatenate(self.test_predictions)
        truth = torch.concatenate(self.test_truth)
        torch.save(predictions, f"{output_dir}/predictions.pt")
        torch.save(truth, f"{output_dir}/truth.pt")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Truth shape: {truth.shape}")
        print(f"Wrote files predictions.pt and truth.pt to {output_dir}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )

        effective_batch_size = (
            (self.trainer.num_devices * self.trainer.num_nodes)  # Total devices
            * self.trainer.accumulate_grad_batches
            * self.trainer.datamodule.hparams.batch_size  # Get batch size from datamodule hparams
        )

        train_ds_size = len(self.trainer.datamodule.ds_train)
        steps_per_epoch = (
            train_ds_size + effective_batch_size - 1
        ) // effective_batch_size

        num_iterations = steps_per_epoch * self.trainer.max_epochs
        T_max = num_iterations - self.hparams.warmup_iterations

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-7,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_iterations,
        )

        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def get_strategy(strategy):
    if strategy == "deep_speed":
        from pytorch_lightning.strategies import DeepSpeedStrategy

        return DeepSpeedStrategy(
            stage=3,  # Stage 3 enables full parameter, gradient, and optimizer sharding
            offload_optimizer=True,  # Offload optimizer states to CPU
            offload_parameters=True,  # Offload parameters to CPU
        )

    return strategy
