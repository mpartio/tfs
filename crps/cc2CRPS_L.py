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
from common.util import (
    get_rank,
    get_latest_run_dir,
    find_latest_checkpoint_path,
    strip_prefix,
    adapt_checkpoint_to_model,
)
from pgu.layers import get_padded_size
from typing import Optional


class cc2CRPSModel(L.LightningModule):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        prognostic_params: list[str],
        forcing_params: list[str],
        static_forcing_params: list[str],
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
        drop_path_rate: float = 0.0,
        learning_rate: float = 1e-3,
        warmup_iterations: int = 1000,
        weight_decay: float = 1e-3,
        rollout_length: int = 1,
        init_weights_from_ckpt: bool = False,
        adapt_ckpt_resolution: bool = False,
        branch_from_run: str = None,
        use_gradient_checkpointing: bool = False,
        add_refinement_head: bool = False,
        model_family: str = "pgu",
        use_scheduled_sampling: bool = False,
        ss_pred_min: float = 0.0,
        ss_pred_max: float = 1.0,
        noise_dim: Optional[int] = None,
        num_members: Optional[int] = None,
        loss_function: str = "huber_loss",
        use_ste: bool = False,
        autoregressive_mode: bool = True,
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
                "drop_path_rate",
                "prognostic_params",
                "forcing_params",
                "static_forcing_params",
                "use_gradient_checkpointing",
                "use_scheduled_sampling",
                "add_refinement_head",
                "noise_dim",
                "num_members",
                "ss_pred_min",
                "ss_pred_max",
                "use_ste",
                "autoregressive_mode",
            ]
        }

        if model_family == "pgu":
            from pgu.cc2 import cc2CRPS
            from pgu.loss import LOSS_FUNCTIONS

            if autoregressive_mode:
                from pgu.util import roll_forecast
            else:
                from pgu.util import roll_forecast_direct as roll_forecast

            loss_fn = LOSS_FUNCTIONS[loss_function]
        elif model_family == "pgu_ens":
            from pgu_ens.cc2 import cc2CRPS
            from pgu_ens.util import roll_forecast
            from pgu_ens.loss import loss_fn

        self.model_class = model_family
        self._roll_forecast = roll_forecast
        self._loss_fn = loss_fn

        self.model = cc2CRPS(config=model_kwargs)

        self.run_name = os.environ["CC2_RUN_NAME"]
        self.run_number = int(os.environ.get("CC2_RUN_NUMBER", -1))
        self.run_dir = os.environ["CC2_RUN_DIR"]

        if self.run_dir and not self.hparams.init_weights_from_ckpt:
            rank_zero_info(
                "Run dir set but weights are not loaded (init_weights_from_ckpt: false)"
            )

        elif self.run_dir and self.hparams.init_weights_from_ckpt:

            if self.hparams.branch_from_run:
                if "/" in self.hparams.branch_from_run:
                    ckpt_dir = "runs/{}".format(self.hparams.branch_from_run)
                else:
                    ckpt_dir = get_latest_run_dir(
                        "runs/" + self.hparams.branch_from_run
                    )

                rank_zero_info(f"Branching from {ckpt_dir}")

            else:
                ckpt_dir = (
                    "/".join(self.run_dir.split("/")[:-1])
                    + "/"
                    + str(self.run_number - 1)
                )
            ckpt_path = find_latest_checkpoint_path(ckpt_dir)

            rank_zero_info(f"Initializing weights from: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            state_dict = ckpt["state_dict"]
            state_dict = strip_prefix(state_dict)

            if self.hparams.adapt_ckpt_resolution:
                rank_zero_info("Adapting checkpoint resolution")

                assert (
                    "hyper_parameters" in ckpt.keys()
                ), "checkpoint does not have key 'hyper_parameters'"

                # We need the old/new size info here.
                # It might come from hparams, or you might need to load
                # the old config associated with the checkpoint.
                # This part requires careful handling of parameters.
                # Let's assume old/new size can be calculated or passed via hparams.

                old_input_resolution = ckpt["hyper_parameters"]["input_resolution"]
                new_input_resolution = self.hparams.input_resolution

                old_patch_size = ckpt["hyper_parameters"]["patch_size"]
                new_patch_size = self.hparams.patch_size

                old_resolution = get_padded_size(
                    *old_input_resolution,
                    old_patch_size,
                )
                old_resolution = (
                    old_resolution[0] // old_patch_size,
                    old_resolution[1] // old_patch_size,
                )

                new_resolution = get_padded_size(*new_input_resolution, new_patch_size)
                new_resolution = (
                    new_resolution[0] // new_patch_size,
                    new_resolution[1] // new_patch_size,
                )

                # Use your existing adaptation function
                state_dict = adapt_checkpoint_to_model(
                    state_dict, self.model.state_dict(), old_resolution, new_resolution
                )

            # Load the (potentially adapted) state dict
            # strict=False allows missing/extra keys (e.g., different final layer)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            rank_zero_info(f"Weight loading results: {load_result}")

        rank = get_rank()

        if self.run_name is not None:
            print(
                "Rank {} starting at {} using run directory {}".format(
                    rank, datetime.now(), self.run_dir
                )
            )

        self.test_predictions = []
        self.test_truth = []
        self.test_dates = []

        self.latest_val_tendencies = None
        self.latest_val_predictions = None
        self.latest_val_data = None

        self.latest_train_tendencies = None
        self.latest_train_predictions = None
        self.latest_train_data = None

        self.use_ste = use_ste
        self.use_scheduled_sampling = use_scheduled_sampling
        self.ss_pred_min = ss_pred_min
        self.ss_pred_max = ss_pred_max
        self.autoregressive_mode = autoregressive_mode

    def on_train_start(self) -> None:
        self.max_epochs = self.trainer.max_epochs
        # max_steps is not typically set (by me)
        if self.trainer.max_steps > 0:
            self.max_steps = self.trainer.max_steps
        else:
            # fallback if only max_epochs is set
            self.max_steps = self.trainer.estimated_stepping_batches

        assert self.max_steps > 0, "Trainer must be configured with max_steps"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # data, forcing, step)

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=self.use_scheduled_sampling,
            step=self.global_step,
            max_step=self.max_steps,
            ss_pred_min=self.ss_pred_min,
            ss_pred_max=self.ss_pred_max,
            pl_module=self,
        )

        self.log("train_loss", loss["loss"], sync_dist=False)

        if batch_idx == 0:
            self.latest_train_tendencies = tendencies
            self.latest_train_predictions = predictions
            self.latest_train_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def validation_step(self, batch, batch_idx):
        data, forcing = batch

        loss, tendencies, predictions = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=False,
        )

        self.log("val_loss", loss["loss"], sync_dist=True)

        if batch_idx == 0:
            self.latest_val_tendencies = tendencies
            self.latest_val_predictions = predictions
            self.latest_val_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def test_step(self, batch, batch_idx):
        data, forcing, dates = batch

        _, tendencies, predictions = self._roll_forecast(
            self,
            data,
            forcing,
            self.hparams.rollout_length,  # Access from hparams
            loss_fn=None,
            use_scheduled_sampling=False,
        )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)

        if self.model_class == "pgu_ens":
            B, T, C, H, W = analysis_time.shape
            analysis_time = analysis_time.unsqueeze(1).expand(
                -1, self.hparams.num_members, -1, -1, -1, -1
            )
            predictions = torch.concatenate((analysis_time, predictions), dim=2)
            y = (
                data[1]
                .unsqueeze(1)
                .expand(-1, self.hparams.num_members, -1, -1, -1, -1)
            )
            truth = torch.concatenate((analysis_time, y), dim=2)
        else:
            predictions = torch.concatenate((analysis_time, predictions), dim=1)
            truth = torch.concatenate((analysis_time, data[1]), dim=1)

        self.test_predictions.append(predictions)
        self.test_truth.append(truth)
        self.test_dates.append(torch.concatenate((dates[0][:, -1:], dates[1]), dim=1))

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
        dates = torch.concatenate(self.test_dates)

        torch.save(predictions, f"{output_dir}/predictions.pt")
        torch.save(truth, f"{output_dir}/truth.pt")
        torch.save(dates, f"{output_dir}/dates.pt")

        print(f"Predictions shape: {predictions.shape}")
        print(f"Truth shape: {truth.shape}")
        print(f"Dates shape: {dates.shape}")
        print(f"Wrote files predictions.pt, truth.pt and dates.pt to {output_dir}")

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
