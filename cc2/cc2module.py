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
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
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
)
from typing import Optional, Callable


class cc2module(L.LightningModule):
    def __init__(
        self,
        history_length: int = 2,
        hidden_dim: int = 96,
        patch_size: int = 4,
        encoder1_depth: int = 2,
        encoder2_depth: int = 2,
        decoder1_depth: int = 2,
        decoder2_depth: int = 2,
        window_size: int = 8,
        window_size_deep: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float | list[float] = 0.0,
        learning_rate: float = 1e-3,
        warmup_iterations: int = 1000,
        weight_decay: float = 1e-3,
        rollout_length: int = 1,
        branch_from_run: str = None,
        use_gradient_checkpointing: bool = False,
        model_family: str = "pgu",
        use_scheduled_sampling: bool = False,
        ss_pred_min: float = 0.0,
        ss_pred_max: float = 1.0,
        freeze_layers: list[str] = [],
        loss_fn: nn.Module | None = None,
        test_output_directory: str | None = None,
        use_rollout_weighting: bool = False,
        use_statistics_from_checkpoint: bool = True,
        force_frozen_backbone_to_eval: bool = False,
        preprocessor: Callable | None = None,
        use_flow_matching: bool = False,
        num_inference_steps: int = 4,
        flow_warm_start: bool = True,
        warm_start_alpha: float = 0.4,
        flow_init_noise_sigma: float = 1.0,
        flow_eta: float = 0.0,
        flow_alpha_a: float = 1.0,
        flow_alpha_b: float = 1.0,
        flow_init_alpha: float = 1.0,
        flow_alpha_schedule_rho: float = 1.0,
        use_advected_persistence: bool = False,
        advection_wind_level: str = "925",
        advection_secondary_level: str | None = None,
        advection_secondary_weight: float = 0.0,
        advection_dt_seconds: float = 3 * 3600.0,
        advection_grid_spacing_m: float = 5000.0,
        advection_flip_v_axis: bool = True,
        use_rolling_advection: bool = False,
        advection_proj_init: str = "zero",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_configured = False

        # Extract only model-specific parameters
        model_kwargs = {
            k: v
            for k, v in self.hparams.items()
            if k
            in [
                "history_length",
                "hidden_dim",
                "patch_size",
                "encoder1_depth",
                "encoder2_depth",
                "decoder1_depth",
                "decoder2_depth",
                "window_size",
                "window_size_deep",
                "num_heads",
                "mlp_ratio",
                "drop_rate",
                "attn_drop_rate",
                "drop_path_rate",
                "use_gradient_checkpointing",
                "use_scheduled_sampling",
                "ss_pred_min",
                "ss_pred_max",
                "use_flow_matching",
                "use_advected_persistence",
            ]
        }

        # hparams serializes values to dict, so assign preprocessor here manually
        model_kwargs["preprocessor"] = preprocessor

        self.model_kwargs = model_kwargs

        self.test_predictions = []
        self.test_truth = []
        self.test_dates = []

        self.latest_val_tendencies = None
        self.latest_val_predictions = None
        self.latest_val_data = None

        self.latest_train_tendencies = None
        self.latest_train_predictions = None
        self.latest_train_data = None

        self.use_scheduled_sampling = use_scheduled_sampling
        self.use_statistics_from_checkpoint = use_statistics_from_checkpoint
        self.ss_pred_min = ss_pred_min
        self.ss_pred_max = ss_pred_max
        self.test_output_directory = test_output_directory

        self._loss_fn = loss_fn

        assert self._loss_fn is not None

    def configure_model(self) -> None:
        if self.model_configured:
            return

        # Read input_resolution from data
        self.input_resolution = self.trainer.datamodule.input_resolution
        rank_zero_info(f"Data resolution is {self.input_resolution}")

        self.model_kwargs.update(
            {
                "input_resolution": self.trainer.datamodule.input_resolution,
                "prognostic_params": self.trainer.datamodule.hparams.prognostic_params,
                "forcing_params": self.trainer.datamodule.hparams.forcing_params,
                "static_forcing_params": self.trainer.datamodule.hparams.static_forcing_params,
            }
        )

        self._build_model()

        self.model_configured = True

    def _store_trainable_module_names(self) -> None:
        trainable_modules = set()
        for module_name, module in self.model.named_modules():
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    trainable_modules.add(module_name)
                    break

        self._trainable_module_names = trainable_modules
        rank_zero_info("Backbone will be set to eval to disable dropouts")

    def _force_frozen_backbone_to_eval(self) -> None:
        if not hasattr(self, "_trainable_module_names"):
            return

        # 1) everything deterministic
        self.model.eval()

        # 2) re-enable training behavior only on modules that own trainable params
        for module_name, module in self.model.named_modules():
            if module_name in self._trainable_module_names:
                module.train()

    def _print_trainable_layers(self):
        trainable = set()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.add(name)

        rank_zero_info(f"Trainable layers: {trainable}")

    def _freeze_layers(self) -> None:
        if len(self.hparams.freeze_layers) == 0:
            return

        frozen = []

        for l in self.hparams.freeze_layers:
            # start with submodules
            for name, module in self.model.named_modules():
                if name.startswith(l):
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(name)

            # also direct parameters on the model (not in submodules)
            for name, param in self.model.named_parameters(recurse=False):
                if name.startswith(l):
                    param.requires_grad = False
                    frozen.append(name)

        rank_zero_info(f"Froze layers: {frozen}")
        self._store_trainable_module_names()
        self._print_trainable_layers()

    def _extend_proj_force_for_flow(self, state_dict: dict) -> dict:
        """
        When use_flow_matching=True the model's proj_force expects extra input
        channels at the END of the forcing tensor. A checkpoint from before flow
        matching (or from CFM-only, when adding advected-persistence on top) has
        a narrower weight matrix. Zero-extend at the END so existing trained
        columns keep their original meaning.

        Channel layout (in tensor and weight column order):
            [..F_original.., x_alpha, alpha_map, (adv_persistence if enabled)]

        Both transitions are handled by the same zero-pad-at-end rule:
        - non-CFM ckpt → CFM model        : +2 cols zero-init at end
        - non-CFM ckpt → CFM+adv model    : +3 cols zero-init at end
        - CFM ckpt → CFM+adv model        : +1 col  zero-init at end
        """
        key_w = "patch_embed.proj_force.weight"

        if key_w not in state_dict:
            return state_dict

        old_w = state_dict[key_w]  # [d_force, Cf_old*ps*ps]
        model_w = self.model.patch_embed.proj_force.weight  # [d_force, Cf_new*ps*ps]

        if old_w.shape == model_w.shape:
            return state_dict  # already the right size

        d_force, old_in = old_w.shape
        new_in = model_w.shape[1]
        new_w = torch.zeros(d_force, new_in, dtype=old_w.dtype, device=old_w.device)
        new_w[:, :old_in] = old_w

        # Non-zero init for the advection column (DE-13 fix).
        # Trigger: extending by exactly one channel (i.e., adding only the adv
        # column on top of an existing CFM ckpt) AND scheme != "zero".
        ps = int(self.hparams.patch_size)
        ps2 = ps * ps
        scheme = getattr(self.hparams, "advection_proj_init", "zero")
        delta_cols = new_in - old_in
        if delta_cols == ps2 and scheme != "zero":
            try:
                fp = list(self.trainer.datamodule.hparams.forcing_params)
                Cf_new = new_in // ps2
                old_3d = new_w[:, :old_in].reshape(d_force, Cf_new - 1, ps2)  # [d_force, Cf-1, ps²]

                if scheme == "uv_blend_925_850":
                    # Init the adv column from a wind-blend of existing forcing
                    # columns. The advection channel itself is computed from
                    # 0.4·u_925 + 0.6·u_850 winds; matching the projection
                    # weights to this blend gives the model a sensible starting
                    # gradient path through u/v-shaped activations.
                    idxs = {n: fp.index(n) for n in ("u_925","u_850","v_925","v_850") if n in fp}
                    assert len(idxs) == 4, f"need u/v at 925+850, found {idxs}"
                    blend = (
                        0.4 * old_3d[:, idxs["u_925"], :]
                      + 0.6 * old_3d[:, idxs["u_850"], :]
                      + 0.4 * old_3d[:, idxs["v_925"], :]
                      + 0.6 * old_3d[:, idxs["v_850"], :]
                    ) * 0.5   # ½ to scale magnitude into the per-column range
                    new_w_3d = new_w.reshape(d_force, Cf_new, ps2)
                    new_w_3d[:, -1, :] = blend
                    new_w = new_w_3d.reshape(d_force, new_in)
                    init_desc = "0.5·(0.4·u_925 + 0.6·u_850 + 0.4·v_925 + 0.6·v_850) column blend"
                elif scheme == "kaiming":
                    fan_in = ps2
                    std = (2.0 / fan_in) ** 0.5
                    new_w_3d = new_w.reshape(d_force, Cf_new, ps2)
                    new_w_3d[:, -1, :] = torch.randn(d_force, ps2, dtype=old_w.dtype) * std
                    new_w = new_w_3d.reshape(d_force, new_in)
                    init_desc = f"Kaiming-normal init, std={std:.4f}"
                else:
                    raise ValueError(f"Unknown advection_proj_init scheme: {scheme}")

                state_dict[key_w] = new_w
                rank_zero_info(
                    f"Extended proj_force.weight [{d_force},{old_in}]→[{d_force},{new_in}] with "
                    f"advection column init: {init_desc}"
                )
                return state_dict
            except Exception as e:
                rank_zero_warn(
                    f"Non-zero advection init failed ({e}); falling back to zero-init."
                )
                # Fall through to plain assignment

        state_dict[key_w] = new_w
        rank_zero_info(
            f"Extended proj_force.weight from [{d_force},{old_in}] to [{d_force},{new_in}] "
            f"(zero-init trailing columns)"
        )
        return state_dict

    def _setup_advection(self) -> None:
        """Cache forcing-tensor indices and m/s denormalization stats for the
        u/v wind components used to advect cloud state. Optionally blends a
        secondary wind level (e.g. 0.4·925 + 0.6·850).

        Lazy: called from `_build_flow_forcing`/`test_step` on first use because
        trainer.datamodule isn't bound at __init__.
        """
        if getattr(self, "_adv_setup_done", False):
            return

        dm = self.trainer.datamodule
        fp = list(dm.hparams.forcing_params)
        ds = dm._get_or_create_full_dataset()
        all_params = list(ds.all_combined_params)

        def _lookup(name: str):
            if name not in fp:
                raise ValueError(
                    f"advection requires {name} in forcing_params; found: {fp}"
                )
            idx = fp.index(name)
            g = all_params.index(name)
            return idx, float(ds.means[0, g, 0, 0]), float(ds.stds[0, g, 0, 0])

        lvl_p = self.hparams.advection_wind_level
        self._adv_u_idx, self._adv_u_mean, self._adv_u_std = _lookup(f"u_{lvl_p}")
        self._adv_v_idx, self._adv_v_mean, self._adv_v_std = _lookup(f"v_{lvl_p}")

        lvl_s = self.hparams.advection_secondary_level
        w_s = float(self.hparams.advection_secondary_weight)
        if lvl_s is not None and w_s > 0.0:
            self._adv2_u_idx, self._adv2_u_mean, self._adv2_u_std = _lookup(f"u_{lvl_s}")
            self._adv2_v_idx, self._adv2_v_mean, self._adv2_v_std = _lookup(f"v_{lvl_s}")
            self._adv_w_secondary = w_s
            blend_desc = f"{1.0 - w_s:.2f}·{lvl_p} + {w_s:.2f}·{lvl_s}"
        else:
            self._adv2_u_idx = None
            self._adv_w_secondary = 0.0
            blend_desc = f"single-level {lvl_p}"

        rank_zero_info(f"Advected-persistence wind: {blend_desc}")
        self._adv_setup_done = True

    def _compute_advection_winds(self, forcing: torch.Tensor):
        """De-standardize and (optionally) blend wind components from the
        analysis-time slot of `forcing`. Returns pixel-per-step displacements
        (u_pix, v_pix) suitable for `advect_semi_lagrangian`.

        Wind is held CONSTANT in time — the analysis-time field is used at
        every rollout step. This matches both frozen-x_0 and rolling-state
        advection schemes (only the input field differs between them).
        """
        from dataloader.advection import winds_to_pixel_per_step

        self._setup_advection()
        B, T_total, _, H, W = forcing.shape
        # Analysis time is the last history slot; we don't have n_step here
        # so we use T-1 of the absolute T_total, knowing history_length=2
        # → analysis is at index history_length-1=1. We index by name elsewhere
        # but here the convention is the same as in _build_flow_forcing.
        T = T_total - self.hparams.rollout_length
        u_norm = forcing[:, T - 1, self._adv_u_idx, :, :]
        v_norm = forcing[:, T - 1, self._adv_v_idx, :, :]
        u_ms = u_norm * self._adv_u_std + self._adv_u_mean
        v_ms = v_norm * self._adv_v_std + self._adv_v_mean

        if self._adv2_u_idx is not None:
            u2_norm = forcing[:, T - 1, self._adv2_u_idx, :, :]
            v2_norm = forcing[:, T - 1, self._adv2_v_idx, :, :]
            u2_ms = u2_norm * self._adv2_u_std + self._adv2_u_mean
            v2_ms = v2_norm * self._adv2_v_std + self._adv2_v_mean
            w = self._adv_w_secondary
            u_ms = (1.0 - w) * u_ms + w * u2_ms
            v_ms = (1.0 - w) * v_ms + w * v2_ms

        return winds_to_pixel_per_step(
            u_ms,
            v_ms,
            dt_seconds=self.hparams.advection_dt_seconds,
            grid_spacing_m=self.hparams.advection_grid_spacing_m,
            flip_v_for_image_origin=self.hparams.advection_flip_v_axis,
        )

    def _compute_advected_persistence(
        self,
        forcing: torch.Tensor,
        current_state: torch.Tensor,
        n_step: int,
    ) -> torch.Tensor:
        """Backward semi-Lagrangian advection of `current_state` at each lead
        k = 1..n_step using the (optionally-blended) analysis-time wind.

        Frozen-x_0 mode: at lead k, advect by k steps with constant wind. This
        is what gets pre-filled before the rollout in `test_step` or used at
        training time (where R=1 means k=1 only). For rolling-state inference,
        see `_compute_advection_winds` and the per-step recomputation in
        `flow_roll_forecast`.
        """
        from dataloader.advection import advect_semi_lagrangian

        u_pix, v_pix = self._compute_advection_winds(forcing)
        B = forcing.shape[0]
        field0 = current_state[:, 0, :, :, :]
        out = torch.zeros(B, n_step, *field0.shape[1:], device=forcing.device, dtype=forcing.dtype)
        for k in range(n_step):
            out[:, k, ...] = advect_semi_lagrangian(field0, u_pix, v_pix, n_steps=k + 1)
        return out

    def _build_model(self) -> None:
        if self.hparams.model_family == "pgu":
            from pgu.cc2 import cc2model
            from pgu.util import roll_forecast

        elif self.hparams.model_family == "swinu":
            from swinu.cc2 import cc2model
            from swinu.util import roll_forecast

        self.model_class = self.hparams.model_family
        self._roll_forecast = roll_forecast

        self.model = cc2model(config=self.model_kwargs)

        self.run_dir = None
        if self.trainer.state.stage in ("train", "test"):
            self.run_name = os.environ["CC2_RUN_NAME"]
            self.run_number = int(os.environ.get("CC2_RUN_NUMBER", -1))
            self.run_dir = os.environ["CC2_RUN_DIR"]

        if self.hparams.branch_from_run:
            if "/" in self.hparams.branch_from_run:
                branch_run_dir = "runs/{}".format(self.hparams.branch_from_run)
            else:
                branch_run_dir = get_latest_run_dir(
                    "runs/" + self.hparams.branch_from_run
                )

            ckpt_path = find_latest_checkpoint_path(branch_run_dir)

            rank_zero_info(
                f"Branching from {branch_run_dir} using weights from: {ckpt_path}"
            )

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            state_dict = ckpt["state_dict"]
            state_dict = strip_prefix(state_dict)

            if self.hparams.use_flow_matching:
                state_dict = self._extend_proj_force_for_flow(state_dict)

            # Load the state dict
            # strict=False allows missing/extra keys (e.g., different final layer)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            rank_zero_info(f"Weight loading results: {load_result}")

        self._freeze_layers()

        rank = get_rank()

        msg = "Rank {} starting at {}".format(get_rank(), datetime.now())

        if self.run_dir is not None:
            msg += f" using run directory {self.run_dir}"

        print(msg)

    def on_train_batch_start(self, batch, batch_idx):
        if getattr(self.hparams, "force_frozen_backbone_to_eval", False):
            self._force_frozen_backbone_to_eval()

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

    def _build_flow_forcing(self, forcing, y, x=None):
        """
        Corrupt the clean targets and inject as extra forcing channels for flow matching.

        forcing : [B, T+n_step, C_force, H, W]  – original forcing (no flow channels)
        y       : [B, n_step,   C_data,  H, W]  – clean future targets
        x       : [B, T,        C_data,  H, W]  – history (required when
                  use_advected_persistence=True; analysis-time frame is x[:, -1])

        Returns: [B, T+n_step, C_force+E, H, W] where E=2 (no adv) or E=3 (with adv).
        Trailing-channel layout at each future time slot:
            (x_alpha, alpha_map[, advected_persistence])
        History slots carry zeros for all extras.
        """
        B, T_total, C_force, H, W = forcing.shape
        _, n_step, C_data, _, _ = y.shape
        T = T_total - n_step
        use_adv = bool(self.hparams.use_advected_persistence)

        # Sample one alpha per sample in the batch.
        # Defaults flow_alpha_a=flow_alpha_b=1.0 reproduce Uniform(0,1); any other
        # pair selects a Beta(a,b) distribution (e.g. Beta(2,5) skews toward small α).
        a, b = self.hparams.flow_alpha_a, self.hparams.flow_alpha_b
        if a == 1.0 and b == 1.0:
            alpha = torch.rand(B, device=forcing.device)  # [B]
        else:
            alpha = (
                torch.distributions.Beta(a, b)
                .sample((B,))
                .to(device=forcing.device, dtype=forcing.dtype)
            )

        # Corrupt target: x_alpha = (1-alpha)*y + alpha*z
        alpha_5d = alpha.view(B, 1, 1, 1, 1)
        z = torch.randn_like(y)
        x_alpha = (1.0 - alpha_5d) * y + alpha_5d * z  # [B, n_step, C_data, H, W]

        n_extra = 3 if use_adv else 2
        extra = torch.zeros(
            B, T_total, n_extra, H, W, device=forcing.device, dtype=forcing.dtype
        )
        for t in range(n_step):
            extra[:, T + t, 0, :, :] = x_alpha[:, t, 0, :, :]
            extra[:, T + t, 1, :, :] = alpha.view(B, 1, 1).expand(B, H, W)

        if use_adv:
            if x is None:
                raise ValueError(
                    "use_advected_persistence=True requires `x` (history) in "
                    "_build_flow_forcing; caller must pass data[0]."
                )
            current_state = x[:, -1, ...].unsqueeze(1)  # [B, 1, C_data, H, W]
            adv = self._compute_advected_persistence(forcing, current_state, n_step)
            for t in range(n_step):
                extra[:, T + t, 2, :, :] = adv[:, t, 0, :, :]

        return torch.cat([forcing, extra], dim=2)

    def training_step(self, batch, batch_idx):
        data, forcing = batch

        if self.hparams.use_flow_matching:
            x, y = data
            forcing = self._build_flow_forcing(forcing, y, x)

        loss, outs = self._roll_forecast(
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
            use_rollout_weighting=self.hparams.use_rollout_weighting,
        )

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]

        self.log("train_loss", loss["loss"], sync_dist=False)

        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                v = torch.sum(v).item()

            self.log(
                f"loss/train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, sync_dist=False)

        if batch_idx == 0:
            self.latest_train_tendencies = tendencies.float()
            self.latest_train_predictions = predictions.float()
            self.latest_train_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def validation_step(self, batch, batch_idx):
        data, forcing = batch

        if self.hparams.use_flow_matching:
            x, y = data
            forcing = self._build_flow_forcing(forcing, y, x)

        loss, outs = self._roll_forecast(
            self.model,
            data,
            forcing,
            self.hparams.rollout_length,
            loss_fn=self._loss_fn,
            use_scheduled_sampling=False,
            use_rollout_weighting=self.hparams.use_rollout_weighting,
            step=self.global_step,
            max_step=getattr(self, "max_steps", None) or self.trainer.max_steps or 1,
        )

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]

        self.log("val_loss", loss["loss"], sync_dist=False)

        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                v = torch.sum(v).item()

            self.log(
                f"loss/val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        if batch_idx == 0:
            self.latest_val_tendencies = tendencies.float()
            self.latest_val_predictions = predictions.float()
            self.latest_val_data = data

        return {
            "loss": loss["loss"],
            "tendencies": tendencies,
            "predictions": predictions,
            "loss_components": loss,
        }

    def test_step(self, batch, batch_idx):
        data, forcing, dates = batch

        if self.hparams.use_flow_matching:
            from swinu.util import flow_roll_forecast

            B, T_total, C_force, H, W = forcing.shape
            use_adv = bool(self.hparams.use_advected_persistence)
            use_rolling = bool(self.hparams.use_rolling_advection)
            if use_rolling and not use_adv:
                raise ValueError(
                    "use_rolling_advection=True requires use_advected_persistence=True"
                )
            n_extra = 3 if use_adv else 2
            extra = torch.zeros(
                B, T_total, n_extra, H, W, device=forcing.device, dtype=forcing.dtype
            )

            rolling_winds = None
            if use_adv:
                x = data[0]
                n_step = self.hparams.rollout_length
                T = T_total - n_step
                if use_rolling:
                    # Channel will be recomputed inside the rollout loop from
                    # the model's evolving prediction. Leave extra channel zero
                    # at entry; flow_roll_forecast overwrites at each step.
                    rolling_winds = self._compute_advection_winds(forcing)
                else:
                    # Frozen-x_0 pre-fill (Path A behaviour)
                    current_state = x[:, -1, ...].unsqueeze(1)
                    adv = self._compute_advected_persistence(forcing, current_state, n_step)
                    for t in range(n_step):
                        extra[:, T + t, 2, :, :] = adv[:, t, 0, :, :]

            flow_forcing = torch.cat([forcing, extra], dim=2)

            _, outs = flow_roll_forecast(
                self.model,
                data,
                flow_forcing,
                self.hparams.rollout_length,
                num_inference_steps=self.hparams.num_inference_steps,
                flow_warm_start=self.hparams.flow_warm_start,
                warm_start_alpha=self.hparams.warm_start_alpha,
                init_noise_sigma=self.hparams.flow_init_noise_sigma,
                eta=self.hparams.flow_eta,
                init_alpha=self.hparams.flow_init_alpha,
                alpha_schedule_rho=self.hparams.flow_alpha_schedule_rho,
                has_advected_persistence=use_adv,
                rolling_advection_winds=rolling_winds,
            )
        else:
            _, outs = self._roll_forecast(
                self.model,
                data,
                forcing,
                self.hparams.rollout_length,
                loss_fn=None,
                use_scheduled_sampling=False,
                use_rollout_weighting=False,
            )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)
        self.test_dates.append(
            torch.concatenate((dates[0][:, -1:], dates[1]), dim=1).cpu()
        )
        truth = torch.concatenate((analysis_time, data[1]), dim=1)
        self.test_truth.append(truth.cpu())

        tendencies = outs["tendencies"]
        predictions = outs["predictions"]
        predictions = torch.concatenate((analysis_time, predictions), dim=1)
        self.test_predictions.append(predictions.cpu())

    def predict_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def _gather(self, predictions, truth, dates):
        # Gather across all DDP ranks
        predictions = self.all_gather(predictions)

        # all_gather adds a new dimension [world_size, ...], so reshape
        predictions = predictions.reshape(-1, *predictions.shape[2:])

        dates = self.all_gather(dates)

        # Dates is 2D [batch, time], so flatten only first two dims
        dates = dates.reshape(-1, dates.shape[-1])

        # Sort by dates to restore chronological order
        sort_idx = torch.argsort(dates[:, 0])
        predictions = predictions[sort_idx]
        dates = dates[sort_idx]

        if len(truth):
            truth = self.all_gather(truth)
            truth = truth.reshape(-1, *truth.shape[2:])
            truth = truth[sort_idx]

        return predictions, truth, dates

    def _write_results_on_test_predict_end(self):
        def _write(tensor, filename):
            torch.save(tensor, filename)
            print(f"Wrote file {filename} (shape: {tensor.shape})")

        # Get the run directory from the checkpoint path
        if self.test_output_directory is None:
            run_dir = self.run_dir if self.run_dir is not None else "."
            self.test_output_directory = f"{run_dir}/test-output"

        os.makedirs(self.test_output_directory, exist_ok=True)

        predictions = torch.concatenate(self.test_predictions)
        dates = torch.concatenate(self.test_dates)

        truth = []
        if len(self.test_truth):
            truth = torch.concatenate(self.test_truth)

        if self.trainer.world_size > 1:
            predictions, truth, dates = self._gather(predictions, truth, dates)

        # Only rank 0 writes to avoid duplicate writes
        if self.trainer.is_global_zero:
            _write(predictions, f"{self.test_output_directory}/predictions.pt")

            if len(truth):
                _write(truth, f"{self.test_output_directory}/truth.pt")

            _write(dates, f"{self.test_output_directory}/dates.pt")

    def on_test_end(self):
        self._write_results_on_test_predict_end()

    def on_predict_end(self):
        self._write_results_on_test_predict_end()

    def configure_optimizers(self):
        decay, no_decay = [], []

        NO_DECAY_KEYS = ["bias", "pos_embed", "norm", "alpha", "gamma", ".A.", ".B."]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in NO_DECAY_KEYS):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
        )

        if self.trainer.max_epochs > 0:
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

        elif self.trainer.max_steps is not None:
            assert (
                self.trainer.max_steps > self.hparams.warmup_iterations
            ), f"max_steps {self.trainer.max_steps} is smaller than warmup_iterations {self.hparams.warmup_iterations}"

            T_max = self.trainer.max_steps - self.hparams.warmup_iterations

        else:
            raise ValueError("Either max_epochs or max_steps needs to be defined")

        assert T_max > 0, f"max iterations is {T_max}"

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

    def on_save_checkpoint(self, checkpoint):
        dm = self.trainer.datamodule
        if dm and dm._full_dataset and hasattr(dm._full_dataset, "data"):
            checkpoint["data_statistics"] = dm._full_dataset.data.statistics
            checkpoint["data_statististics_name_to_index"] = (
                dm._full_dataset.data.name_to_index
            )
        else:
            rank_zero_warn("Statistics not saved to checkpoint")
