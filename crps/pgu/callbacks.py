import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
import os
import warnings
import shutil
from common.util import calculate_wavelet_snr, moving_average, get_rank
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import rank_zero_only
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "black",
    "purple",
    "grey",
    "magenta",
]


def get_gradient_names():
    return [
        "encoder1",
        "encoder2",
        "upsample",
        "downsample",
        "decoder1",
        "decoder2",
        "expand",
    ]


def run_info():
    run_name = os.environ["CC2_RUN_NAME"]
    run_number = os.environ["CC2_RUN_NUMBER"]
    run_dir = os.environ["CC2_RUN_DIR"]

    return run_name, run_number, run_dir


def envelope_binning(x: torch.tensor, n_bins: int = 1000):
    # divide data into bins and calculate min and max from each bin,
    # and plot those
    N = x.shape[0]

    if N == 0:
        empty = x.new_empty(0)
        empty = empty.long()
        return empty, empty, empty

    B = max(1, min(n_bins, N))
    W = N // B  # now W >= 1
    trimmed = x[: B * W]

    blocks = trimmed.view(B, W)
    mins, _ = blocks.min(dim=1)
    maxs, _ = blocks.max(dim=1)
    xs = torch.arange(B) * W
    xs = xs.long()
    return xs, mins, maxs


def dynamic_ma(x: torch.tensor, n_bins: int):
    N = x.shape[0]
    W = max(40, N // n_bins)
    # your existing moving_average(x, W) function
    return moving_average(x, W)


def analyze_gradients(model):
    # Pre-compile patterns to check once
    sections = get_gradient_names()
    gradient_stats = {k: [] for k in sections}

    # Batch all gradient stats by section in a single pass
    with torch.no_grad():  # Avoid tracking history for these operations
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            for section in sections:
                if section in name:
                    gradient_stats[section].append(param.grad.abs().mean())
                    break  # Parameter can only belong to one section

    # Calculate statistics using PyTorch operations
    stats = {}
    for section, grads in gradient_stats.items():
        if grads:
            grads_tensor = torch.stack(grads)
            stats[section] = {
                "mean": grads_tensor.mean().item(),
                "std": grads_tensor.std().item(),
            }

    return stats


class PredictionPlotterCallback(L.Callback):
    def __init__(self):
        pass

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        predictions = pl_module.latest_train_predictions
        x, y = pl_module.latest_train_data

        self.plot(
            x.cpu().detach().to(torch.float32),
            y.cpu().detach().to(torch.float32),
            predictions.cpu().detach().to(torch.float32),
            trainer.current_epoch,
            "train",
            trainer.sanity_checking,
        )

        # Restore model to training mode
        pl_module.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        predictions = pl_module.latest_val_predictions
        x, y = pl_module.latest_val_data

        self.plot(
            x.cpu().detach().to(torch.float32),
            y.cpu().detach().to(torch.float32),
            predictions.cpu().detach().to(torch.float32),
            trainer.current_epoch,
            "val",
            trainer.sanity_checking,
        )

        # Restore model to training mode
        pl_module.train()

    def plot(self, x, y, predictions, epoch, stage, sanity_check=False):
        # y shape: [B, T, 1, 128, 128]
        # prediction shape: [B, T, 1, 128, 128]
        # input_field = x[0].squeeze()
        run_name, run_number, run_dir = run_info()

        truth = y[0]

        x = x[0]  # T, C, H, W
        predictions = predictions[0]  # T, C, H, W (B removed)
        num_truth = truth.shape[0]
        num_hist = x.shape[0]

        rows = num_truth
        cols = num_hist + 1 + 1  # input fields, +1 for prediction, +1 for truth

        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.5))
        ax = np.atleast_2d(ax)

        for i in range(num_truth):  # times
            if i == 0:
                # First prediction: all ground truth
                input_field = x.squeeze()
            elif i <= num_hist:
                # gradually using own predictions as input
                num_ground_truth = max(0, num_hist - i)
                num_pred = num_hist - num_ground_truth
                start_pred = max(0, i - num_hist)

                input_field = torch.cat(
                    [
                        x[i:num_hist].squeeze(1),
                        predictions[start_pred:num_pred].squeeze(1),
                    ]
                )
            elif i > num_hist:
                start_pred = max(0, i - num_hist)
                input_field = predictions[start_pred : start_pred + num_hist].squeeze()

            for j in range(num_hist):
                num = i - num_hist + j + 1
                ax[i, j].imshow(input_field[j])
                ax[i, j].set_title(f"Time T{num:+}")
                ax[i, j].set_axis_off()
            ax[i, num_hist].imshow(predictions[i].squeeze())
            ax[i, num_hist].set_title(f"Time T{i+1:+} prediction")
            ax[i, num_hist].set_axis_off()
            ax[i, num_hist + 1].imshow(truth[i].squeeze())
            ax[i, num_hist + 1].set_title(f"Time T{i+1:+} truth")
            ax[i, num_hist + 1].set_axis_off()

        title = (
            "Training time prediction"
            if stage == "train"
            else "Validation time prediction"
        )

        title = "{} for {} num={} at epoch {} (host={}, time={})".format(
            title,
            run_name,
            run_number,
            epoch,
            platform.node(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = "{}/figures/{}_{}_{}_epoch_{:03d}_{}.png".format(
            run_dir,
            platform.node(),
            run_name,
            run_number,
            epoch,
            stage,
        )

        plt.suptitle(title)

        if sanity_check is False:
            plt.savefig(filename)
        plt.close(fig)


class DiagnosticCallback(L.Callback):
    def __init__(self, check_frequency: int = 50):

        self.train_loss = []
        self.val_loss = []
        self.lr = []
        self.val_snr_real = []
        self.val_snr_pred = []
        self.gradients_mean = {}
        self.gradients_std = {}
        self.train_loss_components = {}
        self.val_loss_components = {}

        self.loss_names = []
        self.grad_names = get_gradient_names()

        self.check_frequency = check_frequency

    def state_dict(self):
        return {
            "gradients_mean": self.gradients_mean,
            "gradients_std": self.gradients_std,
            "lr": self.lr,
            "train_loss": self.train_loss,
            "train_loss_components": self.train_loss_components,
            "val_loss": self.val_loss,
            "val_loss_components": self.val_loss_components,
            "val_snr_real": self.val_snr_real,
            "val_snr_pred": self.val_snr_pred,
        }

    def load_state_dict(self, state_dict):
        try:
            self.gradients_mean = state_dict["gradients_mean"]
            self.gradients_std = state_dict["gradients_std"]
            self.lr = state_dict["lr"]
            self.train_loss = state_dict["train_loss"]
            self.train_loss_components = state_dict["train_loss_components"]
            self.val_loss = state_dict["val_loss"]
            self.val_loss_components = state_dict["val_loss_components"]
            self.val_snr_real = state_dict["val_snr_real"]
            self.val_snr_pred = state_dict["val_snr_pred"]
        except KeyError as e:
            print(
                f"Warning: Missing key in DiagnosticCallback state_dict: {e}. Continuing anyway."
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        # a) train loss
        train_loss = outputs["loss"]
        pl_module.log(
            "train/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        _loss_names = []
        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue

            pl_module.log(
                f"train/{k}",
                torch.sum(v),
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )

            if len(self.loss_names) == 0:
                _loss_names.append(k)

        if len(self.loss_names) == 0:
            self.loss_names = _loss_names

        # b) learning rate
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("lr", lr, on_step=True, on_epoch=False, sync_dist=False)
        return

        # c) gradients
        if batch_idx % self.check_frequency == 0:
            grads = analyze_gradients(pl_module)

            for k in grads.keys():
                mean, std = grads[k]["mean"], grads[k]["std"]
                pl_module.log(
                    f"train/grad_{k}_mean",
                    mean,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
                pl_module.log(
                    f"train/grad_{k}_std",
                    std,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        # a) Validation loss
        val_loss = outputs["loss"]
        pl_module.log(
            "val/loss_epoch",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue

            pl_module.log(
                f"val/{k}",
                torch.sum(v),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        if batch_idx % self.check_frequency == 0:
            # b) signal to noise ratio
            predictions = outputs["predictions"]

            data, _ = batch
            _, y = data

            # Select first of batch and last of time
            truth = y[0][-1].cpu().squeeze().to(torch.float32)

            # ... and first of members
            pred = predictions[0][-1][0].cpu().squeeze().to(torch.float32)

            snr_pred = calculate_wavelet_snr(pred, None)
            snr_real = calculate_wavelet_snr(truth, None)

            pl_module.log(
                "val/snr_real",
                snr_real["snr_db"],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            pl_module.log(
                "val/snr_pred",
                snr_pred["snr_db"],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        current_train_loss = trainer.logged_metrics.get(
            "train/loss_epoch", float("nan")
        )
        self.train_loss.append(current_train_loss)

        for k in self.loss_names:
            val = trainer.logged_metrics.get(f"train/{k}_epoch", float("nan"))

            try:
                self.train_loss_components[k].append(val)
            except KeyError:
                self.train_loss_components[k] = [val]

        for k in self.grad_names:
            val = trainer.logged_metrics.get(f"train/grad_{k}_mean_epoch", float("nan"))

            try:
                self.gradients_mean[k].append(val)
            except KeyError:
                self.gradients_mean[k] = [val]

            val = trainer.logged_metrics.get(f"train/grad_{k}_std_epoch", float("nan"))

            try:
                self.gradients_std[k].append(val)
            except KeyError:
                self.gradients_std[k] = [val]

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        tendencies = pl_module.latest_val_tendencies
        predictions = pl_module.latest_val_predictions
        x, y = pl_module.latest_val_data

        self.plot_visual(
            x[0].cpu().detach().to(torch.float32),
            y[0].cpu().detach().to(torch.float32),
            predictions[0].cpu().detach().to(torch.float32),
            tendencies[0].cpu().detach().to(torch.float32),
            trainer.current_epoch,
            trainer.sanity_checking,
        )

    def plot_visual(self, input_field, truth, pred, tendencies, epoch, sanity_checking):
        run_name, run_number, run_dir = run_info()

        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                run_name,
                run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # input_field T, C, H, W
        # truth T, C, H, W
        # pred T, C, H, W
        # tend T, C, H, W

        input_field = input_field.squeeze(1)
        truth = truth.squeeze(1)  # remove channel dim
        pred = pred.squeeze(1)
        tendencies = tendencies.squeeze(1)

        T = pred.shape[0]

        fig, ax = plt.subplots(T, 5, figsize=(15, 3 * T + 0.5))
        ax = np.atleast_2d(ax)

        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for t in range(T):
            if t == 0:
                true_tendencies = truth[t].squeeze() - input_field[-1].squeeze()
            else:
                true_tendencies = truth[t].squeeze() - truth[t - 1].squeeze()

            im = ax[t, 0].imshow(true_tendencies, cmap=cmap, norm=norm)
            ax[t, 0].set_title(f"True tendencies step={t}")
            ax[t, 0].set_axis_off()

            divider = make_axes_locatable(ax[t, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            ax[t, 1].imshow(tendencies[t], cmap=cmap, norm=norm)
            ax[t, 1].set_title(f"Predicted tendencies step={t}")
            ax[t, 1].set_axis_off()

            data = true_tendencies - tendencies[t]
            ax[t, 2].set_title(f"Tendencies bias step={t}")
            ax[t, 2].set_axis_off()
            im = ax[t, 2].imshow(data.cpu(), cmap=cmap)

            divider_bias = make_axes_locatable(ax[t, 2])
            cax_bias = divider_bias.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax_bias)

            ax[t, 3].set_title(f"True histogram step={t}")
            ax[t, 3].hist(true_tendencies.flatten(), bins=50, density=True)

            ax[t, 4].set_title(f"Predicted histogram step={t}")
            ax[t, 4].hist(tendencies[t].flatten(), bins=50, density=True)

        plt.tight_layout()

        if sanity_checking:
            return

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_analysis.png".format(
                run_dir,
                platform.node(),
                run_name,
                run_number,
                epoch,
            )
        )

        plt.close(fig)


class CleanupFailedRunCallback(L.Callback):
    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        # Ensure cleanup only happens on the main process (rank 0) in distributed settings
        if trainer.is_global_zero:
            run_dir = os.environ.get("CC2_RUN_DIR")

            if run_dir and os.path.isdir(run_dir):
                # Check if the directory is empty or contains only minimal files
                # Adjust this condition based on what you consider "empty"
                # e.g., allow hparams.yaml, empty logs/ dir?
                # A simple check is just len(os.listdir()) == 0 or 1 (maybe hparams.yaml)
                try:
                    files_in_dir = os.listdir(f"{run_dir}/figures")
                    # Define what constitutes an "empty" directory that should be removed
                    # Example: empty or only contains hparams file
                    if len(files_in_dir) == 0:
                        shutil.rmtree(run_dir)
                        print(f"Removed empty run directory: {run_dir}")
                except OSError as e:
                    print(f"\nError during cleanup check/removal of {log_dir}: {e}")
            else:
                print(
                    f"\nDetected exception: {type(exception).__name__}. Could not determine log directory for cleanup."
                )
