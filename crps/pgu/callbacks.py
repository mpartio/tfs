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
import json
import warnings
import shutil
from pgu.util import roll_forecast
from common.util import calculate_wavelet_snr, moving_average, get_rank
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only
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


def run_info():
    run_name = os.environ["CC2_RUN_NAME"]
    run_number = os.environ["CC2_RUN_NUMBER"]
    run_dir = os.environ["CC2_RUN_DIR"]

    return run_name, run_number, run_dir


def envelope_binning(x: torch.tensor, n_bins: int = 1000):
    # divide data into bins and calculate min and max from each bin,
    # and plot those
    N = x.shape[0]
    B = min(n_bins, N)  # <-- never more bins than data
    W = N // B  # now W >= 1
    trimmed = x[: B * W]

    blocks = trimmed.view(B, W)
    mins, _ = blocks.min(dim=1)
    maxs, _ = blocks.max(dim=1)
    xs = torch.arange(B) * W

    return xs, mn, mx


def dynamic_ma(x: torch.tensor, n_bins: int):
    N = x.shape[0]
    W = max(1, N // n_bins)
    # your existing moving_average(x, W) function
    return moving_average(x, W)


def analyze_gradients(model):
    # Pre-compile patterns to check once
    sections = [
        "encoder1",
        "encoder2",
        "upsample",
        "downsample",
        "decoder1",
        "decoder2",
        "expand",
    ]
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

        train_dataloader = trainer.train_dataloader

        # Get a single batch from the dataloader
        data, forcing = next(iter(train_dataloader))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = forcing.to(pl_module.device)

        rollout_length = data[1].shape[1]

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                rollout_length,
                None,
            )

        x, y = data

        self.plot(
            x.cpu().detach(),
            y.cpu().detach(),
            predictions.cpu().detach(),
            trainer.current_epoch,
            "train",
            trainer.sanity_checking,
        )

        # Restore model to training mode
        pl_module.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        val_dataloader = trainer.val_dataloaders

        data, forcing = next(iter(val_dataloader))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = forcing.to(pl_module.device)

        rollout_length = data[1].shape[1]

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                rollout_length,
                None,
            )

        x, y = data

        self.plot(
            x.cpu().detach(),
            y.cpu().detach(),
            predictions.cpu().detach(),
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
        plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, check_frequency: int = 50):
        (
            self.train_loss,
            self.val_loss,
            self.lr,
            self.val_snr,
        ) = ([], [], [], [])

        self.train_loss_components = {}
        self.val_loss_components = {}
        self.gradients_mean = {}
        self.gradients_std = {}

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
            "val_snr": self.val_snr,
        }

    def load_state_dict(self, state_dict):
        self.gradients_mean = state_dict["gradients_mean"]
        self.gradients_std = state_dict["gradients_std"]
        self.lr = state_dict["lr"]
        self.train_loss = state_dict["train_loss"]
        self.train_loss_components = state_dict["train_loss_components"]
        self.val_loss = state_dict["val_loss"]
        self.val_loss_components = state_dict["val_loss_components"]
        self.val_snr = state_dict["val_snr"]

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) train loss
        self.train_loss.append(outputs["loss"].item())
        pl_module.log("train_loss", self.train_loss[-1], prog_bar=True)

        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue
            try:
                self.train_loss_components[k].append(v.cpu())
            except KeyError as e:
                self.train_loss_components[k] = [v.cpu()]

        # b) learning rate
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

        # c) gradients
        if batch_idx % self.check_frequency == 0:
            grads = analyze_gradients(pl_module)

            for k in grads.keys():
                mean, std = grads[k]["mean"], grads[k]["std"]
                try:
                    self.gradients_mean[k].append(mean)
                    self.gradients_std[k].append(std)
                except KeyError as e:
                    self.gradients_mean[k] = [mean]
                    self.gradients_std[k] = [std]

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) Validation loss
        self.val_loss.append(outputs["loss"].detach().cpu())
        pl_module.log("val_loss", self.val_loss[-1].mean(), prog_bar=True)

        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue
            try:
                self.val_loss_components[k].append(v.cpu())
            except KeyError as e:
                self.val_loss_components[k] = [v.cpu()]

        if batch_idx % self.check_frequency == 0:
            # b) signal to noise ratio
            predictions = outputs["predictions"]

            data, _ = batch
            _, y = data

            # Select first of batch and last of time
            truth = y[0][-1].cpu().squeeze()

            # ... and first of members
            pred = predictions[0][-1][0].cpu().squeeze()

            snr_pred = calculate_wavelet_snr(pred, None)
            snr_real = calculate_wavelet_snr(truth, None)
            self.val_snr.append((snr_real["snr_db"], snr_pred["snr_db"]))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.is_global_zero:
            return

        # Get a single batch from the dataloader
        data, forcing = next(iter(trainer.val_dataloaders))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = forcing.to(pl_module.device)

        rollout_length = data[1].shape[1]

        # Perform a prediction
        with torch.no_grad():
            _, tendencies, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                rollout_length,
                loss_fn=None,
            )

        x, y = data

        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()

        assert torch.isfinite(tendencies).all(), "non-finite values in tendencies"
        assert torch.isfinite(predictions).all(), "non-finite values in predictions"

        self.plot_visual(
            x[0].cpu().detach(),
            y[0].cpu().detach(),
            predictions[0].cpu().detach(),
            tendencies[0].cpu().detach(),
            trainer.current_epoch,
            trainer.sanity_checking,
        )

        self.plot_history(trainer.current_epoch, trainer.sanity_checking)

    def plot_visual(self, input_field, truth, pred, tendencies, epoch, sanity_checking):
        run_name, run_number, run_dir = run_info()

        plt.figure(figsize=(24, 8))
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
            cax = divider.append_axes("right", size="0.1%", pad=0.05)
            fig.colorbar(im, cax=cax)

            ax[t, 1].imshow(tendencies[t], cmap=cmap, norm=norm)
            ax[t, 1].set_title(f"Predicted tendencies step={t}")
            ax[t, 1].set_axis_off()

            data = true_tendencies - tendencies[t]
            ax[t, 2].set_title(f"Tendencies bias step={t}")
            ax[t, 2].set_axis_off()
            ax[t, 2].imshow(data.cpu(), cmap=cmap, norm=norm)

            ax[t, 3].set_title(f"True histogram step={t}")
            ax[t, 3].hist(true_tendencies.flatten(), bins=30)

            ax[t, 4].set_title(f"Predicted histogram step={t}")
            ax[t, 4].hist(tendencies[t].flatten(), bins=30)

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

        plt.close()

    def plot_history(self, epoch, sanity_checking=False):
        def clip_to_quantile(tensor: torch.tensor, quantile: float = 0.99):
            if tensor.numel() == 0:
                return tensor
            threshold = torch.quantile(tensor, quantile)
            return torch.clamp(tensor, max=threshold)

        warnings.filterwarnings(
            "ignore", message="No artists with labels found to put in legend"
        )

        run_name, run_number, run_dir = run_info()

        plt.figure(figsize=(16, 16))
        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                run_name,
                run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # TRAIN LOSS

        train_loss = torch.tensor(self.train_loss)
        if len(train_loss) > 2000:
            # Remove first 100 after we have enough data, as they often
            # they contain data messes up the y-axis
            train_loss = train_loss[100:]

        n_bins = 1000
        train_loss = clip_to_quantile(train_loss)
        xs, mn, mx = envelope_binning(train_loss, n_bins=n_bins)

        plt.subplot(331)
        plt.title("Training loss")
        plt.fill_between(xs, mn, mx, color="C0", alpha=0.2)

        ma = dynamic_ma(train_loss, n_bins=n_bins)

        plt.plot(
            xs,
            ma[xs],
            color="C0",
            label="Train Loss",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(torch.tensor(self.lr), label="LR", color="green")
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax2.legend(loc="upper right")

        val_loss = torch.tensor(self.val_loss)
        val_loss = clip_to_quantile(val_loss)

        if len(val_loss) > 500:
            val_loss = val_loss[20:]

        loss_names = self.train_loss_components.keys()

        assert sanity_checking == True or len(loss_names) == 2

        for i, name in enumerate(loss_names):
            plt.subplot(3, 3, 2 + i)

            plt.title(f"Train {name} per step")
            data = torch.stack(self.train_loss_components[name])

            if data.ndim == 1:
                data = data.unsqueeze(1)

            for j in range(data.shape[1]):
                color = colors[j]
                plt.plot(data[:, j], color=color, alpha=0.3)
                plt.plot(moving_average(data[:, j], 60), color=color, label=f"step={j}")
            plt.legend()

        # VALIDATION LOSS

        plt.subplot(334)
        plt.plot(val_loss, color="orange", alpha=0.3)
        plt.plot(
            moving_average(val_loss, 60),
            color="orange",
            label="Val Loss",
        )
        plt.title("Validation loss")
        plt.legend(loc="upper left")

        loss_names = self.val_loss_components.keys()

        for i, name in enumerate(loss_names):
            plt.subplot(3, 3, 5 + i)

            plt.title(f"Validation {name} per step")
            data = torch.stack(self.val_loss_components[name])

            if data.ndim == 1:
                data = data.unsqueeze(1)

            for j in range(data.shape[1]):
                color = colors[j]
                plt.plot(data[:, j], color=color, alpha=0.3)
                plt.plot(moving_average(data[:, j], 60), color=color, label=f"step={j}")
            plt.legend()

        # REST

        plt.subplot(337)
        val_snr = np.array(self.val_snr).T
        snr_real = clip_to_quantile(torch.tensor(val_snr[0]))
        snr_pred = clip_to_quantile(torch.tensor(val_snr[1]))
        plt.plot(snr_real, color="blue", alpha=0.3)
        plt.plot(
            moving_average(snr_real, 30),
            color="blue",
            label="Real",
        )
        plt.plot(snr_pred, color="orange", alpha=0.3)
        plt.plot(moving_average(snr_pred, 30), color="orange", label="Predicted")
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        residual = snr_real - snr_pred

        ax2.plot(residual, color="green", alpha=0.3)
        ax2.plot(moving_average(residual, 30), label="Residual", color="green")
        ax2.legend(loc="upper right")
        plt.title("Signal to Noise Ratio")

        plt.subplot(338)
        plt.yscale("log")
        plt.title("Gradients (mean)")

        for i, section in enumerate(self.gradients_mean.keys()):
            data = self.gradients_mean[section]
            data = torch.tensor(data)
            color = colors[i]
            plt.plot(data, color=color, alpha=0.3)
            plt.plot(moving_average(data, 30), color=color, label=section)
        plt.legend()

        plt.subplot(339)
        plt.yscale("log")
        plt.title("Gradients (std)")
        for i, section in enumerate(self.gradients_std.keys()):
            data = self.gradients_std[section]
            data = torch.tensor(data)
            color = colors[i]
            plt.plot(data, color=color, alpha=0.3)
            plt.plot(moving_average(data, 30), color=color, label=section)
        plt.legend()

        plt.tight_layout()

        if sanity_checking:
            return

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_history.png".format(
                run_dir,
                platform.node(),
                run_name,
                run_number,
                epoch,
            )
        )

        plt.close()


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


class LazyLoggerCallback(L.Callback):
    def __init__(self, run_name: str, run_number: int):
        super().__init__()
        self.run_name = run_name
        self.run_number = run_number
        self.logger_created = False
        self.run_dir = f"runs/{run_name}/{run_number}"

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """This runs after the sanity check is successful."""
        if not self.logger_created and get_rank() == 0:
            trainer.logger = CSVLogger(f"{self.run_dir}/logs")
            self.logger_created = True  # Prevent multiple reassignments
            print(f"Logger initialized at {self.run_dir}/logs")
