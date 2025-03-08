import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
import config
import json
from util import roll_forecast
from common.util import calculate_wavelet_snr, moving_average, get_rank
from datetime import datetime
from dataclasses import asdict
from matplotlib.ticker import ScalarFormatter
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only

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


def analyze_gradients(model):
    # Group gradients by network section
    gradient_stats = {
        "encoder1": [],
        "encoder2": [],
        "upsample": [],
        "downsample": [],
        "decoder1": [],
        "decoder2": [],
        "expand": [],
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            for k in gradient_stats.keys():
                if k in name:
                    grad_norm = param.grad.abs().mean().item()
                    gradient_stats[k].append(grad_norm)

    # Compute statistics for each section
    stats = {}
    for section, grads in gradient_stats.items():
        if grads:
            stats[section] = {
                "mean": np.mean(grads).item(),
                "std": np.std(grads).item(),
                # "min": np.min(grads),
                # "max": np.max(grads),
            }

    return stats


class PredictionPlotterCallback(L.Callback):
    def __init__(self, train_loader, val_loader, config):
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.config = config

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        pl_module.eval()

        # Get a single batch from the dataloader
        data, forcing = next(iter(self.train_dataloader))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = (forcing[0].to(pl_module.device), forcing[1].to(pl_module.device))

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                self.config.rollout_length,
                None,
            )

        x, y = data

        self.plot(
            x.cpu().detach(),
            y.cpu().detach(),
            predictions.cpu().detach(),
            trainer.current_epoch,
            "train",
        )

        # Restore model to training mode
        pl_module.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        pl_module.eval()

        data, forcing = next(iter(self.val_dataloader))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = (forcing[0].to(pl_module.device), forcing[1].to(pl_module.device))

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                self.config.rollout_length,
                None,
            )

        x, y = data

        self.plot(
            x.cpu().detach(),
            y.cpu().detach(),
            predictions.cpu().detach(),
            trainer.current_epoch,
            "val",
        )

        # Restore model to training mode
        pl_module.train()

    def plot(self, x, y, predictions, epoch, stage):
        # y shape: [B, T, 1, 128, 128]
        # prediction shape: [B, T, 1, 128, 128]

        input_field = x[0].squeeze()
        truth = y[0]

        predictions = predictions[0]  # T, C, H, W (B removed)
        num_truth = truth.shape[0]

        rows = num_truth
        cols = 2 + 1 + 1  # +2 for input fields, +1 for prediction, +1 for truth

        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.5))
        ax = np.atleast_2d(ax)

        for i in range(num_truth):  # times
            if i == 0:
                input_field = x[0].squeeze()
            elif i == 1:
                input_field = torch.stack(
                    [x[0, 1].squeeze(), predictions[0][0].squeeze()]
                )
            elif i > 1:
                input_field = torch.stack(
                    [predictions[i - 2, 0].squeeze(), predictions[i - 1, 0].squeeze()]
                )

            ax[i, 0].imshow(input_field[0])
            ax[i, 0].set_title(f"Time T{i-1:+}")
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(input_field[1])
            ax[i, 1].set_title(f"Time T{i:+}")
            ax[i, 1].set_axis_off()
            ax[i, 2].imshow(predictions[i].squeeze())
            ax[i, 2].set_title(f"Time T{i+1:+} prediction")
            ax[i, 2].set_axis_off()
            ax[i, 3].imshow(truth[i].squeeze())
            ax[i, 3].set_title(f"Time T{i+1:+} truth")
            ax[i, 3].set_axis_off()

        title = (
            "Training time prediction"
            if stage == "train"
            else "Validation time prediction"
        )

        title = "{} for {} num={} at epoch {} (host={}, time={})".format(
            title,
            self.config.run_name,
            self.config.run_number,
            epoch,
            platform.node(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = "{}/figures/{}_{}_{}_epoch_{:03d}_{}.png".format(
            self.config.run_dir,
            platform.node(),
            self.config.run_name,
            self.config.run_number,
            epoch,
            stage,
        )

        plt.suptitle(title)
        plt.savefig(filename)
        plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, config, freq=50):
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

        self.freq = freq
        self.config = config

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) train loss
        self.train_loss.append(outputs["loss"].detach().cpu().item())
        pl_module.log("train_loss", self.train_loss[-1], prog_bar=True)

        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue
            try:
                self.train_loss_components[k].append(v.cpu())
            except KeyError as e:
                self.train_loss_components[k] = [v.cpu()]

        # b) learning rate
        assert len(trainer.optimizers) > 0, "No optimizer found"
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

        # c) gradients
        if batch_idx % self.freq == 0:
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

        if batch_idx % self.freq == 0:
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
        if trainer.sanity_checking:
            return

        if not trainer.is_global_zero:
            return

        # Get a single batch from the dataloader
        data, forcing = next(iter(trainer.val_dataloaders))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = (forcing[0].to(pl_module.device), forcing[1].to(pl_module.device))

        # Perform a prediction
        with torch.no_grad():
            _, tendencies, predictions = roll_forecast(
                pl_module,
                data,
                forcing,
                self.config.rollout_length,
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
        )

        self.plot_history(trainer.current_epoch)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Convert any tensors/numpy arrays to Python types
        def convert_to_serializable(obj):
            if hasattr(obj, "tolist"):  # Handle tensors/numpy arrays
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj

        D = {"config": {}, "statistics": {}}

        saved_variables = [
            "train_loss",
            "val_loss",
            "val_snr",
            "lr",
            "gradients_mean",
            "gradients_std",
            "freq",
        ]

        for k in saved_variables:
            D["statistics"][k] = self.__dict__[k]

        for k in self.config.__dict__.keys():
            D["config"][k] = self.config.__dict__[k]

        D["config"]["current_iteration"] = trainer.global_step

        filename = f"{self.config.run_dir}/run-info.json"

        with open(filename, "w") as f:
            json.dump(D, f, indent=4, default=convert_to_serializable)

    def plot_visual(self, input_field, truth, pred, tendencies, epoch):
        plt.figure(figsize=(24, 8))
        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                self.config.run_name,
                self.config.run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # input_field T, H, W, C
        # truth T, H, W, C
        # pred T, C, H, W
        # tend T, C, H, W

        input_field = input_field.squeeze(1)
        truth = truth.squeeze(1)  # remove channel dim
        pred = pred.squeeze(1)
        tendencies = tendencies.squeeze(1)

        T = pred.shape[0]

        if T == 1:
            true_tendencies = truth[-1].squeeze() - input_field[-1].squeeze()
        else:
            true_tendencies = truth[-1].squeeze() - truth[-2].squeeze()

        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        plt.subplot(241)
        plt.imshow(true_tendencies, cmap=cmap, norm=norm)
        plt.title("True Tendencies")
        plt.colorbar()

        plt.subplot(242)
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        plt.imshow(tendencies[-1], cmap=cmap, norm=norm)
        plt.title("Predicted Tendencies")
        plt.colorbar()

        data = pred[-1] - truth[-1]
        plt.subplot(243)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Prediction Bias")
        plt.colorbar()

        data = true_tendencies - tendencies[-1]
        plt.subplot(244)
        plt.title("Tendency Bias")
        plt.imshow(data.cpu(), cmap=cmap, norm=norm)
        plt.colorbar()

        plt.subplot(245)
        plt.hist(truth[-1].flatten(), bins=30)
        plt.title("True histogram")

        plt.subplot(246)
        plt.hist(pred[-1].flatten(), bins=30)
        plt.title("Predicted histogram")

        plt.subplot(247)
        plt.title("True tendencies histogram")
        plt.hist(true_tendencies.flatten(), bins=30)

        plt.subplot(248)
        plt.title("Predicted tendencies histogram")
        plt.hist(tendencies[-1].flatten(), bins=30)

        plt.tight_layout()
        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_analysis.png".format(
                self.config.run_dir,
                platform.node(),
                self.config.run_name,
                self.config.run_number,
                epoch,
            )
        )

        plt.close()

    def plot_history(self, epoch):
        plt.figure(figsize=(16, 16))
        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                self.config.run_name,
                self.config.run_number,
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

        plt.subplot(331)
        plt.title("Training loss")
        plt.plot(train_loss, color="blue", alpha=0.3)
        plt.plot(
            moving_average(train_loss, 60),
            color="blue",
            label="Train Loss",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(torch.tensor(self.lr), label="LR", color="green")
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax2.legend(loc="upper right")

        val_loss = torch.tensor(self.val_loss)
        if len(val_loss) > 500:
            val_loss = val_loss[20:]

        loss_names = self.train_loss_components.keys()

        assert len(loss_names) == 2

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
        snr_real = torch.tensor(val_snr[0])
        snr_pred = torch.tensor(val_snr[1])
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
        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_history.png".format(
                self.config.run_dir,
                platform.node(),
                self.config.run_name,
                self.config.run_number,
                epoch,
            )
        )

        plt.close()


class LazyLoggerCallback(L.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger_created = False

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """This runs after the sanity check is successful."""
        if not self.logger_created and get_rank() == 0:
            trainer.logger = CSVLogger(f"{self.config.run_dir}/logs")
            self.logger_created = True  # Prevent multiple reassignments
            print(f"Logger initialized at {self.config.run_dir}/logs")
