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
import warnings
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


def var_and_mae(predictions, y):
    variance = torch.var(predictions[:, :, -1, ...], dim=2, unbiased=False)
    variance = variance.detach().mean().cpu().numpy().item()

    mean_pred = torch.mean(predictions[:, :, -1, ...], dim=2)
    y_true = y[:, -1, ...]

    mae = torch.mean(torch.abs(y_true - mean_pred)).detach().cpu().numpy().item()

    return variance, mae


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
        forcing = forcing.to(pl_module.device)

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
        forcing = forcing.to(pl_module.device)

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
        # Take first batch member
        # Layout
        # 1. Input field at T-1
        # 2. Input field at T
        # 3. Prediction at T+1 (first prediction)
        # 4. Prediction at T+1 (second prediction)
        # 5. Prediction at T+1 (third prediction)
        # 6. Truth at T+1
        # 7. Prediction at T+2 (first prediction)
        # 8. Prediction at T+2 (second prediction)
        # 9. Prediction at T+2 (third prediction)
        # 10. Truth at T+2
        # ...
        #
        # 1 2 3 4 5 6
        # 2 3 7 8 9 10

        # y shape: [B, 1, 1, 128, 128]
        # prediction shape: [B, 3, 1, 128, 128]

        input_field = x[0].squeeze()
        truth = y[0]

        predictions = predictions[0]  # M, T, C, H, W (B removed)

        num_truth = truth.shape[0]
        num_members = predictions.shape[0]

        rows = num_truth
        cols = num_members + 2 + 1  # +2 for input fields, +1 for truth

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

            for j in range(num_members):  # members
                ax[i, j + 2].imshow(predictions[j, i, ...].squeeze())
                ax[i, j + 2].set_title(f"Time T{i+1:+} member {j}")
                ax[i, j + 2].set_axis_off()

            ax[i, j + 3].imshow(truth[i].squeeze())
            ax[i, j + 3].set_title(f"Time T{i+1:+} truth")
            ax[i, j + 3].set_axis_off()

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
            self.train_mae,
            self.train_var,
            self.val_loss,
            self.lr,
            self.val_snr,
            self.val_mae,
            self.val_var,
        ) = ([], [], [], [], [], [], [], [])

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

        # b) learning rate
        assert len(trainer.optimizers) > 0, "No optimizer found"
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

        if batch_idx % self.freq == 0:
            # c) gradients
            grads = analyze_gradients(pl_module)

            for k in grads.keys():
                mean, std = grads[k]["mean"], grads[k]["std"]
                try:
                    self.gradients_mean[k].append(mean)
                    self.gradients_std[k].append(std)
                except KeyError as e:
                    self.gradients_mean[k] = [mean]
                    self.gradients_std[k] = [std]

            # d) variance and l1

            predictions = outputs["predictions"]  # B, M, T, C, H, W

            data, _ = batch
            _, y = data

            var, mae = var_and_mae(predictions, y)
            self.train_var.append(var)
            self.train_mae.append(mae)

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) Validation loss
        self.val_loss.append(outputs["loss"].detach().cpu().item())
        pl_module.log("val_loss", self.val_loss[-1], prog_bar=True)

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

            # d) variance and l1
            var, mae = var_and_mae(predictions, y)
            self.val_var.append(var)
            self.val_mae.append(mae)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.is_global_zero:
            return

        # Get a single batch from the dataloader
        data, forcing = next(iter(trainer.val_dataloaders))
        data = (data[0].to(pl_module.device), data[1].to(pl_module.device))
        forcing = forcing.to(pl_module.device)

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

        self.plot_visual(
            x[0].cpu().detach(),
            y[0].cpu().detach(),
            predictions[0].cpu().detach(),
            tendencies[0].cpu().detach(),
            trainer.current_epoch,
            trainer.sanity_checking,
        )

        self.plot_history(trainer.current_epoch, trainer.sanity_checking)

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
            "train_mae",
            "train_var",
            "val_loss",
            "val_mae",
            "val_var",
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

    def plot_visual(
        self, input_field, truth, pred, tendencies, epoch, sanity_checking=False
    ):
        # input_field T, C, H, W
        # truth T, C, H, W
        # pred M, T, C, H, W
        # tend M, T, C, H, W

        input_field = input_field.squeeze(1)
        truth = truth.squeeze(1)  # remove channel dim
        pred = pred.squeeze(2)
        tendencies = tendencies.squeeze(2)

        T = pred.shape[1]

        fig, ax = plt.subplots(T, 6, figsize=(18, 3 * T + 1))
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

            im = ax[t, 1].imshow(tendencies[0][t], cmap=cmap, norm=norm)
            ax[t, 1].set_title(f"Predicted tendencies m=0 step={t}")

            data = true_tendencies - tendencies[0][t]
            ax[t, 2].set_title(f"Tendencies bias m=0 step={t}")
            im = ax[t, 2].imshow(data.cpu(), cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax[t, 2])

            ax[t, 3].set_title(f"True histogram step={t}")
            ax[t, 3].hist(true_tendencies.flatten(), bins=30)

            ax[t, 4].set_title(f"Predicted histogram m=0 step={t}")
            ax[t, 4].hist(tendencies[0][t].flatten(), bins=30)

            uncertainty = torch.var(pred[:, t], dim=0)
            plt.title("Tendency Variance")
            plt.imshow(uncertainty.cpu(), cmap="Reds")
            plt.colorbar()

        plt.tight_layout()

        if sanity_checking:
            return

        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                self.config.run_name,
                self.config.run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_val.png".format(
                self.config.run_dir,
                platform.node(),
                self.config.run_name,
                self.config.run_number,
                epoch,
            )
        )

        plt.close()

    def plot_history(self, epoch, sanity_checking=False):
        warnings.filterwarnings(
            "ignore", message="No artists with labels found to put in legend"
        )

        plt.figure(figsize=(20, 8))
        plt.suptitle(
            "{} at epoch {} (host={}, time={})".format(
                sys.argv[0],
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        train_loss = self.train_loss
        if len(train_loss) > 2000:
            # Remove first 100 after we have enough data, as they often
            # they contain data messes up the y-axis
            train_loss = train_loss[100:]

        plt.subplot(241)
        plt.title("Training loss")
        plt.plot(train_loss, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_loss), 60),
            color="blue",
            label="Train Loss",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(torch.tensor(self.lr), label="LR", color="green")
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax2.legend(loc="upper right")

        val_loss = self.val_loss
        if len(val_loss) > 500:
            val_loss = val_loss[20:]

        plt.subplot(242)
        plt.plot(val_loss, color="orange", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_loss), 60),
            color="orange",
            label="Val Loss",
        )
        plt.title("Validation loss")
        plt.legend(loc="upper left")

        plt.subplot(243)
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

        plt.subplot(244)
        plt.yscale("log")
        plt.title("Gradients (mean)")

        for i, section in enumerate(self.gradients_mean.keys()):
            data = self.gradients_mean[section]
            data = torch.tensor(data)
            color = colors[i]
            plt.plot(data, color=color, alpha=0.3)
            plt.plot(moving_average(data, 30), color=color, label=section)
        plt.legend()

        plt.subplot(245)
        plt.yscale("log")
        plt.title("Gradients (std)")

        for i, section in enumerate(self.gradients_std.keys()):
            data = self.gradients_std[section]
            data = torch.tensor(data)
            color = colors[i]
            plt.plot(data, color=color, alpha=0.3)
            plt.plot(moving_average(data, 30), color=color, label=section)
        plt.legend()

        plt.subplot(246)
        plt.plot(self.train_var, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_var), 30),
            color="blue",
            label="Variance",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(self.train_mae, color="orange", alpha=0.3)
        ax2.plot(
            moving_average(torch.tensor(self.train_mae), 30),
            color="orange",
            label="MAE",
        )
        ax2.legend(loc="upper right")
        plt.title("Train Variance vs MAE")

        plt.subplot(247)
        plt.plot(self.val_var, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_var), 30),
            color="blue",
            label="Variance",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(self.val_mae, color="orange", alpha=0.3)
        ax2.plot(
            moving_average(torch.tensor(self.val_mae), 30), color="orange", label="MAE"
        )
        ax2.legend(loc="upper right")
        plt.title("Val Variance vs MAE")

        plt.tight_layout()
        if sanity_checking:
            return

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
