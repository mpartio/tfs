import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
import warnings
import os
import shutil
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


def var_and_mae(predictions, y):
    variance = torch.var(predictions[:, :, -1, ...], dim=2, unbiased=False)
    variance = variance.detach().mean().cpu().numpy().item()

    mean_pred = torch.mean(predictions[:, :, -1, ...], dim=2)
    y_true = y[:, -1, ...]

    mae = torch.mean(torch.abs(y_true - mean_pred)).detach().cpu().numpy().item()

    return variance, mae


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

        run_name, run_number, run_dir = run_info()

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
    def __init__(self, config, check_frequency=50):
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

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) train loss
        train_loss = outputs["loss"].item()

        pl_module.log(
            "train/loss_step", train_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        pl_module.log("train/loss_epoch", train_loss, on_step=False, on_epoch=True)

        _loss_names = []
        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue

            pl_module.log(f"train/{k}", v.cpu(), on_step=True, on_epoch=True)

            if len(self.loss_names) == 0:
                _loss_names.append(k)

        if len(self.loss_names) == 0:
            self.loss_names = _loss_names

        # b) learning rate
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("lr", lr, on_step=True, on_epoch=False)

        # c) gradients
        if batch_idx % self.check_frequency == 0:
            grads = analyze_gradients(pl_module)

            for k in grads.keys():
                mean, std = grads[k]["mean"], grads[k]["std"]
                pl_module.log(f"train/grad_{k}_mean", mean, on_step=True, on_epoch=True)
                pl_module.log(f"train/grad_{k}_std", std, on_step=True, on_epoch=True)

            # d) variance and l1

            predictions = outputs["predictions"]  # B, M, T, C, H, W

            data, _ = batch
            _, y = data

            var, mae = var_and_mae(predictions.to(torch.float32), y.to(torch.float32))
            pl_module.log("train/variance", var)
            pl_module.log("train/mae", mae)

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) Validation loss
        val_loss = outputs["loss"].item()

        pl_module.log(
            "val/loss_epoch", val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        for k, v in outputs["loss_components"].items():
            if k == "loss":
                continue

            pl_module.log(f"val/{k}", v.cpu(), on_step=True, on_epoch=True)

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
                "val/snr_real", snr_real["snr_db"], on_step=False, on_epoch=True
            )
            pl_module.log(
                "val/snr_pred", snr_pred["snr_db"], on_step=False, on_epoch=True
            )

            # d) variance and l1
            var, mae = var_and_mae(predictions.to(torch.float32), y.to(torch.float32))
            pl_module.log("val/variance", var)
            pl_module.log("val/mae", mae)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        current_val_loss = trainer.logged_metrics.get("val/loss_epoch", float("nan"))
        current_lr = trainer.logged_metrics.get("lr", float("nan"))
        current_snr_real = trainer.logged_metrics.get("val/snr_real", float("nan"))
        current_snr_pred = trainer.logged_metrics.get("val/snr_pred", float("nan"))

        # Append to internal history lists
        self.val_loss.append(current_val_loss)
        self.lr.append(current_lr)
        self.val_snr_real.append(current_snr_real)
        self.val_snr_pred.append(current_snr_pred)

        for k in self.loss_names:
            val = trainer.logged_metrics.get(f"val/{k}_epoch", float("nan"))

            try:
                self.val_loss_components[k].append(val)
            except KeyError:
                self.val_loss_components[k] = [val]

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

        self.plot_history(trainer.current_epoch, trainer.sanity_checking)

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

    def plot_visual(
        self, input_field, truth, pred, tendencies, epoch, sanity_checking=False
    ):
        # input_field T, C, H, W
        # truth T, C, H, W
        # pred M, T, C, H, W
        # tend M, T, C, H, W

        run_name, run_number, run_dir = run_info()

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
                run_name,
                run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_val.png".format(
                run_dir,
                platform.node(),
                run_name,
                run_number,
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
        run_name, run_number, run_dir = run_info()

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
