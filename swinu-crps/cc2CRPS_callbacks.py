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
from cc2util import roll_forecast, moving_average, analyze_gradients
from util import calculate_wavelet_snr
from datetime import datetime
from dataclasses import asdict
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")


def var_and_mae(predictions, y):
    variance = torch.var(predictions[:, -1, ...], dim=1, unbiased=False)
    variance = variance.detach().mean().cpu().numpy().item()

    mean_pred = torch.mean(predictions[:, -1, ...], dim=1)
    y_true = y[:, -1, ...]

    mae = torch.mean(torch.abs(y_true - mean_pred)).detach().cpu().numpy().item()

    return variance, mae


class TrainDataPlotterCallback(L.Callback):
    def __init__(self, dataloader, config):
        self.dataloader = dataloader
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        pl_module.eval()

        # Get a single batch from the dataloader
        x, y = next(iter(self.dataloader))

        x = x.to(pl_module.device)

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(
                pl_module,
                x,
                y,
                self.config.rollout_length,
                None,
                num_members=self.config.num_members,
            )

        self.plot(
            x.cpu().detach(), y, predictions.cpu().detach(), trainer.current_epoch
        )

        # Restore model to training mode
        pl_module.train()

    def plot(self, x, y, predictions, epoch):
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

        predictions = predictions[0]  # T, M, C, H, W (B removed)

        num_truth = truth.shape[0]
        num_members = predictions.shape[1]

        rows = num_truth
        cols = num_members + 2 + 1  # +2 for input fields, +1 for truth

        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
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
            ax[i, 0].set_title("Time T-1")
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(input_field[1])
            ax[i, 1].set_title("Time T")
            ax[i, 1].set_axis_off()

            for j in range(num_members):  # members
                ax[i, j + 2].imshow(predictions[i, j, ...].squeeze())
                ax[i, j + 2].set_title(f"Time T+{i+1} member {j}")
                ax[i, j + 2].set_axis_off()

            ax[i, j + 3].imshow(truth[i].squeeze())
            ax[i, j + 3].set_title(f"Time T+{i+1} truth")
            ax[i, j + 3].set_axis_off()

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_train.png".format(
                self.config.run_dir,
                platform.node(),
                self.config.run_name,
                self.config.run_number,
                epoch,
            )
        )
        plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, config, freq=25):
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

        self.gradients = {
            "encoder": [],
            "decoder": [],
            "attention": [],
            "prediction": [],
        }

        self.freq = freq
        self.config = config

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

            for k in self.gradients.keys():
                try:
                    self.gradients[k].append(grads[k]["mean"].item())
                except KeyError as e:
                    pass

            # d) variance and l1

            predictions = outputs["predictions"]

            _, y = batch

            var, mae = var_and_mae(predictions, y)
            self.train_var.append(var)
            self.train_mae.append(mae)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) Validation loss
        self.val_loss.append(outputs["loss"].detach().cpu().item())
        pl_module.log("val_loss", self.val_loss[-1], prog_bar=True)

        if batch_idx % self.freq == 0:
            # b) signal to noise ratio
            predictions = outputs["predictions"]

            _, y = batch

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

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if not trainer.is_global_zero:
            return

        # Get a single batch from the dataloader
        x, y = next(iter(trainer.val_dataloaders))
        x = x[0].unsqueeze(0).to(pl_module.device)
        y = y[0].unsqueeze(0).to(pl_module.device)

        # Perform a prediction
        with torch.no_grad():
            _, tendencies, predictions = roll_forecast(
                pl_module,
                x,
                y,
                self.config.rollout_length,
                loss_fn=None,
                num_members=self.config.num_members,
            )

        self.plot_visual(
            x[0].cpu().detach(),
            y[0].cpu().detach(),
            predictions[0].cpu().detach(),
            tendencies[0].cpu().detach(),
            trainer.current_epoch,
        )

        self.plot_history(trainer.current_epoch)

    def plot_visual(self, input_field, truth, pred, tendencies, epoch):
        plt.figure(figsize=(24, 12))
        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                self.config.run_name,
                self.config.run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        truth = truth.squeeze(1)  # remove channel dim
        pred = pred.squeeze(2)
        tendencies = tendencies.squeeze(2)

        # input_field T, H, W
        # truth T, H, W
        # pred T, M, H, W
        # tend T, M, H, W

        plt.subplot(341)
        plt.imshow(input_field[0])
        plt.title("Input time=T-1")
        plt.colorbar()

        plt.subplot(342)
        plt.imshow(input_field[1])
        plt.title("Input time=T")
        plt.colorbar()

        plt.subplot(343)
        plt.imshow(truth[-1])
        plt.title("Truth")
        plt.colorbar()

        plt.subplot(344)
        plt.imshow(pred[-1][0])
        plt.title("Prediction")
        plt.colorbar()

        plt.subplot(345)
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        plt.imshow(tendencies[-1][0], cmap=cmap, norm=norm)
        plt.title("Tendencies")
        plt.colorbar()

        data = truth[-1] - input_field[-1]
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        plt.subplot(346)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("True Tendencies")
        plt.colorbar()

        data = pred[-1][0] - truth[-1]
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        plt.subplot(347)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Prediction Bias")
        plt.colorbar()

        plt.subplot(348)
        plt.hist(truth[-1].flatten(), bins=20)
        plt.title("Truth histogram")

        plt.subplot(349)
        plt.hist(pred[-1][0].flatten(), bins=20)
        plt.title("Predicted histogram")

        uncertainty = torch.var(pred[-1], dim=0)
        plt.subplot(3, 4, 10)  # Adjust subplot number as needed
        plt.title("Member Variance")
        plt.imshow(uncertainty.cpu(), cmap="Reds")
        plt.colorbar()

        # Calculate mean prediction and error map
        mean_pred = torch.mean(pred[-1], dim=0)
        error_map = torch.abs(mean_pred - truth[-1])

        plt.subplot(3, 4, 11)
        plt.title("Spatial MAE")
        plt.imshow(error_map.cpu())
        plt.colorbar()

        plt.tight_layout()
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

    def plot_history(self, epoch):
        plt.figure(figsize=(16, 8))
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

        plt.subplot(231)
        plt.title("Training loss")
        plt.plot(train_loss, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_loss), 50),
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

        plt.subplot(232)
        plt.plot(val_loss, color="orange", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_loss), 50),
            color="orange",
            label="Val Loss",
        )
        plt.title("Validation loss")
        plt.legend(loc="upper left")

        plt.subplot(233)
        val_snr = np.array(self.val_snr).T
        snr_real = torch.tensor(val_snr[0])
        snr_pred = torch.tensor(val_snr[1])
        plt.plot(snr_real, color="blue", alpha=0.3)
        plt.plot(
            moving_average(snr_real, 10),
            color="blue",
            label="Real",
        )
        plt.plot(snr_pred, color="orange", alpha=0.3)
        plt.plot(moving_average(snr_pred, 10), color="orange", label="Predicted")
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        residual = snr_real - snr_pred

        ax2.plot(residual, color="green", alpha=0.3)
        ax2.plot(moving_average(residual, 10), label="Residual", color="green")
        ax2.legend(loc="upper right")
        plt.title("Signal to Noise Ratio")

        plt.subplot(234)
        plt.yscale("log")
        plt.title("Gradients")
        colors = ["blue", "orange", "green", "red", "black", "purple"]
        for section in self.gradients.keys():
            data = self.gradients[section]
            data = torch.tensor(data)
            color = colors.pop(0)
            plt.plot(data, color=color, alpha=0.3)
            plt.plot(moving_average(data, 10), color=color, label=section)
        plt.legend()

        plt.subplot(235)
        plt.plot(self.train_var, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_var), 10),
            color="blue",
            label="Variance",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(self.train_mae, color="orange", alpha=0.3)
        ax2.plot(
            moving_average(torch.tensor(self.train_mae), 10),
            color="orange",
            label="MAE",
        )
        ax2.legend(loc="upper right")
        plt.title("Train Variance vs MAE")

        plt.subplot(236)
        plt.plot(self.val_var, color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_var), 10),
            color="blue",
            label="Variance",
        )
        plt.legend(loc="upper left")

        ax2 = plt.gca().twinx()
        ax2.plot(self.val_mae, color="orange", alpha=0.3)
        ax2.plot(
            moving_average(torch.tensor(self.val_mae), 10), color="orange", label="MAE"
        )
        ax2.legend(loc="upper right")
        plt.title("Val Variance vs MAE")

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

    def on_fit_end(self, trainer, pl_module):
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

        D = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        filename = f"{self.config.run_dir}/statistics.json"

        with open(filename, "w") as f:
            json.dump(D, f, indent=4)
