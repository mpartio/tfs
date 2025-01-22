import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
from cc2util import roll_forecast, moving_average, analyze_gradients
from util import calculate_wavelet_snr
from datetime import datetime

matplotlib.use("Agg")


class TrainDataPlotterCallback(L.Callback):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        pl_module.eval()

        # Get a single batch from the dataloader
        x, y = next(iter(self.dataloader))

        x = x.to(pl_module.device)

        # Perform a prediction
        with torch.no_grad():
            _, _, predictions = roll_forecast(pl_module, x, y, 1, None)

        self.plot(
            x.cpu().detach(), y, predictions.cpu().detach(), trainer.current_epoch
        )

        # Restore model to training mode
        pl_module.train()

    def plot(self, x, y, predictions, epoch):
        # Take first batch member
        input_field = x[0].squeeze()
        truth = y[0].squeeze()  # By squeezing we remove the time dim
        predictions = predictions[0].squeeze()

        rows = 2
        cols = int(np.ceil((3 + predictions.shape[0]) / 2))
        fig, ax = plt.subplots(rows, cols, figsize=(9, 6))
        ax[0, 0].imshow(input_field[0])
        ax[0, 0].set_title("T-1")
        ax[0, 0].set_axis_off()
        ax[0, 1].imshow(input_field[1])
        ax[0, 1].set_title("T")
        ax[0, 1].set_axis_off()
        ax[0, 2].imshow(truth)
        ax[0, 2].set_title("Truth")
        ax[0, 2].set_axis_off()

        for i in range(predictions.shape[0]):
            ax[1, i].imshow(predictions[i, ...])
            ax[1, i].set_title(f"Pred {i}")
            ax[1, i].set_axis_off()

        plt.savefig("figures/{}_epoch_{:03d}_train.png".format(platform.node(), epoch))
        plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, freq=50):
        (
            self.train_loss,
            self.val_loss,
            self.lr,
            self.snr_db,
        ) = ([], [], [], [])

        self.gradients = {
            "encoder": [],
            "decoder": [],
            "attention": [],
            "norms": [],
            "prediction": [],
        }

        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) train loss
        self.train_loss.append(outputs["loss"].detach().cpu().item())
        pl_module.log("train_loss", self.train_loss[-1], prog_bar=True)

        # b) learning rate
        assert len(trainer.optimizers) > 0, "No optimizer found"
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

        if trainer.global_step % self.freq == 0:
            # c) gradients
            grads = analyze_gradients(pl_module)

            for k in self.gradients.keys():
                try:
                    self.gradients[k].append(grads[k]["mean"].item())
                except KeyError as e:
                    pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        # a) Validation loss
        self.val_loss.append(outputs["loss"].detach().cpu().item())
        pl_module.log("val_loss", self.val_loss[-1], prog_bar=True)

        if trainer.global_step % self.freq == 0:
            self.val_loss.append(outputs["loss"].detach().cpu().item())

            # b) signal to noise ratio
            tendencies = outputs["tendencies"]
            predictions = outputs["predictions"]

            x, y = batch

            truth = y[0].cpu().squeeze()
            # Select first of batch
            pred = predictions[0].cpu().squeeze()
            # pred = torch.mean(pred, dim=1)

            snr_pred = calculate_wavelet_snr(pred[0], None)
            snr_real = calculate_wavelet_snr(truth, None)
            self.snr_db.append((snr_real["snr_db"], snr_pred["snr_db"]))

            self.x = x[0].cpu().squeeze()
            self.y = y[0].cpu().squeeze()
            self.predictions = pred
            self.tendencies = tendencies[0][0].cpu().squeeze()

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
                pl_module.model, x, y, 1, loss_fn=None
            )

        self.plot_visual(
            x.cpu().squeeze().detach(),
            y.cpu().squeeze().detach(),
            predictions[0].cpu().squeeze().detach(),
            tendencies[0].cpu().squeeze().detach(),
            trainer.current_epoch,
        )

        self.plot_history(trainer.current_epoch)

    def plot_visual(self, input_field, truth, pred, tendencies, epoch):
        plt.figure(figsize=(24, 12))
        plt.suptitle(
            "{} at epoch {} (host={}, time={})".format(
                sys.argv[0],
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        plt.subplot(341)
        plt.imshow(input_field[0])
        plt.title("Input time=T-1")
        plt.colorbar()

        plt.subplot(342)
        plt.imshow(input_field[1])
        plt.title("Input time=T")
        plt.colorbar()

        plt.subplot(343)
        plt.imshow(truth)
        plt.title("Truth")
        plt.colorbar()

        plt.subplot(344)
        plt.imshow(pred[0])
        plt.title("Prediction")
        plt.colorbar()

        plt.subplot(345)
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(
            vmin=tendencies.min(), vcenter=0, vmax=tendencies.max()
        )
        plt.imshow(tendencies[0], cmap=cmap, norm=norm)
        plt.title("Tendencies")
        plt.colorbar()

        data = truth - input_field[-1]
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

        plt.subplot(346)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("True Residual")
        plt.colorbar()

        data = pred[0] - input_field[-1]
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

        plt.subplot(347)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Residual of Prediction/L1={:.4f}".format(F.l1_loss(pred[0], truth)))
        plt.colorbar()

        plt.subplot(348)
        plt.hist(truth.flatten(), bins=20)
        plt.title("Truth histogram")

        plt.subplot(349)
        plt.hist(pred[0].flatten(), bins=20)
        plt.title("Predicted histogram")

        uncertainty = torch.var(pred, dim=0)
        plt.subplot(3, 4, 10)  # Adjust subplot number as needed
        plt.title("Prediction Uncertainty")
        plt.imshow(uncertainty.cpu(), cmap="Reds")
        plt.colorbar()

        # Calculate mean prediction and error map
        mean_pred = torch.mean(pred, dim=0)
        error_map = torch.abs(mean_pred - truth)

        plt.subplot(3, 4, 11)
        plt.title("Spatial Error Distribution")
        plt.imshow(error_map.cpu())
        plt.colorbar()

        # Calculate statistics
        mean_error = error_map.mean().item()
        max_error = error_map.max().item()
        std_error = error_map.std().item()

        # Add stats text box
        stats_text = (
            f"Mean Error: {mean_error:.3f}\n"
            f"Max Error: {max_error:.3f}\n"
            f"Std Error: {std_error:.3f}"
        )

        # Place text box in upper left corner with white background
        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Optional: Add percentile information
        p90 = torch.quantile(error_map, 0.90).item()
        p95 = torch.quantile(error_map, 0.95).item()
        plt.axhline(
            p90,
            color="r",
            linestyle="--",
            alpha=0.3,
            label=f"90th percentile: {p90:.3f}",
        )
        plt.axhline(
            p95,
            color="r",
            linestyle=":",
            alpha=0.3,
            label=f"95th percentile: {p95:.3f}",
        )

        plt.tight_layout()
        plt.savefig("figures/{}_epoch_{:03d}_val.png".format(platform.node(), epoch))

        plt.close()

    def plot_history(self, epoch):
        plt.figure(figsize=(12, 8))
        plt.suptitle(
            "{} at epoch {} (host={}, time={})".format(
                sys.argv[0],
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        plt.subplot(221)
        plt.title("Training loss")
        plt.plot(self.train_loss, label="Train Loss", color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_loss), 50),
            label="Moving average",
            color="blue",
        )

        ax2 = plt.gca().twinx()
        ax2.plot(torch.tensor(self.lr) * 1e6, label="LRx1M", color="green")
        ax2.legend(loc="upper right")

        plt.subplot(222)
        plt.plot(self.val_loss, label="Val Loss", color="orange", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_loss), 50),
            label="Moving average",
            color="orange",
        )
        plt.title("Validation loss")
        plt.legend(loc="upper left")

        plt.subplot(223)
        snr_db = np.array(self.snr_db).T
        snr_real = torch.tensor(snr_db[0])
        snr_pred = torch.tensor(snr_db[1])
        plt.plot(snr_real, label="Real", color="blue", alpha=0.3)
        plt.plot(
            moving_average(snr_real, 10),
            label="Moving Average",
            color="blue",
        )
        plt.plot(snr_pred, label="Pred", color="orange", alpha=0.3)
        plt.plot(
            moving_average(snr_pred, 10),
            color="orange",
            label="Moving average",
        )
        plt.legend(loc="center right")

        ax2 = plt.gca().twinx()
        ax2.plot(
            moving_average(snr_real - snr_pred, 20), label="Residual", color="green"
        )
        ax2.legend(loc="upper right")
        plt.title("Signal to Noise Ratio")

        plt.subplot(224)
        plt.yscale("log")
        plt.title("Gradients")
        colors = ["blue", "orange", "green", "red", "black", "purple"]
        for section in self.gradients.keys():
            data = self.gradients[section]
            data = torch.tensor(data)
            color = colors.pop(0)
            plt.plot(data, label=section, color=color, alpha=0.3)
            plt.plot(moving_average(data, 20), color=color)  # , label=f"{section} MA")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            "figures/{}_epoch_{:03d}_history.png".format(platform.node(), epoch)
        )

        plt.close()
