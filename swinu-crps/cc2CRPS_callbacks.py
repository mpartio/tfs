import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
from cc2util import roll_forecast, moving_average, analyze_gradients
from util import calculate_wavelet_snr
from datetime import datetime


class TrainDataPlotterCallback(L.Callback):
    def __init__(self, freq=500):
        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        if trainer.global_step > 0 and trainer.global_step % self.freq == 0:
            x, y = batch
            # Take first batch member
            input_field = x[0].detach().cpu().squeeze()
            truth = (
                y[0].detach().cpu().squeeze()
            )  # By squeezeing we remove the time dim
            predictions = outputs["predictions"]
            predictions = predictions[0].detach().cpu().squeeze()

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

            plt.savefig(
                f"figures/{platform.node()}_train-{trainer.global_step:05d}-predictions.png"
            )
            plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, freq=100):
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
        self.x, self.y, self.predictions, self.tendencies = None, None, None, None
        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if not trainer.is_global_zero:
            return

        if trainer.global_step % self.freq == 0:

            # a) train loss
            self.train_loss.append(outputs["loss"].detach().cpu().item())
            pl_module.log("train_loss", self.train_loss[-1], prog_bar=True)

            # b) learning rate
            assert len(trainer.optimizers) > 0, "No optimizer found"
            self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])

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

        if trainer.global_step % self.freq == 0:
            # a) Validation loss
            self.val_loss.append(outputs["loss"].detach().cpu().item())

            pl_module.log("val_loss", self.val_loss[-1], prog_bar=True)

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

        if self.x is None:
            return

        self.plot(
            self.x,
            self.y,
            self.predictions,
            self.tendencies,
            trainer.current_epoch,
        )

    def plot(self, input_field, truth, pred, tendencies, iteration):
        plt.figure(figsize=(24, 12))
        plt.suptitle(
            "{} at iteration {} (host={}, time={})".format(
                sys.argv[0],
                iteration,
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

        pred = pred[0]  # Select first member
        plt.subplot(344)
        plt.imshow(pred)
        plt.title("Prediction")
        plt.colorbar()

        plt.subplot(345)
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(
            vmin=tendencies.min(), vcenter=0, vmax=tendencies.max()
        )
        plt.imshow(tendencies, cmap=cmap, norm=norm)
        plt.title("Tendencies")
        plt.colorbar()

        data = truth - input_field[-1]
        cmap = plt.cm.coolwarm  # You can also try 'bwr' or other diverging colormaps
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

        plt.subplot(346)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("True Residual")
        plt.colorbar()

        data = pred - input_field[-1]
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        plt.subplot(347)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Residual of Prediction/L1={:.4f}".format(F.l1_loss(pred, truth)))
        plt.colorbar()

        plt.subplot(348)
        plt.title("Losses")
        plt.plot(self.train_loss, label="Train Loss", color="blue", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.train_loss), 50),
            label="Train Loss MA",
            color="blue",
        )
        plt.plot(self.val_loss, label="Val Loss", color="orange", alpha=0.3)
        plt.plot(
            moving_average(torch.tensor(self.val_loss), 50),
            label="Val Loss MA",
            color="orange",
        )

        plt.legend(loc="upper left")
        ax2 = plt.gca().twinx()
        ax2.plot(torch.tensor(self.lr) * 1e6, label="LRx1M", color="green")
        ax2.legend(loc="upper right")

        snr_db = np.array(self.snr_db).T

        plt.subplot(349)
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

        plt.subplot(3, 4, 10)
        plt.hist(truth.flatten(), bins=20)
        plt.title("Truth histogram")

        plt.subplot(3, 4, 11)
        plt.hist(pred.flatten(), bins=20)
        plt.title("Predicted histogram")

        plt.subplot(3, 4, 12)
        plt.yscale("log")
        plt.title("Gradients")
        colors = ["blue", "orange", "green", "red", "black", "purple"]
        for section in self.gradients.keys():
            data = self.gradients[section]
            data = torch.tensor(data)
            color = colors.pop(0)
            plt.plot(data, label=section, color=color, alpha=0.3)
            plt.plot(moving_average(data, 50), color=color)

        plt.legend()
        plt.tight_layout()
        plt.savefig(
            "figures/{}_epoch_{:03d}.png".format(platform.node(), iteration)
        )

        plt.close()
