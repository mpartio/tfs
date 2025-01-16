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
    def __init__(self, freq=10000):
        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.freq == 0:
            rows = 2
            cols = int(np.ceil((3 + predictions.shape[1]) / 2))
            fig, ax = plt.subplots(rows, cols, figsize=(9, 6))
            ax[0, 0].imshow(input_field[0, 0, ...].detach().cpu().squeeze())
            ax[0, 0].set_title("T-1")
            ax[0, 0].set_axis_off()
            ax[0, 1].imshow(input_field[0, 1, ...].detach().cpu().squeeze())
            ax[0, 1].set_title("T")
            ax[0, 1].set_axis_off()
            ax[0, 2].imshow(truth[0, 0, ...].detach().cpu().squeeze())
            ax[0, 2].set_title("Truth")
            ax[0, 2].set_axis_off()

            for i in range(predictions.shape[1]):
                ax[1, i].imshow(predictions[0, i, ...].detach().cpu().squeeze())
                ax[1, i].set_title(f"Pred {i}")
                ax[1, i].set_axis_off()

            plt.savefig(
                f"figures/{platform.node()}_train-{trainer.global_step:05d}-predictions.png"
            )
            plt.close()


class DiagnosticCallback(L.Callback):
    def __init__(self, freq=1000):
        (
            self.iteration,
            self.train_loss,
            self.val_loss,
            self.lr,
            self.mae,
            self.snr_db,
            self.gradients,
        ) = ([], [], [], [], [], [], [])

        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.freq == 0:
            self.iteration.append(trainer.global_step)

            # a) train loss
            self.train_loss.append(outputs["loss"].detach().cpu().item())

            # b) learning rate
            optimizers = trainer.optimizers
            assert len(trainer.optimizers) > 0, "No optimizer found"
            self.lr.append(optimizers.param_groups[0]["lr"])

            # d) gradients
            self.gradients.append(analyze_gradients(pl_module))

            val_loss = np.nan if len(self.val_loss) == 0 else self.val_loss[-1]

            print(
                "Iteration {:05d} Train Loss: {:.4f}, Val Loss: {:.4f}, L1: {:.4f}".format(
                    trainer.global_step,
                    self.train_loss[-1],
                    val_loss,
                    self.mae[-1],
                )
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.freq == 0:
            # a) Validation loss
            val_loss = outputs.detach().cpu()

            self.val_loss.append(val_loss)

            # b) signal to noise ratio
            x, y = batch
            with torch.no_grad():
                _, tendencies, predictions = roll_forecast(
                    pl_module, x, y, 1, loss_fn=None
                )
                truth = y.cpu().squeeze()
                pred = predictions.cpu().squeeze()
                # pred = torch.mean(pred, dim=1)
                pred = pred[:, 0, ...]  # Select first member
                self.mae.append(F.l1_loss(pred, truth).item())

                snr_pred = calculate_wavelet_snr(pred[0], None)
                snr_real = calculate_wavelet_snr(truth[0], None)
                self.snr_db.append((snr_real["snr_db"], snr_pred["snr_db"]))

        if trainer.global_step > 0 and trainer.global_step % (self.freq * 20) == 0:
            self.plot(
                x[0].cpu().squeeze(),
                y[0].cpu().squeeze(),
                predictions.cpu()[0].squeeze(),
                tendencies.cpu()[0, 0, ...].squeeze(),
                trainer.global_step,
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
        plt.hist(pred.flatten(), bins=20)
        plt.title("Gradients")
        colors = ["blue", "orange", "green", "red", "black", "purple"]
        for item in self.gradients:
            section, grads = item.items()
            data = grads["mean"]
            color = colors.pop(0)
            plt.plot(data, label=section, color=color, alpha=0.3)
            plt.plot(moving_average(data, 50), color=color)

        plt.tight_layout()
        plt.savefig(
            "figures/{}_iteration_{:05d}.png".format(platform.node(), iteration)
        )

        plt.close()
