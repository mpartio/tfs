import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import lightning as L
from datetime import datetime, timedelta
from cc2CRPS import cc2CRPS
from crps import AlmostFairCRPSLoss
from util import calculate_wavelet_snr
from cc2CRPS_data import cc2DataModule
from cc2CRPS_callbacks import TrainDataPlotterCallback, DiagnosticCallback
from cc2util import roll_forecast


def convert_delta(dlt: timedelta) -> str:
    minutes, seconds = divmod(int(dlt.total_seconds()), 60)
    return f"{minutes}:{seconds:02}"


def get_lr_schedule(optimizer, warmup_iterations, total_iterations):
    def lr_lambda(current_iteration):
        # Warmup phase
        if current_iteration < warmup_iterations:
            return current_iteration / warmup_iterations

        # Linear decay phase
        else:
            progress = (current_iteration - warmup_iterations) / (
                total_iterations - warmup_iterations
            )
            # Decay from 1.0 to 0.1
            return max(0.1, 1.0 - 0.9 * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




def analyze_gradients(model):
    # Group gradients by network section
    gradient_stats = {
        "encoder": [],  # Encoder blocks
        "attention": [],  # Attention blocks
        "norms": [],  # Layer norms
        "decoder": [],  # Decoder blocks
        "prediction": [],  # Final head
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()

            if "encoder" in name:
                gradient_stats["encoder"].append(grad_norm)
            elif "decoder" in name:
                gradient_stats["decoder"].append(grad_norm)
            elif "attn" in name:
                gradient_stats["attention"].append(grad_norm)
            elif "norm" in name:
                gradient_stats["norms"].append(grad_norm)
            elif "prediction_head" in name:
                gradient_stats["prediction"].append(grad_norm)

    # Compute statistics for each section
    stats = {}
    for section, grads in gradient_stats.items():
        if grads:
            stats[section] = {
                "mean": np.mean(grads),
                "std": np.std(grads),
                "min": np.min(grads),
                "max": np.max(grads),
            }

    return stats


class cc2CRPSModel(L.LightningModule):
    def __init__(self, n_y=1):
        super().__init__()
        self.model = cc2CRPS(
            dim=128, input_resolution=(128, 128), n_members=3, n_layers=4
        )
        self.crps_loss = AlmostFairCRPSLoss(alpha=0.95)
        self.max_iterations = int(5e5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, _, _ = roll_forecast(
            self.model, x, y, 1, loss_fn=self.crps_loss
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, _, _ = roll_forecast(
            self.model, x, y, 1, loss_fn=self.crps_loss
        )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05
        )

        scheduler = get_lr_schedule(
            optimizer, warmup_iterations=1000, total_iterations=self.max_iterations
        )

        return [optimizer], [scheduler]


iterations = int(5e5)

model = cc2CRPSModel()
cc2Data = cc2DataModule(batch_size=8, n_x=2, n_y=1)

trainer = L.Trainer(
    max_steps=iterations,
    precision="16-mixed",
    accelerator="cuda",
    devices=1,
    callbacks=[TrainDataPlotterCallback(), DiagnosticCallback()],
)

torch.set_float32_matmul_precision('high')

trainer.fit(model, cc2Data.train_dataloader(), cc2Data.val_dataloader())

print("Done at {}".format(datetime.now()))
