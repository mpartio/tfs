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
from cc2CRPS_data import cc2DataModule, cc2ZarrModule
from cc2CRPS_callbacks import TrainDataPlotterCallback, DiagnosticCallback
from cc2util import roll_forecast
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR


# def get_lr_schedule(optimizer, warmup_iterations, total_iterations):
#    def lr_lambda(current_iteration):
#        # Warmup phase
#        if current_iteration < warmup_iterations:
#            return current_iteration / warmup_iterations
#
#        # Linear decay phase
#        else:
#            progress = (current_iteration - warmup_iterations) / (
#                total_iterations - warmup_iterations
#            )
#            # Decay from 1.0 to 0.1
#            return max(0.1, 1.0 - 0.9 * progress)
#
#    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class cc2CRPSModel(L.LightningModule):
    def __init__(self, max_iterations, n_y=1):
        super().__init__()
        self.model = cc2CRPS(
            dim=128, input_resolution=(128, 128), n_members=3, n_layers=4, window_size=8
        )
        self.crps_loss = AlmostFairCRPSLoss(alpha=0.95)
        self.max_iterations = max_iterations

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, tendencies, predictions = roll_forecast(
            self.model, x, y, 1, loss_fn=self.crps_loss
        )
        self.log("train_loss", loss)
        return {"loss": loss, "tendencies": tendencies, "predictions": predictions}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, tendencies, predictions = roll_forecast(
            self.model, x, y, 1, loss_fn=self.crps_loss
        )
        self.log("val_loss", loss)
        return {"loss": loss, "tendencies": tendencies, "predictions": predictions}

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05
        )

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_iterations - 1000,  # Remaining steps after warmup
            eta_min=0,
        )
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        # scheduler = get_lr_schedule(
        #    optimizer, warmup_iterations=1000, total_iterations=self.max_iterations
        # )

        # return [optimizer], [scheduler]


iterations = int(5e5)

model = cc2CRPSModel(iterations)
# cc2Data = cc2DataModule(batch_size=24, n_x=2, n_y=1)
cc2Data = cc2ZarrModule(
    zarr_path="../data/nwcsaf-128x128.zarr", batch_size=18, n_x=2, n_y=1
)

trainer = L.Trainer(
    max_steps=iterations,
    precision="16-mixed",
    accelerator="cuda",
    devices=1,
    callbacks=[
        TrainDataPlotterCallback(),
        DiagnosticCallback(),
        ModelSummary(max_depth=-1),
        ModelCheckpoint(monitor="val_loss", dirpath="models"),
    ],
)

torch.set_float32_matmul_precision("high")

trainer.fit(model, cc2Data.train_dataloader(), cc2Data.val_dataloader())

print("Done at {}".format(datetime.now()))
