import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = torch.abs(y_true - y_pred).mean()
        assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
        return {"loss": loss, "mae_loss": loss}
