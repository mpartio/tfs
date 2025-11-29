import torch
import torch.nn as nn
import torch.nn.functional as F
from pgu.mse_plus_amse import mse_plus_amse_loss
from pgu.weighted_mse_plus_amse import weighted_mse_plus_amse_loss


def mse_loss(y_true: torch.tensor, y_pred: torch.tensor):
    loss = ((y_true - y_pred) ** 2).mean()
    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
    return {"loss": loss, "mse_loss": loss}


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return mse_loss(y_true, y_pred)


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = torch.abs(y_true - y_pred).mean()
        assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
        return {"loss": loss, "mae_loss": loss}


LOSS_FUNCTIONS = {
    "mse_loss": mse_loss,
    "mse_plus_amse_loss": mse_plus_amse_loss,
    "weighted_mse_plus_amse_loss": weighted_mse_plus_amse_loss,
}
