import torch
import torch.nn as nn
import torch.nn.functional as F
from pgu.mse_plus_amse import mse_plus_amse_loss
from pgu.weighted_mse_plus_amse import weighted_mse_plus_amse_loss


def mse_loss(y_true: torch.tensor, y_pred: torch.tensor):

    loss = ((y_true - y_pred) ** 2).mean()
    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)

    return {"loss": loss, "mse_loss": loss}


LOSS_FUNCTIONS = {
    "mse_loss": mse_loss,
    "mse_plus_amse_loss": mse_plus_amse_loss,
    "weighted_mse_plus_amse_loss": weighted_mse_plus_amse_loss,
}
