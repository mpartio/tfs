import torch
import torch.nn as nn
import torch.nn.functional as F
from pgu.mse_plus_amse import mse_plus_amse_loss
from pgu.weighted_mse_plus_amse import weighted_mse_plus_amse_loss


class WeightedMSELoss(nn.Module):
    def __init__(self, d0: float = 0.1, alpha: float = 1.0, w_max: float = 3.0):
        super().__init__()
        self.d0 = d0
        self.alpha = alpha
        self.w_max = w_max

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Weighted MSE loss.

        Weights are larger where |y_true| is large (i.e. where the true delta
        is large), to emphasise dynamically active pixels.

        Args:
            y_true: [B, C, H, W] tensor of targets (deltas or absolute values).
            y_pred: [B, C, H, W] tensor of predictions.
            d0: scale for |y_true| at which weighting starts to ramp up.
            alpha: strength of additional weighting.
            w_max: maximum weight (to avoid a few pixels dominating).

        Returns:
            dict with total loss and the weighted MSE scalar.
        """
        residual = y_pred - y_true  # [B,C,H,W]
        abs_delta = y_true.abs()

        # Raw weights: 1 + alpha * (|Δ| / d0)
        weights = 1.0 + self.alpha * (abs_delta / self.d0)
        weights = torch.clamp(weights, 1.0, self.w_max)

        # Normalise to keep overall scale comparable to plain MSE
        weights = weights / weights.mean().detach()

        loss = (weights * residual**2).mean()

        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss, "mse_loss": loss}


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
