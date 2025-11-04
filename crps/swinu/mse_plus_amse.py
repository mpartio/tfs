import torch
import torch.nn as nn
import torch.nn.functional as F
from swinu.amse import AMSELoss

def mse_plus_amse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_bins: int = None,
    lambda_spectral: float = 0.05,
):
    """
    Combined pixel-wise MSE + spectral AMSE.
    """
    if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    # pixelwise MSE
    mse_loss = F.mse_loss(y_pred.float(), y_true.float())

    # spectral AMSE
    amse_loss = AMSELoss()(y_pred, y_true)["total_loss"]

    total_loss = mse_loss + lambda_spectral * amse_loss
    assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss}"

    return {
        "loss": total_loss,
        "pixel_mse": mse_loss.detach(),
        "spectral_amse": amse_loss,
    }


class MSEAMSELoss(nn.Module):
    def __init__(
        self,
        n_bins: int = None,
        lambda_spectral: float = 0.05,
    ):
        super().__init__()

        self.n_bins = n_bins
        self.lambda_spectral = lambda_spectral

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return mse_plus_amse_loss(
            y_true,
            y_pred,
            n_bins=self.n_bins,
            lambda_spectral=self.lambda_spectral,
        )
