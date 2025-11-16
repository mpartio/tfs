import torch
import torch.nn as nn
import torch.nn.functional as F
from pgu.spectrally_adjusted_huber import huber_with_tendency_and_spectral_loss
from pgu.spectrally_adjusted_mse import amse2d_loss, amse2d_with_tendency_loss
from pgu.mse_plus_amse import mse_plus_amse_loss
from pgu.weighted_mse_plus_amse import weighted_mse_plus_amse_loss


def mse_loss(y_true: torch.tensor, y_pred: torch.tensor):

    loss = ((y_true - y_pred) ** 2).mean()
    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)

    return {"loss": loss, "mse_loss": loss}


def huber_with_tendency_loss(
    y_true: torch.tensor,
    y_pred: torch.tensor,
    alpha: float = 2.0,  # scaler for tendency loss
    tau: float = 0.02,  # lower bound for "noise" change,
    e_std: float = 0.32,  # empirical tendency std, change point for huber
):
    huber = nn.SmoothL1Loss(beta=e_std, reduction="none")

    # rollout_len starts with 0
    B, T, C, H, W = y_true.shape

    # base loss
    base = huber(y_pred, y_true)
    step_loss = base.mean(dim=(0, 2, 3, 4))  # [T]

    # change-aware tendency loss
    # ignore tiny changes below tau, and scale up other by alpha
    magnitude = (torch.abs(y_true) - tau).clamp_min(0)

    # scale the magnitude, as volatile batches might result into higher loss
    # than calm batches
    magnitude_mean = magnitude.mean(dim=(0, 2, 3, 4), keepdim=True)  # [1,T,1,1,1]
    magnitude = magnitude / (magnitude_mean + 1e-8)  # [B,T,C,H,W]
    magnitude = magnitude * alpha

    tendency_loss = (magnitude * base).mean(dim=(0, 2, 3, 4))  # [T]

    loss = step_loss.mean() + tendency_loss.mean()

    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)

    return {"loss": loss, "step_loss": step_loss, "tendency_loss": tendency_loss}


def _down2(x: torch.Tensor) -> torch.Tensor:
    # average-pool by 2 (anti-aliasing-ish) on H,W; keeps gradients simple
    B, T, C, H, W = x.shape
    x2 = F.avg_pool2d(x.view(B * T, C, H, W), kernel_size=2, stride=2)
    H2, W2 = x2.shape[-2:]
    return x2.view(B, T, C, H2, W2)


def _multiscale_huber(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_scales: int = 3,
    weights: torch.Tensor | None = None,  # shape [n_scales]
    e_std: float = 0.32,  # empirical tendency std, change point for huber
) -> torch.Tensor:
    huber = nn.SmoothL1Loss(beta=e_std, reduction="none")

    # Default: heavier weight on coarse scales, sums to 1
    if weights is None:
        w = torch.tensor([0.5, 0.35, 0.15], device=y_true.device)[:n_scales]
        weights = w / w.sum()

    yt, yp = y_true, y_pred
    per_scale = []
    for s in range(n_scales):
        base = huber(yp, yt).mean(dim=(0, 2, 3, 4))  # [T]
        per_scale.append(weights[s] * base)  # weight this scale
        if s < n_scales - 1:
            yt = _down2(yt)
            yp = _down2(yp)
    # sum scales -> [T], then mean over time -> scalar
    return torch.stack(per_scale, dim=0).sum(dim=0).mean()


def multiscale_huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 2.0,  # scaler for tendency term
    tau: float = 0.02,  # lower bound for "noise" change
    e_std: float = 0.32,  # empirical tendency std, change point for huber
    n_scales: int = 3,
    scale_weights: torch.Tensor | None = None,
):
    # shapes: [B,T,C,H,W]
    # --- multi-scale base loss (replaces single-scale mean) ---
    ms_loss = _multiscale_huber(
        y_true, y_pred, n_scales=n_scales, weights=scale_weights
    )

    # --- your change-aware tendency weighting (kept as-is) ---
    huber = nn.SmoothL1Loss(beta=e_std, reduction="none")
    base = huber(y_pred, y_true)  # [B,T,C,H,W]
    magnitude = (torch.abs(y_true) - tau).clamp_min(0)
    mag_mean = magnitude.mean(dim=(0, 2, 3, 4), keepdim=True)  # [1,T,1,1,1]
    magnitude = magnitude / (mag_mean + 1e-8)
    tendency_loss = (alpha * magnitude * base).mean(dim=(0, 2, 3, 4)).mean()  # scalar

    loss = ms_loss + tendency_loss
    assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
    return {"loss": loss, "ms_loss": ms_loss, "tendency_loss": tendency_loss}


LOSS_FUNCTIONS = {
    "mse_loss": mse_loss,
    "huber_loss": huber_with_tendency_loss,
    "multiscale_huber_loss": multiscale_huber_loss,
    "spectrally_adjusted_huber_loss": huber_with_tendency_and_spectral_loss,
    "spectrally_adjusted_mse_loss": amse2d_loss,
    "spectrally_adjusted_mse_with_tendency_loss": amse2d_with_tendency_loss,
    "mse_plus_amse_loss": mse_plus_amse_loss,
    "weighted_mse_plus_amse_loss": weighted_mse_plus_amse_loss,
}
