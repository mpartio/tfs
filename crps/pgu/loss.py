import torch
import torch.nn as nn

e_std = 0.32  # empirical tendency std, change point for huber
huber = nn.SmoothL1Loss(beta=e_std, reduction="none")


def loss_fn(
    y_true: torch.tensor,
    y_pred: torch.tensor,
    alpha: float = 2.0,  # scaler for tendency loss
    tau: float = 0.02,  # lower bound for "noise" change
):
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
