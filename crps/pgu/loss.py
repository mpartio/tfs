import torch
import torch.nn as nn

# step_weights = torch.tensor([1, 1.2, 1.5, 1.8, 2.2, 3.0])
# tendency_weights = torch.tensor([2, 2.2, 2.5, 2.8, 3.2, 4.0])
step_weights = torch.tensor([3.0, 1.5, 1.0, 0.8, 0.5, 0.3])
e_std = 0.32  # empirical tendency std
huber = nn.SmoothL1Loss(beta=e_std, reduction="none")


def loss_fn(
    y_true: torch.tensor, y_pred: torch.tensor, rollout_len: int, alpha: int = 2
):
    # rollout_len starts with 0
    B, T, C, H, W = y_true.shape

    weights = step_weights.to(y_true.device)

    err = y_pred - y_true

    step_loss = huber(y_pred, y_true)
    step_loss = step_loss.mean(dim=(0, 2, 3, 4))
    step_loss = weights[0 : rollout_len + 1] * step_loss

    tendency_importance = alpha * torch.abs(y_true)
    tendency_map = tendency_importance * err**2
    tendency_loss = tendency_map.mean(dim=(0, 2, 3, 4))
    tendency_loss = weights[0 : rollout_len + 1] * tendency_loss

    loss = torch.stack((step_loss, tendency_loss)).sum()

    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)

    return {"loss": loss, "step_loss": step_loss, "tendency_loss": tendency_loss}
