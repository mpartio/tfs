import torch
import torch.nn as nn

step_weights = torch.tensor([1, 1.2, 1.5, 1.8, 2.2, 3.0])
tendency_weights = torch.tensor([2, 2.2, 2.5, 2.8, 3.2, 4.0])


def loss_fn(y_true: torch.tensor, y_pred: torch.tensor, rollout_len: int):
    weights = torch.stack(
        (step_weights[rollout_len], tendency_weights[rollout_len])
    ).to(y_true.device)

    step_loss = torch.mean(nn.MSELoss()(y_true, y_pred))
    tendency_importance = 1.0 + 5.0 * torch.abs(y_true)
    tendency_loss = torch.mean(tendency_importance * (y_pred - y_true) ** 2)

    loss = torch.stack((step_loss, tendency_loss)) * weights

    assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)

    return {"loss": torch.sum(loss), "step_loss": loss[0], "tendency_loss": loss[1]}
