import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = ((y_true - y_pred) ** 2).mean()
        assert torch.isfinite(loss).all(), "Non-finite values at loss: {}".format(loss)
        return {"loss": loss, "mse_loss": loss}
