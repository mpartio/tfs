import torch
import torch.nn as nn


class RectifiedFlowLoss(nn.Module):
    """
    MSE loss on the velocity field for Rectified Flow training.

    The model predicts tendency ≈ v = z - y_true (noise minus clean next state).
    _compute_step_loss passes this velocity target as y_true_full (since training
    replaces data[1] with z - y_true). We compare against y_pred_delta (tendency).
    """

    def forward(
        self, y_pred_delta: torch.Tensor, y_true_full: torch.Tensor, **kwargs
    ) -> dict:
        loss = ((y_true_full - y_pred_delta) ** 2).mean()
        assert torch.isfinite(loss).all(), f"Non-finite RF loss: {loss}"
        return {"loss": loss}
