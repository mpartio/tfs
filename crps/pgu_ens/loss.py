import torch
import torch.nn as nn


def loss_fn(y_true, y_pred):
    return AlmostFairCRPSLoss()(y_pred, y_true)


class AlmostFairCRPSLoss(nn.Module):
    def __init__(self, alpha=0.95, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, predictions, target):
        """
        Args:
            predictions: [B, M, 1, H, W]
            target: [B, 1, H, W]
        """
        assert (
            predictions.ndim == 5
        ), "predictions shape needs to be: B, M, C, H, W: {}".format(predictions.shape)
        assert target.ndim == 4, "target shape needs to be: B, C, H, W: {}".format(
            target.shape
        )

        B, M, C, H, W = predictions.shape
        epsilon = (1 - self.alpha) / M

        # Reshape target to match predictions dimensions
        target = target.unsqueeze(1)  # [B, 1, 1, H, W]

        # Compute differences all at once
        # Using broadcasting for both target differences and member differences
        pred_target_diff = torch.abs(predictions - target)  # [B, M, 1, H, W]

        # Compute member differences using broadcasting
        x_i = predictions.unsqueeze(2)  # [B, M, 1, 1, H, W]
        x_j = predictions.unsqueeze(1)  # [B, 1, M, 1, H, W]
        pred_pred_diff = torch.abs(x_i - x_j)  # [B, M, M, 1, H, W]

        # Sum across members using the formula
        first_term = pred_target_diff.mean(dim=1)  # [B, 1, H, W]
        second_term = (1 - epsilon) * pred_pred_diff.mean(dim=[1, 2])  # [B, 1, H, W]

        # Compute mean across spatial dimensions
        loss = (first_term - 0.5 * second_term).mean(dim=[1, 2, 3])

        loss = loss.mean()
        assert (
            loss == loss
        ), "NaN in loss, predictions min/mean/max/nans: {:.3f}/{:.3f}/{:.3f}/{}, target min/mean/max/nans: {:.3f}/{:.3f}/{:.3f}/{}".format(
            torch.min(predictions),
            torch.mean(predictions),
            torch.max(predictions),
            torch.isnan(predictions).sum(),
            torch.min(target),
            torch.mean(target),
            torch.max(target),
            torch.isnan(target).sum(),
        )

        return loss
