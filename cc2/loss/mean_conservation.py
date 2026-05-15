import torch
import torch.nn as nn


class MeanConservationLoss(nn.Module):
    """Penalises drift of the predicted field mean from the true field mean.

    Computes the per-sample, per-rollout-step spatial mean of both predicted and
    true fields, takes the absolute difference, and averages across batch and
    rollout dimensions. Symmetric by default (penalises over- and under-forecast
    equally). Set ``one_sided_positive=True`` to penalise only positive drift
    (over-forecast), which is the dominant error mode for the current refinement.
    """

    def __init__(self, one_sided_positive: bool = False, use_full_state: bool = True):
        super().__init__()
        self.one_sided_positive = one_sided_positive
        self.use_full_state = use_full_state

    def forward(
        self,
        y_pred_delta: torch.Tensor,
        y_true_delta: torch.Tensor,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs,
    ):
        pred = y_pred_full if self.use_full_state else y_pred_delta
        true = y_true_full if self.use_full_state else y_true_delta

        mean_pred = pred.mean(dim=(-1, -2))
        mean_true = true.mean(dim=(-1, -2))
        diff = mean_pred - mean_true

        if self.one_sided_positive:
            diff = torch.clamp(diff, min=0.0)

        loss = diff.abs().mean()
        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss}
