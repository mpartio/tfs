import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    """
    1D Wasserstein distance (Earth Mover's Distance) between the empirical
    value distributions of prediction and target.

    Computed by sorting both tensors and taking mean |sort(pred) - sort(target)|.
    This is differentiable: gradients flow through the sorted prediction values
    (the sort permutation is treated as fixed for the backward pass).

    Targets are detached (we only shape the prediction distribution).
    This loss does NOT enforce spatial correspondence — pair with MAE.
    """

    def __init__(self, n_quantiles: int | None = None):
        super().__init__()
        # n_quantiles: if set, subsample both distributions to this many points
        # before computing W1. Reduces memory for large spatial fields.
        # None = use all pixels.
        self.n_quantiles = n_quantiles

    def forward(
        self,
        y_pred_full: torch.Tensor,
        y_true_full: torch.Tensor,
        **kwargs,
    ):
        yp = y_pred_full.float().flatten()
        yt = y_true_full.float().flatten().detach()

        if self.n_quantiles is not None and yp.numel() > self.n_quantiles:
            idx = torch.randperm(yp.numel(), device=yp.device)[: self.n_quantiles]
            yp = yp[idx]
            yt = yt[idx]

        yp_sorted = yp.sort()[0]
        yt_sorted = yt.sort()[0]

        loss = (yp_sorted - yt_sorted).abs().mean()

        assert torch.isfinite(loss), f"Non-finite Wasserstein loss: {loss}"
        return {"loss": loss}
