import torch
import torch.nn as nn


class MeanConservationLoss(nn.Module):
    """Penalises drift of the predicted field mean from the true field mean.

    The loss is mathematically equivalent to ``H*W * |mean(p) - mean(y)|`` —
    the spatial-sum form. The naive ``|mean(p) - mean(y)|`` form has per-pixel
    gradient ``sign(Δmean) / (H*W) ≈ 4e-6`` for 475×535 fields, which is six
    orders of magnitude smaller than BCE's per-pixel gradient (~O(1)) and is
    effectively zero under Adam. Scaling internally by ``H*W`` makes the per-
    pixel gradient ``O(sign)`` and lets typical weights (0.05–0.5) compete
    with BCE.

    Args:
        one_sided_positive: if True, penalise only positive drift (over-
            forecast). Zero gradient when under-forecasting. Targets the cc2
            cloud-over-creation failure mode without suppressing real lysis.
        use_full_state: compute the loss on absolute predictions vs absolute
            truth (True) instead of on tendencies (False).
        hinge_tolerance: dead-zone threshold; if |mean_drift| ≤ tolerance the
            loss contribution is zero. Defaults to 0 (always fires). Set to
            e.g. 0.005 to ignore sub-percent bias drift.
    """

    def __init__(
        self,
        one_sided_positive: bool = False,
        use_full_state: bool = True,
        hinge_tolerance: float = 0.0,
    ):
        super().__init__()
        self.one_sided_positive = one_sided_positive
        self.use_full_state = use_full_state
        self.hinge_tolerance = float(hinge_tolerance)

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

        H, W = pred.shape[-2], pred.shape[-1]
        n_pixels = float(H * W)

        mean_pred = pred.mean(dim=(-1, -2))
        mean_true = true.mean(dim=(-1, -2))
        diff = mean_pred - mean_true

        if self.one_sided_positive:
            diff = torch.clamp(diff, min=0.0)

        abs_diff = diff.abs()

        if self.hinge_tolerance > 0.0:
            abs_diff = torch.clamp(abs_diff - self.hinge_tolerance, min=0.0)

        # Multiply by H*W so per-pixel gradient is O(sign), not O(sign/(H*W)).
        # See module docstring.
        loss = (abs_diff * n_pixels).mean()
        assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"
        return {"loss": loss}
