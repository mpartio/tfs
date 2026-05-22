"""Tendency-magnitude matching loss.

Diagnostic on candied-circle@α=0.85 (May 2026) showed the model produces only
~half of truth's per-step change magnitude at long leads (truth flat at ~0.17,
predictions decay from 0.13 → 0.08). This is *temporal smoothing*: the model's
outputs at consecutive forecast steps are too similar to each other.

This loss penalises the model for *under-producing* per-step change magnitude.
Asymmetric: we don't penalise over-production, only under-production. The
choice is intentional — over-production is a noise-emission failure mode we
do NOT want to encourage.

Formula (per-sample field-mean variant — DEFAULT):
    pred_mag_b   = mean_pixels(|y_pred_delta_b|)
    truth_mag_b  = mean_pixels(|y_true_delta_b|)
    loss         = mean_b( max(0, truth_mag_b - pred_mag_b) )

Per-pixel variant (more rigid; available via flag):
    loss = mean_pixels( max(0, |y_true_delta| - |y_pred_delta|) )
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TendencyMatchLoss(nn.Module):
    def __init__(
        self,
        per_pixel: bool = False,
        symmetric: bool = False,
    ):
        super().__init__()
        self.per_pixel = per_pixel
        self.symmetric = symmetric

    def forward(
        self,
        y_pred_delta: torch.Tensor,
        y_true_delta: torch.Tensor,
        **kwargs,
    ) -> dict:
        pred = y_pred_delta.float()
        truth = y_true_delta.float()

        if self.per_pixel:
            diff = truth.abs() - pred.abs()
            if self.symmetric:
                loss = diff.abs().mean()
            else:
                loss = torch.clamp(diff, min=0.0).mean()
        else:
            B = pred.shape[0]
            pred_mag = pred.abs().reshape(B, -1).mean(dim=1)
            truth_mag = truth.abs().reshape(B, -1).mean(dim=1)
            diff = truth_mag - pred_mag
            if self.symmetric:
                loss = diff.abs().mean()
            else:
                loss = torch.clamp(diff, min=0.0).mean()

        assert torch.isfinite(loss), f"Non-finite TendencyMatch loss: {loss}"
        return {"loss": loss}
