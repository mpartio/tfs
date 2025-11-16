import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, rfftfreq, fftfreq


bin_weights = None


def build_highk_ramp(n_bins: int, gamma: float = 1.0, device=None, dtype=None):
    rho = torch.linspace(0, 1, steps=n_bins, device=device, dtype=dtype)
    return rho.pow(gamma)  # will be normalized inside the loss


def _radial_bins_rfft(Hf, Wf, device, n_bins=None):
    fy = fftfreq(Hf, d=1.0, device=device)  # [-0.5,0.5)
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # [0,0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    r = r / r.max().clamp(min=1e-8)
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(0, n_bins - 1)
    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, counts, n_bins


def _amse2d_per_time(y_pred, y_true, n_bins=None, hann_window=True, eps=1e-8):
    """
    y_pred, y_true: [B,T,C,H,W]
    Returns per-time AMSE values: [T]
    """
    B, T, C, H, W = y_pred.shape
    device = y_pred.device

    # optional Hann windowing
    if hann_window:
        wh = torch.hann_window(H, device=device).unsqueeze(1)
        ww = torch.hann_window(W, device=device).unsqueeze(0)
        win = (wh @ ww).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        yp = y_pred * win
        yt = y_true * win
    else:
        yp, yt = y_pred, y_true

    yp = yp.to(torch.float32)
    yt = yt.to(torch.float32)

    # rfft2 over H,W; average PSD/cross over channels
    X = rfft2(yp, dim=(-2, -1), norm="ortho")  # [B,T,C,Hf,Wf]
    Y = rfft2(yt, dim=(-2, -1), norm="ortho")

    Hf, Wf = X.shape[-2], X.shape[-1]
    bin_index, counts, n_bins = _radial_bins_rfft(Hf, Wf, device)

    # Magnitudes & cross, channel-mean
    PX = (X.real**2 + X.imag**2).mean(dim=2)  # [B,T,Hf,Wf]
    PY = (Y.real**2 + Y.imag**2).mean(dim=2)
    Sxy = (X * torch.conj(Y)).mean(dim=2)  # complex [B,T,Hf,Wf]

    flat_idx = bin_index.flatten()

    def reduce_bt(Z):  # Z: [B,T,Hf,Wf] -> [B*T, n_bins]
        Zbt = Z.reshape(B * T, Hf * Wf)
        sums = torch.zeros(B * T, n_bins, device=device, dtype=Z.dtype)
        sums.index_add_(1, flat_idx, Zbt)
        return sums / counts

    PSDx = reduce_bt(PX).clamp_min(eps)  # [B*T, n_bins]
    PSDy = reduce_bt(PY).clamp_min(eps)
    Sxy_mag = reduce_bt(torch.abs(Sxy)).clamp_min(eps)

    sqrtx = PSDx.sqrt()
    sqrty = PSDy.sqrt()
    denom = (sqrtx * sqrty).clamp_min(eps)

    # coherence magnitude
    Coh = (Sxy_mag / denom).clamp(0.0, 1.0)

    amp_term = (sqrtx - sqrty) ** 2
    coh_term = 2.0 * (1.0 - Coh) * torch.maximum(PSDx, PSDy)

    loss_bins = amp_term + coh_term  # [B*T, n_bins]

    global bin_weights

    if bin_weights is None:
        bin_weights = build_highk_ramp(
            n_bins, device=loss_bins.device, dtype=loss_bins.dtype
        )
        # normalize to keep lambda_spectral comparable across choices
        bin_weights = bin_weights / (bin_weights.mean() + eps)

        if bin_weights.numel() != n_bins:
            raise ValueError(
                f"bin_weights length {bin_weights.numel()} != n_bins {n_bins}"
            )

    amse_bt = (loss_bins * bin_weights.unsqueeze(0)).mean(dim=1)  # [B*T]

    amse_t = amse_bt.view(B, T).mean(dim=0)  # [T]
    return amse_t


def weighted_mse_plus_amse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    n_bins: int = None,
    hann_window: bool = True,
    eps: float = 1e-8,
    lambda_spectral: float = 0.05,
):
    """
    Combined pixel-wise MSE + spectral AMSE.
    y_*: [B,T,C,H,W] (or [B,C,H,W] -> treated as T=1)
    """
    if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    # pixelwise MSE
    mse_loss = F.mse_loss(y_pred, y_true)

    # spectral AMSE
    amse_t = _amse2d_per_time(
        y_pred, y_true, n_bins=n_bins, hann_window=hann_window, eps=eps
    )  # [T]
    amse_loss = amse_t.mean()

    total_loss = mse_loss + lambda_spectral * amse_loss
    assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss}"

    return {
        "loss": total_loss,
        "pixel_mse": mse_loss.detach(),
        "spectral_amse": amse_t,
    }
