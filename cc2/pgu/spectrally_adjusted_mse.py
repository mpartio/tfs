import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, rfftfreq, fftfreq


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


def _amse2d_per_time(y_pred, y_true, n_bins=None, hann_window=False, eps=1e-8):
    """
    y_pred, y_true: [B,T,C,H,W]
    Returns per-time AMSE values: [T]
    """
    B, T, C, H, W = y_pred.shape
    device = y_pred.device

    if hann_window:
        wh = torch.hann_window(H, device=device).unsqueeze(1)
        ww = torch.hann_window(W, device=device).unsqueeze(0)
        win = (wh @ ww).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        yp = y_pred * win
        yt = y_true * win
    else:
        yp, yt = y_pred, y_true

    yp = y_pred.to(torch.float32)
    yt = y_true.to(torch.float32)

    # rfft2 over H,W; average PSD/cross over channels
    X = rfft2(yp, dim=(-2, -1), norm="ortho")  # [B,T,C,Hf,Wf]
    Y = rfft2(yt, dim=(-2, -1), norm="ortho")

    Hf, Wf = X.shape[-2], X.shape[-1]
    bin_index, counts, n_bins = _radial_bins_rfft(Hf, Wf, device)

    # Magnitudes & cross, channel-mean
    PX = (X.real**2 + X.imag**2).mean(dim=2)  # [B,T,Hf,Wf]
    PY = (Y.real**2 + Y.imag**2).mean(dim=2)
    Rxy = (X * torch.conj(Y)).mean(dim=2).real  # [B,T,Hf,Wf]

    flat_idx = bin_index.flatten()

    def reduce_bt(Z):  # Z: [B,T,Hf,Wf] -> [B*T, n_bins]
        Zbt = Z.reshape(B * T, Hf * Wf)
        sums = torch.zeros(B * T, n_bins, device=device, dtype=Z.dtype)
        sums.index_add_(1, flat_idx, Zbt)
        return sums / counts

    PSDx = reduce_bt(PX).clamp_min(eps)  # [B*T, n_bins]
    PSDy = reduce_bt(PY).clamp_min(eps)
    Rxyb = reduce_bt(Rxy)

    sqrtx = PSDx.sqrt()
    sqrty = PSDy.sqrt()
    denom = (sqrtx * sqrty).clamp_min(eps)
    Coh = (Rxyb / denom).clamp(-1.0, 1.0)  # [B*T, n_bins]

    amp_term = (sqrtx - sqrty) ** 2
    max_psd = torch.maximum(PSDx, PSDy)
    coh_term = 2.0 * max_psd * (1.0 - Coh)

    amse_bt = (amp_term + coh_term).mean(dim=1)  # [B*T]
    amse_t = amse_bt.view(B, T).mean(dim=0)  # [T]
    return amse_t


def amse2d_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    n_bins: int = None,
    hann_window: bool = False,
    eps: float = 1e-8,
):
    """
    Pure AMSE-2D (Fourier analog of paper's AMSE). No pixelwise term.
    y_*: [B,T,C,H,W] (or [B,C,H,W] -> treated as T=1)
    """
    if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    amse_t = _amse2d_per_time(
        y_pred, y_true, n_bins=n_bins, hann_window=hann_window, eps=eps
    )  # [T]
    loss = amse_t.mean()
    assert torch.isfinite(loss), f"Non-finite AMSE: {loss}"
    return {"loss": loss, "spectral_amse": amse_t}


def amse2d_with_tendency_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    alpha: float = 2.0,  # tendency weight (optional)
    tau: float = 0.02,  # ignore tiny changes below tau (optional)
    lambda_spec: float = 0.25,  # weight for spectral AMSE-2D
    n_bins: int = None,
    hann_window: bool = False,
    eps: float = 1e-8,
):
    """
    y_*: [B,T,C,H,W] (or [B,C,H,W] -> treated as T=1)
    Returns dict with total loss and components.
    """
    if y_true.dim() == 4:
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    B, T, C, H, W = y_true.shape

    # ---- Base MSE ----
    base = (y_pred - y_true) ** 2  # [B,T,C,H,W]
    step_loss = base.mean(dim=(0, 2, 3, 4))  # [T]

    # ---- Change-aware tendency (optional; can set alpha=0 to disable) ----
    if alpha != 0.0:
        magnitude = (torch.abs(y_true) - tau).clamp_min(0.0)  # [B,T,C,H,W]
        magnitude_mean = magnitude.mean(dim=(0, 2, 3, 4), keepdim=True)
        magnitude = magnitude / (magnitude_mean + 1e-8)
        magnitude = magnitude * alpha
        tendency_loss = (magnitude * base).mean(dim=(0, 2, 3, 4))  # [T]
    else:
        tendency_loss = torch.zeros(T, device=y_true.device, dtype=y_true.dtype)

    # ---- Spectral AMSE-2D ----
    amse_t = _amse2d_per_time(
        y_pred, y_true, n_bins=n_bins, hann_window=hann_window, eps=eps
    )  # [T]

    # ---- Combine ----
    loss = step_loss.mean() + tendency_loss.mean() + (lambda_spec * amse_t.mean())
    assert torch.isfinite(loss), f"Non-finite values at loss: {loss}"

    return {
        "loss": loss,
        "step_loss": step_loss,  # [T]
        "tendency_loss": tendency_loss,  # [T]
        "spectral_amse": amse_t,  # [T]
        "lambda_spec": torch.tensor(lambda_spec, device=y_true.device),
    }
