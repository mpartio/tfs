import torch
import torch.nn as nn
import torch.fft as tfft
import math


def _hann2d(H, W, device, dtype):
    # separable Hann window
    w_h = torch.hann_window(H, dtype=dtype, device=device)
    w_w = torch.hann_window(W, dtype=dtype, device=device)
    return w_h[:, None] * w_w[None, :]


def _radial_bins(H, W, n_bins=None, device=None):
    """
    Returns: bin_index [H, W] of ints in [0, n_bins-1], and counts per bin [n_bins]
    Uses full symmetric frequency grid indices for rfft2-compatible magnitudes.
    """
    # Frequencies corresponding to FFT grid (use same as rfft2 magnitudes)
    fy = torch.fft.fftfreq(H, d=1.0, device=device)  # [-0.5, 0.5)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device)  # [0, 0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)  # radial frequency
    r = r / r.max().clamp(min=1e-8)  # normalize to [0,1]

    if n_bins is None:
        # ~ one bin per ~pixel scale up to Nyquist along the shorter axis
        n_bins = max(8, min(H, W) // 2)

    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1  # [H*W]
    bin_index = bin_index.reshape(H, FX.shape[1]).clamp(min=0, max=n_bins - 1)

    # counts per bin
    counts = torch.bincount(bin_index.flatten(), minlength=n_bins)
    counts = counts.clamp(min=1)  # avoid div by zero

    return bin_index, counts, n_bins


def _spectra_and_coherence_radial(
    y_pred, y_true, hann_window=False, n_bins=None, eps=1e-8
):
    """
    y_pred, y_true: [B, T, C, H, W]
    Returns per-time AMSE-like loss components aggregated over bins: [T]
    """
    B, T, C, H, W = y_pred.shape
    device = y_pred.device
    dtype = y_pred.dtype

    # Convert to float32 if bfloat16 (FFT doesn't support bf16)
    if dtype == torch.bfloat16:
        y_pred = y_pred.float()
        y_true = y_true.float()
        compute_dtype = torch.float32
    else:
        compute_dtype = dtype

    # Optional window to reduce leakage (same for pred and true)
    if hann_window:
        win = _hann2d(H, W, device, compute_dtype)
        y_pred_w = y_pred * win
        y_true_w = y_true * win
    else:
        y_pred_w = y_pred
        y_true_w = y_true

    # 2D FFT (rfft2 to save compute), normalize='ortho' keeps scale-stable
    # Shape after rfft2: [B, T, C, H, W_r] where W_r = W//2 + 1
    X = tfft.rfft2(y_pred_w, dim=(-2, -1), norm="ortho")
    Y = tfft.rfft2(y_true_w, dim=(-2, -1), norm="ortho")

    # Power Spectral Density per (B,T,C) & bin
    # PSD = mean over channels of |F|^2 (you can choose sum if channels should aggregate energy)
    # Cross spectrum: mean over channels of F_x * conj(F_y)
    # Then radial-bin average.
    Hf, Wf = X.shape[-2], X.shape[-1]
    bin_index, counts, n_bins = _radial_bins(
        Hf, 2 * (Wf - 1), n_bins=n_bins, device=device
    )  # reconstruct original W for bin gen
    # BUT we actually need bins for rfft2 grid size (Hf, Wf):
    # Build bins on rfft grid instead:
    fy = torch.fft.fftfreq(Hf, d=1.0, device=device)
    fx = torch.fft.rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # original W from Wf
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    r = r / r.max().clamp(min=1e-8)
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(min=0, max=n_bins - 1)
    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)

    # Flatten spatial FFT dims for bin-reduce
    def _bin_reduce_mean(z):  # z: [..., Hf, Wf]
        zb = z.reshape(-1, Hf * Wf)  # merge leading dims
        # sum per bin
        sums = torch.zeros(zb.shape[0], n_bins, dtype=z.dtype, device=z.device)
        sums.index_add_(1, bin_index.flatten(), zb)
        return sums / counts  # [.., n_bins]

    # Compute per-channel PSD and cross
    # |F|^2
    PX = X.real**2 + X.imag**2  # [B,T,C,Hf,Wf]
    PY = Y.real**2 + Y.imag**2
    # cross-spectrum real part: Re(Fx * conj(Fy)) = Re(Fx.real*Fy.real + Fx.imag*Fy.imag, Fx.imag*Fy.real - Fx.real*Fy.imag) -> simpler: (Fx*conj(Fy)).real
    Cxy = (X * torch.conj(Y)).real  # [B,T,C,Hf,Wf]

    # Average over channels first (you can experiment: avg vs sum)
    PXc = PX.mean(dim=2)  # [B,T,Hf,Wf]
    PYc = PY.mean(dim=2)
    Cxyc = Cxy.mean(dim=2)

    # Radial-bin means
    PSDx = _bin_reduce_mean(
        PXc
    )  # [B*T, n_bins] after reshape; but we merged dims above; handle carefully
    # Correct reshape for _bin_reduce_mean:
    def reduce_over_bt(z):
        # z: [B, T, Hf, Wf]
        z_flat = z.reshape(B * T, Hf, Wf)
        zb = z_flat.reshape(B * T, Hf * Wf)
        sums = torch.zeros(B * T, n_bins, dtype=z.dtype, device=z.device)
        sums.index_add_(1, bin_index.flatten(), zb)
        return sums / counts  # [B*T, n_bins]

    PSDx = reduce_over_bt(PXc)
    PSDy = reduce_over_bt(PYc)
    Rxy = reduce_over_bt(Cxyc)  # real cross-power

    # Amplitude ratio terms use sqrt(PSD). Coherence ~ Rxy / sqrt(PSDx*PSDy)
    sqrt_PSDx = torch.sqrt(PSDx.clamp_min(eps))
    sqrt_PSDy = torch.sqrt(PSDy.clamp_min(eps))
    denom = (sqrt_PSDx * sqrt_PSDy).clamp_min(eps)
    Coh = (Rxy / denom).clamp(-1.0, 1.0)  # [B*T, n_bins], real

    # AMSE per-bin per-(B*T)
    # (sqrt(PSDx) - sqrt(PSDy))^2 + 2 * max(PSDx, PSDy) * (1 - Coh)
    amp_term = (sqrt_PSDx - sqrt_PSDy) ** 2
    max_psd = torch.maximum(PSDx, PSDy)
    coh_term = 2.0 * max_psd * (1.0 - Coh)

    amse_bt_bin = amp_term + coh_term  # [B*T, n_bins]

    # Average over bins (equal weight). You can optionally weight bins ~ area or k.
    amse_bt = amse_bt_bin.mean(dim=1)  # [B*T]
    amse_bt = amse_bt.reshape(B, T)

    # Return mean over batch, keep time dimension (like your step_loss/tendency_loss)
    amse_t = amse_bt.mean(dim=0)  # [T]
    return amse_t  # [T]


def huber_with_tendency_and_spectral_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    alpha: float = 2.0,  # scaler for tendency loss
    tau: float = 0.02,  # lower bound for "noise" change
    e_std: float = 0.32,  # Huber delta (your setting)
    lambda_spec: float = 0.25,  # weight for spectral AMSE term
    n_bins: int = None,
    hann_window: bool = False,
    eps: float = 1e-8,
):
    """
    y_true, y_pred: [B, T, C, H, W]
    Returns dict with total loss and components.
    """
    huber = nn.SmoothL1Loss(beta=e_std, reduction="none")
    B, T, C, H, W = y_true.shape

    # ---- Base Huber (per-pixel), keep your reduction scheme ----
    base = huber(y_pred, y_true)  # [B,T,C,H,W]
    step_loss = base.mean(dim=(0, 2, 3, 4))  # [T]

    # ---- Change-aware tendency loss (your formulation) ----
    magnitude = (torch.abs(y_true) - tau).clamp_min(0.0)  # [B,T,C,H,W]
    magnitude_mean = magnitude.mean(dim=(0, 2, 3, 4), keepdim=True)  # [1,T,1,1,1]
    magnitude = magnitude / (magnitude_mean + 1e-8)
    magnitude = magnitude * alpha
    tendency_loss = (magnitude * base).mean(dim=(0, 2, 3, 4))  # [T]

    # ---- Spectral AMSE-2D (radial Fourier) ----
    amse_t = _spectra_and_coherence_radial(
        y_pred, y_true, hann_window=hann_window, n_bins=n_bins, eps=eps
    )  # [T]

    # ---- Combine ----
    loss = step_loss.mean() + tendency_loss.mean() + lambda_spec * amse_t.mean()

    assert torch.isfinite(loss).all(), f"Non-finite values at loss: {loss}"

    return {
        "loss": loss,
        "step_loss": step_loss,  # [T]
        "tendency_loss": tendency_loss,  # [T]
        "spectral_amse": amse_t,  # [T]
        "lambda_spec": torch.tensor(lambda_spec, device=y_true.device),
    }
