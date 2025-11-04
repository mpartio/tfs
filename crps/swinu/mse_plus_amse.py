import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, rfftfreq, fftfreq


def _radial_bins_rfft(Hf, Wf, device, n_bins):
    fy = fftfreq(Hf, d=1.0, device=device)  # [-0.5,0.5)
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # [0,0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    is_dc = r == 0.0
    r = r / r.max().clamp(min=1e-8)
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(1e-8, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index[is_dc.flatten()] = -1  # exclude dc bins
    bin_index = bin_index.reshape(Hf, Wf)
    mask = bin_index >= 0
    counts = torch.bincount(bin_index[mask].flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, mask, counts, n_bins


def _amse2d_per_time(y_pred, y_true, n_bins):
    eps = 1e-8

    B, T, C, H, W = y_pred.shape

    assert C == 1, f"Support only one output channel (tcc), got: {C}"
    device = y_pred.device

    wh = torch.hann_window(H, device=device).unsqueeze(1)
    ww = torch.hann_window(W, device=device).unsqueeze(0)
    win = (wh @ ww).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    yp = y_pred * win
    yt = y_true * win

    yp = yp.to(torch.float32)
    yt = yt.to(torch.float32)

    # rfft2 over H,W; average PSD/cross over channels
    X = rfft2(yp, dim=(-2, -1), norm="ortho").squeeze(dim=2)  # [B,T,Hf,Wf]
    Y = rfft2(yt, dim=(-2, -1), norm="ortho").squeeze(dim=2)

    Hf, Wf = X.shape[-2], X.shape[-1]
    bin_index, mask, counts, n_bins = _radial_bins_rfft(Hf, Wf, device, n_bins)

    # Magnitudes & cross, channel-mean
    PX = X.real**2 + X.imag**2  # [B,T,Hf,Wf]
    PY = Y.real**2 + Y.imag**2

    # Cross spectrum
    Sxy = X * torch.conj(Y)  # complex [B,T,Hf,Wf]

    flat_idx = bin_index[mask].flatten()

    def reduce_bt(Z):  # Z: [B,T,Hf,Wf] -> [B*T, n_bins]
        Zbt = Z.reshape(B * T, Hf * Wf)[:, mask.flatten()]
        sums = torch.zeros(B * T, n_bins, device=device, dtype=Z.dtype)
        sums.index_add_(1, flat_idx, Zbt)
        return sums / counts

    PSDx = reduce_bt(PX).clamp_min(eps)  # [B*T, n_bins]
    PSDy = reduce_bt(PY).clamp_min(eps)
    Sxy_mag = torch.abs(reduce_bt(Sxy)).clamp_min(eps)

    sqrtx = PSDx.sqrt()
    sqrty = PSDy.sqrt()
    denom = (sqrtx * sqrty).clamp_min(eps)

    # coherence magnitude
    Coh = (Sxy_mag / denom).clamp(0.0, 1.0)

    amp_term = (sqrtx - sqrty) ** 2
    coh_term = 2.0 * (1.0 - Coh) * torch.maximum(PSDx, PSDy)

    amse_bt = (amp_term + coh_term).mean(dim=1)  # [B*T]
    amse_t = amse_bt.view(B, T).mean(dim=0)  # [T]
    return amse_t


def mse_plus_amse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    n_bins: int = None,
    lambda_spectral: float = 0.05,
):
    """
    Combined pixel-wise MSE + spectral AMSE.
    """
    if y_true.dim() == 4:  # [B,C,H,W] -> [B,1,C,H,W]
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

    # pixelwise MSE
    mse_loss = F.mse_loss(y_pred.float(), y_true.float())

    # spectral AMSE
    amse_t = _amse2d_per_time(y_pred, y_true, n_bins)  # [T]
    amse_loss = amse_t.mean()

    total_loss = mse_loss + lambda_spectral * amse_loss
    assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss}"

    return {
        "loss": total_loss,
        "pixel_mse": mse_loss.detach(),
        "spectral_amse": amse_t,
    }


class MSEAMSELoss(nn.Module):
    def __init__(
        self,
        n_bins: int = None,
        lambda_spectral: float = 0.05,
    ):
        super().__init__()

        self.n_bins = n_bins
        self.lambda_spectral = lambda_spectral

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        return mse_plus_amse_loss(
            y_true,
            y_pred,
            n_bins=self.n_bins,
            lambda_spectral=self.lambda_spectral,
        )
