import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import rfft2, rfftfreq, fftfreq
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import lightning as L


def _ensure_btchw(y):
    # Accept [B,T,C,H,W] or [B,C,H,W]. Return [B,T,C,H,W], T>=1
    if y.dim() == 5:
        return y
    elif y.dim() == 4:
        B, C, H, W = y.shape
        return y.view(B, 1, C, H, W)
    raise ValueError(f"Expected 4D or 5D, got {y.shape}")


def _mae(y, x):
    return (y - x).abs().mean().item()


def _bias(y, x):
    return (y - x).mean().item()


def _var_ratio(y, x, eps=1e-8):
    vy = y.var(unbiased=False)
    vx = x.var(unbiased=False)
    return (vy / (vx + eps)).item()


def _radial_bins(Hf, Wf, device, n_bins=None):
    # For rfft grid: spatial size in FFT domain is [Hf, Wf], where Wf = W//2 + 1
    fy = fftfreq(Hf, d=1.0, device=device)  # [-0.5,0.5)
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # [0,0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    r = r / r.max().clamp(min=1e-8)  # normalize to [0,1]
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(0, n_bins - 1)
    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, counts, n_bins


def _psd_coherence_once(
    y_pred, y_true, top_frac=0.20, k_mid=0.30, k_high=0.60, eps=1e-8
):
    """
    Inputs: [C,H,W] tensors (single frame). Returns dict scalars.
    """
    device = y_pred.device
    dtype = y_pred.dtype
    C, H, W = y_pred.shape

    # FFT on each channel; use orthonormal to keep scales stable
    X = rfft2(y_pred, norm="ortho")  # [C,H,Wf]
    Y = rfft2(y_true, norm="ortho")  # [C,H,Wf]
    Hf, Wf = X.shape[-2], X.shape[-1]

    # PSD per-channel then average across channels
    PX = (X.real**2 + X.imag**2).mean(dim=0)  # [Hf,Wf]
    PY = (Y.real**2 + Y.imag**2).mean(dim=0)
    Cxy = (X * torch.conj(Y)).mean(dim=0).real  # [Hf,Wf], real cross-power

    # Radial binning
    bin_index, counts, n_bins = _radial_bins(Hf, Wf, device)
    flat_idx = bin_index.flatten()

    def _bin_mean(Z):
        Zb = Z.reshape(-1, Hf * Wf)
        sums = torch.zeros(Zb.shape[0], n_bins, dtype=Zb.dtype, device=device)
        sums.index_add_(1, flat_idx, Zb)
        return sums / counts  # [1,n_bins] in our usage

    PSDx = _bin_mean(PX[None, ...]).squeeze(0)  # [n_bins]
    PSDy = _bin_mean(PY[None, ...]).squeeze(0)
    Rxy = _bin_mean(Cxy[None, ...]).squeeze(0)

    # High-k power ratio (top_frac of bins)
    k0 = int((1.0 - top_frac) * n_bins)
    k0 = max(0, min(n_bins - 1, k0))
    hi_pred = PSDx[k0:].mean()
    hi_true = PSDy[k0:].mean()
    high_k_ratio = (hi_pred / (hi_true + eps)).item()

    # Coherence as Rxy / sqrt(Px*Py)
    denom = (PSDx.sqrt() * PSDy.sqrt()).clamp_min(eps)
    Coh = (Rxy / denom).clamp(-1.0, 1.0)  # [n_bins]

    def _pick_band(target):
        idx = int(target * (n_bins - 1))
        idx = max(0, min(n_bins - 1, idx))
        return Coh[idx].item()

    coh_mid = _pick_band(k_mid)
    coh_high = _pick_band(k_high)

    return {
        "high_k_power_ratio": high_k_ratio,
        "coherence_mid": coh_mid,
        "coherence_high": coh_high,
    }


def _fss_once(y_pred, y_true, threshold=0.5, radius=10, eps=1e-8):
    """
    Fraction Skill Score at one threshold & radius on a single frame.
    y_pred, y_true: [C,H,W] or [H,W] with values in [0,1] (cloud cover/occupancy).
    We binarize at threshold, then compute fractional cover via box filter (ones kernel).
    """
    if y_pred.dim() == 3:
        # average channels first (if multiple)
        y_pred = y_pred.mean(dim=0)
        y_true = y_true.mean(dim=0)
    # Binarize exceedance
    yp = (y_pred >= threshold).to(torch.float32)[None, None]  # [1,1,H,W]
    yt = (y_true >= threshold).to(torch.float32)[None, None]

    k = 2 * radius + 1
    kernel = torch.ones((1, 1, k, k), device=yp.device) / (k * k)
    pad = radius
    # fractional coverage in window
    yp_f = F.conv2d(F.pad(yp, (pad, pad, pad, pad), mode="reflect"), kernel)
    yt_f = F.conv2d(F.pad(yt, (pad, pad, pad, pad), mode="reflect"), kernel)

    num = (yp_f - yt_f).pow(2).mean()
    den = (yt_f.pow(2) + yp_f.pow(2)).mean() + eps
    fss = 1.0 - (num / den)
    return fss.item()


def _compute_metrics_pack(y_pred_btchw, y_true_btchw, fss_threshold=0.5, fss_radius=10):
    """
    y_*: [B,T,C,H,W] (or [B,C,H,W]) tensors, same shapes
    Uses only the first frame T=0 for speed/consistency (your stored sample is single-step).
    Aggregates metrics across batch (mean of frame-wise metrics).
    """
    y_pred = _ensure_btchw(y_pred_btchw)
    y_true = _ensure_btchw(y_true_btchw)
    B, T, C, H, W = y_pred.shape
    # Use first step (single-step training), average over batch
    step = 0
    yp = y_pred[:, step]  # [B,C,H,W]
    yt = y_true[:, step]

    mae_list, fss_list, vk_list, coh_mid_list, coh_hi_list = [], [], [], [], []
    for b in range(yp.shape[0]):
        yp_b = yp[b]
        yt_b = yt[b]

        mae_list.append(_mae(yp_b, yt_b))
        vk_list.append(_var_ratio(yp_b, yt_b))

        coh = _psd_coherence_once(yp_b, yt_b)  # dict
        coh_mid_list.append(coh["coherence_mid"])
        coh_hi_list.append(coh["coherence_high"])
        fss_list.append(
            _fss_once(yp_b, yt_b, threshold=fss_threshold, radius=fss_radius)
        )

    # High-k ratio can be averaged too (use last computed dict from loop)
    # Better: recompute per-sample and average
    hk_list = []
    for b in range(yp.shape[0]):
        hk = _psd_coherence_once(yp[b], yt[b])["high_k_power_ratio"]
        hk_list.append(hk)

    return {
        "mae": float(sum(mae_list) / len(mae_list)),
        "variance_ratio": float(sum(vk_list) / len(vk_list)),
        "high_k_power_ratio": float(sum(hk_list) / len(hk_list)),
        "coherence_mid": float(sum(coh_mid_list) / len(coh_mid_list)),
        "coherence_high": float(sum(coh_hi_list) / len(coh_hi_list)),
        "fss": float(sum(fss_list) / len(fss_list)),
    }


class EarlyWarningMetricsCallback(L.Callback):
    """
    Logs quick diagnostics to MLflow at epoch end using predictions already
    stored on the pl_module:
      - latest_train_predictions, latest_train_data = (x, y)
      - latest_val_predictions,   latest_val_data   = (x, y)

    It does NOT run any forward passes itself.
    """

    def __init__(
        self,
        fss_threshold: float = 0.5,
        fss_radius: int = 10,
        log_train: bool = True,
        log_val: bool = True,
    ):
        super().__init__()
        self.fss_threshold = fss_threshold
        self.fss_radius = fss_radius
        self.log_train = log_train
        self.log_val = log_val

    def _mlflow_log_metrics(self, pl_module, metrics: dict, prefix: str):
        for k, v in metrics.items():
            pl_module.log(
                f"metrics/{prefix}/{k}",
                float(v),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (
            self.log_train
            and hasattr(pl_module, "latest_train_predictions")
            and hasattr(pl_module, "latest_train_data")
        ):
            with torch.no_grad():
                y_pred = pl_module.latest_train_predictions
                _, y_true = pl_module.latest_train_data
                # Ensure on same device & dtype if needed
                metrics = _compute_metrics_pack(
                    y_pred,
                    y_true,
                    fss_threshold=self.fss_threshold,
                    fss_radius=self.fss_radius,
                )
            self._mlflow_log_metrics(pl_module, metrics, prefix="train")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (
            self.log_val
            and hasattr(pl_module, "latest_val_predictions")
            and hasattr(pl_module, "latest_val_data")
        ):
            with torch.no_grad():
                y_pred = pl_module.latest_val_predictions
                _, y_true = pl_module.latest_val_data
                metrics = _compute_metrics_pack(
                    y_pred,
                    y_true,
                    fss_threshold=self.fss_threshold,
                    fss_radius=self.fss_radius,
                )
            self._mlflow_log_metrics(pl_module, metrics, prefix="val")
