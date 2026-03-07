#!/usr/bin/env python3
"""
composite_score_v2.py

A cleaner composite score that (a) rewards fine-scale structure explicitly, and
(b) avoids common pitfalls (NaN propagation, overly “binary” hk/var penalties).

Expected `values` input (dict[str, pd.DataFrame]):

Required keys:
  - "mae": DataFrame with columns: ["model","timestep","mae"]  (timestep optional)
  - "fss": DataFrame with columns: ["model","timestep","fss"] plus ONE of:
           ["mask","mask_size","ws","window","scale_px"] and optionally ["thr"]

Optional keys:
  - "variance_ratio": columns ["model","timestep","variance_ratio"]
  - "highk_power_ratio": columns ["model","timestep","highk_power_ratio"]
  - "spectral_coherence": columns ["model","timestep","coherence_mid","coherence_high"]
  - "change_metrics": columns ["model","timestep","f1","tendency_corr","stationarity_ratio"]
  - "ssim": columns ["model","timestep","ssim"]

This script computes:
  - S_mae         (bounded [0,1], smooth)
  - S_fss_small   (fine-scale FSS aggregation; masks default [2,4,6,9] px)
  - S_fss_meso    (mesoscale FSS aggregation; masks default [11,16,21,29] px)
  - S_hk, S_var   (log-Gaussian around 1.0; smooth)
  - S_coh         (mean of mid/high clipped to [0,1])
  - S_change      (weighted mean of f1/tendency_corr/stationarity_score)
  - S_ssim        (as-is, clipped to [0,1])

The final composite is a weighted average over available components with
automatic re-normalization when some are missing.

You can tune mask sets and weights in `CompositeConfig`.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_mean(x: pd.Series) -> float:
    v = _to_numeric(x).dropna()
    return float(v.mean()) if len(v) else float("nan")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = _to_numeric(values)
    w = _to_numeric(weights)
    ok = v.notna() & w.notna()
    if not ok.any():
        return float("nan")
    num = float((v[ok] * w[ok]).sum())
    den = float(w[ok].sum())
    return num / den if den > 0 else float("nan")


def _clip01(x: float) -> float:
    if np.isnan(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))


def _ideal_one_log_gauss(r: float, sigma_log: float) -> float:
    """
    Smooth score around 1.0 using a Gaussian in log-space:
      score = exp(-(log(r)/sigma_log)^2)

    sigma_log ~ 0.2 means 1.22x (or 0.82x) gives exp(-(0.2/0.2)^2)=e^-1~0.37.
    """
    if np.isnan(r) or r <= 0:
        return float("nan")
    z = math.log(r) / max(1e-8, sigma_log)
    return float(math.exp(-(z * z)))


def _ideal_one_linear(r: float, tol: float) -> float:
    """Your original style (kept for reference): 1 - |r-1|/tol, clipped."""
    if np.isnan(r):
        return float("nan")
    return _clip01(1.0 - abs(r - 1.0) / max(1e-8, tol))


def _get_scale_col(df_fss: pd.DataFrame) -> Optional[str]:
    for c in ["mask", "mask_size", "ws", "window", "scale_px"]:
        if c in df_fss.columns:
            return c
    return None


def _avg_leads(
    df: pd.DataFrame, col: str, lead_weights: Optional[Dict[int, float]]
) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    if not lead_weights or "timestep" not in df.columns:
        return _safe_mean(df[col])
    g = df.copy()
    g["w"] = g["timestep"].map(lead_weights).fillna(1.0)
    return _weighted_mean(g[col], g["w"])


# -----------------------------
# Config
# -----------------------------


@dataclass
class CompositeConfig:
    # FSS scale splits (px). You can tune these aggressively to reward fine-scale.
    fss_small_masks: Tuple[int, ...] = (2, 4, 6, 9)
    fss_meso_masks: Tuple[int, ...] = (11, 16, 21, 29)

    # If you have multiple thresholds in df_fss, you can:
    #  - set thr_select=None to average all
    #  - or provide e.g. (0.2, 0.5, 0.8)
    thr_select: Optional[Tuple[float, ...]] = None

    # MAE reference choice:
    #  - if mae_ref is None, prefer persistence; else fallback to median.
    mae_ref: Optional[float] = None

    # “How strict” for hk/var. log-Gauss is smoother than linear tol.
    hk_sigma_log: float = 0.25
    var_sigma_log: float = 0.25

    # Change stationarity ideal=1; score with log-Gauss or linear. Use log-Gauss by default.
    stat_sigma_log: float = 0.20

    # Coherence: optional extra scaling (often coherence ranges are modest).
    coh_scale: float = (
        1.0  # set to >1 if you want coherence to matter more before clipping
    )

    # Component weights (will be re-normalized over available components)
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            # Default: make fine-scale FSS explicit and important.
            self.weights = {
                "mae": 0.20,
                "fss_small": 0.30,
                "fss_meso": 0.10,
                "hk": 0.15,
                "var": 0.10,
                "coh": 0.05,
                "change": 0.05,
                "ssim": 0.05,
            }


# -----------------------------
# Core computation
# -----------------------------


def compute_composite_v2(
    values: Dict[str, pd.DataFrame],
    lead_weights: Optional[Dict[int, float]] = None,
    save_path: str = "runs/verification",
    out_csv: str = "results/composite_scores_v2.csv",
) -> pd.DataFrame:

    cfg = CompositeConfig(
        fss_small_masks=(2, 4, 6, 9),
        fss_meso_masks=(11, 16, 21, 29),
        thr_select=None,  # or (0.2,0.5,0.8) if you have "thr"
        hk_sigma_log=0.25,
        var_sigma_log=0.25,
        stat_sigma_log=0.20,
        coh_scale=1.0,
        weights={
            "mae": 0.20,
            "fss_small": 0.35,
            "fss_meso": 0.10,
            "hk": 0.15,
            "var": 0.10,
            "coh": 0.05,
            "change": 0.05,
            "ssim": 0.00,  # set to 0 if SSIM is noisy/unhelpful
        },
    )

    return _compute_composite_v2(values, cfg, lead_weights, save_path, out_csv)


def _compute_composite_v2(
    values: Dict[str, pd.DataFrame],
    config: CompositeConfig = CompositeConfig(),
    lead_weights: Optional[Dict[int, float]] = None,
    save_path: str = "runs/verification",
    out_csv: str = "results/composite_scores_v2.csv",
) -> pd.DataFrame:
    os.makedirs(os.path.join(save_path, os.path.dirname(out_csv)), exist_ok=True)

    df_mae = values.get("mae", pd.DataFrame())
    df_fss = values.get("fss", pd.DataFrame())
    df_var = values.get("variance_ratio", pd.DataFrame())
    df_hk = values.get("highk_power_ratio", pd.DataFrame())
    df_coh = values.get("spectral_coherence", pd.DataFrame())
    df_chg = values.get("change_metrics", pd.DataFrame())
    df_ssim = values.get("ssim", pd.DataFrame())

    # -------------------------
    # MAE ref
    # -------------------------
    mae_ref = config.mae_ref
    if mae_ref is None and not df_mae.empty and "model" in df_mae.columns:
        pers = df_mae[df_mae["model"].str.contains("persistence", case=False, na=False)]
        if not pers.empty and "mae" in pers.columns:
            mae_ref = _safe_mean(pers["mae"])
        else:
            tmp = df_mae.groupby("model", dropna=True).apply(
                lambda g: _safe_mean(g["mae"])
            )
            mae_ref = float(tmp.median()) if len(tmp) else 1.0
    if mae_ref is None or not np.isfinite(mae_ref) or mae_ref <= 0:
        mae_ref = 1.0

    # -------------------------
    # Model list
    # -------------------------
    models = set()
    for df in [df_mae, df_fss, df_var, df_hk, df_coh, df_chg, df_ssim]:
        if not df.empty and "model" in df.columns:
            models |= set(df["model"].dropna().unique().tolist())
    models = sorted(models)

    # -------------------------
    # FSS aggregation helpers
    # -------------------------
    scale_col = _get_scale_col(df_fss) if not df_fss.empty else None

    def _fss_by_mask(m: str, masks: Sequence[int]) -> float:
        if df_fss.empty or "model" not in df_fss.columns or "fss" not in df_fss.columns:
            return float("nan")
        g = df_fss[df_fss["model"] == m]
        if g.empty:
            return float("nan")

        if config.thr_select is not None and "thr" in g.columns:
            g = g[g["thr"].isin(config.thr_select)]
            if g.empty:
                return float("nan")

        if scale_col is None:
            # No mask column -> just average fss
            return _avg_leads(g, "fss", lead_weights)

        # Filter to selected masks (robust numeric compare)
        gg = g.copy()
        gg[scale_col] = _to_numeric(gg[scale_col])
        gg = gg[gg[scale_col].isin(list(masks))]
        if gg.empty:
            return float("nan")

        # Average within each mask first, then average masks (prevents one mask dominating by row count)
        per_mask = []
        for ms in masks:
            gm = gg[gg[scale_col] == ms]
            if gm.empty:
                continue
            per_mask.append(_avg_leads(gm, "fss", lead_weights))
        if not per_mask:
            return float("nan")
        return float(np.nanmean(per_mask))

    rows = []
    for m in models:
        # --- MAE -> score in [0,1], smoother than exp(-(x/ref)^2) saturation:
        # Use exp(-((mae/ref - 1)/sigma)^2) with sigma controlling strictness.
        mae_val = _avg_leads(
            df_mae[df_mae.get("model", pd.Series(dtype=str)) == m], "mae", lead_weights
        )
        if np.isnan(mae_val):
            S_mae = float("nan")
        else:
            ratio = float(mae_val / mae_ref)
            # sigma=0.25 means ratio=1.25 gives exp(-(1)^2)=0.37
            sigma = 0.25
            S_mae = float(math.exp(-(((ratio - 1.0) / sigma) ** 2)))

        # --- FSS small/meso
        S_fss_small = _fss_by_mask(m, config.fss_small_masks)
        S_fss_meso = _fss_by_mask(m, config.fss_meso_masks)

        # --- hk/var ratios (smooth around 1)
        hk_val = _avg_leads(
            df_hk[df_hk.get("model", pd.Series(dtype=str)) == m],
            "highk_power_ratio",
            lead_weights,
        )
        var_val = _avg_leads(
            df_var[df_var.get("model", pd.Series(dtype=str)) == m],
            "variance_ratio",
            lead_weights,
        )
        S_hk = _ideal_one_log_gauss(hk_val, config.hk_sigma_log)
        S_var = _ideal_one_log_gauss(var_val, config.var_sigma_log)

        # --- coherence (mid/high)
        gcoh = df_coh[df_coh.get("model", pd.Series(dtype=str)) == m]
        coh_mid = (
            _avg_leads(gcoh, "coherence_mid", lead_weights)
            if not gcoh.empty
            else float("nan")
        )
        coh_high = (
            _avg_leads(gcoh, "coherence_high", lead_weights)
            if not gcoh.empty
            else float("nan")
        )
        if np.isnan(coh_mid) or np.isnan(coh_high):
            S_coh = float("nan")
        else:
            S_coh = _clip01(config.coh_scale * 0.5 * (float(coh_mid) + float(coh_high)))

        # --- change: weighted mean over available parts (NO NaN propagation)
        gchg = df_chg[df_chg.get("model", pd.Series(dtype=str)) == m]
        f1 = _avg_leads(gchg, "f1", lead_weights) if not gchg.empty else float("nan")
        tc = (
            _avg_leads(gchg, "tendency_corr", lead_weights)
            if not gchg.empty
            else float("nan")
        )
        stat = (
            _avg_leads(gchg, "stationarity_ratio", lead_weights)
            if not gchg.empty
            else float("nan")
        )
        S_stat = _ideal_one_log_gauss(stat, config.stat_sigma_log)

        change_parts = []
        change_w = []
        if np.isfinite(f1):
            change_parts.append(_clip01(float(f1)))
            change_w.append(0.5)
        if np.isfinite(tc):
            change_parts.append(_clip01(float(tc)))
            change_w.append(0.3)
        if np.isfinite(S_stat):
            change_parts.append(_clip01(float(S_stat)))
            change_w.append(0.2)
        if change_parts:
            S_change = float(np.average(change_parts, weights=change_w))
        else:
            S_change = float("nan")

        # --- ssim
        ssim_val = _avg_leads(
            df_ssim[df_ssim.get("model", pd.Series(dtype=str)) == m],
            "ssim",
            lead_weights,
        )
        S_ssim = _clip01(float(ssim_val)) if np.isfinite(ssim_val) else float("nan")

        components = {
            "S_mae": S_mae,
            "S_fss_small": S_fss_small,
            "S_fss_meso": S_fss_meso,
            "S_hk": S_hk,
            "S_var": S_var,
            "S_coh": S_coh,
            "S_change": S_change,
            "S_ssim": S_ssim,
            "mae": mae_val,
            "mae_ref": mae_ref,
            "hk_ratio": hk_val,
            "var_ratio": var_val,
            "coh_mid": coh_mid,
            "coh_high": coh_high,
        }

        # Composite with renormalization over available components
        w = config.weights
        map_w = {
            "S_mae": "mae",
            "S_fss_small": "fss_small",
            "S_fss_meso": "fss_meso",
            "S_hk": "hk",
            "S_var": "var",
            "S_coh": "coh",
            "S_change": "change",
            "S_ssim": "ssim",
        }

        num, den = 0.0, 0.0
        for k, wname in map_w.items():
            v = components[k]
            if np.isfinite(v):
                wk = float(w.get(wname, 0.0))
                num += wk * float(v)
                den += wk
        score = num / den if den > 0 else float("nan")

        rows.append({"model": m, "score": score, **components})

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(os.path.join(save_path, out_csv), index=False)
    return out


DEFAULT_COMPONENTS = [
    "S_mae",
    "S_fss_small",
    "S_fss_meso",
    "S_hk",
    "S_var",
    "S_coh",
    "S_change",
    "S_ssim",
]

DEFAULT_LABELS = {
    "S_mae": "MAE",
    "S_fss_small": "FSS (small)",
    "S_fss_meso": "FSS (meso)",
    "S_hk": "High-k",
    "S_var": "Variance",
    "S_coh": "Coherence",
    "S_change": "Change",
    "S_ssim": "SSIM",
}

# Your weights (must match your composite definition)
DEFAULT_WEIGHTS = {
    "S_mae": 0.15,
    "S_ssim": 0.10,
    "S_fss_small": 0.10,
    "S_fss_meso": 0.10,
    "S_coh": 0.15,
    "S_hk": 0.10,
    "S_var": 0.15,
    "S_change": 0.15,
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _luminance(rgba):
    r, g, b, _ = rgba
    # Relative luminance (sRGB-ish)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_component_bars_stacked(
    df,
    save_path: str = "runs/verification",
    filename: str = "figures/component_scores_stacked_v3.png",
    top_n: int = 12,
    exclude_persistence: bool = True,
    components: list[str] | None = None,
    labels: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
    mode: str = "weighted",  # "raw" or "weighted"
    annotate_threshold: float = 0.14,  # only write numbers on segments >= this
):
    """
    Cleaner stacked bars.

    mode="raw":
        Stack raw component scores (0..1). Visual sum is meaningless but shows profile.
        Composite score shown as a vertical marker.

    mode="weighted":
        Stack weighted contributions normalized to sum to the composite score.
        This makes the stack interpretable: stack == score.

    df is expected to contain:
      - model (str)
      - score (float)
      - component columns (floats)
    """
    if components is None:
        components = DEFAULT_COMPONENTS
    if labels is None:
        labels = DEFAULT_LABELS
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dfp = df.copy()
    if exclude_persistence and "model" in dfp.columns:
        dfp = dfp[~dfp["model"].str.contains("persistence", case=False, na=False)]

    dfp = dfp.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    # keep only existing components
    comps = [c for c in components if c in dfp.columns]
    if not comps:
        raise ValueError("No component columns found in df.")

    models = dfp["model"].astype(str).tolist()
    y = np.arange(len(models))

    # Matrix of raw component scores
    raw = dfp[comps].to_numpy(dtype=float)
    raw = np.where(np.isfinite(raw), raw, np.nan)

    # Build data to stack
    if mode == "raw":
        stack = np.where(np.isfinite(raw), raw, 0.0)
        xlabel = "Stacked raw component scores (sum is NOT the composite)"
        title = "Component Scores (raw) – stacked profile + composite marker"
        score_x = dfp["score"].to_numpy(dtype=float)
    elif mode == "weighted":
        # contribution = w_i/Σw_available * raw_i, then rescale to sum to score
        score_x = dfp["score"].to_numpy(dtype=float)
        stack = np.zeros_like(raw, dtype=float)

        for i in range(raw.shape[0]):
            avail = []
            for j, c in enumerate(comps):
                if np.isfinite(raw[i, j]):
                    avail.append(c)

            if not avail or not np.isfinite(score_x[i]):
                continue

            wsum = sum(weights.get(c, 0.0) for c in avail)
            if wsum <= 0:
                continue

            # unscaled weighted parts
            parts = []
            for j, c in enumerate(comps):
                if np.isfinite(raw[i, j]) and c in avail:
                    parts.append((j, (weights.get(c, 0.0) / wsum) * raw[i, j]))
            total = sum(v for _, v in parts)
            scale = (score_x[i] / total) if total > 0 else 0.0

            for j, v in parts:
                stack[i, j] = v * scale

        xlabel = "Stacked weighted contributions (stack SUM == composite score)"
        title = "Component Contributions (weighted) – stacked bars sum to composite"
    else:
        raise ValueError('mode must be "raw" or "weighted"')

    # Figure sizing
    fig_h = max(4.0, 0.6 * len(models))
    fig_w = 13.5
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    # Consistent palette
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(comps))]

    left = np.zeros(len(models), dtype=float)

    for j, c in enumerate(comps):
        vals = stack[:, j]
        ax.barh(
            y,
            vals,
            left=left,
            height=0.78,
            label=labels.get(c, c),
            color=colors[j],
            edgecolor="black",
            linewidth=0.25,
        )

        # sparse annotations
        for i, v in enumerate(vals):
            if v >= annotate_threshold:
                rgba = colors[j]
                txt_color = "white" if _luminance(rgba) < 0.45 else "black"
                ax.text(
                    left[i] + v / 2.0,
                    y[i],
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=txt_color,
                )

        left += vals

    # Composite score marker + label (works in both modes)
    scores = dfp["score"].to_numpy(dtype=float)
    for i, s in enumerate(scores):
        if np.isfinite(s):
            ax.vlines(s, i - 0.39, i + 0.39, linewidth=2.0)  # marker line
            ax.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # x-limits based on stack totals (raw can exceed 1 easily; weighted should be <=1)
    stack_totals = left
    xmax = float(np.max(stack_totals) * 1.08) if len(stack_totals) else 1.0
    xmax = max(xmax, float(np.nanmax(scores) * 1.15) if np.isfinite(scores).any() else 1.0)
    ax.set_xlim(0.0, xmax)

    # Legend outside
    ax.legend(title="Component", bbox_to_anchor=(1.01, 1.0), loc="upper left", frameon=True)

    plt.tight_layout()
    _ensure_dir(os.path.join(save_path, os.path.dirname(filename)))
    plt.savefig(os.path.join(save_path, filename), dpi=200)
    plt.close()



def plot_component_table_v2(
    df: pd.DataFrame,
    save_path: str = "runs/verification",
    filename: str = "results/composite_components_v2.csv",
    cols: Optional[Sequence[str]] = None,
):
    os.makedirs(os.path.join(save_path, os.path.dirname(filename)), exist_ok=True)
    if cols is None:
        cols = [
            "model",
            "score",
            "S_fss_small",
            "S_hk",
            "S_var",
            "S_mae",
            "S_fss_meso",
            "S_coh",
            "S_change",
            "S_ssim",
            "mae",
            "hk_ratio",
            "var_ratio",
        ]
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(os.path.join(save_path, filename), index=False)
