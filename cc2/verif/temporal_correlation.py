"""
Per-pixel temporal correlation between predicted and true cloud-cover evolution.

For each (i, j) spatial location, computes Pearson correlation across init times
between the model's predicted field and the truth field at a given lead time.
Then averages over space to produce a single scalar per lead.

Rationale (campaign-internal, 2026-05): high-K power and FSS are both subject to
"sharpness without correctness" — a sparser-than-truth model has higher high-K
mechanically (zero-vs-nonzero contrast), and FSS pooling is satisfied by
matching neighbourhood means without per-pixel correctness. The multi-case
dynamic-event gate (`dynamic_case_gate.py`) catches this but is expensive
(needs to enumerate top-quintile dynamic cases and score each). This metric is
a cheap scalar that:

- requires per-pixel correctness (sparse-but-wrong predictions don't pass)
- is bias-invariant (correlation is shift-invariant)
- is scale-invariant (so over-/under-prediction doesn't game it)

What it catches that high-K misses:
- the S2 / α=0.50 failure mode where the model becomes spectrally sharper but
  places its sparse clouds in the wrong locations
- the candied-circle dynamic-event smoothing — predictions are right on average
  but the time-evolution at each pixel is wrong

Expected ranges (calibrated on existing models):
- kinetic-creative-12h:  ~0.55 at t+12 (target ceiling)
- trusting-radar-12h:    ~0.50 at t+12 (the dynamic-reference floor)
- candied-circle-12h:    ~0.40 at t+12
- reduced-arena-12h:     ~0.20 at t+12 (sharpness-without-correctness)
- cloudcast-production:  ~0.45 at t+12

Suggested operational threshold: spatial-mean correlation ≥ 0.45 at t+12.

Usage:
    python3 verif/temporal_correlation.py \\
        --candidates /data/tfs/runs/ED12h/<model>-12h ...
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


def load_run(run_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = torch.load(f"{run_dir}/predictions.pt", weights_only=False, map_location="cpu")
    t = torch.load(f"{run_dir}/truth.pt", weights_only=False, map_location="cpu")
    d = torch.load(f"{run_dir}/dates.pt", weights_only=False, map_location="cpu")
    return p, t, d


def _per_pixel_correlation_across_cases(
    p: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Pearson correlation per (i,j) across the batch dim. p, t are [N, H, W]."""
    p_mean = p.mean(dim=0, keepdim=True)
    t_mean = t.mean(dim=0, keepdim=True)
    p_dev = p - p_mean
    t_dev = t - t_mean

    num = (p_dev * t_dev).sum(dim=0)
    den = ((p_dev**2).sum(dim=0).sqrt()) * ((t_dev**2).sum(dim=0).sqrt())
    den = den.clamp(min=1e-12)

    corr = num / den
    corr = torch.where(corr.isnan(), torch.zeros_like(corr), corr)
    return corr


def per_pixel_temporal_correlation(
    pred: torch.Tensor, truth: torch.Tensor, lead: int
) -> Dict[str, float]:
    """Per-pixel correlation between pred and truth at a given lead.

    For each (i, j), correlate pred_n[i,j] vs truth_n[i,j] across init times n.
    Returns spatial-mean + percentile breakdown.

    Measures: does the per-pixel evolution across forecasts agree with truth?
    Cheap, bias-invariant, catches "sharpness-without-correctness" failure
    modes where high-K rises but per-pixel evolution drifts.
    """
    p = pred[:, lead].float().squeeze(1)  # [N, H, W]
    t = truth[:, lead].float().squeeze(1)

    nan_mask = (p.isnan().any(dim=(-1, -2))) | (t.isnan().any(dim=(-1, -2)))
    valid = ~nan_mask
    if valid.sum() < 3:
        return {"mean": float("nan"), "n_valid": int(valid.sum())}

    p, t = p[valid], t[valid]
    corr = _per_pixel_correlation_across_cases(p, t)
    flat = corr.flatten()
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p10": float(np.percentile(flat.numpy(), 10)),
        "p90": float(np.percentile(flat.numpy(), 90)),
        "n_valid": int(valid.sum()),
    }


def per_pixel_tendency_correlation(
    pred: torch.Tensor, truth: torch.Tensor, lead: int
) -> Dict[str, float]:
    """Per-pixel correlation of TENDENCIES (lead-to-lead deltas) across init times.

    For each (i, j), correlate (pred[lead] - pred[lead-1])_n vs the corresponding
    truth tendency, across init times n. Requires lead >= 1.

    Measures: does the model get the per-step change at each pixel right?
    This is the metric the TemporalDifferenceLoss directly optimises; logging
    it as a monitor closes a useful loop.

    Bias-invariant by construction: if pred is uniformly shifted, the tendency
    is unaffected. So this metric is sensitive to *evolution* not *level*.
    """
    if lead < 1:
        raise ValueError(f"Tendency correlation requires lead >= 1, got {lead}")

    p_delta = (pred[:, lead] - pred[:, lead - 1]).float().squeeze(1)  # [N, H, W]
    t_delta = (truth[:, lead] - truth[:, lead - 1]).float().squeeze(1)

    nan_mask = (
        p_delta.isnan().any(dim=(-1, -2)) | t_delta.isnan().any(dim=(-1, -2))
    )
    valid = ~nan_mask
    if valid.sum() < 3:
        return {"mean": float("nan"), "n_valid": int(valid.sum())}

    p_delta, t_delta = p_delta[valid], t_delta[valid]
    corr = _per_pixel_correlation_across_cases(p_delta, t_delta)
    flat = corr.flatten()
    return {
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p10": float(np.percentile(flat.numpy(), 10)),
        "p90": float(np.percentile(flat.numpy(), 90)),
        "n_valid": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="One or more run directories containing predictions.pt and truth.pt",
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        default=[1, 4, 8, 12],
        help="Lead indices to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["state", "tendency", "both"],
        default="both",
        help="state = correlate pred[lead] vs truth[lead]; "
             "tendency = correlate (pred[lead] - pred[lead-1]) vs truth's; "
             "both = print both tables",
    )
    args = parser.parse_args()

    if args.mode in ("state", "both"):
        print(f"\nState correlation: corr_n(pred_n[lead, i, j], truth_n[lead, i, j])")
        print(f"------------------------------------------------------------------\n")
        cols = [f"t+{L}" for L in args.leads]
        header = f"{'model':<50}  " + "  ".join(f"{c:>8}" for c in cols)
        print(header)
        print("-" * len(header))
        for cand_dir in args.candidates:
            name = os.path.basename(cand_dir.rstrip("/"))
            try:
                pred, truth, _ = load_run(cand_dir)
            except Exception as e:
                print(f"{name:<50}  load error: {e}")
                continue
            cells = [
                f"{per_pixel_temporal_correlation(pred, truth, L)['mean']:>8.4f}"
                for L in args.leads
            ]
            print(f"{name:<50}  " + "  ".join(cells))
        print()
        print("Observed range on the campaign comparison set:")
        print("  trusting-radar (dynamic-reference)   ~0.44 at t+12")
        print("  good candidates (kinetic, candied)   ~0.33-0.36 at t+12")
        print("  broken candidates (α=0.50, S2)       ~0.28-0.30 at t+12")
        print("Suggested floor: spatial-mean ≥ 0.32 at t+12 ('not catastrophically broken').")

    if args.mode in ("tendency", "both"):
        # Tendency is meaningful for leads >= 1
        td_leads = [L for L in args.leads if L >= 1]
        print(f"\nTendency correlation: corr_n(pred_n[lead] - pred_n[lead-1], truth's)")
        print(f"------------------------------------------------------------------\n")
        print("Bias-invariant; sensitive to evolution, not level.")
        print("This is the metric TemporalDifferenceLoss directly optimises.\n")
        cols = [f"t+{L}" for L in td_leads]
        header = f"{'model':<50}  " + "  ".join(f"{c:>8}" for c in cols)
        print(header)
        print("-" * len(header))
        for cand_dir in args.candidates:
            name = os.path.basename(cand_dir.rstrip("/"))
            try:
                pred, truth, _ = load_run(cand_dir)
            except Exception as e:
                print(f"{name:<50}  load error: {e}")
                continue
            cells = [
                f"{per_pixel_tendency_correlation(pred, truth, L)['mean']:>8.4f}"
                for L in td_leads
            ]
            print(f"{name:<50}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
