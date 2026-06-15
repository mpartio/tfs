"""
Multi-case dynamic-event regression gate.

Picks the top-quintile init times by how much the truth field evolves over the
forecast horizon, then scores how well each candidate model captures those
dynamic events compared to a reference (typically trusting-radar, the model we
visually confirmed retains dynamic structure).

Use this as a gate when evaluating new refinement protocols: a model that wins
on aggregate composite but fails the dynamic-case gate is smoothing over the
events that matter operationally.

Usage:
    python3 verif/dynamic_case_gate.py \
        --reference /data/tfs/runs/ED12h/trusting-radar-12h \
        --candidate /data/tfs/runs/ED12h/candied-circle-aclamp85-12h \
        --top_quintile 0.20 \
        --capture_tolerance 0.10

Outputs a per-case capture/miss table and the overall capture rate. Default
gate is 70% capture rate over the top-quintile dynamic cases.

How "dynamic" is defined:
    D(init_time) = MSE(truth[t+12], truth[t+0])
    Sorted descending; top-quintile cases retained.

How "captured" is defined:
    A model is captured on case i if
        MSE(pred_i[t+12], truth_i[t+12]) <= (1+tol) * MSE(ref_i[t+12], truth_i[t+12])
    i.e. the candidate is within `tol` (default 10%) of the reference's pixel
    accuracy at lead t+12 on the dynamic case.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_run(run_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load predictions, truth, dates from a run directory.

    Expected files: predictions.pt, truth.pt, dates.pt
    Shapes: predictions/truth = [N, T, C, H, W]; dates = [N, T] (unix seconds)
    """
    p = torch.load(f"{run_dir}/predictions.pt", weights_only=False, map_location="cpu")
    t = torch.load(f"{run_dir}/truth.pt", weights_only=False, map_location="cpu")
    d = torch.load(f"{run_dir}/dates.pt", weights_only=False, map_location="cpu")
    return p, t, d


def intersect_runs(*runs):
    """Filter all runs to the intersection of their analysis times (column 0 of dates)."""
    date_sets = []
    for _, _, d in runs:
        # use the analysis time (first lead column) as the index key
        date_sets.append({float(x): i for i, x in enumerate(d[:, 0].tolist())})
    common = set(date_sets[0].keys())
    for s in date_sets[1:]:
        common &= set(s.keys())
    common_sorted = sorted(common)

    filtered = []
    for (p, t, d), s in zip(runs, date_sets):
        idx = [s[k] for k in common_sorted]
        filtered.append((p[idx], t[idx], d[idx]))
    return filtered, common_sorted


def compute_D(truth: torch.Tensor) -> np.ndarray:
    """D(init) = MSE(truth[t+12], truth[t+0]) per init time."""
    # truth shape: [N, T, C, H, W]. T=13 (lead 0..12). Lead 0 is analysis time, lead 12 is t+12.
    t0 = truth[:, 0]
    t12 = truth[:, -1]
    sq_diff = (t12 - t0) ** 2
    return sq_diff.float().mean(dim=(-1, -2, -3)).numpy()  # [N]


def case_mse(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """MSE(pred[t+12], truth[t+12]) per init time."""
    return ((pred[:, -1] - truth[:, -1]) ** 2).float().mean(dim=(-1, -2, -3)).numpy()


def case_tendency_corr(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """Per-case Pearson correlation between pred-delta and truth-delta.

    Tendency = pred[t+12] - pred[t+0] vs truth[t+12] - truth[t+0].
    Positive means the model got the direction-of-change right.

    Added (Phase-1, verificationist veto): guards against the "high-variance
    model wins gate via MSE averaging" failure mode. A captured case must
    have positive tendency correlation, i.e., not just be MSE-close to
    reference by accident of variance.
    """
    pred_delta = (pred[:, -1] - pred[:, 0]).float()  # [N, C, H, W]
    truth_delta = (truth[:, -1] - truth[:, 0]).float()
    N = pred_delta.shape[0]
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        p = pred_delta[i].flatten()
        t = truth_delta[i].flatten()
        # Pearson correlation; guard against zero-variance cases
        if p.std() < 1e-8 or t.std() < 1e-8:
            out[i] = 0.0
        else:
            out[i] = float(torch.corrcoef(torch.stack([p, t]))[0, 1])
    return out


def classify_event(truth: torch.Tensor, idx: int) -> str:
    """Classify the dynamic event at one init time.

    Simple proxy:
        delta_mean = mean(truth[t+12]) - mean(truth[t+0])
    If delta_mean > +0.02: genesis (more cloud).
    If delta_mean < -0.02: lysis (less cloud).
    Else: translation (mean roughly preserved; structure changed).
    """
    t0 = truth[idx, 0].float().mean().item()
    t12 = truth[idx, -1].float().mean().item()
    dm = t12 - t0
    if dm > 0.02:
        return "genesis"
    if dm < -0.02:
        return "lysis"
    return "translation"


def run_gate(
    reference_dir: str,
    candidate_dir: str,
    top_quintile: float = 0.20,
    capture_tolerance: float = 0.10,
    gate_threshold: float = 0.70,
) -> dict:
    """Compute the dynamic-case gate.

    Returns dict with capture_rate, n_cases, per-event-type breakdown, passed.
    """
    ref = load_run(reference_dir)
    cand = load_run(candidate_dir)
    (ref, cand), common_times = intersect_runs(ref, cand)

    ref_p, ref_t, ref_d = ref
    cand_p, cand_t, cand_d = cand

    # Sanity: truth tensors should be equal up to filtering noise (occasional NaN-filled rows)
    # but we use cand_t for the D calculation since cand has the same intersected set
    D = compute_D(cand_t)  # [N]
    n_total = len(D)
    n_top = max(1, int(round(top_quintile * n_total)))
    top_idx = np.argsort(D)[::-1][:n_top]

    # Sort top_idx by D descending for nicer reporting
    top_idx_sorted = top_idx[np.argsort(-D[top_idx])]

    ref_mse = case_mse(ref_p, ref_t)
    cand_mse = case_mse(cand_p, cand_t)
    # Phase-1: tendency-correlation guard against high-variance gaming
    cand_tendcorr = case_tendency_corr(cand_p, cand_t)

    captured = np.zeros(n_top, dtype=bool)
    event_types = []
    rows = []
    for i, idx in enumerate(top_idx_sorted):
        ref_m = ref_mse[idx]
        cand_m = cand_mse[idx]
        tendcorr = cand_tendcorr[idx]
        # Capture criteria: MSE proximity AND positive tendency correlation
        # (the latter blocks the "noisy model wins by variance averaging" mode)
        mse_ok = cand_m <= (1.0 + capture_tolerance) * ref_m
        tend_ok = tendcorr > 0.0
        is_captured = mse_ok and tend_ok
        captured[i] = is_captured
        etype = classify_event(cand_t, idx)
        event_types.append(etype)
        rows.append(
            (
                int(idx),
                float(common_times[idx]),
                float(D[idx]),
                float(ref_m),
                float(cand_m),
                float(cand_m / max(ref_m, 1e-12)),
                float(tendcorr),
                etype,
                bool(is_captured),
            )
        )

    capture_rate = captured.mean()
    by_event = {}
    for etype in {"genesis", "lysis", "translation"}:
        mask = np.array([t == etype for t in event_types])
        if mask.sum() > 0:
            by_event[etype] = {
                "n": int(mask.sum()),
                "rate": float(captured[mask].mean()),
            }

    return {
        "n_total_cases": int(n_total),
        "n_dynamic_cases": int(n_top),
        "capture_rate": float(capture_rate),
        "gate_threshold": gate_threshold,
        "passed": bool(capture_rate >= gate_threshold),
        "by_event": by_event,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Reference run dir (e.g. trusting-radar)")
    parser.add_argument("--candidate", required=True, help="Candidate run dir to evaluate")
    parser.add_argument("--top_quintile", type=float, default=0.20)
    parser.add_argument("--capture_tolerance", type=float, default=0.10)
    parser.add_argument("--gate_threshold", type=float, default=0.70)
    parser.add_argument("--verbose", action="store_true", help="Print every case")
    args = parser.parse_args()

    result = run_gate(
        args.reference,
        args.candidate,
        top_quintile=args.top_quintile,
        capture_tolerance=args.capture_tolerance,
        gate_threshold=args.gate_threshold,
    )

    print(f"\nDynamic-case gate result")
    print(f"------------------------")
    print(f"Reference:  {args.reference}")
    print(f"Candidate:  {args.candidate}")
    print(f"")
    print(f"Total init times (intersected): {result['n_total_cases']}")
    print(f"Top quintile retained:           {result['n_dynamic_cases']} ({args.top_quintile:.0%})")
    print(f"Capture tolerance:               {args.capture_tolerance:.0%}")
    print(f"")
    print(f"OVERALL CAPTURE RATE:  {result['capture_rate']:.1%}")
    print(f"Gate threshold:        {result['gate_threshold']:.0%}")
    print(f"Status:                {'PASS' if result['passed'] else 'FAIL'}")
    print(f"")
    print(f"By event type:")
    for etype, stats in sorted(result["by_event"].items()):
        print(f"  {etype:<12} n={stats['n']:<3}  rate={stats['rate']:.1%}")

    if args.verbose:
        print(f"\nPer-case detail (sorted by D descending):")
        print(f"  {'idx':>4} {'D':>9} {'ref_mse':>9} {'cand_mse':>9} {'ratio':>7} {'tendcorr':>9} {'event':<12} {'captured'}")
        for idx, _, D_v, ref_m, cand_m, ratio, tendcorr, etype, capt in result["rows"]:
            print(
                f"  {idx:>4} {D_v:>9.5f} {ref_m:>9.5f} {cand_m:>9.5f} {ratio:>7.3f} {tendcorr:>9.3f} {etype:<12} {capt}"
            )


if __name__ == "__main__":
    main()
