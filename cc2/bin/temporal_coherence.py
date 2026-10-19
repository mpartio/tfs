#!/usr/bin/env python3
"""Temporal coherence of a model's hourly test output, against its own truth.

For every case and every consecutive pair of frames (t+k -> t+k+1, k = 0..11;
frame 0 is the analysis) it measures, for the forecast f and the truth o:

  mad    mean |x_{k+1} - x_k|                 (how much the field changes per hour)
  corr   Pearson(x_k, x_{k+1})                (frame-to-frame pattern persistence)
  chg_r  Pearson(f_{k+1}-f_k, o_{k+1}-o_k)    (does the forecast change where/how the truth changes)

at three scales: raw, and after a Gaussian low-pass of sigma = 3 px (~15 km)
and 8 px (~40 km). The low-pass separates sampled fine detail from the
large-scale evolution.

Displacement (subsample of cases): per 128x128 tile, the shift that best maps
x_k onto x_{k+1} on the 15-km-smoothed fields (FFT cross-correlation, not
whitened, parabolic sub-pixel), for forecast and truth at the same valid times.
Reports forecast/truth speed ratio and the angle between the two shift vectors.
No wind or NWP field is used.

Inputs are the standard test-output directories (predictions.pt / truth.pt,
[N, 13, H, W]); they are memory-mapped, never loaded whole.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SIGMAS = (0, 3, 8)
TILE = 128
_P = _T = None


def _init(pred_path, truth_path):
    global _P, _T
    _P = torch.load(pred_path, mmap=True, weights_only=True)
    _T = torch.load(truth_path, mmap=True, weights_only=True)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _shift(a, b):
    a = a - a.mean()
    b = b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    aw, bw = a * w, b * w
    c = np.real(np.fft.ifft2(np.fft.fft2(bw) * np.conj(np.fft.fft2(aw))))
    c /= np.sqrt((aw * aw).sum() * (bw * bw).sum()) + 1e-12
    i, j = np.unravel_index(np.argmax(c), c.shape)
    n0, n1 = c.shape

    def sub(c0, cm, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if d == 0 else 0.5 * (cm - cp) / d
    dy = i + sub(c[i, j], c[i - 1, j], c[(i + 1) % n0, j])
    dx = j + sub(c[i, j], c[i, j - 1], c[i, (j + 1) % n1])
    if dy > n0 / 2:
        dy -= n0
    if dx > n1 / 2:
        dx -= n1
    return dy, dx, float(c.max())


def _case(args):
    c, do_disp = args
    f = _P[c].numpy().astype(np.float32).reshape(_P.shape[1], *_P.shape[-2:])   # [L, (1,) H, W] -> [L, H, W]
    o = _T[c].numpy().astype(np.float32).reshape(_T.shape[1], *_T.shape[-2:])
    L = f.shape[0]
    out = {s: {"mad_f": [], "mad_o": [], "corr_f": [], "corr_o": [], "chg_r": []} for s in SIGMAS}
    sm = {}
    for s in SIGMAS:
        fs = np.stack([gaussian_filter(x, s) if s else x for x in f])
        os_ = np.stack([gaussian_filter(x, s) if s else x for x in o])
        sm[s] = (fs, os_)
        for k in range(L - 1):
            df, do = fs[k + 1] - fs[k], os_[k + 1] - os_[k]
            out[s]["mad_f"].append(float(np.abs(df).mean()))
            out[s]["mad_o"].append(float(np.abs(do).mean()))
            out[s]["corr_f"].append(_pearson(fs[k], fs[k + 1]))
            out[s]["corr_o"].append(_pearson(os_[k], os_[k + 1]))
            out[s]["chg_r"].append(_pearson(df, do))
    disp = []
    if do_disp:
        fs, os_ = sm[3]
        H, W = fs.shape[1:]
        for k in (1, 5, 10):
            for y0 in range(0, H - TILE + 1, TILE):
                for x0 in range(0, W - TILE + 1, TILE):
                    sl = (slice(y0, y0 + TILE), slice(x0, x0 + TILE))
                    a_f, b_f, a_o, b_o = fs[k][sl], fs[k + 1][sl], os_[k][sl], os_[k + 1][sl]
                    if min(a_f.std(), b_f.std(), a_o.std(), b_o.std()) < 0.12:
                        continue
                    fy, fx, pf = _shift(a_f, b_f)
                    oy, ox, po = _shift(a_o, b_o)
                    if pf < 0.5 or po < 0.5:
                        continue
                    disp.append((k, fy, fx, oy, ox))
    return c, out, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th case")
    ap.add_argument("--disp-stride", type=int, default=7, help="displacement on every n-th used case")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--match-dir", default=None,
                    help="only use cases whose analysis time is also in <match-dir>/dates.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pp, tp = os.path.join(a.dir, "predictions.pt"), os.path.join(a.dir, "truth.pt")
    _init(pp, tp)
    N, L = _P.shape[0], _P.shape[1]
    cases = list(range(0, N, a.stride))
    if a.match_dir:
        mine = torch.load(os.path.join(a.dir, "dates.pt"), weights_only=True)[:, 0].numpy()
        want = set(torch.load(os.path.join(a.match_dir, "dates.pt"), weights_only=True)[:, 0].numpy().tolist())
        cases = [c for c in cases if mine[c] in want]
    jobs = [(c, i % a.disp_stride == 0) for i, c in enumerate(cases)]
    with mp.Pool(a.workers, initializer=_init, initargs=(pp, tp)) as pool:
        res = pool.map(_case, jobs, chunksize=4)

    summary = {"label": a.label, "dir": a.dir, "n_cases": len(cases), "pairs": [f"t+{k}->t+{k+1}" for k in range(L - 1)]}
    for s in SIGMAS:
        d = {}
        for key in ("mad_f", "mad_o", "corr_f", "corr_o", "chg_r"):
            arr = np.array([r[1][s][key] for r in res])        # [cases, L-1]
            d[key] = np.nanmean(arr, axis=0).tolist()
        d["mad_ratio"] = (np.array(d["mad_f"]) / np.array(d["mad_o"])).tolist()
        summary[f"sigma{s}"] = d
    disp = np.array([x for r in res for x in r[2]])
    if len(disp):
        dd = {}
        for k in (1, 5, 10):
            m = disp[disp[:, 0] == k]
            if not len(m):
                continue
            sf, so = np.hypot(m[:, 1], m[:, 2]), np.hypot(m[:, 3], m[:, 4])
            keep = so > 0.5                                      # truth actually moved
            ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(m[:, 1], m[:, 2]) - np.arctan2(m[:, 3], m[:, 4]))))))
            dd[f"t+{k}->t+{k+1}"] = {
                "n_tiles": int(len(m)), "median_speed_fc_px_h": float(np.median(sf)),
                "median_speed_obs_px_h": float(np.median(so)),
                "median_ratio_fc_obs": float(np.median(sf[keep] / so[keep])) if keep.any() else None,
                "median_angle_deg": float(np.median(ang[keep])) if keep.any() else None}
        summary["displacement_sigma3"] = dd
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(cases)} cases, {len(disp)} displacement tiles)")


if __name__ == "__main__":
    main()
