#!/usr/bin/env python3
"""Score a sub-hourly (e.g. 15-min) test output against its own truth.

Reads <dir>/{predictions,truth,dates}.pt as written by cc2module test_step
([N, 1+R, C, H, W], frame 0 = analysis) and writes per-case, per-lead arrays:

  mae, bias                [N, 1+R]   pixel-wise, tcc
  chg_f, chg_t             [N, R]     mean |frame(k) - frame(k-1)|, forecast / truth (raw pixels)
  chg40_f, chg40_t         [N, R]     the same after 8x8 average pooling (~40 km)
  corr40_f, corr40_t       [N, R]     spatial correlation of consecutive ~40 km frames
  lead_minutes             [1+R]
  init_unix                [N]

Change metrics compare consecutive frames, so the first entry is analysis -> first lead.
Usage:  python3 score_subhourly.py --dir <test-output> --save row.pt
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def unix_seconds(d):
    d = d.numpy().astype(np.int64)
    return d // 10**9 if d.max() > 10**12 else d


def corr(a, b):  # [B, H, W] -> [B]
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()

    P = torch.load(f"{a.dir}/predictions.pt", mmap=True, weights_only=False)
    T = torch.load(f"{a.dir}/truth.pt", mmap=True, weights_only=False)
    d = unix_seconds(torch.load(f"{a.dir}/dates.pt", weights_only=False))
    N, L = P.shape[:2]
    lead_min = (d[0] - d[0, 0]) // 60
    assert all(np.array_equal((d[i] - d[i, 0]) // 60, lead_min) for i in range(N)), "lead layout differs between cases"
    out = {k: np.zeros((N, L)) for k in ("mae", "bias")}
    out.update({k: np.zeros((N, L - 1)) for k in ("chg_f", "chg_t", "chg40_f", "chg40_t", "corr40_f", "corr40_t")})
    for i in range(N):
        p, t = P[i, :, 0].float(), T[i, :, 0].float()               # [L, H, W]
        out["mae"][i] = (p - t).abs().mean((1, 2)).numpy()
        out["bias"][i] = (p - t).mean((1, 2)).numpy()
        p40 = F.avg_pool2d(p[:, None], 8)[:, 0]
        t40 = F.avg_pool2d(t[:, None], 8)[:, 0]
        out["chg_f"][i] = (p[1:] - p[:-1]).abs().mean((1, 2)).numpy()
        out["chg_t"][i] = (t[1:] - t[:-1]).abs().mean((1, 2)).numpy()
        out["chg40_f"][i] = (p40[1:] - p40[:-1]).abs().mean((1, 2)).numpy()
        out["chg40_t"][i] = (t40[1:] - t40[:-1]).abs().mean((1, 2)).numpy()
        out["corr40_f"][i] = corr(p40[1:], p40[:-1]).numpy()
        out["corr40_t"][i] = corr(t40[1:], t40[:-1]).numpy()
    out["lead_minutes"] = lead_min
    out["init_unix"] = d[:, 0]
    torch.save(out, a.save)
    m = out["mae"].mean(0)
    pick = [k for k, v in enumerate(lead_min) if v in (15, 30, 45, 60, 120, 180, 360, 720)]
    print(f"n={N}  MAE  " + "  ".join(f"+{lead_min[k]}m {m[k]:.4f}" for k in pick))
    r = out["chg40_f"].mean(0) / out["chg40_t"].mean(0)
    print("40 km change ratio fcst/truth  " + "  ".join(f"->{lead_min[k]}m {r[k - 1]:.2f}" for k in pick))
    print(f"saved {a.save}")


if __name__ == "__main__":
    main()
