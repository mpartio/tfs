#!/usr/bin/env python3
"""Read-only scan of an anemoi cc2 zarr for dead / suspicious frames.

Per time step (declared-missing dates are skipped):
  tcc: NaN fraction, mean, std, fraction exactly 0 / 1 / 0.5, and a hash to find
       frames identical to the previous non-missing frame.
  every other variable: NaN count and spatial std (frozen field), except the
       spatially constant clock fields (cos/sin_julian_day) and insolation (can
       legitimately be all 0 at night).
Flags:
  tcc_all_zero   > 99 % of pixels exactly 0
  tcc_all_one    > 99 % exactly 1
  tcc_constant   std < 1e-6
  tcc_nan        > 1 % NaN on a date not declared missing
  tcc_half_fill  > 5 % exactly 0.5 (the NWCSAF 0.5-fill)
  tcc_duplicate  bit-identical to the previous available frame
  tcc_spike      domain mean differs from BOTH neighbours by > 0.12 in the same direction
  forcing_nan / forcing_frozen  per variable
Writes <out>.json (flags) and <out>.npz (per-frame tcc stats).
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import zarr

SKIP_CONST = {"cos_julian_day", "sin_julian_day", "insolation"}
_D = None


def _init(path):
    global _D
    _D = zarr.open(path, mode="r")["data"]


def _chunk(args):
    t0, t1, itcc, other = args
    blk = np.asarray(_D[t0:t1, :, 0, :])          # [t, var, grid]
    out = []
    for k in range(blk.shape[0]):
        f = blk[k, itcc]
        nan = np.isnan(f)
        g = f[~nan]
        rec = {
            "t": t0 + k, "nan": float(nan.mean()),
            "mean": float(g.mean()) if g.size else np.nan, "std": float(g.std()) if g.size else np.nan,
            "f0": float((g == 0).mean()) if g.size else np.nan, "f1": float((g == 1).mean()) if g.size else np.nan,
            "f05": float((g == 0.5).mean()) if g.size else np.nan,
            "hash": hashlib.md5(f.tobytes()).hexdigest(),
            "forcing": [],
        }
        for name, j in other:
            x = blk[k, j]
            n_nan = int(np.isnan(x).sum())
            if n_nan:
                rec["forcing"].append((name, "nan", n_nan))
            elif name not in SKIP_CONST and float(np.std(x)) < 1e-9:
                rec["forcing"].append((name, "frozen", 0))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    from anemoi.datasets import open_dataset
    ds = open_dataset(a.zarr)
    n2i = ds.name_to_index
    dates = np.array(ds.dates)
    missing = set(int(m) for m in ds.missing)
    d = zarr.open(a.zarr, mode="r")["data"]
    T, ct = d.shape[0], d.chunks[0]
    itcc = n2i["tcc"]
    other = [(n, j) for n, j in n2i.items() if n != "tcc"]
    jobs = [(t, min(t + ct, T), itcc, other) for t in range(0, T, ct)]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.zarr,)) as ex:
        recs = [r for chunk in ex.map(_chunk, jobs, chunksize=8) for r in chunk]
    recs.sort(key=lambda r: r["t"])

    flags = []
    def flag(t, kind, detail=""):
        flags.append({"index": t, "date": str(dates[t]), "flag": kind, "detail": detail})

    avail = [r for r in recs if r["t"] not in missing]
    for i, r in enumerate(avail):
        t = r["t"]
        if r["nan"] > 0.01:
            flag(t, "tcc_nan", f"{r['nan']:.3f}")
        if r["f0"] > 0.99:
            flag(t, "tcc_all_zero", f"{r['f0']:.4f}")
        if r["f1"] > 0.99:
            flag(t, "tcc_all_one", f"{r['f1']:.4f}")
        if r["std"] < 1e-6:
            flag(t, "tcc_constant", f"mean {r['mean']:.3f}")
        if r["f05"] > 0.05:
            flag(t, "tcc_half_fill", f"{r['f05']:.3f}")
        if i and r["hash"] == avail[i - 1]["hash"]:
            flag(t, "tcc_duplicate", f"same as {dates[avail[i - 1]['t']]}")
        if 0 < i < len(avail) - 1:
            dp, dn = r["mean"] - avail[i - 1]["mean"], r["mean"] - avail[i + 1]["mean"]
            if abs(dp) > 0.12 and abs(dn) > 0.12 and np.sign(dp) == np.sign(dn):
                flag(t, "tcc_spike", f"mean {avail[i-1]['mean']:.3f} -> {r['mean']:.3f} -> {avail[i+1]['mean']:.3f}")
        for name, kind, n in r["forcing"]:
            flag(t, f"forcing_{kind}", f"{name} {n}")

    summary = {}
    for f in flags:
        summary[f["flag"]] = summary.get(f["flag"], 0) + 1
    json.dump({"zarr": a.zarr, "n_frames": T, "n_declared_missing": len(missing), "summary": summary,
               "flags": flags}, open(a.out + ".json", "w"), indent=1)
    np.savez(a.out + ".npz", t=np.array([r["t"] for r in recs]), mean=np.array([r["mean"] for r in recs]),
             std=np.array([r["std"] for r in recs]), nan=np.array([r["nan"] for r in recs]),
             f0=np.array([r["f0"] for r in recs]), f1=np.array([r["f1"] for r in recs]),
             f05=np.array([r["f05"] for r in recs]), dates=dates.astype("datetime64[s]").astype(np.int64))
    print(f"{T} frames, {len(missing)} declared missing; flags: {summary}")


if __name__ == "__main__":
    main()
