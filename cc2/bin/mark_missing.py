#!/usr/bin/env python3
"""Mark dates as missing IN PLACE in an anemoi zarr (cc2 datasets).

anemoi-datasets keeps missing dates in the zarr's .zattrs as "missing_dates"
(ISO strings); open_dataset() then reports them in .missing, raises
MissingDateError on access, and cc2data skips every sample window touching them.
This script
  1. backs up .zattrs and the data chunk files holding those dates,
  2. adds the dates to .zattrs["missing_dates"] (sorted, de-duplicated),
  3. optionally sets the listed variables (default tcc) to NaN at those dates,
     so raw-zarr readers see NaN instead of the bad values. Other variables
     (forcing) are left untouched.
Dry run by default; pass --apply to write.

  python3 mark_missing.py <zarr> 2025-10-06T12:00 2025-11-16T05:00 --apply
"""
import argparse
import datetime as dt
import json
import os
import shutil

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--nan-vars", default="tcc", help="comma list of variables to set NaN ('' = none)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = zarr.open(a.zarr, mode="r+" if a.apply else "r")
    d = root["data"]
    dates = root["dates"][:].astype("datetime64[s]")
    n2i = {v: i for i, v in enumerate(root.attrs["variables"])}
    nan_vars = [v for v in a.nan_vars.split(",") if v]
    old = list(root.attrs.get("missing_dates", []))
    fmt = old[0] if old else "2000-01-01T00:00:00"
    want = [np.datetime64(x, "s") for x in a.dates]
    idx = []
    for w in want:
        hit = np.where(dates == w)[0]
        assert len(hit) == 1, f"{w} not in dataset"
        idx.append(int(hit[0]))
    as_str = [str(w) for w in want]          # 'YYYY-MM-DDTHH:MM:SS', the format anemoi writes
    assert len(fmt) == len(as_str[0]), f"missing_dates format differs: {fmt!r} vs {as_str[0]!r}"
    new = sorted(set(old) | set(as_str))
    print(f"missing_dates: {len(old)} -> {len(new)}  (adding {sorted(set(as_str) - set(old))})")
    for t, w in zip(idx, want):
        vals = {v: float(np.nanmean(d[t, n2i[v], 0, :])) for v in nan_vars}
        print(f"  {w} index {t}: current mean {vals}")
    if not a.apply:
        print("dry run; pass --apply to write")
        return

    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    bak = os.path.join(a.zarr, f".backup-mark-missing-{stamp}")
    os.makedirs(bak)
    shutil.copy2(os.path.join(a.zarr, ".zattrs"), bak)
    ct = d.chunks[0]
    for t in idx:
        key = f"{t // ct}.0.0.0"
        src = os.path.join(a.zarr, "data", key)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bak, "data." + key))
    json.dump({"dates": as_str, "indices": idx, "nan_vars": nan_vars, "old_missing_dates": old},
              open(os.path.join(bak, "README.json"), "w"), indent=1)

    root.attrs["missing_dates"] = new
    for t in idx:
        for v in nan_vars:
            d[t, n2i[v], 0, :] = np.nan
    print(f"applied; backup in {bak}")


if __name__ == "__main__":
    main()
