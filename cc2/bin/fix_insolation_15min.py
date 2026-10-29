#!/usr/bin/env python3
"""Recompute `insolation` IN PLACE in a sub-hourly anemoi cc2 zarr.

anemoi's `insolation` forcing is earthkit.meteo's cos_solar_zenith_angle, whose
solar hour angle uses the integer `date.hour`: at :15/:30/:45 it returns the
value of the preceding full hour (only the day-of-year term moves), i.e. an
hourly staircase. This script recomputes the field with the fractional hour,
otherwise reproducing earthkit's formula operation for operation, so full-hour
frames come out bit-identical (they are left untouched anyway) and sub-hourly
frames get the true solar angle. Only `insolation` is rewritten; tcc and the
clock features (which earthkit already computes from the full timestamp) are
read and written back unchanged. Chunk files are replaced atomically by zarr.
The insolation entries of the statistics arrays are recomputed over
[statistics_start_date, statistics_end_date] from the corrected field.

Dry run by default (reports the change on a sample day); pass --apply to write.

  python3 fix_insolation_15min.py /x/datasets/cc2/nwcsaf-475x535-2024-2026-15min-v5.zarr --apply
"""
import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import zarr

DAYS_PER_YEAR = 365.25


def cos_sza(date, lat, lon):
    """earthkit.meteo.solar.array.cos_solar_zenith_angle with the fractional hour."""
    delta = date - dt.datetime(date.year, 1, 1)
    angle = (delta.days + delta.seconds / 86400.0) / DAYS_PER_YEAR * np.pi * 2
    declination = (0.396372 - 22.91327 * np.cos(angle) + 4.025430 * np.sin(angle)
                   - 0.387205 * np.cos(2 * angle) + 0.051967 * np.sin(2 * angle)
                   - 0.154527 * np.cos(3 * angle) + 0.084798 * np.sin(3 * angle))
    time_correction = (0.004297 + 0.107029 * np.cos(angle) - 1.837877 * np.sin(angle)
                       - 0.837378 * np.cos(2 * angle) - 2.340475 * np.sin(2 * angle))
    declination = np.deg2rad(declination)
    latr = np.deg2rad(lat)
    hour = date.hour if date.minute == 0 and date.second == 0 else date.hour + date.minute / 60 + date.second / 3600
    solar_angle = np.deg2rad((hour - 12) * 15 + lon + time_correction)
    z = np.sin(declination) * np.sin(latr) + np.cos(declination) * np.cos(latr) * np.cos(solar_angle)
    return np.clip(z, 0, None)


def to_dt(t):
    return dt.datetime.fromisoformat(str(t.astype("datetime64[s]")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("--sample-day", default="2025-06-10")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = zarr.open(a.zarr, mode="r+" if a.apply else "r")
    d = root["data"]
    dates = np.asarray(root["dates"]).astype("datetime64[s]")
    names = list(root.attrs["variables"])
    ii = names.index("insolation")
    lat, lon = np.asarray(root["latitudes"]), np.asarray(root["longitudes"])
    ct = d.chunks[0]

    # sample-day report + full-hour reproduction check
    on_day = np.where(dates.astype("datetime64[D]") == np.datetime64(a.sample_day))[0]
    assert len(on_day), f"--sample-day {a.sample_day} not in the dataset ({dates[0]} .. {dates[-1]})"
    k0, k1 = int(on_day[0]), int(on_day[-1]) + 1
    old = np.asarray(d[k0:k1, ii, 0, :])
    new = np.stack([cos_sza(to_dt(t), lat, lon).astype(d.dtype) for t in dates[k0:k1]])
    full = np.array([to_dt(t).minute == 0 for t in dates[k0:k1]])
    print(f"{a.sample_day}: full-hour frames reproduced bit-identically: "
          f"{np.array_equal(old[full], new[full])}; max |new-old| at sub-hourly frames {np.abs(new[~full] - old[~full]).max():.4f}")
    g = 267 * 475 + 237
    print("  mid-domain old:", np.round(old[24:32, g], 4))
    print("  mid-domain new:", np.round(new[24:32, g], 4))
    if not np.array_equal(old[full], new[full]):
        raise SystemExit("full-hour frames NOT reproduced -- formula differs from the one that built the zarr; aborting")
    if not a.apply:
        print("dry run; pass --apply to write")
        return

    s0 = np.datetime64(root.attrs["statistics_start_date"]); s1 = np.datetime64(root.attrs["statistics_end_date"])
    missing = np.array(root.attrs.get("missing_dates", []), dtype="datetime64[s]")   # NaN frames, excluded like anemoi does
    n = 0; sm = 0.0; sq = 0.0; mn = np.inf; mx = -np.inf
    t0 = time.time()
    for c in range(0, len(dates), ct):
        sl = slice(c, min(c + ct, len(dates)))
        blk = np.asarray(d[sl])                       # [t, var, 1, grid]
        changed = False
        for k, t in enumerate(dates[sl]):
            T = to_dt(t)
            if T.minute or T.second:
                blk[k, ii, 0] = cos_sza(T, lat, lon)
                changed = True
        if changed:
            d[sl] = blk
        inwin = (dates[sl] >= s0) & (dates[sl] <= s1) & ~np.isin(dates[sl], missing)
        if inwin.any():
            x = blk[inwin, ii, 0].astype(np.float64)
            n += x.size; sm += x.sum(); sq += (x * x).sum(); mn = min(mn, x.min()); mx = max(mx, x.max())
        if (c // ct) % 200 == 0:
            print(f"  {dates[sl][0]}  {time.time() - t0:.0f}s", flush=True)
    mean = sm / n; std = np.sqrt(max(sq / n - mean * mean, 0.0))
    for key, val in (("mean", mean), ("stdev", std), ("minimum", mn), ("maximum", mx), ("sums", sm), ("squares", sq)):
        arr = root[key]; v = np.asarray(arr[:]); old_v = v[ii]; v[ii] = val; arr[:] = v
        print(f"  statistics {key}[insolation]: {old_v:.6g} -> {val:.6g}")
    root.attrs["insolation_fix"] = json.dumps({"script": os.path.basename(__file__), "applied_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
                                               "note": "sub-hourly insolation recomputed with the fractional hour; full hours untouched"})
    print(f"applied in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
