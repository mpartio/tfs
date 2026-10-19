#!/usr/bin/env python3
"""15-min output eyeball test for a direct-prediction CFM checkpoint.

The direct model has no lead embedding: each lead is decoded from the shared
analysis encode and the forcing VALID AT THAT LEAD (swinu/util.py
direct_flow_forecast). So sub-hourly output only needs sub-hourly forcing.
This script bypasses the hourly dataloader and feeds arbitrary valid times.

Inputs: an anemoi zarr on a 15-min axis (built from the operational
prepare-data.py path, see scratchpad mnwc15/prep_15min.py + recipe15.yaml).
Normalisation and static forcings come from cc2data.AnemoiDataset itself, with
the checkpoint's statistics (mapped by variable name).

insolation: earthkit's cos_solar_zenith_angle uses the INTEGER hour, so on a
15-min axis the channel is a staircase. Training only saw whole hours, where
the formula is exact. --insolation continuous recomputes it with the fractional
hour (identical at whole hours, asserted).

Noise: direct_flow_forecast draws an independent initial noise per lead (and,
with eta > 0, independent per-step noise). --noise shared uses ONE noise field
for every lead (and the same per-step noise), which should give temporally
coherent frames.

Writes <out>.pt with predictions [L, H, W] (lead 0 = analysis), valid times and
the run settings, plus frame-to-frame change diagnostics in <out>.json.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cc2trainer  # noqa: F401  (registers numpy safe globals for torch.load)
from dataloader.cc2data import AnemoiDataset, get_default_normalization_methods
from swinu.cc2 import cc2model

MODEL_KEYS = [
    "history_length", "hidden_dim", "patch_size", "encoder1_depth", "encoder2_depth",
    "decoder1_depth", "decoder2_depth", "window_size", "window_size_deep", "num_heads",
    "mlp_ratio", "drop_rate", "attn_drop_rate", "drop_path_rate", "use_gradient_checkpointing",
    "use_scheduled_sampling", "ss_pred_min", "ss_pred_max", "use_flow_matching",
    "direct_prediction",
]


def load_model(ckpt_path, input_resolution, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp, dh = ck["hyper_parameters"], ck["datamodule_hyper_parameters"]
    kw = {k: hp[k] for k in MODEL_KEYS}
    kw["use_gradient_checkpointing"] = False
    kw.update(input_resolution=input_resolution, prognostic_params=dh["prognostic_params"],
              forcing_params=dh["forcing_params"], static_forcing_params=dh["static_forcing_params"],
              preprocessor=None)
    model = cc2model(config=kw)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    res = model.load_state_dict(sd, strict=True)
    print("weights:", res)
    return model.to(device).eval(), ck, hp, dh


def stats_in_zarr_order(ck, zarr_n2i):
    n2i = ck["data_statististics_name_to_index"]
    st = ck["data_statistics"]
    out = {}
    for key, arr in st.items():
        arr = np.asarray(arr)
        new = np.full(len(zarr_n2i), np.nan)
        for name, j in zarr_n2i.items():
            new[j] = arr[n2i[name]]
        out[key] = new
    return out


def frames(ds, idx):
    """cc2data.AnemoiDataset.__getitem__ for an arbitrary list of time indices."""
    raw = np.stack([ds.data[i, ds.combined_dynamic_indexes, ...] for i in idx])
    T = raw.shape[0]
    raw = torch.from_numpy(raw.reshape(T, len(ds.combined_dynamic_indexes), *ds.input_resolution))
    norm = ds.normalize(raw, ds.combined_dynamic_params)
    npg = len(ds.data_indexes)
    prog, dyn = norm[:, :npg], norm[:, npg:]
    sf = ds.static_forcings
    if sf.shape[-1] > 2000:
        sf = sf.reshape(sf.shape[0], sf.shape[1], *ds.input_resolution)
    forcing = torch.cat((dyn, sf.repeat(T, 1, 1, 1)), dim=1)
    return prog, forcing


def continuous_insolation(dates, lat, lon):
    from earthkit.meteo.solar.array import solar_declination_angle
    out = []
    for d in dates:
        d = d.astype("datetime64[s]").astype(dt.datetime)
        decl, tc = solar_declination_angle(d)
        decl = np.deg2rad(decl)
        la = np.deg2rad(lat)
        hour = d.hour + d.minute / 60.0 + d.second / 3600.0
        ang = np.deg2rad((hour - 12) * 15 + lon + tc)
        z = np.sin(decl) * np.sin(la) + np.cos(decl) * np.cos(la) * np.cos(ang)
        out.append(np.clip(z, 0, None))
    return np.stack(out)


@torch.no_grad()
def forecast(model, x, forcing, n_lead, K, eta, sigma, noise, seed, device):
    """swinu.util.direct_flow_forecast with controllable noise sharing."""
    B, T, C, H, W = x.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    extra = torch.zeros(B, forcing.shape[1], 2, H, W, device=device, dtype=forcing.dtype)
    ff_all = torch.cat([forcing, extra], dim=2)
    analysis, analysis_curr = x[:, -T:], x[:, -1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded, skip, pad = model.encode_context(analysis, ff_all[:, :T])
    shape = analysis_curr.shape
    shared0 = torch.randn(shape, generator=gen, device=device)
    shared_eta = [torch.randn(shape, generator=gen, device=device) for _ in range(K)]
    alphas = torch.linspace(1.0, 0.05, K + 1, device=device)
    preds = []
    for i in range(n_lead):
        ff = ff_all[:, T + i:T + i + 1].clone()
        x_alpha = sigma * (shared0 if noise == "shared" else torch.randn(shape, generator=gen, device=device))
        x1 = analysis_curr
        for k in range(K):
            a, an = alphas[k], alphas[k + 1]
            ff[:, 0, -2] = x_alpha[:, 0, 0]
            ff[:, 0, -1] = float(a)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tend = model.decode_lead(encoded, skip, pad, ff, i)
            x1 = analysis_curr + tend.float()
            ac = a.clamp(min=1e-6)
            z_est = (x_alpha - (1.0 - ac) * x1) / ac
            x_alpha = (1.0 - an) * x1 + an * z_est
            if eta > 0:
                e = shared_eta[k] if noise == "shared" else torch.randn(shape, generator=gen, device=device)
                x_alpha = x_alpha + eta * torch.sqrt(an) * e
        preds.append(x1.clamp(0, 1)[0, 0, 0].float().cpu())
    return torch.stack(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--static", default="/x/datasets/cc2/meps-const-v4.zarr")
    p.add_argument("--analysis", default="2026-10-14T09:00")
    p.add_argument("--step-min", type=int, default=15, help="lead spacing in minutes (60 = hourly)")
    p.add_argument("--last", default="2026-10-14T20:00")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--noise", choices=["independent", "shared"], default="independent")
    p.add_argument("--insolation", choices=["continuous", "staircase"], default="continuous")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = "cuda"
    from anemoi.datasets import open_dataset
    z = open_dataset(a.zarr)
    model, ck, hp, dh = load_model(a.ckpt, tuple(z.field_shape), device)
    stats = stats_in_zarr_order(ck, z.name_to_index)
    ds = AnemoiDataset(
        zarr_path=[a.zarr], group_size=1, prognostic_params=dh["prognostic_params"],
        forcing_params=dh["forcing_params"], static_forcing_path=a.static,
        static_forcing_params=dh["static_forcing_params"],
        normalization_methods=get_default_normalization_methods(dh["normalization"]),
        statistics=stats, history_length=hp["history_length"])

    dates = np.array(z.dates)
    t0 = np.datetime64(a.analysis)
    hist = [t0 - np.timedelta64(60 * (hp["history_length"] - 1 - j), "m") for j in range(hp["history_length"])]
    leads = np.arange(t0 + np.timedelta64(a.step_min, "m"), np.datetime64(a.last) + np.timedelta64(1, "m"),
                      np.timedelta64(a.step_min, "m"))
    want = list(hist) + list(leads)
    idx = [int(np.where(dates == t)[0][0]) for t in want]
    prog, forcing = frames(ds, idx)
    assert not torch.isnan(prog[: len(hist)]).any(), "NaN in analysis frames"
    assert not torch.isnan(forcing).any(), "NaN in forcing"

    if a.insolation == "continuous":
        j = dh["forcing_params"].index("insolation")
        lat = np.asarray(z.latitudes).reshape(z.field_shape)
        lon = np.asarray(z.longitudes).reshape(z.field_shape)
        ins = torch.from_numpy(continuous_insolation(np.array(want), lat, lon)).float()
        whole = [k for k, t in enumerate(want) if t.astype("datetime64[m]").astype(int) % 60 == 0]
        d = (ins[whole] - forcing[whole, j]).abs().max().item()
        assert d < 1e-4, f"continuous insolation differs from zarr at whole hours by {d}"
        print(f"insolation: continuous (max |diff| vs zarr at whole hours {d:.2e})")
        forcing[:, j] = ins

    x = prog[: len(hist)].unsqueeze(0).to(device)          # [1, T, 1, H, W]
    f = forcing.unsqueeze(0).to(device)                     # [1, T+L, Cf, H, W]
    pred = forecast(model, x, f, len(leads), a.K, a.eta, a.sigma, a.noise, a.seed, device)
    pred = torch.cat([prog[len(hist) - 1, 0].unsqueeze(0).float(), pred])   # lead 0 = analysis
    valid = [str(t0)] + [str(t) for t in leads]

    d1 = (pred[1:] - pred[:-1]).abs().mean(dim=(1, 2)).tolist()
    diag = {"valid": valid, "mean_tcc": pred.mean(dim=(1, 2)).tolist(),
            "mean_abs_change_from_previous_frame": [None] + d1,
            "settings": vars(a), "ckpt_global_step": int(ck.get("global_step", -1))}
    torch.save({"predictions": pred, "valid": valid, "settings": vars(a)}, a.out + ".pt")
    json.dump(diag, open(a.out + ".json", "w"), indent=1)
    print(f"wrote {a.out}.pt {tuple(pred.shape)}; mean |frame change| {np.mean(d1):.4f}")


if __name__ == "__main__":
    main()
