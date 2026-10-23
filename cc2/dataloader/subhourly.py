"""Opt-in sub-hourly samples: sub-hourly truth + time-interpolated hourly forcing.

EXPERIMENT (15-min training, 2026-10): enabled only by `data.subhourly` in the config;
without it cc2data.py is unchanged. To remove: delete this file and the `subhourly`
argument/branches in cc2data.py.

A sample is a set of frame times relative to the analysis time t0:
    history_minutes  e.g. [-60, 0]          (len == model history_length)
    lead_minutes     e.g. [15, 30, ..., 720] candidate leads
    leads_per_sample k: training draws a random sorted k-subset of the candidates per sample,
                     val/test use a fixed subset per window (stable val loss); None = all leads
    init_minutes     analysis-time grid in minutes (60 = full hours only)

Every variable present in the sub-hourly zarr (tcc and the computed forcings: insolation,
clocks) is read from it at the exact frame time. All other variables come from the hourly
zarr (`data.data_path`), linearly interpolated in time between the bracketing hours
(exact at full hours). Variable order, statistics and normalisation are those of the
hourly zarr, so checkpoint statistics apply unchanged. A window is valid only if every
candidate frame exists and is not missing in the sub-hourly zarr and every bracketing
hour is not missing in the hourly zarr.
"""
import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.datasets.data import MissingDateError
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from dataloader.cc2data import AnemoiDataset


def _read_rows(ds, rows, var_idx):
    """{row: [var, grid]} for the needed time rows: one contiguous read (chunk-efficient), or
    row by row when the span crosses a declared-missing date that the sample does not use."""
    rows = np.unique(rows)
    a, b = int(rows.min()), int(rows.max()) + 1
    try:
        blk = ds[a:b, var_idx, 0, :]
        return {int(r): blk[r - a] for r in rows}
    except MissingDateError:
        return {int(r): ds[int(r):int(r) + 1, var_idx, 0, :][0] for r in rows}


class SubHourlyDataset(AnemoiDataset):
    def __init__(self, subhourly: dict, **kwargs):
        cfg = dict(subhourly)
        self.history_minutes = [int(m) for m in cfg["history_minutes"]]
        self.lead_minutes = [int(m) for m in cfg["lead_minutes"]]
        self.leads_per_sample = cfg.get("leads_per_sample")
        self.init_minutes = int(cfg.get("init_minutes", 60))
        k = self.leads_per_sample or len(self.lead_minutes)
        assert len(self.history_minutes) == kwargs["history_length"]
        assert kwargs["group_size"] == len(self.history_minutes) + k, (
            f"rollout_length must equal leads_per_sample (or len(lead_minutes)): group_size "
            f"{kwargs['group_size']} != {len(self.history_minutes)} + {k}")
        assert self.history_minutes[-1] == 0 and sorted(self.lead_minutes) == self.lead_minutes
        assert self.lead_minutes[0] > 0

        # the hourly zarr provides names, statistics, normalisation and static forcings
        super().__init__(**kwargs)

        self.sub = open_dataset(cfg["path"], **self.data_options)
        assert tuple(self.sub.field_shape) == tuple(self.input_resolution), "grid mismatch"
        sub_names = set(self.sub.name_to_index)
        self.names = self.prognostic_params + self.forcing_params
        self.from_sub = [n in sub_names for n in self.names]
        assert self.from_sub[: len(self.prognostic_params)] == [True] * len(self.prognostic_params), (
            "prognostic variables must come from the sub-hourly zarr")
        self.sub_idx = [self.sub.name_to_index[n] for n, s in zip(self.names, self.from_sub) if s]
        self.hour_idx = [self.data.name_to_index[n] for n, s in zip(self.names, self.from_sub) if not s]

        sub_dates = np.asarray(self.sub.dates).astype("datetime64[s]")
        hour_dates = np.asarray(self.data.dates).astype("datetime64[s]")
        self.sub_dates, self.hour_dates = sub_dates, hour_dates
        sub_pos = {t: i for i, t in enumerate(sub_dates.tolist())}
        hour_pos = {t: i for i, t in enumerate(hour_dates.tolist())}
        sub_miss = {int(i) for i in self.sub.missing}
        hour_miss = {int(i) for i in self.data.missing}

        offs = np.array(self.history_minutes + self.lead_minutes, dtype="timedelta64[m]")
        h0 = offs[0]
        valid = []
        for i, start in enumerate(sub_dates):
            t0 = start - h0
            if (t0.astype("datetime64[m]").astype(np.int64) % self.init_minutes) != 0:
                continue
            ok = True
            for tau in (t0 + offs).astype("datetime64[s]").tolist():
                j = sub_pos.get(tau)
                if j is None or j in sub_miss:
                    ok = False
                    break
                lo = np.datetime64(tau, "h").astype("datetime64[s]")
                hi = lo if np.datetime64(tau, "s") == lo else lo + np.timedelta64(1, "h")
                for b in {lo.tolist(), hi.tolist()}:
                    jb = hour_pos.get(b)
                    if jb is None or jb in hour_miss:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid.append(i)
        self.valid_indices = valid
        self.dates = sub_dates           # cc2DataModule.setup: dates[valid_indices] = first history frame
        self.time_steps = len(sub_dates)
        self._sub_pos, self._hour_pos = sub_pos, hour_pos
        rank_zero_info(
            f"SubHourlyDataset: {len(valid)} windows; history {self.history_minutes} min, "
            f"{len(self.lead_minutes)} candidate leads {self.lead_minutes[0]}..{self.lead_minutes[-1]} min, "
            f"{k} per sample, init every {self.init_minutes} min; from sub-hourly zarr: "
            f"{[n for n, s in zip(self.names, self.from_sub) if s]}")

    def _lead_subset(self, window_start_idx: int, random_leads: bool) -> list[int]:
        n, k = len(self.lead_minutes), self.leads_per_sample
        if not k or k >= n:
            return list(self.lead_minutes)
        if random_leads:
            pick = torch.randperm(n)[:k]
        else:
            g = torch.Generator().manual_seed(int(window_start_idx))
            pick = torch.randperm(n, generator=g)[:k]
        return [self.lead_minutes[j] for j in sorted(pick.tolist())]

    def load(self, idx: int, random_leads: bool = False):
        start = self.valid_indices[idx]
        t0 = self.sub_dates[start] - np.timedelta64(self.history_minutes[0], "m")
        minutes = self.history_minutes + self._lead_subset(start, random_leads)
        times = (t0 + np.array(minutes, dtype="timedelta64[m]")).astype("datetime64[s]")
        T, (H, W) = len(times), self.input_resolution
        out = np.empty((T, len(self.names), H * W), dtype=np.float32)

        sj = np.array([self._sub_pos[t] for t in times.tolist()])
        sub_rows = _read_rows(self.sub, sj, self.sub_idx)
        lo = times.astype("datetime64[h]").astype("datetime64[s]")
        w = ((times - lo).astype(np.int64) / 3600.0).astype(np.float32)
        hj_lo = np.array([self._hour_pos[t] for t in lo.tolist()])
        hj_hi = hj_lo + (w > 0)
        hour_rows = _read_rows(self.data, np.r_[hj_lo, hj_hi], self.hour_idx) if self.hour_idx else None

        cs = [c for c, s in enumerate(self.from_sub) if s]
        ch = [c for c, s in enumerate(self.from_sub) if not s]
        for f in range(T):
            out[f, cs] = sub_rows[sj[f]]
            if ch:
                x0 = hour_rows[hj_lo[f]]
                out[f, ch] = x0 if w[f] == 0 else (1 - w[f]) * x0 + w[f] * hour_rows[hj_hi[f]]
        return out.reshape(T, len(self.names), H, W), times

    def __getitem__(self, idx):
        return self.get(idx, random_leads=False)

    def get(self, idx: int, random_leads: bool):
        raw, times = self.load(idx, random_leads)
        combined = self.normalize(torch.from_numpy(raw), self.names)
        P = len(self.prognostic_params)
        data, forcing = combined[:, :P], combined[:, P:]
        if self.static_forcings is not None:
            if self.static_forcings.shape[-1] > 2000:
                s = self.static_forcings.shape
                self.static_forcings = self.static_forcings.reshape(s[0], s[1], *self.input_resolution)
            forcing = torch.cat((forcing, self.static_forcings.repeat(len(times), 1, 1, 1)), dim=1)
        if self.return_metadata:
            return data, forcing, times.astype(np.float64)
        return data, forcing


class SubHourlySubset(torch.utils.data.Dataset):
    """Subset whose lead sampling is fixed per split (random for train, deterministic otherwise)."""

    def __init__(self, dataset: SubHourlyDataset, indices, random_leads: bool):
        self.dataset, self.indices, self.random_leads = dataset, np.asarray(indices), random_leads

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset.get(int(self.indices[i]), self.random_leads)
