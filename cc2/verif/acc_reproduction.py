"""Reproduce the paper's anomaly correlation coefficient (ACC).

The paper metric uses a separate per-grid-point climatology for the
observations and for each forecast system.  Climatologies are pooled over all
forecast leads and binned by valid calendar month and UTC hour (12 * 24 bins).
For each case and lead, ACC is the spatial cosine correlation between forecast
and observed anomalies relative to their respective climatologies.

Example
-------
python acc_reproduction.py \
    --truth /path/to/truth.pt \
    --dates /path/to/dates.pt \
    --forecast cfm_k4=/path/to/predictions.pt \
    --forecast bland_layer=/path/to/predictions.pt \
    --saved-acc /Users/partio/latex-link/data/plotdata/acc.pt \
    --meta /Users/partio/latex-link/data/plotdata/meta.pt \
    --output recomputed_acc.pt

Expected tensor shapes are [case, lead, channel, height, width] for fields and
[case, lead] for Unix valid times.  A singleton channel dimension is optional.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch


N_CLIMATOLOGY_BINS = 12 * 24


def _load_tensor(path: str | Path) -> torch.Tensor:
    """Load a tensor, accepting either a bare tensor or a one-tensor mapping."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("tensor", "predictions", "truth", "dates"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
    raise TypeError(f"{path} does not contain a recognizable tensor")


def _flatten_fields(fields: torch.Tensor) -> torch.Tensor:
    """Return fields as [case * lead, pixel] in float64."""
    if fields.ndim == 5:
        if fields.shape[2] != 1:
            raise ValueError(
                "ACC expects one forecast channel, "
                f"but received shape {tuple(fields.shape)}"
            )
        fields = fields[:, :, 0]
    if fields.ndim != 4:
        raise ValueError(
            "Fields must have shape [case, lead, (channel,) height, width], "
            f"but received {tuple(fields.shape)}"
        )
    return fields.reshape(fields.shape[0] * fields.shape[1], -1).to(torch.float64)


def valid_time_bins(dates: torch.Tensor) -> torch.Tensor:
    """Map Unix valid times to zero-based month-by-hour climatology bins."""
    if dates.ndim != 2:
        raise ValueError(
            f"Dates must have shape [case, lead], but received {tuple(dates.shape)}"
        )
    bins = []
    for timestamp in dates.detach().cpu().reshape(-1).tolist():
        valid_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        bins.append((valid_time.month - 1) * 24 + valid_time.hour)
    return torch.tensor(bins, dtype=torch.long)


def per_pixel_climatology(
    fields_flat: torch.Tensor,
    bins: torch.Tensor,
    *,
    chunk_frames: int = 32,
) -> torch.Tensor:
    """Calculate a [288, pixel] climatology using all cases and leads."""
    if fields_flat.shape[0] != bins.numel():
        raise ValueError("Field-frame and valid-time counts differ")
    if not torch.isfinite(fields_flat).all():
        raise ValueError(
            "Non-finite field values found. The paper dataset excluded missing "
            "scans; reproduce that filtering before calculating ACC."
        )

    sums = torch.zeros(
        (N_CLIMATOLOGY_BINS, fields_flat.shape[1]), dtype=torch.float64
    )
    for start in range(0, fields_flat.shape[0], chunk_frames):
        stop = min(start + chunk_frames, fields_flat.shape[0])
        sums.index_add_(0, bins[start:stop], fields_flat[start:stop])

    counts = torch.bincount(bins, minlength=N_CLIMATOLOGY_BINS)
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Climatology bins without samples: {missing}")
    return sums / counts.to(torch.float64).unsqueeze(1)


def own_climatology_acc(
    forecast: torch.Tensor,
    observation: torch.Tensor,
    dates: torch.Tensor,
    *,
    chunk_frames: int = 32,
) -> torch.Tensor:
    """Compute per-case, per-lead spatial ACC using separate climatologies."""
    if forecast.shape != observation.shape:
        raise ValueError(
            f"Forecast shape {tuple(forecast.shape)} does not match "
            f"observation shape {tuple(observation.shape)}"
        )

    n_cases, n_leads = forecast.shape[:2]
    if tuple(dates.shape) != (n_cases, n_leads):
        raise ValueError(
            f"Dates shape {tuple(dates.shape)} does not match "
            f"field case/lead shape {(n_cases, n_leads)}"
        )

    forecast_flat = _flatten_fields(forecast)
    observation_flat = _flatten_fields(observation)
    bins = valid_time_bins(dates)

    forecast_climatology = per_pixel_climatology(
        forecast_flat, bins, chunk_frames=chunk_frames
    )
    observation_climatology = per_pixel_climatology(
        observation_flat, bins, chunk_frames=chunk_frames
    )

    scores = torch.empty(forecast_flat.shape[0], dtype=torch.float64)
    for start in range(0, forecast_flat.shape[0], chunk_frames):
        stop = min(start + chunk_frames, forecast_flat.shape[0])
        chunk_bins = bins[start:stop]
        forecast_anomaly = (
            forecast_flat[start:stop] - forecast_climatology[chunk_bins]
        )
        observation_anomaly = (
            observation_flat[start:stop] - observation_climatology[chunk_bins]
        )
        numerator = (forecast_anomaly * observation_anomaly).sum(dim=1)
        denominator = torch.sqrt(
            forecast_anomaly.square().sum(dim=1)
            * observation_anomaly.square().sum(dim=1)
        )
        scores[start:stop] = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.full_like(numerator, torch.nan),
        )

    return scores.reshape(n_cases, n_leads).to(torch.float32)


def compare_scores(
    calculated: torch.Tensor, saved: torch.Tensor
) -> tuple[float, float, int]:
    """Return maximum error, mean error, and finite/non-finite mismatches."""
    if calculated.shape != saved.shape:
        raise ValueError(
            f"Calculated shape {tuple(calculated.shape)} does not match "
            f"saved shape {tuple(saved.shape)}"
        )
    finite_calculated = torch.isfinite(calculated)
    finite_saved = torch.isfinite(saved)
    finite_mismatches = int((finite_calculated != finite_saved).sum())
    common = finite_calculated & finite_saved
    if not common.any():
        return float("nan"), float("nan"), finite_mismatches
    difference = (calculated[common] - saved[common]).abs()
    return float(difference.max()), float(difference.mean()), finite_mismatches


def _parse_forecast(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Forecast must be specified as MODEL=/path/to/predictions.pt"
        ) from error
    if not name or not path:
        raise argparse.ArgumentTypeError(
            "Forecast must be specified as MODEL=/path/to/predictions.pt"
        )
    return name, Path(path)


def _self_test() -> None:
    """Exercise the implementation on a small deterministic synthetic dataset."""
    n_cases, n_leads, height, width = 288, 2, 3, 4
    base = torch.arange(
        n_cases * n_leads * height * width, dtype=torch.float32
    ).reshape(n_cases, n_leads, 1, height, width)
    observation = torch.sin(base / 11.0)
    forecast = 0.8 * observation + 0.1 * torch.cos(base / 7.0) + 0.25

    # Populate every month/hour bin twice so that anomalies are nonzero.
    timestamps = []
    for _ in range(2):
        for month in range(1, 13):
            for hour in range(24):
                timestamps.append(
                    int(
                        datetime(
                            2025, month, 1, hour, tzinfo=timezone.utc
                        ).timestamp()
                    )
                )
    dates = torch.tensor(timestamps, dtype=torch.float64).reshape(n_cases, n_leads)

    score = own_climatology_acc(forecast, observation, dates, chunk_frames=7)
    if score.shape != (n_cases, n_leads) or not torch.isfinite(score).all():
        raise AssertionError("Synthetic ACC self-test failed")
    print(
        "self-test passed:",
        f"shape={tuple(score.shape)}",
        f"range=[{float(score.min()):.6f}, {float(score.max()):.6f}]",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--dates", type=Path)
    parser.add_argument(
        "--forecast",
        action="append",
        type=_parse_forecast,
        default=[],
        metavar="MODEL=PATH",
    )
    parser.add_argument("--saved-acc", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    required = {
        "--truth": args.truth,
        "--dates": args.dates,
        "--forecast": args.forecast,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    truth = _load_tensor(args.truth)
    dates = _load_tensor(args.dates)
    calculated_by_name = {}
    for model_name, forecast_path in args.forecast:
        print(f"calculating ACC for {model_name} from {forecast_path}")
        forecast = _load_tensor(forecast_path)
        calculated_by_name[model_name] = own_climatology_acc(
            forecast, truth, dates, chunk_frames=args.chunk_frames
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": list(calculated_by_name),
                "tensor": torch.stack(list(calculated_by_name.values())),
                "dims": ["model", "case", "lead"],
            },
            args.output,
        )
        print(f"wrote {args.output}")

    if args.saved_acc or args.meta:
        if not args.saved_acc or not args.meta:
            parser.error("--saved-acc and --meta must be supplied together")
        saved = _load_tensor(args.saved_acc)
        metadata = torch.load(args.meta, map_location="cpu", weights_only=False)
        model_names = metadata["models"]
        for model_name, calculated in calculated_by_name.items():
            if model_name not in model_names:
                raise KeyError(f"{model_name!r} is absent from saved metadata")
            saved_row = saved[model_names.index(model_name)]
            maximum, mean, finite_mismatches = compare_scores(calculated, saved_row)
            print(
                f"{model_name}: max_abs_error={maximum:.9g}, "
                f"mean_abs_error={mean:.9g}, "
                f"finite_mismatches={finite_mismatches}"
            )


if __name__ == "__main__":
    main()
