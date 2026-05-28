"""
Apply per-lead mean-bias correction to an existing ED12h predictions.pt.

This is the cheapest possible falsifier for "is the +0.002/step bias drift
fixable without retraining" — it post-processes an existing predictions
tensor by subtracting each lead's signed mean bias (read from a bias.csv
produced by `verify.py`), then re-clamps to [0, 1].

If the corrected predictions raise the composite materially while leaving
S_FSS, S_GEN, S_GATE, and tendency-correlation unchanged, the bias is an
inference-time mean-shift problem and a training-protocol intervention is
not required to fix it.

Usage:
    python3 verif/apply_bias_correction.py \\
        --source /data/tfs/runs/ED12h/<model>-12h \\
        --bias-csv /data/tfs/runs/verification/<tag>/results/bias.csv \\
        --bias-model <model-name-as-it-appears-in-csv> \\
        --dest /data/tfs/runs/ED12h/<model>-12h-biascorrected

The model-name in the CSV is the `--path_name` basename used when verify.py
was run (e.g. `kinetic-creative-k16-eta0p05-sig1p0`).
"""
import argparse
import csv
from pathlib import Path

import torch


def load_bias_table(csv_path: str, model: str, n_leads: int) -> torch.Tensor:
    table = torch.zeros(n_leads, dtype=torch.float32)
    found = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            t = int(row["timestep"])
            if 0 <= t < n_leads:
                table[t] = float(row["bias"])
                found.append(t)
    if not found:
        raise ValueError(f"no rows for model={model!r} in {csv_path}")
    missing = [t for t in range(n_leads) if t not in found]
    if missing:
        print(f"warning: no bias for leads {missing} (leaving as 0)")
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="ED12h dir containing predictions/truth/dates.pt")
    p.add_argument("--bias-csv", required=True)
    p.add_argument("--bias-model", required=True, help="model name as it appears in bias.csv")
    p.add_argument("--dest", required=True, help="output dir (will be created)")
    args = p.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)

    pred = torch.load(src / "predictions.pt", weights_only=False, map_location="cpu")
    truth = torch.load(src / "truth.pt", weights_only=False, map_location="cpu")
    dates = torch.load(src / "dates.pt", weights_only=False, map_location="cpu")

    n_leads = pred.shape[1]
    bias = load_bias_table(args.bias_csv, args.bias_model, n_leads)

    print(f"per-lead bias table for {args.bias_model!r}:")
    for t in range(n_leads):
        print(f"  t+{t:<2}  {bias[t].item():+.5f}")

    corrected = pred - bias.view(1, n_leads, 1, 1, 1)
    corrected.clamp_(0.0, 1.0)

    torch.save(corrected, dst / "predictions.pt")
    torch.save(truth, dst / "truth.pt")
    torch.save(dates, dst / "dates.pt")
    print(f"\nwrote corrected predictions: {dst}")
    print(f"  pred shape {tuple(corrected.shape)}, range [{corrected.min():.4f}, {corrected.max():.4f}]")


if __name__ == "__main__":
    main()
