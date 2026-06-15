"""
Case-study stamps generator.

Reads predictions for a fixed list of forecaster-curated case dates and
produces ONE PNG per case showing truth + all baselines + all candidates
side-by-side.

Layout per case:
    Rows: [Truth, kinetic-creative, candied-circle, cloudcast-production,
           black-henry-aclamp0p85, bipartite-fort, candied-circle-aclamp85]
          (missing models simply skipped — e.g., cloudcast for cases without
           2026 predictions)
    Cols: [t+0, t+4, t+8, t+12]  (clipped to available leads for shorter rollouts)

Cases (see memory/case_studies.md):
    1. 2025-07-01T03:00Z — western Finland clearing (DE-12)
    2. 2026-04-28T08:00Z — northern Finland over-forecast (kinetic-creative fail)
    3. 2026-04-24T08:00Z — northern Finland under-forecast (kinetic-creative fail)
    4. 2026-05-25T09:00Z — southern Baltic over-creation (PENDING)

Usage:
    python3 verif/case_study_stamps.py --out-dir /tmp/case_study_stamps

The model list and per-case cache root are hardcoded; this is the entry point
for the case-study chain, run once after all per-(model, case) inferences are
in the cache at /data/tfs/runs/case-study/<tag>/case<N>/.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CASES = [
    {
        "id": 1,
        "init_time": "2025-07-01T03:00:00",
        "region": "western Finland",
        "failure": "candied-circle smooths over a real clearing",
    },
    {
        "id": 2,
        "init_time": "2026-04-28T08:00:00",
        "region": "northern Finland",
        "failure": "kinetic-creative over-forecasts cloud",
    },
    {
        "id": 3,
        "init_time": "2026-04-24T08:00:00",
        "region": "northern Finland",
        "failure": "kinetic-creative under-forecasts (too clear)",
    },
    {
        "id": 4,
        "init_time": "2026-05-25T09:00:00",
        "region": "southern Baltic",
        "failure": "cloud over-creation",
    },
]

# Display order: Truth first, then these persistent baselines, then the
# candidate passed via --candidate-tag (appended at runtime in main()).
# Missing models are silently skipped (e.g., cloudcast-production has no
# 2026 predictions — included only for case 1).
MODEL_ORDER = [
    ("kinetic-creative", "baseline"),
    ("cloudcast-production", "baseline"),
    ("bland-layer-3654", "baseline"),
]

CASE_STUDY_ROOT = "/data/tfs/runs/case-study"

# Target lead indices for stamp columns. Code clips to actual T-1 if shorter.
LEAD_INDICES_TARGET = [0, 4, 8, 12]


def lead_indices_for(T: int):
    return [min(L, T - 1) for L in LEAD_INDICES_TARGET]


def load_case(model_tag: str, case_id: int, base_dir: str = None):
    # base_dir overrides the default CASE_STUDY_ROOT/<tag> lookup (used for the
    # candidate when --candidate-dir points outside the standard cache root).
    root = Path(base_dir) if base_dir else Path(CASE_STUDY_ROOT) / model_tag
    case_dir = root / f"case{case_id}"
    pp = case_dir / "predictions.pt"
    if not pp.exists():
        return None
    p = torch.load(pp, weights_only=False, map_location="cpu")
    t = torch.load(case_dir / "truth.pt", weights_only=False, map_location="cpu")
    d = torch.load(case_dir / "dates.pt", weights_only=False, map_location="cpu")
    return p, t, d


def plot_case(case, model_predictions, out_path):
    """Render ONE PNG for ONE case showing truth + all available models as rows.

    model_predictions: list of (tag, kind, (pred_tensor, truth_tensor, date_tensor))
                       in display order, missing models pre-filtered out.
    """
    cmap = "Blues"

    # Determine actual T from first model's data (truth dimension)
    first_truth = model_predictions[0][2][1]
    T = first_truth.shape[1]
    leads = lead_indices_for(T)
    ncols = len(leads)

    # Rows: truth + each model
    nrows = 1 + len(model_predictions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"Case {case['id']} — {case['init_time']} — {case['region']}\n"
        f"(reference failure mode: {case['failure']})",
        fontsize=15,
    )

    # Truth from first model's bundled truth (truth.pt is the same across models for a given case)
    truth_field = first_truth[0]  # [T, C, H, W]

    last_im = None
    for r in range(nrows):
        if r == 0:
            label = "Truth"
            img_src = truth_field  # [T, C, H, W]
        else:
            tag, kind, (pred, _, _) = model_predictions[r - 1]
            label = tag if kind == "baseline" else f"{tag} *"  # mark candidates
            img_src = pred[0]  # [T, C, H, W]

        for c, lead in enumerate(leads):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=8)
            if r == 0:
                ax.set_title(f"t+{lead}h", fontsize=12)
            last_im = ax.imshow(
                img_src[lead, 0].numpy(), cmap=cmap, vmin=0, vmax=1
            )

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.93, 0.05, 0.012, 0.85])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Cloud cover (0=clear, 1=overcast)", rotation=90, labelpad=12, fontsize=11)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/case_study_stamps")
    parser.add_argument(
        "--cases", type=int, nargs="+", default=[1, 2, 3],
        help="Case IDs to render"
    )
    parser.add_argument(
        "--candidate-tag", default=None,
        help="Tag of the new model to append below the persistent baselines",
    )
    parser.add_argument(
        "--candidate-dir", default=None,
        help="Dir holding the candidate's case<N>/ predictions "
        "(defaults to CASE_STUDY_ROOT/<candidate-tag>)",
    )
    args = parser.parse_args()

    # Persistent baselines + the candidate (if any), in display order.
    models = list(MODEL_ORDER)
    base_dir_for = {}
    if args.candidate_tag:
        models = [(t, k) for t, k in models if t != args.candidate_tag]
        models.append((args.candidate_tag, "candidate"))
        if args.candidate_dir:
            base_dir_for[args.candidate_tag] = args.candidate_dir

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    for case_id in args.cases:
        case = next((c for c in CASES if c["id"] == case_id), None)
        if case is None:
            print(f"Unknown case id {case_id}, skipping.")
            continue

        # Gather predictions for all models that have data for this case
        model_data = []
        skipped = []
        for tag, kind in models:
            loaded = load_case(tag, case_id, base_dir=base_dir_for.get(tag))
            if loaded is None:
                skipped.append(tag)
                continue
            model_data.append((tag, kind, loaded))

        if not model_data:
            print(f"Case {case_id}: no model predictions found. Skipping.")
            continue

        if skipped:
            print(f"Case {case_id}: skipping (no data): {', '.join(skipped)}")

        out_path = Path(args.out_dir) / f"case{case_id}.png"
        plot_case(case, model_data, out_path)
        print(f"Stamp written: {out_path}  ({len(model_data)} models)")
        out_paths.append(str(out_path))

    # Manifest for orchestrator
    if out_paths:
        print("PATHS=" + ",".join(out_paths))


if __name__ == "__main__":
    main()
