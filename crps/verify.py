import torch
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.util import get_latest_run_dir
from tqdm import tqdm
import matplotlib.colors as mcolors
from verif.mae import mae, mae2d
from verif.psd import psd


def get_scores():
    return ["mae", "mae2d", "psd"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, nargs="+")
    parser.add_argument(
        "--score",
        type=str,
        required=False,
        nargs="+",
        choices=get_scores(),
        help="Score to produce",
    )
    args = parser.parse_args()

    if args.score is None:
        args.score = ["mae"]
    return args


def read_data(run_name):
    run_dir = get_latest_run_dir(f"runs/{run_name}")
    file_path = f"{run_dir}/test-output"

    predictions = torch.load(
        f"{file_path}/predictions.pt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    truth = torch.load(
        f"{file_path}/truth.pt", map_location=torch.device("cpu"), weights_only=True
    )
    dates = torch.load(
        f"{file_path}/dates.pt", map_location=torch.device("cpu"), weights_only=True
    )

    return truth, predictions, dates


def union(all_truth, all_predictions, all_dates):
    # 1. Build union of all forecast sequences
    all_sets = [set(map(tuple, d.numpy())) for d in all_dates]
    union_keys = set().union(*all_sets)

    print(f"Total unique forecast sequences before union: {len(union_keys)}")

    # 2. Sort the union chronologically (by first lead time)
    sorted_union = sorted(union_keys, key=lambda row: row[0])

    # 3. For each run, filter and reorder according to sorted_union
    new_truth, new_predictions, new_dates = [], [], []

    for i, (truth, pred, dates) in enumerate(
        zip(all_truth, all_predictions, all_dates)
    ):
        date_dict = {tuple(row): idx for idx, row in enumerate(dates.numpy())}

        filtered_truth = []
        filtered_pred = []
        filtered_dates = []

        for key in sorted_union:
            if key in date_dict:
                idx = date_dict[key]
                filtered_truth.append(truth[idx])
                filtered_pred.append(pred[idx])
                filtered_dates.append(torch.tensor(key, dtype=torch.float64))

        if filtered_truth:
            new_truth.append(torch.stack(filtered_truth))
            new_predictions.append(torch.stack(filtered_pred))
            new_dates.append(torch.stack(filtered_dates))
        else:
            print(f"{run_name[i]}: No overlapping forecasts found!")

    print(f"Total forecast sequences after union: {new_predictions[0].shape[0]}")
    return new_truth, new_predictions, new_dates


def equalize_datasets(run_name, all_truth, all_predictions, all_dates):
    need_union = False

    for i in tqdm(range(1, len(all_dates)), desc="Equalizing"):
        if all_dates[i - 1].shape != all_dates[i].shape:
            print(
                "Different shape of dates for {} ({}) and {} ({})".format(
                    args.run_name[i - 1],
                    all_dates[i - 1].shape,
                    args.run_name[i],
                    all_dates[i].shape,
                )
            )

            a = all_dates[i]
            b = all_dates[i - 1]

            a = set(map(tuple, a.numpy()))
            b = set(map(tuple, b.numpy()))

            A_not_in_B = a - b
            B_not_in_A = b - a

            # Print rows

            if len(A_not_in_B):
                print("Rows in {} but not in {}:".format(run_name[i - 1], run_name[i]))
                for row in A_not_in_B:
                    d_s = [np.datetime64(int(x), "s") for x in row]
                    print(d_s)

            if len(B_not_in_A):
                print("Rows in {} but not in {}:".format(run_name[i], run_name[i - 1]))
                for row in B_not_in_A:
                    d_s = [np.datetime64(int(x), "s") for x in row]
                    print(d_s)

            need_union = True

    if need_union:
        # Filter predictions, truth and dates so that a union of both datasets
        # is picked

        return union(all_truth, all_predictions, all_dates)

    return all_truth, all_predictions, all_dates


def prepare_data(args):
    all_truth, all_predictions, all_dates = [], [], []

    for run_name in tqdm(args.run_name, desc="Reading data"):

        truth, predictions, dates = read_data(run_name)

        all_truth.append(truth)
        all_predictions.append(predictions)
        all_dates.append(dates)

    return equalize_datasets(args.run_name, all_truth, all_predictions, all_dates)


def plot_mae_timeseries(
    df: pd.DataFrame, save_path="runs/verification/mae_timeseries.png"
):
    if df.empty:
        print("No results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1
    num_models = df["model"].nunique()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="mae",
        hue="model",
        style="model",  # Optional: Different marker/line style for each model (useful for B&W)
        markers=True,  # Show markers on the points
        dashes=False,  # Use solid lines unless you have many models
    )

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.title("Model Comparison: MAE per Forecast Timestep")
    plt.xticks(range(num_timesteps))  # Ensure ticks for each integer timestep
    plt.legend(title="Model ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_mae2d(
    run_name: list[str],
    results: torch.tensor,
    save_path: str = "runs/verification/mae2d.png",
):
    # results shape: num_models, steps, height, width
    num_models = len(results)
    num_timesteps = results[0].shape[0] - 1  # skip first step as mae = 0
    nrows = num_models
    ncols = num_timesteps

    fig_width = ncols * 2.5
    fig_height = nrows * 2.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    vmin, vmax = 0, 1

    fig.suptitle(f"Spatial MAE", fontsize=16)

    axes = axes if axes.ndim == 2 else axes.reshape((nrows, ncols))

    for r in range(nrows):
        y_pred = results[r]
        for c in range(ncols):
            ax = axes[r, c]

            if c == 0:
                ax.set_ylabel(run_name[r], fontsize=10, rotation=90, labelpad=10)

            if r == 0:
                ax.set_title(f"Timestep {c+1}h", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            im = ax.imshow(y_pred[c + 1], cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar at the far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MAE", fontsize=10)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_stamps(
    run_name: list[str],
    all_truth,
    all_predictions,
    all_dates,
    save_path: str = "runs/verification/example_timeseries.png",
):

    truth = all_truth[0][0]

    num_timesteps = truth.shape[0]
    num_models = len(all_predictions)
    nrows = 1 + num_models  # 1 row for truth + N rows for models
    ncols = num_timesteps

    # Dynamically adjust figsize - very rough estimate
    fig_width = ncols * 2.5
    fig_height = nrows * 2.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    # If only one row/col, axes might not be a 2D array, handle this:
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])  # Make it 2D Nrows x 1 col

    fig.suptitle(f"Ground Truth vs. Model Predictions", fontsize=16)
    cmap = "viridis"

    for r in range(nrows):
        if r == 0:
            img_data = truth
        else:
            img_data = all_predictions[r - 1][
                0
            ]  # select first forecast from each model

        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if c == 0:  # Set row labels on the first column
                if r == 0:
                    ax.set_ylabel("Ground Truth", fontsize=10, rotation=90, labelpad=10)
                else:
                    ax.set_ylabel(
                        f"{args.run_name[r-1]}", fontsize=10, rotation=90, labelpad=10
                    )

            ax.imshow(img_data[c, 0], cmap=cmap, vmin=0, vmax=1)

            if r == 0:  # Top row: Ground Truth
                ax.set_title(f"Timestep {c}", fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Example timeseries plot saved to {save_path}")
    plt.close(fig)


def plot_psd(
    run_name: list[str],
    obs_psd: dict,
    pred_psds: list[dict],
    save_path: str = "runs/verification/psd.png",
):

    plt.figure()
    plt.xlabel("Horizontal Scale (km)", fontsize=12)
    plt.ylabel(
        "PSD", fontsize=12
    )  # Add units if clear, e.g., '(Cloud Cover Fraction)$^2$ / (km$^{-2}$)'
    #    plt.xscale("log")
    #    plt.yscale("log")
    plt.title("Power Spectral Density Comparison", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.7)  # Grid for major and minor ticks

    # scales = obs_psd["scales"]
    sx = obs_psd["sx"]
    psd = obs_psd["psd"]
    plt.loglog(sx, psd, label="Observed", linewidth=1, color="black")

    for i in range(len(run_name)):
        sx = pred_psds[i]["sx"]
        # sort_indices = np.argsort(scales)[::-1] # Sort scales descending
        psd = pred_psds[i]["psd"]
        plt.loglog(sx, psd, label=run_name[i], linewidth=2)

    plt.gca().invert_xaxis()

    plt.legend(fontsize=10)
    plt.savefig(save_path)

    print(f"Plot saved to {save_path}")

    plt.close()


if __name__ == "__main__":
    args = get_args()
    all_truth, all_predictions, all_dates = prepare_data(args)

    pivot_df = None
    for score in args.score:
        if score == "mae":
            results = mae(args.run_name, all_truth, all_predictions)
            plot_mae_timeseries(results)
            pivot_df = results.pivot(index="model", columns="timestep", values="mae")

        elif score == "mae2d":
            results = mae2d(all_truth, all_predictions)
            plot_mae2d(args.run_name, results)

        elif score == "psd":
            obs_psd, pred_psd = psd(all_truth, all_predictions)
            plot_psd(args.run_name, obs_psd, pred_psd)

    plot_stamps(args.run_name, all_truth, all_predictions, all_dates)

    if pivot_df is not None:
        print(pivot_df)
