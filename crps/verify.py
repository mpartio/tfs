import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common.util import get_latest_run_dir
from tqdm import tqdm
from verif.mae import mae, mae2d, plot_mae_timeseries, plot_mae2d
from verif.psd import psd, plot_psd
from verif.fss import fss, plot_fss


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, nargs="+")
    parser.add_argument(
        "--score",
        type=str,
        required=False,
        nargs="+",
        choices=["mae", "mae2d", "psd", "fss"],
        help="Score to produce",
    )

    args = parser.parse_args()

    if args.score is None:
        args.score = ["mae"]
    return args


def read_data(run_name):
    if "/" in run_name:
        run_dir = f"runs/{run_name}"
    else:
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

    if len(all_dates) == 1:
        return all_truth, all_predictions, all_dates

    return equalize_datasets(args.run_name, all_truth, all_predictions, all_dates)


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

        elif score == "fss":
            results = fss(all_truth, all_predictions)
            plot_fss(args.run_name, results)

    plot_stamps(args.run_name, all_truth, all_predictions, all_dates)

    if pivot_df is not None:
        print(pivot_df)
