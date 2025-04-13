import torch
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.util import get_latest_run_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, nargs="+")
    parser.add_argument("--scores", type=str, required=False, nargs="+")
    args = parser.parse_args()

    if args.scores is None:
        args.scores = ["mae"]
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

    return truth, predictions


def calculate_mae(y_true: torch.tensor, y_pred: torch.tensor):
    assert y_pred.shape == y_true.shape
    return torch.mean(torch.abs(y_pred - y_true)).item()


def calculate_mae_per_timestep(y_pred: torch.tensor, y_true: torch.tensor):
    assert y_pred.shape == y_true.shape

    # Calculate absolute difference
    abs_diff = torch.abs(y_pred - y_true)

    # Define dimensions to average over (all except time dimension, which is dim=1)
    # Get all dimension indices: list(range(pred_tensor.ndim))
    # Remove the time dimension index (1)
    dims_to_average = [
        i for i in range(y_pred.ndim) if i != 1
    ]  # e.g., [0, 2, 3, 4] for 5D tensor

    mae_per_step = torch.mean(abs_diff, dim=dims_to_average)

    return mae_per_step.tolist()


def evaluate_models(args):
    results = []
    for run_name in args.run_name:
        truth, predictions = read_data(run_name)
        mae_score = calculate_mae_per_timestep(predictions, truth)

        # Append results in long format
        for timestep_index, mae_score in enumerate(mae_score):
            results.append(
                {
                    "model": run_name,
                    "timestep": timestep_index,  # Timestep index (0, 1, ...)
                    "mae": mae_score,
                }
            )

    # Create DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by="mae", ascending=True
        )  # Rank by MAE (lower is better)

    return results_df


def plot_results(df, save_path="/data/runs/verification/mae_comparison.png"):
    """Plots the MAE scores using Seaborn."""
    if df.empty:
        print("No results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1
    num_models = df["model"].nunique()
    print(f"Plotting {num_models} models over {num_timesteps} timesteps.")

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="mae",
        hue="model",  # Different colored line for each model
        style="model",  # Optional: Different marker/line style for each model (useful for B&W)
        markers=True,  # Show markers on the points
        dashes=False,  # Use solid lines unless you have many models
    )

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.title("Model Comparison: MAE per Forecast Timestep")
    plt.xticks(range(num_timesteps))  # Ensure ticks for each integer timestep
    plt.legend(
        title="Model ID", bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # Place legend outside plot
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)  # Add horizontal grid lines
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_timeseries(args, save_path="/data/runs/verification/example_timeseries.png"):
    truth, all_predictions = None, []

    for run_name in args.run_name:
        if truth is None:
            truth, predictions = read_data(run_name)
            truth = truth[0]
            all_predictions.append(predictions[0])
        else:
            _, predictions = read_data(run_name)
            all_predictions.append(predictions[0])

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

            if r == 0:  # Top row: Ground Truth
                img_data = truth[c]  # Shape: [channel, H, W]
                if img_data.shape[0] == 1:  # Remove channel dim if it's 1
                    img_data = img_data.squeeze(0)
                ax.imshow(img_data.cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"Timestep {c}", fontsize=10)
            else:
                pred_data = all_predictions[r - 1]  # current_model_id]
                img_data = pred_data[c]  # Shape: [channel, H, W]
                if img_data.shape[0] == 1:
                    img_data = img_data.squeeze(0)
                ax.imshow(img_data.cpu().numpy(), cmap=cmap, vmin=0, vmax=1)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Example timeseries plot saved to {save_path}")
    plt.close(fig)


args = get_args()
results = evaluate_models(args)
plot_results(results)
plot_timeseries(args)
print("\n--- Evaluation Results ---")
# print(results)
pivot_df = results.pivot(index="model", columns="timestep", values="mae")
print(pivot_df)
print("------------------------\n")
