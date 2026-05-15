import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def bias(
    run_name: list[str],
    all_truth: list,
    all_predictions: list,
    save_path: str,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]

        diff = predictions - truth
        dims = [j for j in range(predictions.ndim) if j != 1]
        bias_per_step = torch.mean(diff, dim=dims).tolist()

        for timestep_index, score in enumerate(bias_per_step):
            results.append(
                {
                    "model": run_name[i],
                    "timestep": timestep_index,
                    "bias": score,
                }
            )

    # Eulerian persistence baseline
    base_truth = all_truth[0]
    pers = base_truth[:, 0:1].expand_as(base_truth)
    bias_per_step = torch.mean(pers - base_truth, dim=(0, 2, 3, 4)).tolist()
    for t, score in enumerate(bias_per_step):
        results.append({"model": "eulerian-persistence", "timestep": t, "bias": score})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["timestep", "model"])
    results_df.to_csv(f"{save_path}/results/bias.csv", index=False)

    return results_df


def plot_bias(df: pd.DataFrame, save_path: str):
    if df.empty:
        return

    num_timesteps = df["timestep"].max() + 1

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="timestep",
        y="bias",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
    )
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Mean Bias (pred − truth)")
    plt.title("Model Comparison: Bias per Forecast Timestep")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"{save_path}/figures/bias_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
