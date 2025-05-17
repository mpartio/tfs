import torch
import pandas as pd


def calculate_mae_per_timestep(y_pred: torch.tensor, y_true: torch.tensor):
    assert y_pred.shape == y_true.shape
    assert y_pred.max() < 2, "y pred scaled incorrectly"
    assert y_true.max() < 2, "y true scaled incorrectly"

    # Calculate absolute difference


def mae(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]

        abs_diff = torch.abs(predictions - truth)
        dims_to_average = [i for i in range(predictions.ndim) if i != 1]
        mae_per_step = torch.mean(abs_diff, dim=dims_to_average)
        mae_per_step = mae_per_step.tolist()

        # Append results in long format
        for timestep_index, mae_score in enumerate(mae_per_step):
            results.append(
                {
                    "model": run_name[i],
                    "timestep": timestep_index,
                    "mae": mae_score,
                }
            )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="mae", ascending=True)

    return results_df


def mae2d(
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
):
    results = []

    for i in range(len(all_predictions)):
        y_pred = all_predictions[i]
        y_true = all_truth[i]

        abs_diff = torch.abs(y_pred - y_true)

        assert len(abs_diff.shape) == 5, "Invalid shape: {}".format(
            abs_diff.shape
        )  # B, C, 1, H, W

        mae2d = torch.mean(abs_diff, dim=(0, 2))

        results.append(mae2d)

    return results
