import torch
import pandas as pd

def mae(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]
        mae_score = calculate_mae_per_timestep(predictions, truth)

        # Append results in long format
        for timestep_index, mae_score in enumerate(mae_score):
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
