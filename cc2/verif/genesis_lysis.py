import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def _compute_patch_stats_with_pixel_counts(
    field_t0,
    field_t1,
    patch_size=10,
    clear_threshold=0.2,
    cloudy_threshold=0.8,
):
    H, W = field_t0.shape
    n_patches_y = H // patch_size
    n_patches_x = W // patch_size

    patch_changes = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)
    patch_clear_fraction = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)
    patch_cloudy_fraction = np.zeros((n_patches_y, n_patches_x), dtype=np.float32)

    pixels_per_patch = float(patch_size * patch_size)

    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y0 = i * patch_size
            y1 = y0 + patch_size
            x0 = j * patch_size
            x1 = x0 + patch_size

            patch_t0 = field_t0[y0:y1, x0:x1]
            patch_t1 = field_t1[y0:y1, x0:x1]

            patch_changes[i, j] = patch_t1.mean() - patch_t0.mean()
            patch_clear_fraction[i, j] = (patch_t0 < clear_threshold).sum() / pixels_per_patch
            patch_cloudy_fraction[i, j] = (patch_t0 > cloudy_threshold).sum() / pixels_per_patch

    return patch_changes, patch_clear_fraction, patch_cloudy_fraction


def _compute_genesis_lysis_metrics_pixel_count(
    obs_changes,
    obs_clear_frac,
    obs_cloudy_frac,
    pred_changes,
    change_threshold=0.25,
    clear_core_threshold=0.85,
    cloudy_core_threshold=0.85,
):
    obs_flat = obs_changes.flatten()
    pred_flat = pred_changes.flatten()

    if np.allclose(np.std(obs_flat), 0.0) or np.allclose(np.std(pred_flat), 0.0):
        correlation = np.nan
    else:
        correlation = np.corrcoef(obs_flat, pred_flat)[0, 1]
    rmse = np.sqrt(((obs_flat - pred_flat) ** 2).mean())

    obs_genesis = (obs_changes >= change_threshold) & (obs_clear_frac > clear_core_threshold)
    obs_lysis = (obs_changes <= -change_threshold) & (obs_cloudy_frac > cloudy_core_threshold)
    pred_genesis = (pred_changes >= change_threshold) & (obs_clear_frac > clear_core_threshold)
    pred_lysis = (pred_changes <= -change_threshold) & (obs_cloudy_frac > cloudy_core_threshold)

    genesis_hits = int((obs_genesis & pred_genesis).sum())
    genesis_misses = int((obs_genesis & ~pred_genesis).sum())
    genesis_false_alarms = int((~obs_genesis & pred_genesis).sum())
    genesis_csi = genesis_hits / (genesis_hits + genesis_misses + genesis_false_alarms + 1e-10)

    lysis_hits = int((obs_lysis & pred_lysis).sum())
    lysis_misses = int((obs_lysis & ~pred_lysis).sum())
    lysis_false_alarms = int((~obs_lysis & pred_lysis).sum())
    lysis_csi = lysis_hits / (lysis_hits + lysis_misses + lysis_false_alarms + 1e-10)

    return {
        "correlation": correlation,
        "rmse": rmse,
        "genesis_csi": genesis_csi,
        "lysis_csi": lysis_csi,
        "genesis_hits": genesis_hits,
        "genesis_misses": genesis_misses,
        "genesis_false_alarms": genesis_false_alarms,
        "lysis_hits": lysis_hits,
        "lysis_misses": lysis_misses,
        "lysis_false_alarms": lysis_false_alarms,
        "n_genesis": int(obs_genesis.sum()),
        "n_lysis": int(obs_lysis.sum()),
    }


def genesis_lysis(
    run_name,
    all_truth,
    all_predictions,
    save_path,
    patch_size=10,
    change_threshold=0.25,
    clear_threshold=0.2,
    cloudy_threshold=0.8,
    core_threshold=0.85,
    window_stride=3,
):
    results = []

    for i, model_name in enumerate(run_name):
        truth = all_truth[i].detach().cpu().numpy()
        pred = all_predictions[i].detach().cpu().numpy()

        n_forecasts, n_steps, _, _, _ = truth.shape
        time_windows = []
        for t0 in range(0, n_steps - 1, window_stride):
            t1 = t0 + window_stride
            if t1 < n_steps:
                time_windows.append((t0, t1, f"t+{t0}-{t1}h"))

        if not time_windows:
            raise ValueError(
                f"Not enough timesteps ({n_steps}) for genesis/lysis windows with stride {window_stride}"
            )

        for forecast_idx in tqdm(range(n_forecasts), desc=f"Genesis/Lysis {model_name}"):
            for t0, t1, window_label in time_windows:
                field_t0 = truth[forecast_idx, t0, 0]
                truth_t1 = truth[forecast_idx, t1, 0]
                pred_t1 = pred[forecast_idx, t1, 0]

                obs_changes, obs_clear_frac, obs_cloudy_frac = _compute_patch_stats_with_pixel_counts(
                    field_t0,
                    truth_t1,
                    patch_size=patch_size,
                    clear_threshold=clear_threshold,
                    cloudy_threshold=cloudy_threshold,
                )
                pred_changes, _, _ = _compute_patch_stats_with_pixel_counts(
                    field_t0,
                    pred_t1,
                    patch_size=patch_size,
                    clear_threshold=clear_threshold,
                    cloudy_threshold=cloudy_threshold,
                )

                metrics = _compute_genesis_lysis_metrics_pixel_count(
                    obs_changes,
                    obs_clear_frac,
                    obs_cloudy_frac,
                    pred_changes,
                    change_threshold=change_threshold,
                    clear_core_threshold=core_threshold,
                    cloudy_core_threshold=core_threshold,
                )

                results.append(
                    {
                        "model": model_name,
                        "forecast_idx": forecast_idx,
                        "time_window": window_label,
                        "timestep": t1,
                        "t0": t0,
                        "t1": t1,
                        "patch_size": patch_size,
                        "change_threshold": change_threshold,
                        "clear_threshold": clear_threshold,
                        "cloudy_threshold": cloudy_threshold,
                        "core_threshold": core_threshold,
                        **metrics,
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["model", "t1", "forecast_idx"])

    df.to_csv(f"{save_path}/results/genesis_lysis.csv", index=False)
    return df


def plot_genesis_lysis(df, save_path="runs/verification"):
    if df.empty:
        print("No genesis/lysis results to plot.")
        return

    summary = (
        df.groupby(["model", "t1"], as_index=False)
        .agg(
            genesis_csi=("genesis_csi", "mean"),
            lysis_csi=("lysis_csi", "mean"),
        )
        .sort_values(["t1", "model"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=False)
    for ax, metric, title in [
        (axes[0], "genesis_csi", "Genesis CSI"),
        (axes[1], "lysis_csi", "Lysis CSI"),
    ]:
        sns.lineplot(
            data=summary,
            x="t1",
            y=metric,
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_xlabel("Window End Lead Time (h)")
        ax.set_ylabel("CSI")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    axes[0].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].legend().remove()
    plt.tight_layout()
    filename = f"{save_path}/figures/genesis_lysis_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
