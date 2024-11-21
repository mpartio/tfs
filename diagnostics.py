import matplotlib.pyplot as plt
import torch
import numpy as np
from copy import deepcopy


class DistributionDiagnostics:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.train_history = {
            "variance": [],
            "alpha_mean": [],
            "beta_mean": [],
            "error": [],
            "batch_mean": [],
            "batch_std": [],
            "batch_zeros": [],
            "batch_ones": [],
            "batch_size": [],
            "error_high": [],
            "error_low": [],
            "error_mid": [],
            "alpha0_error_corr": [],
            "beta0_error_corr": [],
#            "alpha1_error_corr": [],
#            "beta1_error_corr": [],
            "var_row_means": [],
            "var_col_means": [],
            "var_row_std": [],
            "var_col_std": [],
            "var_top_left_mean": [],
            "var_top_right_mean": [],
            "var_bottom_left_mean": [],
            "var_bottom_right_mean": [],
            "var_top_left_std": [],
            "var_top_right_std": [],
            "var_bottom_left_std": [],
            "var_bottom_right_std": [],
            "viewing_angle_weighted_var": [],
            "distance_profile": [],
            "radial_profile": [],
        }
        self.val_history = deepcopy(self.train_history)

    def analyze_batch(self, alpha, beta, weights, samples, targets, inputs):
        B, T, H, W, num_mix = alpha.shape

        batch_size = targets.numel() if targets is not None else alpha.numel()

        alpha = alpha.squeeze(-1)  # Remove mixture dimension
        beta = beta.squeeze(-1)  # Remove mixture dimension

        # Compute variance map
        variance = samples.var(dim=0)

        alpha_mean = alpha.mean().item()
        beta_mean = beta.mean().item()

        result = {
            "variance": variance.mean().item(),
            "alpha_mean": alpha_mean,
            "beta_mean": beta_mean,
        }

        if targets is not None:
            result["last_targets"] = targets.detach().cpu().numpy()
            result["last_inputs"] = inputs.detach().cpu().numpy()

            mean_prediction = samples.mean(dim=0)  # Remove samples dimension

            # Expand targets to match prediction shape or squeeze prediction to match targets
            if targets.shape[1] == 1:
                targets = targets.squeeze(1).squeeze(-1)  # Remove singleton dimensions
                mean_prediction = mean_prediction.squeeze(1).squeeze(-1)

            assert mean_prediction.shape == targets.shape

            error = torch.abs(mean_prediction - targets)
            result["error"] = error.mean().item()
            result["batch_mean"] = targets.mean().item()
            result["batch_std"] = targets.std().item()
            result["batch_zeros"] = (targets == 0).float().sum().item()
            result["batch_ones"] = (targets == 1).float().sum().item()

            error = error.view(-1)
            targets = targets.view(-1)

            # Error by region
            result.update(
                {
                    "error_high": error[targets > 0.8].mean().item()
                    if (targets > 0.8).any()
                    else 0,
                    "error_low": error[targets < 0.2].mean().item()
                    if (targets < 0.2).any()
                    else 0,
                    "error_mid": error[(targets >= 0.2) & (targets <= 0.8)]
                    .mean()
                    .item()
                    if ((targets >= 0.2) & (targets <= 0.8)).any()
                    else 0,
                }
            )

            if len(alpha.shape) == 4:
                alpha = alpha.unsqueeze(-1)
                beta = beta.unsqueeze(-1)
            # Parameter-error correlations

            error = error.flatten()

            for n in range(alpha.shape[-1]):
                alpha_ = alpha[..., n].flatten()
                beta_ = beta[..., n].flatten()

                result.update(
                    {
                        f"alpha{n}_error_corr": torch.corrcoef(
                            torch.stack([alpha_, error])
                        )[0, 1].item(),
                        f"beta{n}_error_corr": torch.corrcoef(
                            torch.stack([beta_, error])
                        )[0, 1].item(),
                    }
                )

            # Spatial variance analysis
            var_map = samples.var(dim=0)  # [B, H, W]

            spatial_stats = self.analyze_spatial_patterns(var_map)
            result.update(spatial_stats)

        result["batch_size"] = batch_size

        return result

    def analyze_spatial_patterns(self, var_map):
        """Comprehensive spatial analysis of variance patterns"""
        # var_map shape: B, T, H, W

        var_map = var_map.mean(dim=0)  # Average over batch
        if len(var_map.shape) > 2:
            var_map = var_map.mean(dim=0)  # Average over time/channels if present

        H, W = var_map.shape

        result = {}
        # 1. Viewing Angle Analysis
        y_coords = np.linspace(0, 1, H)
        viewing_angle_weights = y_coords.reshape(-1, 1)  # Higher weights at bottom
        weighted_variance = var_map * torch.from_numpy(viewing_angle_weights).to(
            var_map.device
        )

        result["viewing_angle_weighted_var"] = weighted_variance.mean().item()

        # 2. Distance-from-nadir Analysis
        y, x = np.ogrid[:H, :W]
        nadir_y, nadir_x = H, W // 2
        distance = np.sqrt((y - nadir_y) ** 2 + (x - nadir_x) ** 2)
        max_dist = np.max(distance)
        distance_normalized = distance / max_dist

        # Create distance bins and compute mean variance in each bin
        num_bins = 10
        distance_bins = np.linspace(0, 1, num_bins)
        binned_variances = []

        for i in range(num_bins - 1):
            mask = torch.from_numpy(
                (distance_normalized >= distance_bins[i])
                & (distance_normalized < distance_bins[i + 1])
            ).to(var_map.device)
            bin_mean = var_map[mask].mean().item() if mask.any() else 0
            binned_variances.append(bin_mean)

        result["distance_profile"] = binned_variances

        # 3. Radial Analysis
        center_y, center_x = H // 2, W // 2
        y_grid, x_grid = np.ogrid[:H, :W]
        radius = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
        max_radius = np.max(radius)
        radius_normalized = radius / max_radius

        # Create radial bins and compute mean variance in each ring
        num_rings = 8
        ring_bins = np.linspace(0, 1, num_rings)
        ring_variances = []

        for i in range(num_rings - 1):
            mask = torch.from_numpy(
                (radius_normalized >= ring_bins[i])
                & (radius_normalized < ring_bins[i + 1])
            ).to(var_map.device)
            ring_mean = var_map[mask].mean().item() if mask.any() else 0
            ring_variances.append(ring_mean)

        result["radial_profile"] = ring_variances

        # Analyze quadrants
        h_mid, w_mid = H // 2, W // 2

        quadrants = {
            "top_left": var_map[..., :h_mid, :w_mid],
            "top_right": var_map[..., :h_mid, w_mid:],
            "bottom_left": var_map[..., h_mid:, :w_mid],
            "bottom_right": var_map[..., h_mid:, w_mid:],
        }

        # Compute stats for each quadrant
        for quad_name, quad_data in quadrants.items():
            result.update(
                {
                    f"var_{quad_name}_mean": quad_data.mean().item(),
                    f"var_{quad_name}_std": quad_data.std().item(),
                }
            )

        # Add row/column analysis
        result.update(
            {
                "var_row_means": var_map.mean(dim=-1)
                .mean()
                .item(),  # Average along rows
                "var_col_means": var_map.mean(dim=-2)
                .mean()
                .item(),  # Average along columns
                "var_row_std": var_map.mean(dim=-1).std().item(),
                "var_col_std": var_map.mean(dim=-2).std().item(),
            }
        )

        return result

    def plot_cloud_patterns(self, epoch):
        """Analyze relationship between variance and cloud patterns"""

        var_map = self.val_history["last_variance_map"]
        targets = self.val_history["last_targets"]
        inputs = self.val_history["last_inputs"]

        var_map = var_map.mean(axis=(0, 1))
        cloud_mean = targets.mean(axis=(0, 1))
        cloud_std = targets.std(axis=(0, 1))
        # Create figure with multiple analyses
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Average Cloud Cover (left panel)
        im1 = axes[0, 0].imshow(cloud_mean)
        axes[0, 0].set_title("Mean Cloud Cover")
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Cloud Variability (middle panel)
        im2 = axes[0, 1].imshow(cloud_std)
        axes[0, 1].set_title("Cloud Cover Variability")
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Variance Map (right panel)
        im3 = axes[0, 2].imshow(var_map)
        axes[0, 2].set_title("Model Variance")
        plt.colorbar(im3, ax=axes[0, 2])

        # 4. Scatter: Cloud Cover vs Variance (bottom left)
        axes[1, 0].scatter(cloud_mean.flatten(), var_map.flatten(), alpha=0.1)
        axes[1, 0].set_xlabel("Cloud Cover")
        axes[1, 0].set_ylabel("Model Variance")
        axes[1, 0].set_title("Variance vs Cloud Cover")

        # 5. Scatter: Cloud Variability vs Variance (bottom middle)
        axes[1, 1].scatter(cloud_std.flatten(), var_map.flatten(), alpha=0.1)
        axes[1, 1].set_xlabel("Cloud Variability")
        axes[1, 1].set_ylabel("Model Variance")
        axes[1, 1].set_title("Variance vs Cloud Variability")

        # 6. Profile Comparison (bottom right)
        # Average along rows to compare patterns
        cloud_profile = cloud_mean.mean(axis=1)
        var_profile = var_map.mean(axis=1)

        ax2 = axes[1, 2].twinx()
        (l1,) = axes[1, 2].plot(cloud_profile, "b-", label="Cloud Cover")
        (l2,) = ax2.plot(var_profile, "r-", label="Variance")
        axes[1, 2].set_xlabel("South → North")
        axes[1, 2].set_ylabel("Cloud Cover", color="b")
        ax2.set_ylabel("Variance", color="r")

        # Combine legends
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        axes[1, 2].legend(lines, labels)

        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/cloud_variance_analysis_epoch_{epoch:03d}.png")
        plt.close()

    def plot_diagnostics(self, epoch, phase="val"):

        # Get both histories
        val_hist = self.val_history
        train_hist = self.train_history

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Plot variance history with both phases
        axes[0, 0].plot(train_hist["variance"], label="Train Instant", alpha=0.3)
        axes[0, 0].plot(val_hist["variance"], label="Val Instant", alpha=0.3)

        # Keep variance trend but show both
        window = min(10, len(train_hist["variance"]))

        train_mean = np.convolve(
            train_hist["variance"], np.ones(window) / window, mode="valid"
        )
        val_mean = np.convolve(
            val_hist["variance"], np.ones(window) / window, mode="valid"
        )

        axes[0, 0].plot(
            range(window - 1, len(train_mean) + window - 1),
            train_mean,
            label="Train Mean",
            color="blue",
            linewidth=2,
        )
        axes[0, 0].plot(
            range(window - 1, len(val_mean) + window - 1),
            val_mean,
            label="Val Mean",
            color="orange",
            linewidth=2,
        )

        axes[0, 0].set_title("Prediction Variance (Instant and Rolling Mean)")

        #        axes[0, 0].plot(train_mean, label="Train Rolling Mean")
        #        axes[0, 0].plot(val_mean, label="Val Rolling Mean")
        axes[0, 0].set_xlabel("Batch (x50)")
        axes[0, 0].set_ylabel("Mean Variance")
        axes[0, 0].legend()

        axes[0, 0].set_title("Average Prediction Variance")
        axes[0, 0].set_xlabel("Batch (x50)")
        axes[0, 0].set_ylabel("Variance")
        axes[0, 0].legend()

        # Plot alpha/beta history for both phases
        axes[0, 1].plot(train_hist["alpha_mean"], label="Train Alpha", alpha=0.6)
        axes[0, 1].plot(train_hist["beta_mean"], label="Train Beta", alpha=0.6)
        axes[0, 1].plot(
            val_hist["alpha_mean"], label="Val Alpha", linestyle="--", alpha=0.6
        )
        axes[0, 1].plot(
            val_hist["beta_mean"], label="Val Beta", linestyle="--", alpha=0.6
        )
        axes[0, 1].set_title("Distribution Parameters")
        axes[0, 1].set_xlabel("Batch (x50)")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].legend()

        # Error vs Variance scatter with temporal coloring
        n_train = len(train_hist["variance"])
        n_val = len(val_hist["variance"])

        # Create color arrays that go from light to dark
        train_colors = plt.cm.Blues(np.linspace(0.2, 1, n_train))
        val_colors = plt.cm.Oranges(np.linspace(0.2, 1, n_val))

        # Error vs Variance scatter for both phases
        axes[0, 2].scatter(
            train_hist["variance"],
            train_hist["error"],
            c=train_colors,
            alpha=0.3,
            label="Train",
        )
        axes[0, 2].scatter(
            val_hist["variance"],
            val_hist["error"],
            c=val_colors,
            alpha=0.3,
            label="Val",
        )
        axes[0, 2].set_title("Error vs Variance (darker=earlier)")
        axes[0, 2].set_xlabel("Variance")
        axes[0, 2].set_ylabel("Error")
        axes[0, 2].legend()

        train_zeros_prop = [
            zeros / size
            for zeros, size in zip(train_hist["batch_zeros"], train_hist["batch_size"])
        ]
        train_ones_prop = [
            ones / size
            for ones, size in zip(train_hist["batch_ones"], train_hist["batch_size"])
        ]
        val_zeros_prop = [
            zeros / size
            for zeros, size in zip(val_hist["batch_zeros"], val_hist["batch_size"])
        ]
        val_ones_prop = [
            ones / size
            for ones, size in zip(val_hist["batch_ones"], val_hist["batch_size"])
        ]

        # Add batch diversity plots
        axes[1, 0].plot(train_hist["batch_mean"], label="Train Mean", alpha=0.6)
        axes[1, 0].plot(train_hist["batch_std"], label="Train Std", alpha=0.6)
        axes[1, 0].plot(
            val_hist["batch_mean"], label="Val Mean", linestyle="--", alpha=0.6
        )
        axes[1, 0].plot(
            val_hist["batch_std"], label="Val Std", linestyle="--", alpha=0.6
        )
        axes[1, 0].set_title("Batch Statistics")
        axes[1, 0].set_xlabel("Batch (x50)")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].legend()

        # Plot binary distribution
        axes[1, 1].plot(train_zeros_prop, label="Train Zeros %", alpha=0.6)
        axes[1, 1].plot(train_ones_prop, label="Train Ones %", alpha=0.6)
        axes[1, 1].plot(val_zeros_prop, label="Val Zeros %", linestyle="--")
        axes[1, 1].plot(val_ones_prop, label="Val Ones %", linestyle="--")
        axes[1, 1].set_title("Binary Distribution per Batch")
        axes[1, 1].set_xlabel("Batch (x50)")
        axes[1, 1].set_ylabel("Proportion")
        axes[1, 1].legend()

        # Regional Error Analysis

        assert "error_high" in train_hist
        for prefix, style in [("Train", "-"), ("Val", "--")]:
            hist = train_hist if prefix == "Train" else val_hist

            axes[1, 2].plot(
                hist["error_high"], label=f"{prefix} High", linestyle=style, alpha=0.6
            )
            axes[1, 2].plot(
                hist["error_low"], label=f"{prefix} Low", linestyle=style, alpha=0.6
            )
            axes[1, 2].plot(
                hist["error_mid"], label=f"{prefix} Mid", linestyle=style, alpha=0.6
            )

        axes[1, 2].set_title("Error by Region")
        axes[1, 2].set_xlabel("Batch (x50)")
        axes[1, 2].set_ylabel("Error")
        axes[1, 2].legend()

        # Parameter-Error Correlations
        for prefix, style in [("Train", "-"), ("Val", "--")]:
            hist = train_hist if prefix == "Train" else val_hist
            for n in range(1):
                if len(hist[f"alpha{n}_error_corr"]) > 0:
                    axes[2, 0].plot(
                        hist[f"alpha{n}_error_corr"],
                        label=f"{prefix} Alpha{n}-Error",
                        linestyle=style,
                        alpha=0.6,
                    )
                axes[2, 0].plot(
                    hist[f"beta{n}_error_corr"],
                    label=f"{prefix} Beta{n}-Error",
                    linestyle=style,
                    alpha=0.6,
                )
        axes[2, 0].set_title("Parameter-Error Correlations")
        axes[2, 0].set_xlabel("Batch (x50)")
        axes[2, 0].set_ylabel("Correlation")
        axes[2, 0].legend()

        # Error Distribution
        # Add a histogram or density plot of errors
        for prefix, style in [("Train", "-"), ("Val", "--")]:
            hist = train_hist if prefix == "Train" else val_hist
            axes[2, 1].hist(
                hist["error"],
                bins=30,
                alpha=0.5,
                label=prefix,
                histtype="step",
                linewidth=2,
            )
        axes[2, 1].set_title("Error Distribution")
        axes[2, 1].set_xlabel("Error")
        axes[2, 1].set_ylabel("Count")
        axes[2, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/diagnostics_epoch_{epoch:03d}.png")
        plt.close()

        self.plot_spatial_patterns(epoch)
        # self.plot_detailed_spatial_analysis(epoch)
        # self.plot_cloud_patterns(epoch)

    def plot_spatial_patterns(self, epoch):
        # Get both histories
        val_hist = self.val_history
        train_hist = self.train_history

        fig, axes = plt.subplots(2, 3, figsize=(25, 15))

        # Updated Spatial Analysis
        for prefix, hist in [("Train", train_hist), ("Val", val_hist)]:
            style = "--" if prefix == "Val" else "-"
            # Plot quadrant means
            for quad in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                axes[0, 0].plot(
                    hist[f"var_{quad}_mean"],
                    label=f'{prefix} {quad.replace("_", " ")}',
                    linestyle=style,
                    alpha=0.6,
                )

        axes[0, 0].set_title("Spatial Variance by Quadrant")
        axes[0, 0].set_xlabel("Batch (x50)")
        axes[0, 0].set_ylabel("Mean Variance")
        axes[0, 0].legend()

        # Add Row/Column analysis for Train
        for score in ("means", "std"):
            style = "--" if score == "std" else "-"
            axes[0, 1].plot(
                train_hist[f"var_row_{score}"],
                label=f"Train Row {score}",
                linestyle=style,
                alpha=0.6,
            )
            axes[0, 1].plot(
                train_hist[f"var_col_{score}"],
                label=f"Train Col {score}",
                linestyle=style,
                alpha=0.6,
            )

        axes[0, 1].set_title("Train Row vs Column Mean and Variance")
        axes[0, 1].set_xlabel("Batch (x50)")
        axes[0, 1].set_ylabel("Mean Mean and Variance")
        axes[0, 1].legend()

        # Add Row/Column analysis for Val
        for score in ("means", "std"):
            style = "--" if score == "std" else "-"
            axes[0, 2].plot(
                val_hist[f"var_row_{score}"],
                label=f"Val Row {score}",
                linestyle=style,
                alpha=0.6,
            )
            axes[0, 2].plot(
                val_hist[f"var_col_{score}"],
                label=f"Val Col {score}",
                linestyle=style,
                alpha=0.6,
            )

        axes[0, 2].set_title("Val Row vs Column Mean and Variance")
        axes[0, 2].set_xlabel("Batch (x50)")
        axes[0, 2].set_ylabel("Mean Mean and Variance")
        axes[0, 2].legend()

        # Plot viewing angle weighted variance
        for prefix, hist in [("Train", train_hist), ("Val", val_hist)]:
            style = "--" if prefix == "Val" else "-"
            axes[1, 0].plot(
                hist["viewing_angle_weighted_var"], label=f"{prefix}", linestyle=style
            )
        axes[1, 0].set_title("Viewing Angle Weighted Variance")
        axes[1, 0].set_xlabel("Batch (x50)")
        axes[1, 0].set_ylabel("Weighted Variance")
        axes[1, 0].legend()

        # Plot distance from nadir profile
        latest_train = train_hist["distance_profile"][-1]
        latest_val = val_hist["distance_profile"][-1]

        # Create bins to match the number of values (one less than bin edges)
        bins = np.linspace(0, 1, len(latest_train) + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # Use bin centers for x-axis

        axes[1, 1].plot(bin_centers, latest_train, label="Train", marker="o")
        axes[1, 1].plot(
            bin_centers, latest_val, label="Val", linestyle="--", marker="o"
        )
        axes[1, 1].set_title("Variance vs Distance from Nadir")
        axes[1, 1].set_xlabel("Normalized Distance")
        axes[1, 1].set_ylabel("Mean Variance")
        axes[1, 1].legend()

        # Plot radial profile
        latest_train_radial = train_hist["radial_profile"][-1]
        latest_val_radial = val_hist["radial_profile"][-1]
        rings = np.linspace(0, 1, len(latest_train_radial) + 1)
        ring_centers = (rings[:-1] + rings[1:]) / 2  # Compute centers

        axes[1, 2].plot(ring_centers, latest_train_radial, label="Train", marker="o")
        axes[1, 2].plot(
            ring_centers, latest_val_radial, label="Val", linestyle="--", marker="o"
        )
        axes[1, 2].set_title("Variance vs Radial Distance")
        axes[1, 2].set_xlabel("Normalized Radius")
        axes[1, 2].set_ylabel("Mean Variance")
        axes[1, 2].legend()

        #        # Add a schematic of the viewing geometry in the last panel
        #        axes[2, 0].axis("off")
        #        axes[2, 0].text(
        #            0.5,
        #            0.5,
        #            "Satellite Viewing Geometry\n\n"
        #            + "Top: Oblique view\n"
        #            + "Bottom: Near nadir\n"
        #            + "Center: Reference point",
        #            ha="center",
        #            va="center",
        #            bbox=dict(facecolor="white", alpha=0.8),
        #        )

        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/spatial_diagnostics_epoch_{epoch:03d}.png")
        plt.close()
