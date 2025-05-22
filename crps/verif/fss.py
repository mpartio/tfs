import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_fss_per_leadtime(
    truth: torch.Tensor,
    preds: torch.Tensor,
    thresholds: list[tuple[float, float]],
    mask_sizes: list[int]
) -> torch.Tensor:
    """
    Compute Fractions Skill Score (FSS) separately for each lead time.

    Args:
        truth (torch.Tensor): Ground truth of shape [N, T, 1, H, W].
        preds (torch.Tensor): Predictions of shape [N, T, 1, H, W].
        thresholds (list of (low, high)): Cloud-cover bins.
        mask_sizes (list of int): Square mask sizes in pixels.

    Returns:
        torch.Tensor: FSS scores of shape [T, len(thresholds), len(mask_sizes)].
    """
    N, T, C, H, W = truth.shape
    n_thresh = len(thresholds)
    n_sizes = len(mask_sizes)
    device = truth.device

    # Prepare output: per lead time
    fss_scores = torch.zeros((T, n_thresh, n_sizes), device=device)

    # Loop over lead times
    for t in range(T):
        truth_t = truth[:, t, :, :, :]  # [N,1,H,W]
        pred_t = preds[:, t, :, :, :]

        # For each category
        for i, (low, high) in enumerate(thresholds):
            truth_bin = ((truth_t >= low) & (truth_t < high)).float()
            pred_bin = ((pred_t >= low) & (pred_t < high)).float()

            # Loop over mask sizes
            for j, size in enumerate(mask_sizes):
                pad = size // 2
                # Compute local fractions
                frac_truth = F.avg_pool2d(truth_bin, kernel_size=size, stride=1, padding=pad)
                frac_pred = F.avg_pool2d(pred_bin, kernel_size=size, stride=1, padding=pad)

                # Flatten to vector
                f_t = frac_truth.view(-1)
                f_o = frac_pred.view(-1)

                # Compute FSS
                mse = torch.mean((f_o - f_t) ** 2)
                mse_ref = torch.mean(f_o ** 2 + f_t ** 2)
                fss_scores[t, i, j] = 1 - mse / mse_ref

    return fss_scores

def compute_fss_for_model(
    truth: torch.Tensor,
    preds: torch.Tensor,
    thresholds: list[tuple[float, float]],
    mask_sizes: list[int],
) -> torch.Tensor:
    """
    Compute Fractions Skill Score (FSS) for a single model prediction against truth.

    Args:
        truth (torch.Tensor): Ground truth tensor of shape [N, T, 1, H, W], values in [0, 1].
        preds (torch.Tensor): Prediction tensor of shape [N, T, 1, H, W], values in [0, 1].
        thresholds (list of (float, float)): List of (lower, upper) thresholds defining categories.
        mask_sizes (list of int): List of square mask sizes (in pixels) for computing neighborhood fractions.

    Returns:
        torch.Tensor: FSS scores of shape [len(thresholds), len(mask_sizes)].
    """
    n_thresh = len(thresholds)
    n_sizes = len(mask_sizes)
    fss_scores = torch.zeros((n_thresh, n_sizes), device=truth.device)

    # Flatten spatial dims only when pooling
    N, T, C, H, W = truth.shape
    flat_dims = N * T

    for i, (low, high) in enumerate(thresholds):
        # Binary fields for this category
        truth_bin = ((truth >= low) & (truth < high)).float()
        pred_bin = ((preds >= low) & (preds < high)).float()

        # Reshape to [N*T, 1, H, W] for pooling
        truth_flat = truth_bin.view(flat_dims, 1, H, W)
        pred_flat = pred_bin.view(flat_dims, 1, H, W)

        for j, size in enumerate(mask_sizes):
            pad = size // 2
            # Local fraction via average pooling
            frac_truth = F.avg_pool2d(
                truth_flat, kernel_size=size, stride=1, padding=pad
            )
            frac_pred = F.avg_pool2d(pred_flat, kernel_size=size, stride=1, padding=pad)

            # Flatten all values
            f_t = frac_truth.view(-1)
            f_o = frac_pred.view(-1)

            # Compute MSE and reference
            mse = torch.mean((f_o - f_t) ** 2)
            mse_ref = torch.mean(f_o**2 + f_t**2)
            fss_scores[i, j] = 1 - mse / mse_ref

    return fss_scores


def fss(
    all_truth: torch.Tensor,
    all_predictions: list[torch.Tensor],
    thresholds: list[tuple[float, float]] = None,
    mask_sizes: list[int] = None,
) -> torch.Tensor:
    """
    Compute Fractions Skill Score (FSS) for multiple model predictions across categories and mask sizes.

    Args:
        all_truth (torch.Tensor): Ground truth tensor of shape [N, T, 1, H, W].
        all_predictions (list of torch.Tensor): List of prediction tensors, each of shape [N, T, 1, H, W].
        thresholds (optional): List of (lower, upper) thresholds. Defaults to
            [(0, 0.0625), (0.0625, 0.5625), (0.5625, 0.9375), (0.9375, 1.01)].
        mask_sizes (optional): List of mask sizes (pixels). Defaults to first 10 odd sizes [1, 3, ..., 19].

    Returns:
        torch.Tensor: FSS scores of shape [M, len(thresholds), len(mask_sizes)], where M is # of models.
    """
    if thresholds is None:
        thresholds = [(0, 0.0625), (0.0625, 0.5625), (0.5625, 0.9375), (0.9375, 1.01)]
    if mask_sizes is None:
        mask_sizes = list(range(1, 20, 2))[:6]

    n_models = len(all_predictions)
    device = all_truth[0].device
    n_thresh = len(thresholds)
    n_sizes = len(mask_sizes)

    results = []

    for idx, preds in enumerate(all_predictions):
        r = compute_fss_per_leadtime(all_truth[idx], preds, thresholds, mask_sizes)
        results.append(r)

    # obs
    truth_t = all_truth[0]
    observed_categories = []
    observed_sum = 0

    for i, (low, high) in enumerate(thresholds):                         
        truth_bin = ((truth_t >= low) & (truth_t < high)).float()
        observed_categories.append(torch.sum(truth_bin))
        observed_sum += observed_categories[-1]
   
    observed_categories_frac = [x / observed_sum for x in observed_categories]
    results.append(observed_categories_frac)

    return results



def plot_fss(
    run_name: list[str],
    results: torch.tensor,
    save_path: str = "runs/verification/fss.png",
):
    plt.close("all")
    domain_x = 2370  # km
    domain_y = 2670

    observed = results.pop()

    # results: list of fss scores, each with shape (num_leadtimes, num_categories, num_masks)
    num_leadtimes = results[0].shape[0]
    num_categories = results[0].shape[1]
    num_masks = results[0].shape[2]

    categories = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]

    for i, r in enumerate(results):
        plt.figure(figure(), figsize=(8, 8))

        fss_good = 0.5 + observed[i] * 0.5

        dx = int(np.ceil(domain_x / float(img_sizes[i][0])))
        x = np.arange(num_leadtimes)
        y = np.arange(num_masks)

        xx, yy = np.meshgrid(x, y)
        v = np.mean(r, axis=1)

        levels = np.linspace(0.3, 1.0, 21)
        plt.contourf(xx, yy, v, levels=levels)
        plt.colorbar()
        plt.title(
            "FSS for category '{}' (FSS_good={:.2f}) '{}'".format(
                categories[i], fss_good, run_name[i]
            )
        )
        plt.xlabel("Leadtime (minutes)")
        plt.ylabel("Mask size (km)")
        CS = plt.contour(xx, yy, v, [fss_good])
        plt.clabel(CS, inline=True, fontsize=10)

        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        break

