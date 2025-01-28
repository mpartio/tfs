import torch
import torch.nn.functional as F
import numpy as np
import os
import time


def gaussian_smooth(x, sigma=0.8, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # Create 1D Gaussian kernel
    gauss = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Create 2D kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel = kernel / kernel.sum()

    # Reshape kernel for PyTorch conv2d
    kernel = kernel[None, None, :, :]

    # Add batch dimension if needed
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    # Move kernel to same device as input
    kernel = kernel.to(x.device)

    # Apply smoothing channel by channel
    pad = kernel_size // 2
    smoothed = []
    for c in range(x.shape[1]):
        channel = x[:, c : c + 1]
        channel = F.pad(channel, (pad, pad, pad, pad), mode="reflect")
        channel = F.conv2d(channel, kernel)
        smoothed.append(channel)

    x = torch.cat(smoothed, dim=1)
    x = torch.clamp(x, 0, 1)

    return x.squeeze(0) if squeeze_back else x


def moving_average(arr, window_size):
    """
    Calculate the running mean of a 1D array using a sliding window.

    Parameters:
    - arr (torch.Tensor): 1D input array.
    - window_size (int): The size of the sliding window.

    Returns:
    - torch.Tensor: The running mean array.
    """
    # Ensure input is a 1D tensor
    arr = arr.reshape(1, 1, -1)

    if window_size >= arr.shape[-1]:
        return torch.full((arr.shape[-1],), float("nan"))

    # Create a uniform kernel
    kernel = torch.ones(1, 1, window_size) / window_size

    # Apply 1D convolution
    running_mean = torch.nn.functional.conv1d(
        arr, kernel, padding=0, stride=1
    ).squeeze()

    nan_padding = torch.full((window_size - 1,), float("nan"))

    result = torch.cat((nan_padding, running_mean))

    return result


def roll_forecast(model, x, y, n_steps, loss_fn):
    total_loss = []
    total_tendencies = []
    total_predictions = []

    assert x.ndim == 4, "invalid dimensions for x: {}".format(x.shape)
    assert y.ndim == 5, "invalid dimensions for y: {}".format(y.shape)

    weights = torch.ones(n_steps).to(x.device)

    for step in range(n_steps):
        y_true = y[:, step, :, :, :]

        # X dim: B, C=2, H, W
        # Y dim: B, C=1, H, W
        assert x.ndim == 4, "invalid dimensions for x: {}".format(x.shape)
        assert y_true.ndim == 4, "invalid dimensions for y: {}".format(y_true.shape)

        assert (
            x.shape[-2:] == y_true.shape[-2:]
        ), "x shape does not match y shape: {} vs {}".format(x.shape, y_true.shape)

        assert (
            x.ndim == y_true.ndim
        ), "x and y need to have equal number of dimensions: {} vs {}".format(
            x.shape, y_true.shape
        )
        # Forward pass

        tendencies, predictions = model(x)

        assert torch.isnan(x).sum() == 0, "NaNs in predictions"

        if loss_fn is not None:
            loss = loss_fn(predictions, y_true)

            total_loss.append(loss)

        total_tendencies.append(tendencies)
        total_predictions.append(predictions)

        if n_steps > 1:
            n_members = tendencies.shape[1]
            chosen_member = torch.randint(n_members, (1,)).item()

            last_x = x[:, -1:, :, :]
            chosen_prediction = predictions[:, chosen_member, :, :, :]

            x = torch.cat((last_x, chosen_prediction), dim=1)

    if len(total_loss) > 0:
        total_loss = torch.stack(total_loss)
        total_loss *= weights

    tendencies = torch.stack(total_tendencies).permute(
        1, 0, 2, 3, 4, 5
    )  # B, T, M, C, H, W
    predictions = torch.stack(total_predictions).permute(
        1, 0, 2, 3, 4, 5
    )  # B, T, M, C, H, W

    return total_loss, tendencies, predictions


def analyze_gradients(model):
    # Group gradients by network section
    gradient_stats = {
        "encoder": [],  # Encoder blocks
        "attention": [],  # Attention blocks
        "decoder": [],  # Decoder blocks
        "prediction": [],  # Final head
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()

            if "encoder" in name:
                gradient_stats["encoder"].append(grad_norm)
            elif "decoder" in name:
                gradient_stats["decoder"].append(grad_norm)
            elif "bridge" in name:
                gradient_stats["attention"].append(grad_norm)
            elif "prediction_head" in name:
                gradient_stats["prediction"].append(grad_norm)

    # Compute statistics for each section
    stats = {}
    for section, grads in gradient_stats.items():
        if grads:
            stats[section] = {
                "mean": np.mean(grads),
                "std": np.std(grads),
                "min": np.min(grads),
                "max": np.max(grads),
            }

    return stats


def get_next_run_number(base_dir):
    rank = int(os.environ.get("SLURM_PROCID", 0))
    next_run_file = f"next_run.txt"

    if rank == 0:
        # Only rank 0 determines the next run number
        if not os.path.exists(base_dir):
            next_num = 1
        else:
            subdirs = [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
            ]
            next_num = max([int(d) for d in subdirs], default=0) + 1

        # Write for other ranks to read
        os.makedirs(base_dir, exist_ok=True)
        with open(f"next_run.txt", "w") as f:
            f.write(str(next_num))

    # All ranks wait for the file
    while not os.path.exists(f"next_run.txt"):
        time.sleep(0.1)

    # All ranks read the same number
    with open(f"next_run.txt", "r") as f:
        return int(f.read().strip())


def get_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None

    subdirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    if not subdirs:
        return None
    latest = max([int(d) for d in subdirs])
    return os.path.join(base_dir, str(latest))
