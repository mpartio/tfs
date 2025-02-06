import torch
import torch.nn.functional as F
import numpy as np
import os
import time


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


def roll_forecast(model, x, y, n_steps, loss_fn, num_members=3):
    total_loss = []
    total_tendencies = []
    total_predictions = []

    assert x.ndim == 4, "invalid dimensions for x: {}".format(x.shape)
    assert y.ndim == 5, "invalid dimensions for y: {}".format(y.shape)

    weights = torch.ones(n_steps).to(x.device)

    for step in range(n_steps):
        y_true = y[:, step, :, :, :]

        if x.ndim == 4:
            B, C, H, W = x.shape
            # Must add member dimension
            x = x.unsqueeze(1).expand(B, num_members, C, H, W)

        # X dim: B, C=2, H, W
        # Y dim: B, C=1, H, W
        assert x.ndim == 5, "invalid dimensions for x: {}".format(x.shape)
        assert y_true.ndim == 4, "invalid dimensions for y: {}".format(y_true.shape)

        assert (
            x.shape[-2:] == y_true.shape[-2:]
        ), "x shape does not match y shape: {} vs {}".format(x.shape, y_true.shape)

        # Forward pass

        tendencies, predictions = model(x, step + 1)

        if loss_fn is not None:
            loss = loss_fn(predictions, y_true)

            total_loss.append(loss)

        total_tendencies.append(tendencies)
        total_predictions.append(predictions)

        if n_steps > 1:
            last_x = x[:, :, -1, ...].unsqueeze(2)  # B, M, C, H, W
            x = torch.cat((last_x, predictions), dim=2)

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
