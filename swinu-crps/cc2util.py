import torch
import numpy as np

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
    assert y.shape[1] == n_steps

    total_loss = []

    assert x.ndim == 4, "invalid dimensions for x: {}".x.shape
    assert y.ndim == 5, "invalid dimensions for y: {}".y.shape

    for step in range(n_steps):
#        if x.ndim == 5:
#            x = x.squeeze(1)  # Remove "time" -> B, C, H, W

        y_true = y[:, step, :, :, :]

        # X dim: B, C=2, H, W
        # Y dim: B, C=1, H, W
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

        assert n_steps == 1
    #        if n_steps > 1:
    #            x = predictions

    if len(total_loss) > 0:
        total_loss = torch.stack(total_loss)
    return total_loss, tendencies, predictions


def analyze_gradients(model):
    # Group gradients by network section
    gradient_stats = {
        "encoder": [],  # Encoder blocks
        "attention": [],  # Attention blocks
        "norms": [],  # Layer norms
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
            elif "attn" in name:
                gradient_stats["attention"].append(grad_norm)
            elif "norm" in name:
                gradient_stats["norms"].append(grad_norm)
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

