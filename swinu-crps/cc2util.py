import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from glob import glob


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


def roll_forecast(model, data, forcing, n_steps, loss_fn, num_members=1):
    total_loss = []
    total_tendencies = []
    total_predictions = []

    x = torch.concat((data[0], forcing[0]), dim=2)

    assert x.ndim == 5, "invalid dimensions for x: {}".format(x.shape)

    weights = torch.ones(n_steps).to(x.device)

    for step in range(n_steps):

        if loss_fn is not None:
            y_true = data[1][:, step]
            assert y_true.ndim == 4, "invalid dimensions for y: {}".format(y_true.shape)

        # Forward pass

        last_state = x[:, -1, 0].unsqueeze(1)

        tendencies = model(x, last_state, step + 1)
        predictions = last_state + tendencies

        if loss_fn is not None:
            assert (
                predictions.shape == y_true.shape
            ), "shapes don't match: {} vs {}".format(predictions.shape, y_true.shape)
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

    # B, T, C, H, W
    tendencies = torch.stack(total_tendencies).permute(1, 0, 2, 3, 4)
    predictions = torch.stack(total_predictions).permute(1, 0, 2, 3, 4)

    return total_loss, tendencies, predictions


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


def read_checkpoint(file_path, model):
    try:
        # Find latest checkpoint
        checkpoints = glob(f"{file_path}/*.ckpt")
        assert checkpoints, "No model checkpoints found in directory {}".format(
            file_path
        )
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, weights_only=False)
        new_state_dict = {}
        state_dict = ckpt["state_dict"]

        for k, v in state_dict.items():
            new_k = k.replace("model.", "")
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)

        return model

    except ValueError as e:
        print("Model checkpoint file not found from path: ", file_path)
        raise e
