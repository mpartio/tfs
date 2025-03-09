import numpy as np
import pywt
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch
import time
from scipy.signal import medfilt2d
from glob import glob


def interpolate_pos_embed(pos_embed, new_grid_size, old_grid_size, num_extra_tokens=0):
    embed_dim = pos_embed.shape[-1]

    # Separate extra tokens (like a class token) from the regular positional tokens.
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]

    # Reshape tokens into a 2D grid.
    pos_tokens = pos_tokens.reshape(
        1, old_grid_size[0], old_grid_size[1], embed_dim
    ).permute(0, 3, 1, 2)

    # Interpolate to the new grid size.
    pos_tokens = F.interpolate(
        pos_tokens,
        size=new_grid_size,
        mode="bicubic",
        align_corners=False,
    )

    # Reshape back to the original format.
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
        1, new_grid_size[0] * new_grid_size[1], embed_dim
    )

    # Concatenate the extra tokens back.
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed


def read_checkpoint(
    file_path,
    model,
    interpolate_positional_embeddings=False,
    old_size=None,
    new_size=None,
):
    try:
        # Find latest checkpoint
        checkpoints = glob(f"{file_path}/*.ckpt")
        assert checkpoints, "No model checkpoints found in directory {}".format(
            file_path
        )
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, weights_only=False)
        state_dict = ckpt["state_dict"]

        old_pos_embed_shape = state_dict["pos_embed"].shape
        new_pos_embed_shape = model.state_dict()["pos_embed"].shape

        if (
            interpolate_positional_embeddings
            and old_pos_embed_shape != new_pos_embed_shape
        ):
            print(
                "Different resolutions for pos_embed: checkpoint: {} vs model: {}, interpolating to match".format(
                    list(old_pos_embed_shape),
                    list(new_pos_embed_shape),
                )
            )
            new_pos_embed = interpolate_pos_embed(
                state_dict["pos_embed"], new_size, old_size
            )

            # Update the checkpoint.
            state_dict["pos_embed"] = new_pos_embed

        model.load_state_dict(state_dict)

        return model

    except ValueError as e:
        print("Model checkpoint file not found from path: ", file_path)
        raise e


def get_next_run_number(base_dir):
    rank = get_rank()
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


def get_rank():
    # If distributed is initialized, use its rank
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # Otherwise, try SLURM_PROCID first, then LOCAL_RANK, default to 0.
    return int(os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", 0)))


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


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_wavelet_snr(prediction, reference=None, wavelet="db2", level=2):
    """
    Calculate SNR using wavelet decomposition, specifically designed for neural network outputs
    with values between 0 and 1 and sharp features.

    Parameters:
    prediction (numpy.ndarray): Predicted field from neural network (values 0-1)
    reference (numpy.ndarray, optional): Ground truth field if available
    wavelet (str): Wavelet type to use (default: 'db2' which preserves edges well)
    level (int): Decomposition level

    Returns:
    dict: Dictionary containing SNR metrics and noise field
    """
    if prediction.ndim != 2:
        raise ValueError("Input must be 2D array: {}".format(prediction.shape))
    if reference is not None and reference.ndim != 2:
        raise ValueError("Reference must be 2D array: {}".format(reference.shape))

    # If we have reference data, we can calculate noise directly
    if reference is not None:
        noise_field = prediction - reference
        _noise_field = noise_field.numpy()

    # If no reference, estimate noise using wavelet decomposition
    else:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(prediction, wavelet, level=level)

        # Get highest frequency details (typically noise)
        cH1, cV1, cD1 = coeffs[1]

        # Estimate noise standard deviation using MAD estimator
        # MAD is more robust to outliers than standard deviation
        noise_std = np.median(np.abs(cD1)) / 0.6745  # 0.6745 is the MAD scaling factor

        # Reconstruct noise field
        coeffs_noise = [np.zeros_like(coeffs[0])]  # Set approximation to zero
        coeffs_noise.extend([(cH1, cV1, cD1)])  # Keep finest details
        coeffs_noise.extend(
            [tuple(np.zeros_like(d) for d in coeff) for coeff in coeffs[2:]]
        )  # Set coarser details to zero

        noise_field = pywt.waverec2(coeffs_noise, wavelet)

        # Normalize noise field to match input scale
        _noise_field = noise_field * (noise_std / np.std(noise_field))

    _prediction = prediction.numpy().astype(np.float32)

    # Calculate signal power (using smoothed prediction as signal estimate)
    smooth_pred = medfilt2d(_prediction, kernel_size=3)
    signal_power = np.mean(smooth_pred**2)

    # Calculate noise power
    noise_power = np.mean(_noise_field**2)

    # Calculate SNR
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    # Calculate local SNR map to identify problematic regions
    local_noise_power = medfilt2d(_noise_field**2, kernel_size=5)
    local_noise_power = local_noise_power[
        : smooth_pred.shape[0], : smooth_pred.shape[1]
    ]

    local_snr = 10 * np.log10(1e-9 + smooth_pred**2 / (local_noise_power + 1e-9))
    return {
        "snr_db": snr_db,
        "noise_field": _noise_field,
        "local_snr_map": local_snr,
    }
