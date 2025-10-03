import numpy as np
import pywt
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch
import time
import importlib
from scipy.signal import medfilt2d
from glob import glob


def strip_prefix(state_dict: dict, prefix: str = "model."):
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def adapt_patch_embed(old_weight, new_weight_shape):
    # Zero-pad the weights to match new kernel size
    new_weight = torch.zeros(
        new_weight_shape, device=old_weight.device, dtype=old_weight.dtype
    )

    # Center the old weights in the new tensor
    pad_h = (new_weight_shape[2] - old_weight.shape[2]) // 2
    pad_w = (new_weight_shape[3] - old_weight.shape[3]) // 2

    new_weight[
        :, :, pad_h : pad_h + old_weight.shape[2], pad_w : pad_w + old_weight.shape[3]
    ] = old_weight

    return new_weight


def adapt_pos_embed(pos_embed, new_grid_size, old_grid_size, num_extra_tokens=0):
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


def adapt_checkpoint_to_model(ckpt_state_dict, model_state_dict, old_size, new_size):
    old_pos_embed_shape = ckpt_state_dict["pos_embed"].shape
    new_pos_embed_shape = model_state_dict["pos_embed"].shape

    # Positional embeddings are sized differently when input data resolution
    # is changed

    if old_pos_embed_shape != new_pos_embed_shape:
        print(
            "Different resolutions for pos_embed: checkpoint: {} vs model: {}, interpolating to match".format(
                list(old_pos_embed_shape),
                list(new_pos_embed_shape),
            )
        )
        new_pos_embed = adapt_pos_embed(
            ckpt_state_dict["pos_embed"], new_size, old_size
        )

        # Update the checkpoint.
        ckpt_state_dict["pos_embed"] = new_pos_embed

    # Patch embeddings are changed if patch size is changed

    keys = ["patch_embed.data_proj.weight", "patch_embed.forcing_proj.weight"]
    old_patch_embed_shape = ckpt_state_dict[keys[0]].shape
    new_patch_embed_shape = model_state_dict[keys[0]].shape

    if old_patch_embed_shape != new_patch_embed_shape:
        print(
            "Different kernel sizes for {}: checkpoint: {} vs model: {}, adapting to match".format(
                keys[0], list(old_patch_embed_shape), list(new_patch_embed_shape)
            )
        )

        for k in keys:
            new_patch_embed = adapt_patch_embed(
                ckpt_state_dict[k], model_state_dict[k].shape
            )
            ckpt_state_dict[k] = new_patch_embed

    keys = ["final_expand.0.weight", "final_expand.0.bias"]
    old_param = ckpt_state_dict[keys[0]]
    new_param_shape = model_state_dict[keys[0]].shape

    if old_param.shape != new_param_shape:
        print(
            "Different dimensions for {}: checkpoint: {} vs model: {}, adapting to match".format(
                keys[0], list(old_param.shape), list(new_param_shape)
            )
        )

        for k in keys:
            old_param = ckpt_state_dict[k]
            new_param_shape = model_state_dict[k].shape

            scale_factor = new_param_shape[0] // old_param.shape[0]
            if (
                scale_factor > 0
                and new_param_shape[0] == old_param.shape[0] * scale_factor
            ):
                # Repeat weights to match new size
                new_param = old_param.repeat_interleave(scale_factor, dim=0)

            ckpt_state_dict[k] = new_param

    return ckpt_state_dict


def find_latest_checkpoint_path(checkpoint_directory):
    assert checkpoint_directory is not None, "checkpoint_directory is 'None'"
    try:
        # Find latest checkpoint
        checkpoints = glob(f"{checkpoint_directory}/checkpoints/epoch*.ckpt")
        assert (
            checkpoints
        ), f"No model checkpoints found in directory {checkpoint_directory}/checkpoints, cwd={os.getcwd()}"
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        return latest_ckpt
    except ValueError as e:
        print("Model checkpoint file not found from path: {}/models".format(file_path))
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
    # 1) If DDP/FSDP has already inited, use it
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    # 2) torchrun / Lightning will set RANK and LOCAL_RANK
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    # 3) SLURM’s local‐id (if you still need it for sbatch)
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    # 4) last‐resort
    return 0


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


def string_to_type(type_str: str) -> type:
    assert "." in type_str
    module_name, _, type_name = type_str.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, type_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import type '{type_str}': {e}") from e
    return cls


def effective_parameters(num_devices, batch_size, lr, total_iterations):
    if num_devices == 1:
        # keep current batch size
        return batch_size, lr

    effective_bs = batch_size * num_devices
    effective_lr = lr * num_devices

    return batch_size, effective_lr


def create_directory_structure(base_directory):
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(f"{base_directory}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_directory}/logs", exist_ok=True)
    os.makedirs(f"{base_directory}/figures", exist_ok=True)
