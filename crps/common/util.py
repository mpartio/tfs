import numpy as np
import pywt
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch
import time
import importlib
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

    assert rank == 0, f"rank is {rank}, not 0"

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

    return next_num


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


def create_directory_structure(base_directory):
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(f"{base_directory}/checkpoints", exist_ok=True)
    os.makedirs(f"{base_directory}/logs", exist_ok=True)
    os.makedirs(f"{base_directory}/figures", exist_ok=True)
