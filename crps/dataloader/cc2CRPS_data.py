import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import lightning as L
import numpy as np
import zarr
from anemoi.datasets import open_dataset
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset


def get_default_normalization_methods():
    return {
        "insolation": "none",
        "t_1000": "standard",
        "t_500": "standard",
        "t_700": "standard",
        "t_850": "standard",
        "t_925": "standard",
        "tcc": "none",
        "u_isobaricInhPa_1000": "standard",
        "u_isobaricInhPa_500": "standard",
        "u_isobaricInhPa_700": "standard",
        "u_isobaricInhPa_850": "standard",
        "u_isobaricInhPa_925": "standard",
        "v_isobaricInhPa_1000": "standard",
        "v_isobaricInhPa_500": "standard",
        "v_isobaricInhPa_700": "standard",
        "v_isobaricInhPa_850": "standard",
        "v_isobaricInhPa_925": "standard",
        "z_isobaricInhPa_1000": "standard",
        "z_isobaricInhPa_500": "standard",
        "z_isobaricInhPa_700": "standard",
        "z_isobaricInhPa_850": "standard",
        "z_isobaricInhPa_925": "standard",
    }


def augment_data(x, y):
    xshape, yshape = x.shape, y.shape
    if torch.rand(1).item() > 0.6:
        return x, y

    # Random flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [-2])  # Horizontal flip
        y = torch.flip(y, [-2])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [-1])  # Vertical flip
        y = torch.flip(y, [-1])

    # Random 90-degree rotations
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k, [-2, -1])
    y = torch.rot90(y, k, [-2, -1])

    assert x.shape == xshape, "Invalid y shape after augmentation: {} vs {}".format(
        x.shape, xshape
    )
    assert y.shape == yshape, "Invalid x shape after augmentation: {} vs {}".format(
        y.shape, yshape
    )

    return x, y


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

    orig_shape = x.shape

    # Add batch dimension if needed
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        B, C, H, W = x.shape
    elif len(x.shape) == 5:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
    else:
        B, C, H, W = x.shape

    assert C < 50, "The order should be B, C, H, W, but got: {}".format(x.shape)

    assert x.ndim == 4, "data needs to have 4 dimensions, got: {}".format(x.shape)

    # Move kernel to same device as input
    kernel = kernel.to(x.device)

    # Apply smoothing channel by channel
    pad = kernel_size // 2
    smoothed = []
    for c in range(C):
        channel = x[:, c : c + 1, ...]
        channel = F.pad(channel, (pad, pad, pad, pad), mode="reflect")
        channel = F.conv2d(channel, kernel)
        smoothed.append(channel)

    x = torch.cat(smoothed, dim=1)
    x = torch.clamp(x, 0, 1)

    x = x.reshape(orig_shape)
    return x


class AnemoiDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        group_size: int,
        input_resolution: tuple,
        prognostic_params: list,
        forcing_params: list,
        normalization_methods: dict,
    ):
        self.data = open_dataset(zarr_path)
        self.group_size = group_size
        self.time_steps = len(self.data.dates)

        self.prognostic_params = prognostic_params
        self.forcing_params = forcing_params

        self.data_indexes = [self.data.name_to_index[x] for x in self.prognostic_params]
        self.forcings_indexes = [
            self.data.name_to_index[x] for x in self.forcing_params
        ]
        self.input_resolution = input_resolution
        assert self.time_steps >= group_size

        self.missing_indices = set(self.data.missing)
        self.valid_indices = [
            i
            for i in range(self.time_steps - self.group_size - 1)
            if not self._sequence_has_missing_data(i)
        ]

        self.statistics = self.data.statistics

        self.normalization_methods = normalization_methods

    def _sequence_has_missing_data(self, start_idx):
        for i in range(start_idx, start_idx + self.group_size):
            if i in self.missing_indices:
                return True
        return False

    def __len__(self):
        return len(self.valid_indices)

    def normalize(self, tensor: torch.tensor, params: list):
        T, C, H, W = tensor.shape

        methods = [self.normalization_methods[k] for k in params]

        mins = self.statistics["minimum"].reshape(1, C, 1, 1)
        maxs = self.statistics["maximum"].reshape(1, C, 1, 1)
        means = self.statistics["mean"].reshape(1, C, 1, 1)
        stds = self.statistics["stdev"].reshape(1, C, 1, 1) + 1e-8

        dtype = tensor.dtype
        device = tensor.device

        # Create masks for each normalization type
        minmax_mask = torch.tensor(
            [m == "minmax" for m in methods], dtype=dtype, device=device
        ).view(1, C, 1, 1)
        std_mask = torch.tensor(
            [m == "standard" for m in methods], dtype=dtype, device=device
        ).view(1, C, 1, 1)
        none_mask = torch.tensor(
            [m == "none" for m in methods], dtype=dtype, device=device
        ).view(1, C, 1, 1)

        # Apply normalizations to all data
        minmax_normalized = (tensor - mins) / (maxs - mins)
        std_normalized = (tensor - means) / stds

        # Combine results using masks (with no normalization option)
        result = (
            minmax_mask * minmax_normalized
            + std_mask * std_normalized
            + none_mask * tensor
        )

        result = result.to(dtype)

        return result

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        data = self.data[
            actual_idx : actual_idx + self.group_size, self.data_indexes, ...
        ]

        # (3, 1, 1, 16384)

        T, C, E, HW = data.shape

        data = data.reshape(
            T, C * E, self.input_resolution[0], self.input_resolution[1]
        )
        data = torch.tensor(data)

        forcing = self.data[
            actual_idx : actual_idx + self.group_size, self.forcings_indexes, ...
        ]
        T, C, E, HW = forcing.shape
        forcing = forcing.reshape(
            T, C * E, self.input_resolution[0], self.input_resolution[1]
        )
        forcing = torch.tensor(forcing)
        forcing = self.normalize(forcing, self.forcing_params)

        return data, forcing


class HourlyZarrDataset(Dataset):
    def __init__(self, zarr_path, group_size):
        # Open the zarr array without loading data
        self.data = zarr.open(zarr_path, mode="r")
        self.group_size = group_size
        self.time_steps, _, _, _ = self.data.shape

        assert self.time_steps >= group_size

    def __len__(self):
        return self.time_steps - self.group_size - 1

    def __getitem__(self, idx):
        # Get consecutive samples
        samples = self.data[idx : idx + self.group_size]

        # Convert to tensor
        samples = torch.from_numpy(samples)

        return samples


class HourlyStreamZarrDataset(Dataset):
    def __init__(self, zarr_path, group_size):
        # Open the zarr array without loading data
        self.data = zarr.open(zarr_path, mode="r")
        self.num_streams, self.time_steps, _, _, _ = self.data.shape

        assert self.num_streams == 4, "Only 4 streams are supported"

        self.group_size = group_size

        self.valid_starts = self._get_valid_starts()

    def _get_valid_starts(self):
        # We need enough room for group_size consecutive samples
        max_start = self.time_steps - self.group_size + 1
        return np.arange(0, max_start)

    def __len__(self):
        return len(self.valid_starts) * self.num_streams

    def __getitem__(self, idx):
        # Convert flat index to (stream_idx, start_idx)
        stream_idx = idx % self.num_streams
        start_pos = self.valid_starts[idx // self.num_streams]

        # Get consecutive samples for this stream
        samples = self.data[stream_idx, start_pos : start_pos + self.group_size]

        # Convert to tensor
        samples = torch.from_numpy(samples)

        return samples


class SplitWrapper:
    def __init__(self, dataset, n_x, apply_smoothing=False):
        self.dataset = dataset
        self.n_x = n_x
        self.apply_smoothing = apply_smoothing

    def __getitem__(self, idx):
        data, forcing = self.dataset[idx]
        # Split into x and y
        data_x = data[: self.n_x]
        data_y = data[self.n_x :]

        if self.apply_smoothing:
            data_x = gaussian_smooth(data_x)
            data_y = gaussian_smooth(data_y)
            forcing = gaussian_smooth(forcing)

        assert data_x.ndim == 4, "Invalid data_x shape: {}".format(data_x.shape)
        assert data_y.ndim == 4, "Invalid data_y shape: {}".format(data_y.shape)
        assert forcing.ndim == 4, "Invalid forcing shape: {}".format(forcing.shape)

        return [(data_x, data_y), forcing]

    def __len__(self):
        return len(self.dataset)


class cc2DataModule(L.LightningDataModule):
    def __init__(self, config):
        self.n_x = config.history_length
        self.n_y = config.rollout_length
        self.val_split = 0.1

        self.cc2_train = None
        self.cc2_val = None
        self.cc2_test = None

        self.config = config
        print("Reading data from {}".format(config.data_path))
        self.setup(config.data_path)

    def setup(self, zarr_path):
        ds = AnemoiDataset(
            zarr_path,
            group_size=self.n_x + self.n_y,
            input_resolution=self.config.input_resolution,
            prognostic_params=self.config.prognostic_params,
            forcing_params=self.config.forcing_params,
            normalization_methods=get_default_normalization_methods(),
        )
        indices = np.arange(len(ds))

        if self.config.limit_data_to is not None:
            indices = indices[: self.config.limit_data_to]
        rng = np.random.RandomState(0)
        rng.shuffle(indices)

        val_size = int(self.val_split * len(indices))
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        self.cc2_train = Subset(ds, train_indices)
        self.cc2_val = Subset(ds, val_indices)

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            SplitWrapper(
                dataset, n_x=self.n_x, apply_smoothing=self.config.apply_smoothing
            ),
            batch_size=self.config.batch_size,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            shuffle=shuffle,
            prefetch_factor=3,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.cc2_train, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.cc2_val)
