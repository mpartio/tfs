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
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset


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
        samples = self.dataset[idx]  # shape is T, H, W, C

        # Split into x and y
        x = samples[: self.n_x]  # shape: [Tx, H, W, C]
        y = samples[self.n_x :]  # shape: [Ty, H, W, C]
        x = x.squeeze(-1)  # shape: [Tx, H, W]

        # Reshape y: move C after T
        y = y.permute(0, 3, 1, 2)  # shape: [Ty, C, H, W]

        if self.apply_smoothing:
            x = gaussian_smooth(x)
            y = gaussian_smooth(y)

        return x, y

    def __len__(self):
        return len(self.dataset)


class cc2DataModule(L.LightningDataModule):
    def __init__(
        self,
        zarr_path: str,
        batch_size: int,
        n_x: int = 1,
        n_y: int = 1,
        limit_to: int = None,
        apply_smoothing: bool = False,
    ):
        self.batch_size = batch_size
        self.n_x = n_x
        self.n_y = n_y
        self.val_split = 0.1

        self.cc2_train = None
        self.cc2_val = None
        self.cc2_test = None
        self.limit_to = limit_to
        self.apply_smoothing = apply_smoothing

        print("Reading data from {}".format(zarr_path))
        self.setup(zarr_path)

    def setup(self, zarr_path):
        if "era5" in zarr_path:
            ds = HourlyZarrDataset(zarr_path, group_size=self.n_x + self.n_y)
        else:
            ds = HourlyStreamZarrDataset(zarr_path, group_size=self.n_x + self.n_y)
        indices = np.arange(len(ds))

        if self.limit_to is not None:
            indices = indices[: self.limit_to]
        rng = np.random.RandomState(0)
        rng.shuffle(indices)

        val_size = int(self.val_split * len(indices))
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        self.cc2_train = Subset(ds, train_indices)
        self.cc2_val = Subset(ds, val_indices)

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            SplitWrapper(dataset, n_x=self.n_x),
            batch_size=self.batch_size,
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
