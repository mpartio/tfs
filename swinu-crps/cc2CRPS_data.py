import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from zarr_dataset import HourlyStreamZarrDataset, SplitWrapper


def smooth_data(data: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0):
    """
    Smooths 2D data using Gaussian blur

    Args:
        data: Input tensor of shape (B, C, H, W) or (C, H, W)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
    """
    # Add batch dimension if needed
    if data.dim() == 3:
        data = data.unsqueeze(0)

    # Create Gaussian kernel
    channels = data.size(1)
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size))

    # Fill kernel with Gaussian values
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - center
            dy = y - center
            kernel[0, 0, x, y] = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Apply to all channels
    kernel = kernel.to(data.device)
    smoothed = F.conv2d(data, kernel, padding=center, groups=channels)

    # Ensure output stays in [0,1]
    smoothed = torch.clamp(smoothed, 0, 1)

    return smoothed.squeeze(0) if data.dim() == 3 else smoothed


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


def partition(tensor, n_x, n_y):

    if tensor.ndim == 5:
        S, T, H, W, C = tensor.shape
    else:
        # era5 data
        tensor = tensor.unsqueeze(-1)
        T, H, W, C = tensor.shape

    group_size = n_x + n_y

    # Reshape into groups of (n_x + n_y), with padding if necessary
    num_groups = T // group_size
    new_length = num_groups * group_size

    if tensor.ndim == 5:
        padded_tensor = tensor[:, :new_length]

        # Reshape into groups
        reshaped = padded_tensor.reshape(S, -1, group_size, H, W)

        # Merge streams into one dim
        reshaped = reshaped.reshape(
            S * reshaped.shape[1], group_size, H, W
        )  # N, G, H, W
    else:
        padded_tensor = tensor[:new_length]

        # Reshape into groups
        reshaped = padded_tensor.reshape(-1, group_size, H, W)

    # Extract single elements and blocks
    x = reshaped[:, :n_x]
    y = reshaped[:, n_x:]

    assert x.shape[0] > 0, "Not enough elements"

    return x, y


def shuffle_to_hourly_streams(data):
    T, H, W, C = data.shape
    N = (T // 4) * 4
    data = data[:N, ...]

    data = data.reshape(-1, 4, H, W, C).transpose(1, 0, 2, 3, 4)

    return data


def read_data(stage, dataset_size, n_x, n_y):

    filename = f"../data/{stage}-{dataset_size}.npz"
    data = np.load(filename)["arr_0"]

    if dataset_size != "era5":
        data = shuffle_to_hourly_streams(data)
    data = torch.tensor(data)
    x_data, y_data = partition(data, n_x, n_y)

    if y_data.ndim == 4:
        y_data = y_data.unsqueeze(1)

    assert x_data.ndim == 4, "invalid dimensions for x: {}".x_data.shape
    assert y_data.ndim == 5, "invalid dimensions for y: {}".y_data.shape

    return TensorDataset(x_data, y_data)


class cc2DataModule(L.LightningDataModule):
    def __init__(self, batch_size, n_x=1, n_y=1, dataset_size="150k"):
        self.batch_size = batch_size
        self.n_x = n_x
        self.n_y = n_y

        self.cc2_train = None
        self.cc2_val = None
        self.cc2_test = None

        self.setup(dataset_size, n_x, n_y)

    def setup(self, dataset_size, n_x, n_y):
        self.cc2_val = read_data("val", dataset_size, n_x, n_y)
        self.cc2_train = read_data("train", dataset_size, n_x, n_y)

        try:
            self.cc2_test = read_data("test", dataset_size, n_x, n_y)
        except FileNotFoundError as e:
            print("Test data not found for dataset {}".format(dataset_size))

    def train_dataloader(self):
        return DataLoader(
            self.cc2_train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.cc2_val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        if self.cc2_test is not None:
            return DataLoader(self.cc2_test, batch_size=self.batch_size)


class cc2ZarrModule(L.LightningDataModule):
    def __init__(self, zarr_path: str, batch_size: int, n_x: int = 1, n_y: int = 1):
        self.batch_size = batch_size
        self.n_x = n_x
        self.n_y = n_y
        self.val_split = 0.1

        self.cc2_train = None
        self.cc2_val = None
        self.cc2_test = None

        self.setup(zarr_path)

    def setup(self, zarr_path):
        samples_per_stream = self.n_x + self.n_y

        ds = HourlyStreamZarrDataset(zarr_path, samples_per_stream=samples_per_stream)
        indices = np.arange(len(ds))

        rng = np.random.RandomState(0)
        rng.shuffle(indices)

        val_size = int(self.val_split * len(ds))
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        self.cc2_train = Subset(ds, train_indices)
        self.cc2_val = Subset(ds, val_indices)

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            SplitWrapper(dataset, n_x=self.n_x),
            batch_size=self.batch_size,
            num_workers=6,
            pin_memory=True,
            shuffle=shuffle,
            prefetch_factor=3,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.cc2_train, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.cc2_val)
