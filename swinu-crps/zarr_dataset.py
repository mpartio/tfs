import torch
import zarr
import numpy as np
from torch.utils.data import Dataset


class HourlyZarrDataset(Dataset):
    def __init__(self, zarr_path, group_size, shuffle=True):
        # Open the zarr array without loading data
        self.data = zarr.open(zarr_path, mode="r")
        self.group_size = group_size
        self.time_steps, _, _, _ = self.data.shape

    def __len__(self):
        return self.time_steps

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
    def __init__(self, dataset, n_x):
        self.dataset = dataset
        self.n_x = n_x

    def __getitem__(self, idx):
        samples = self.dataset[idx]  # shape is T, H, W, C

        # Split into x and y
        x = samples[: self.n_x]  # shape: [Tx, H, W, C]
        y = samples[self.n_x :]  # shape: [Ty, H, W, C]
        x = x.squeeze(-1)  # shape: [H, W, C]

        # Reshape y: move C after T
        y = y.permute(0, 3, 1, 2)  # shape: [Ty, C, H, W]

        return x, y

    def __len__(self):
        return len(self.dataset)
