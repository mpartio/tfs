import torch
import zarr
import numpy as np
from torch.utils.data import Dataset
from cc2util import partition


class HourlyStreamZarrDataset(Dataset):
    def __init__(self, zarr_path, samples_per_stream, shuffle=True):
        # Open the zarr array without loading data
        self.data = zarr.open(zarr_path, mode="r")
        self.num_streams, self.time_steps, self.H, self.W, self.C = self.data.shape

        assert self.num_streams == 4, "Only 4 streams are supported"

        self.samples_per_stream = samples_per_stream

        self.valid_starts = self._get_valid_starts()

        # self.index = list(range(len(self.data)))
        # if shuffle:
        # shuffle(self.index)

    def _get_valid_starts(self):
        # We need enough room for samples_per_stream consecutive samples
        max_start = self.time_steps - self.samples_per_stream + 1
        return np.arange(0, max_start)

    def __len__(self):
        return len(self.valid_starts) * self.num_streams

    def __getitem__(self, idx):
        # Convert flat index to (stream_idx, start_idx)
        stream_idx = idx % self.num_streams
        start_pos = self.valid_starts[idx // self.num_streams]

        # Get consecutive samples for this stream
        samples = self.data[stream_idx, 
                          start_pos:start_pos + self.samples_per_stream]

        # Convert to tensor
        samples = torch.from_numpy(samples)

        return samples


class SplitWrapper:
    def __init__(self, dataset, n_x):
        self.dataset = dataset
        self.n_x = n_x

    def __getitem__(self, idx):
        samples = self.dataset[idx] # shape is T, H, W, C

        # Split into x and y
        x = samples[:self.n_x]    # shape: [Tx, H, W, C]
        y = samples[self.n_x:]    # shape: [Ty, H, W, C]
        x = x.squeeze(-1)         # shape: [H, W, C]

        # Reshape y: move C after T
        y = y.permute(0, 3, 1, 2)    # shape: [Ty, C, H, W]

        return x, y

    def __len__(self):
        return len(self.dataset)
