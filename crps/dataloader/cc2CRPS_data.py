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
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def get_default_normalization_methods(custom_methods):
    default_methods = {
        "insolation": "none",
        "t_1000": "standard",
        "t_500": "standard",
        "t_700": "standard",
        "t_850": "standard",
        "t_925": "standard",
        "tcc": "none",
        "u_1000": "standard",
        "u_500": "standard",
        "u_700": "standard",
        "u_850": "standard",
        "u_925": "standard",
        "v_1000": "standard",
        "v_500": "standard",
        "v_700": "standard",
        "v_850": "standard",
        "v_925": "standard",
        "z_1000": "standard",
        "z_500": "standard",
        "z_700": "standard",
        "z_850": "standard",
        "z_925": "standard",
    }

    if custom_methods is not None:
        default_methods.update(custom_methods)

    rank_zero_info(f"normalization methods: {default_methods}")

    return default_methods


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
        zarr_path: list[str],
        group_size: int,
        prognostic_params: list[str],
        forcing_params: list[str],
        normalization_methods: dict,
        disable_normalization: bool,
        input_resolution: tuple[int, int],
        return_metadata: bool = False,
    ):
        self.data = open_dataset(zarr_path)
        self.group_size = group_size
        self.time_steps = len(self.data.dates)

        assert type(prognostic_params) == list
        assert type(forcing_params) == list

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

        self.disable_normalization = disable_normalization

        if self.disable_normalization is False:
            self.normalization_methods = normalization_methods
            assert self.normalization_methods is not None

            self._setup_normalization()

        self.input_resolution = self.data.field_shape  # H, W

        self.return_metadata = return_metadata
        self.dates = self.data.dates

    def _setup_normalization(self):
        # Pre-compute combined indexes and params
        self.combined_indexes = self.data_indexes + self.forcings_indexes
        self.combined_params = self.prognostic_params + self.forcing_params
        self.total_channels = len(
            self.combined_indexes
        )  # Number of channels before E multiplication

        # default:
        dtype = torch.float32
        methods = [self.normalization_methods[k] for k in self.combined_params]

        # Pre-reshape statistics for all combined channels
        C = self.total_channels  # This will be multiplied by E in __getitem__
        self.mins = torch.tensor(
            self.data.statistics["minimum"][self.combined_indexes].reshape(1, C, 1, 1),
            dtype=dtype,
        )
        self.maxs = torch.tensor(
            self.data.statistics["maximum"][self.combined_indexes].reshape(1, C, 1, 1),
            dtype=dtype,
        )
        self.means = torch.tensor(
            self.data.statistics["mean"][self.combined_indexes].reshape(1, C, 1, 1),
            dtype=dtype,
        )
        self.stds = torch.tensor(
            self.data.statistics["stdev"][self.combined_indexes].reshape(1, C, 1, 1),
            dtype=dtype,
        )
        # Create a single method tensor for indexing
        self.method_tensor = torch.tensor(
            [0 if m == "none" else 1 if m == "minmax" else 2 for m in methods],
            dtype=torch.long,
        ).view(1, C, 1, 1)

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
        indices = [self.data.name_to_index[x] for x in params]

        # Early return for "none" case
        if len(methods) == 1 and methods[0] == "none":
            return tensor

        # Compute normalizations only where needed
        result = tensor.clone()  # Avoid modifying input tensor directly
        minmax_mask = self.method_tensor == 1
        std_mask = self.method_tensor == 2

        if minmax_mask.any():
            result = torch.where(
                minmax_mask,
                (tensor - self.mins)
                / (self.maxs - self.mins + 1e-8),  # Add epsilon for stability
                result,
            )
        if std_mask.any():
            result = torch.where(
                std_mask,
                (tensor - self.means) / (self.stds + 1e-8),  # Add epsilon only for std
                result,
            )

        return result.to(tensor.dtype)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # Extract the full slice (data + forcings) from self.data
        combined_indexes = self.data_indexes + self.forcings_indexes
        combined_params = self.prognostic_params + self.forcing_params

        # Get the combined data
        combined = self.data[
            actual_idx : actual_idx + self.group_size, combined_indexes, ...
        ]

        T, C, E, HW = combined.shape

        assert E == 1

        combined = combined.reshape(
            T, C * E, self.input_resolution[0], self.input_resolution[1]
        )
        combined = torch.from_numpy(combined)

        # Normalize the combined tensor
        if self.disable_normalization is False:
            combined = self.normalize(combined, combined_params)

        # Split into data and forcings based on original channel counts
        data_channels = len(self.data_indexes) * E
        data = combined[:, :data_channels, :, :]
        forcing = combined[:, data_channels:, :, :]

        if self.return_metadata:
            sequence_dates = self.dates[actual_idx : actual_idx + self.group_size]

            return data, forcing, sequence_dates.astype(np.float64)

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


class SplitWrapper:
    def __init__(self, dataset, n_x, apply_smoothing=False):
        self.dataset = dataset
        self.n_x = n_x
        self.apply_smoothing = apply_smoothing
        self.return_metadata = getattr(dataset, "return_metadata", False) or getattr(
            getattr(dataset, "dataset", None), "return_metadata", False
        )

    def __getitem__(self, idx):
        if self.return_metadata:
            data, forcing, sequence_dates = self.dataset[idx]
        else:
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

        if self.return_metadata:
            dates_x = sequence_dates[: self.n_x]
            dates_y = sequence_dates[self.n_x :]

            return [(data_x, data_y), forcing, (dates_x, dates_y)]

        return [(data_x, data_y), forcing]

    def __len__(self):
        return len(self.dataset)


class cc2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: list[str],
        input_resolution: tuple[int, int],
        prognostic_params: tuple[str, ...],
        forcing_params: tuple[str, ...],
        history_length: int = 2,
        rollout_length: int = 1,
        val_split: float = 0.1,
        test_split: float = 0.0,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 6,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 3,
        pin_memory: bool = True,
        normalization: dict[str, str] | None = None,
        disable_normalization: bool = False,
        apply_smoothing: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.ds_train: Subset | None = None
        self.ds_val: Subset | None = None
        self.ds_test: Subset | None = None
        self._full_dataset: AnemoiDataset | None = None  # Cache the full dataset

    def _get_or_create_full_dataset(self) -> AnemoiDataset:
        if self._full_dataset is None:
            norm_methods = None
            if not self.hparams.disable_normalization:
                norm_methods = get_default_normalization_methods(
                    self.hparams.normalization
                )

            self._full_dataset = AnemoiDataset(
                zarr_path=self.hparams.data_path,
                group_size=self.hparams.history_length
                + self.hparams.rollout_length,  # n_x + n_y
                input_resolution=self.hparams.input_resolution,
                prognostic_params=self.hparams.prognostic_params,
                forcing_params=self.hparams.forcing_params,
                normalization_methods=norm_methods,
                disable_normalization=self.hparams.disable_normalization,
            )
        return self._full_dataset

    def setup(self, stage: str | None = None):
        if stage == "fit" and self.ds_train is not None and self.ds_val is not None:
            return
        if stage == "test" and self.ds_test is not None:
            return
        if stage == "predict" and hasattr(self, "ds_predict"):
            return  # Use test for predict if no specific ds_predict

        ds_full = self._get_or_create_full_dataset()
        num_total_valid_samples = len(ds_full)

        indices = np.arange(num_total_valid_samples)
        num_samples_to_split = num_total_valid_samples

        # Shuffle indices before splitting
        rng = np.random.RandomState(self.hparams.seed)
        rng.shuffle(indices)

        # Calculate split sizes based on the potentially limited number of samples
        val_size = int(self.hparams.val_split * num_samples_to_split)
        test_size = int(self.hparams.test_split * num_samples_to_split)
        train_size = num_samples_to_split - val_size - test_size

        if train_size < 0:
            raise ValueError(
                f"Dataset size {num_samples_to_split} (after limit/validation) is too small for val_split={self.hparams.val_split} and test_split={self.hparams.test_split}"
            )

        rank_zero_info(
            f"Dataset split sizes (based on {num_samples_to_split} samples): Train={train_size}, Validation={val_size}, Test={test_size}"
        )

        # Get the actual indices corresponding to the shuffled array segments
        train_indices = indices[val_size + test_size :]
        val_indices = indices[:val_size]
        test_indices = indices[val_size : val_size + test_size]

        # Create Subset datasets based on the stage
        # stage='fit' or stage=None will setup train and val
        if stage == "fit" or stage is None:
            self.ds_train = Subset(ds_full, train_indices)
            self.ds_val = Subset(ds_full, val_indices)

        # stage='test' or stage=None will setup test
        if stage == "test" or stage is None:
            self.ds_test = Subset(ds_full, test_indices)

        # stage='predict' or stage=None will setup predict (using test set here)
        if stage == "predict" or stage is None:
            if (
                self.ds_test is None
            ):  # Ensure test set is created if only predict stage runs
                self.ds_test = Subset(ds_full, test_indices)
            self.ds_predict = self.ds_test  # Assign test dataset to predict dataset

    def _get_dataloader(
        self, dataset: Subset | None, shuffle: bool = False, stage: str | None = None
    ) -> DataLoader:
        if dataset is None:
            raise ValueError(
                "Dataset not available. Ensure setup() has been called for the correct stage."
            )

        if hasattr(dataset, "dataset") and isinstance(dataset.dataset, AnemoiDataset):
            # Set flag based on stage
            is_test_or_predict = stage == "test" or stage == "predict"
            dataset.dataset.return_metadata = is_test_or_predict

        wrapped_dataset = SplitWrapper(
            dataset=dataset,
            n_x=self.hparams.history_length,  # Use hparams
            apply_smoothing=self.hparams.apply_smoothing,  # Use hparams
        )

        return DataLoader(
            wrapped_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            prefetch_factor=(
                self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
            ),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.ds_train, shuffle=True, stage="fit")

    def val_dataloader(self):
        return self._get_dataloader(self.ds_val, shuffle=False, stage="fit")

    def test_dataloader(self):
        return self._get_dataloader(self.ds_test, shuffle=False, stage="test")

    def predict_dataloader(self):
        if hasattr(self, "ds_predict"):
            return self._get_dataloader(self.ds_predict, shuffle=False)
        else:  # Fallback if predict stage wasn't explicitly setup
            print("Predict dataset not explicitly set up, using test dataset.")
            return self._get_dataloader(self.ds_test, shuffle=False, stage="predict")
