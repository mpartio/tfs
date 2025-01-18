import numpy as np
import torch
import os
import xarray as xr
import zarr
from glob import glob
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.kumaraswamy import Kumaraswamy
import pywt
from scipy.signal import medfilt2d


def hourly_split_1_1(data):
    # special case when n_hist = 1 and n_futu = 1 and hourly = True
    T, H, W, C = data.shape
    N = (T // 4) * 4
    data = data[:N, ...]
    data = data.reshape(-1, 4, H, W, C).transpose(1, 0, 2, 3, 4).reshape(-1, H, W, C)
    x_data = np.expand_dims(
        data[1::2,],
        axis=1,
    )
    y_data = np.expand_dims(data[::2], axis=1)
    return x_data, y_data


def instant_split(data, n_hist, n_futu):
    T, H, W, C = data.shape
    s_len = n_hist + n_futu
    N = (T // s_len) * s_len
    data = data[:N, ...]
    data = data.reshape(N // s_len, s_len, H, W, C)
    x_data = data[:, :n_hist, ...]
    y_data = data[:, n_hist : n_hist + n_futu, ...]
    return x_data, y_data


def read_zarr(filename):
    train_loader, val_loader = None, None

    ds = xr.open_mfdataset(filename, engine="zarr", data_vars="minimal")
    ec = ds["effective-cloudiness_heightAboveGround_0"]

    data = np.expand_dims(ec.values, axis=-1)
    data = data * 0.01

    time_step = (ds.time[1] - ds.time[0]).values.astype("timedelta64[s]").astype(int)

    assert time_step == 3600, "Time step is not 1 hour"

    T, H, W, C = data.shape

    N = int(T * 0.8)

    train_data = data[:N]
    val_data = data[N:]

    return train_data, val_data


def read_data(n_hist=1, n_futu=1, dataset_size="10k", batch_size=8, hourly=False):
    def to_dataset(arr):
        x_data, y_data = instant_split(arr, n_hist, n_futu)
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)
        return TensorDataset(x_data, y_data)

    if dataset_size.endswith(".zarr"):
        train_data, val_data = read_zarr(dataset_size)

        train_loader = DataLoader(
            to_dataset(train_data), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            to_dataset(val_data), batch_size=batch_size, shuffle=True
        )
        print("train number of samples: {}".format(len(train_loader.dataset)))
        print("val number of samples: {}".format(len(val_loader.dataset)))

        return train_loader, val_loader

    train_loader, val_loader = None, None

    for ds in ("train", "val"):
        s_len = n_hist + n_futu

        data = np.load(f"data/{ds}-{dataset_size}.npz")["arr_0"]

        if not hourly:
            x_data, y_data = instant_split(data, n_hist, n_futu)
        else:
            if n_hist == 1 and n_futu == 1:
                x_data, y_data = hourly_split_1_1(data)
            else:
                raise ValueError("hourly split only works for n_hist = 1 and n_fut = 1")

        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)

        print("{} number of samples: {}".format(ds, x_data.shape[0]))
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if ds == "train":
            train_loader = dataloader
        else:
            val_loader = dataloader

    return train_loader, val_loader


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_training_history(run_name, latest_only=False):
    files = glob(f"runs/{run_name}/202*.json")
    files = [x for x in files if "config" not in x]
    files.sort()

    if latest_only:
        latest_time = files[-1].split("/")[-1].split("-")[0]
        files = [x for x in files if latest_time in x]

        print("Latest time:", latest_time)
    files.sort()
    return files


def fast_sample_kumaraswamy(alpha, beta, weights, num_samples=1):
    B, T, Y, X, num_mix = alpha.shape

    # Sample uniform random variables
    u = torch.rand(num_samples, B, T, Y, X, num_mix, device=alpha.device)

    # Transform to Kumaraswamy samples with careful handling of numerical stability
    # K(u; a, b) = (1 - (1 - u^(1/b))^(1/a))

    # Step 1: u^(1/b)
    u_safe = torch.clamp(u, min=1e-6, max=1 - 1e-6)  # Prevent 0 and 1
    pow_1 = torch.pow(u_safe, 1.0 / beta.unsqueeze(0))

    # Step 2: 1 - u^(1/b)
    term_1 = torch.clamp(1 - pow_1, min=1e-6, max=1 - 1e-6)

    # Step 3: (1 - u^(1/b))^(1/a)
    samples = 1 - torch.pow(term_1, 1.0 / alpha.unsqueeze(0))

    # Weight the samples
    weighted_samples = (samples * weights.unsqueeze(0)).sum(dim=-1)

    return weighted_samples


def sample_kumaraswamy(alpha, beta, weights, num_samples=1):
    """
    Sample from a mixture of Kumaraswamy distributions.

    Parameters:
    - alpha (torch.Tensor): Tensor of alpha (concentration1) parameters, shape [num_mix].
    - beta (torch.Tensor): Tensor of beta (concentration0) parameters, shape [num_mix].
    - weights (torch.Tensor): Tensor of weights for each Kumaraswamy distribution in the mixture, shape [num_mix].
    - num_samples (int): Number of samples to draw for each distribution in the mixture.

    Returns:
    - torch.Tensor: Sampled values from the mixture, shape [num_samples].
    """
    B, T, Y, X, num_mix = alpha.shape

    # Ensure weights sum to 1 for valid mixture
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Sample from each Kumaraswamy component
    samples = []
    for i in range(num_mix):
        # Define a Kumaraswamy distribution with given alpha and beta for each component
        kumaraswamy_dist = Kumaraswamy(
            concentration1=alpha[..., i], concentration0=beta[..., i]
        )
        samples_i = kumaraswamy_dist.rsample(
            sample_shape=(num_samples,)
        )  # Shape: [num_samples]
        samples.append(samples_i)

    # Stack samples and weight each component
    samples = torch.stack(samples, dim=-1)  # Shape: [num_samples, num_mix]

    # Weighted sum over components, shape: [num_samples]
    weighted_samples = (samples * weights.unsqueeze(0)).sum(dim=-1)

    return weighted_samples


def sample_beta(alpha, beta, weights, num_samples=1):
    """
    Sample from a mixture of Beta distributions.

    Args:
        alpha (torch.Tensor): Tensor of alpha parameters, shape [batch_size, num_mix, ...].
        beta (torch.Tensor): Tensor of beta parameters, shape [batch_size, num_mix, ...].
        weights (torch.Tensor): Tensor of weights for each Beta distribution, shape [batch_size, num_mix, ...].
        num_samples (int): Number of samples to draw for each distribution.

    Returns:
        torch.Tensor: Sampled values, shape [batch_size, num_samples, ...].
    """
    batch_size, steps, H, W, num_mix = alpha.shape

    # Flatten batch and steps dimensions for easier handling
    alpha = alpha.view(batch_size * steps, num_mix, H, W)
    beta = beta.view(batch_size * steps, num_mix, H, W)
    weights = weights.reshape(batch_size * steps, num_mix, H, W)

    # Sample from each Beta distribution
    beta_distributions = torch.distributions.Beta(alpha, beta)

    # Shape: [batch_size * steps, num_mix, H, W]
    samples = beta_distributions.sample()

    # Weight the samples according to the weights of the Beta mixture
    # Reshape weights to match sample dimensions and apply softmax to ensure they sum to 1
    # Shape: [batch_size, num_mix, ...]
    weights = weights.softmax(dim=1)
    samples = (samples * weights).sum(dim=1)  # Weighted sum over num_mix

    # Reshape back to [batch_size, steps, H, W]
    samples = samples.view(batch_size, steps, H, W, 1)

    return samples


def sample_gaussian(mean, stde, weights, num_samples=1, aggregation="median"):
    num_mix = mean.shape[-1]

    samples = []

    for _ in range(num_samples):
        sample = torch.zeros(mean.shape[:-1])

        for i in range(num_mix):
            m = mean[..., i]
            s = stde[..., i]
            w = weights[..., i]

            sample = torch.normal(m, s)
            sample += w * sample

        samples.append(sample)

    if num_samples == 1:
        sample = samples[0]
    else:
        if aggregation == "mean":
            sample = torch.mean(samples, axis=0)
        elif aggregation == "median":
            sample = torch.median(samples, axis=0)

    return sample


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


class Dummy:
    """Dummy element that can be called with everything."""

    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


def setup_mlflow(experiment, url="https://mlflow.apps.ock.fmi.fi"):
    mlflow_enabled = os.environ.get("MLFLOW_DISABLE", None) is None

    if not mlflow_enabled:
        mlflow = Dummy()
        print("mlflow disabled")
        return mlflow

    try:
        import mlflow
    except ModuleNotFoundError:
        mlflow = Dummy()
        print("mlflow disabled")

    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment)

    return mlflow


def beta_function(x, y):
    """
    Compute the Beta function B(x, y) = Γ(x) * Γ(y) / Γ(x + y)
    using the log-Gamma function for numerical stability.
    """
    return torch.exp(torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y))


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
    local_snr = 10 * np.log10(smooth_pred**2 / (local_noise_power + 1e-9))
    return {
        "snr_db": snr_db,
        "noise_field": _noise_field,
        "local_snr_map": local_snr,
    }
