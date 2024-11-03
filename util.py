import numpy as np
import torch
from glob import glob
from torch.utils.data import DataLoader, TensorDataset


def read_data(n_hist=1, n_futu=1, dataset_size="10k", batch_size=8, hourly=False):
    def hourly_split_1_1(data):
        # special case when n_hist = 1 and n_futu = 1 and hourly = True
        T, H, W, C = data.shape
        N = (T // 4) * 4
        data = data[:N, ...]
        data = (
            data.reshape(-1, 4, H, W, C).transpose(1, 0, 2, 3, 4).reshape(-1, H, W, C)
        )
        x_data = np.expand_dims(
            data[
                1::2,
            ],
            axis=1,
        )
        y_data = np.expand_dims(data[::2], axis=1)
        return x_data, y_data

    def instant_split(data, s_len):
        T, H, W, C = data.shape
        N = (T // s_len) * s_len
        data = data[:N, ...]
        data = data.reshape(N // s_len, s_len, H, W, C)
        x_data = data[:, :n_hist, ...]
        y_data = data[:, n_hist : n_hist + n_futu, ...]
        return x_data, y_data

    train_loader, val_loader = None, None

    for ds in ("train", "val"):
        s_len = n_hist + n_futu

        data = np.load(f"data/{ds}-{dataset_size}.npz")["arr_0"]

        if not hourly:
            x_data, y_data = instant_split(data, s_len)
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

    files.sort()

    files = [x for x in files if "config" not in x]

    if latest_only:
        latest_time = files[-1].split("/")[-1].split("-")[0]
        files = [x for x in files if latest_time in x]

    return files
