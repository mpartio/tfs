import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.normal import Normal

# from torch.distributions.kumaraswamy import Kumaraswamy
from util import sample_kumaraswamy


def approximate_pairwise_diff(samples, num_pairs):
    """
    Approximates the mean pairwise absolute difference by sampling a subset of pairs.

    Parameters:
    - samples (torch.Tensor): Samples with shape [num_samples, -1].
    - num_pairs (int): Number of random pairs to sample for approximation.

    Returns:
    - torch.Tensor: Approximate second expectation term, shape [batch, height, width].
    """

    N, _ = samples.shape
    pairwise_diffs = []

    # Randomly sample pairs of indices to compute differences
    for _ in range(num_pairs):
        i, j = random.sample(range(N), 2)
        diff = torch.abs(samples[i] - samples[j])  # Shape: [batch, height, width]
        pairwise_diffs.append(diff)

    # Stack and compute the mean across sampled pairs

    # Shape: [num_pairs, batch, height, width]
    pairwise_diffs = torch.stack(pairwise_diffs, dim=0)

    return pairwise_diffs


class CRPSKumaraswamyLoss(nn.Module):
    def __init__(self, num_samples=100):
        super(CRPSKumaraswamyLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, alpha, beta, weights, y_true):
        """
        Compute the CRPS for a single mixture of Kumaraswamy distributions shared across all observations using Monte Carlo sampling.

        Parameters:
        - y_true (torch.Tensor): Tensor of true target values of shape (B, T, Y, X).
        - alpha (torch.Tensor): Alpha parameters for each cell, shape [batch, height, width, num_mix].
        - beta (torch.Tensor): Beta parameters for each cell, shape [batch, height, width, num_mix].
        - weights (torch.Tensor): Tensor of mixture weights of shape (num_mix,).
        - num_samples (int): Number of samples to draw per mixture component.

        Returns:
        - torch.Tensor: The average CRPS over all observations.
        """

        B, T, H, W, C = y_true.shape

        # Shape: [num_samples, batch, height, width]
        samples = sample_kumaraswamy(alpha, beta, weights, num_samples=self.num_samples)

        # Add channels dimension
        samples = samples.unsqueeze(-1)

        # Expand y_true to match samples for broadcasting
        y_true_expanded = y_true.unsqueeze(0)  # Shape: [1, batch, height, width]

        # First Expectation: Mean absolute difference between samples and true values
        # Shape: [batch, height, width]
        term1 = torch.mean(torch.abs(samples - y_true_expanded), dim=0)

        # Second Expectation: Mean absolute difference between pairs of samples for each cell
        # Reshape samples for pairwise difference calculation
        # Shape: [num_samples, batch * height * width]
        samples_flat = samples.view(self.num_samples, -1)

        # Pairwise differences, shape: [num_samples, num_samples, batch * height * width]
        pairwise_diff = approximate_pairwise_diff(samples_flat, num_pairs=96)

        # Produce second expectation and reshape to match y_true
        term2 = (0.5 * pairwise_diff.mean(dim=0)).view(y_true.shape)

        assert term1.shape == term2.shape

        # Compute CRPS
        crps = torch.mean(term1 - term2)

        return crps


class CRPSGaussianLoss(nn.Module):
    def __init__(self):
        super(CRPSGaussianLoss, self).__init__()

    def forward(self, y_pred_mean, y_pred_std, y_true):
        """
        Compute CRPS for Gaussian predictions.

        :param y_pred_mean: Tensor of predicted means
        :param y_pred_std: Tensor of predicted standard deviations (must be positive)
        :param y_true: Tensor of observed y_true values
        :return: CRPS loss
        """
        # Ensure standard deviations are positive
        y_pred_std = torch.abs(y_pred_std)

        # Compute the standardized difference
        z = (y_true - y_pred_mean) / y_pred_std

        # CDF and PDF of the standard normal distribution
        normal = Normal(0, 1)
        cdf = normal.cdf(z)
        pdf = normal.log_prob(z).exp()

        # CRPS for Gaussian distributions
        crps = y_pred_std * (
            z * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(torch.pi))
        )

        # Return mean CRPS across batch
        return crps.mean()
