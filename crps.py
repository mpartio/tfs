import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.normal import Normal

# from torch.distributions.kumaraswamy import Kumaraswamy
from util import fast_sample_kumaraswamy, sample_kumaraswamy


def approximate_pairwise_diff(samples, num_pairs, method="random"):
    """
    Approximates the mean pairwise absolute difference by sampling a subset of pairs.
    Sampling is done within-batch.

    Parameters:
    - samples (torch.Tensor): Samples with shape [num_samples, -1].
    - num_pairs (int): Number of random pairs to sample for approximation.

    Returns:
    - torch.Tensor: Approximate second expectation term, shape [batch, height, width].
    """

    N, _ = samples.shape
    pairwise_diffs = []

    if method == "random":
        # Randomly sample pairs of indices to compute differences
        for _ in range(num_pairs):
            i, j = random.sample(range(N), 2)
            diff = torch.abs(samples[i] - samples[j])  # Shape: [batch, height, width]
            pairwise_diffs.append(diff)
    elif method == "deterministic":
        # Deterministically sample pairs of indices to compute differences
        for i in range(0, num_pairs - 1, 2):
            diff = torch.abs(samples[i] - samples[i + 1])
            pairwise_diffs.append(diff)
    elif method == "systematic":
        i = torch.arange(num_pairs, device=samples.device)
        j = (i + N // 2) % N  # This ensures better coverage
        diff = torch.abs(samples[i] - samples[j])
        return diff
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
        samples = fast_sample_kumaraswamy(
            alpha, beta, weights, num_samples=self.num_samples
        )

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
        pairwise_diff = approximate_pairwise_diff(
            samples_flat, num_pairs=32, method="systematic"
        )

        # Produce second expectation and reshape to match y_true
        term2 = (0.5 * pairwise_diff.mean(dim=0)).view(y_true.shape)

        assert term1.shape == term2.shape

        # Compute CRPS
        crps = torch.mean(term1 - term2)

        if torch.isnan(crps):
            raise ValueError(f"NaN in CRPS: term1={term1.mean()}, term2={term2.mean()}")

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


class EmpiricalCRPS(nn.Module):
    def __init__(self):
        super(EmpiricalCRPS, self).__init__()

    def forward(self, ensemble_preds, y_true):
        """
        Compute CRPS loss between ensemble predictions and target.

        Args:
            ensemble_preds: tensor of shape (num_samples, batch, 1, H, W)
            target: tensor of shape (batch, 1, H, W)
        """

        M = ensemble_preds.shape[0]  # ensemble size

        # First term: mean absolute error between each member and target
        first_term = torch.mean(torch.abs(ensemble_preds - target))

        # Second term: mean absolute difference between all pairs of members
        # Use einsum for efficient computation
        second_term = torch.einsum("ibhw,jbhw->ijbhw", ensemble_preds, ensemble_preds)
        second_term = torch.mean(torch.abs(second_term)) / (2 * M * M)

        return first_term - second_term


class AlmostFairCRPSLoss(nn.Module):
    def __init__(self, alpha=0.95, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, predictions, target):
        """
        Args:
            predictions: [B, M, 1, H, W]
            target: [B, 1, H, W]
        """
        B, M, C, H, W = predictions.shape
        epsilon = (1 - self.alpha) / M

        # Reshape target to match predictions dimensions
        target = target.unsqueeze(1)  # [B, 1, 1, H, W]

        # Compute differences all at once
        # Using broadcasting for both target differences and member differences
        pred_target_diff = torch.abs(predictions - target)  # [B, M, 1, H, W]

        # Compute member differences using broadcasting
        x_i = predictions.unsqueeze(2)  # [B, M, 1, 1, H, W]
        x_j = predictions.unsqueeze(1)  # [B, 1, M, 1, H, W]
        pred_pred_diff = torch.abs(x_i - x_j)  # [B, M, M, 1, H, W]

        # Sum across members using the formula
        first_term = pred_target_diff.mean(dim=1)  # [B, 1, H, W]
        second_term = (1 - epsilon) * pred_pred_diff.mean(dim=[1, 2])  # [B, 1, H, W]

        # Compute mean across spatial dimensions
        loss = (first_term - 0.5 * second_term).mean(dim=[1, 2, 3])

        loss = loss.mean()
        assert (
            loss == loss
        ), "NaN in loss, predictions min/mean/max/nans: {:.3f}/{:.3f}/{:.3f}/{}, target min/mean/max/nans: {:.3f}/{:.3f}/{:.3f}/{}".format(
            torch.min(predictions),
            torch.mean(predictions),
            torch.max(predictions),
            torch.isnan(predictions).sum(),
            torch.min(target),
            torch.mean(target),
            torch.max(target),
            torch.isnan(target).sum(),
        )

        return loss


class WeightedAlmostFairCRPSLoss(AlmostFairCRPSLoss):
    def __init__(self, alpha=0.95, spatial_weight=None, eps=1e-6):
        super().__init__(alpha, eps)
        self.spatial_weight = spatial_weight

    def forward(self, predictions, target):
        B, M, H, W = predictions.shape
        base_loss = super().forward(predictions, target)

        if self.spatial_weight is not None:
            # Apply spatial weighting
            weight = self.spatial_weight.expand(B, 1, H, W)
            weighted_loss = base_loss * weight
            return weighted_loss.mean()

        return base_loss
