import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kumaraswamy import Kumaraswamy


class CRPSKumaraswamyLoss(nn.Module):
    def __init__(self, z=100):
        super(CRPSBetaLoss, self).__init__()
        self.z = z

    def forward(self, alpha, beta, weight, y_true):
        """
        Compute the CRPS for a single mixture of Kumaraswamy distributions shared across all observations using Monte Carlo sampling.

        Parameters:
        y_true (torch.Tensor): Tensor of true target values of shape (B, T, Y, X).
        alpha (torch.Tensor): Tensor of 'alpha' parameters of shape (num_mix,).
        beta (torch.Tensor): Tensor of 'beta' parameters of shape (num_mix,).
        weights (torch.Tensor): Tensor of mixture weights of shape (num_mix,).
        num_samples (int): Number of samples to draw per mixture component.

        Returns:
        torch.Tensor: The average CRPS over all observations.
        """

        B, T, Y, X = y_true.shape

        # Expand y_true for broadcasting with samples
        y_true = y_true.view(1, B, T, Y, X)  # Shape: (1, B, T, Y, X)

        # Sample from each component of the mixture Kumaraswamy distribution
        samples = []

        num_mix = alpha.shape[0]
        for i in range(num_mix):
            kumaraswamy_dist = Kumaraswamy(concentration1=a[i], concentration0=b[i])
            # Shape: (num_samples,)
            samples_i = kumaraswamy_dist.rsample(sample_shape=(num_samples,))
            samples.append(samples_i)

        # Stack samples across components and apply weights
        # Shape: (num_samples, num_mix)
        samples = torch.stack(samples, dim=-1)
        # Weighted sum over components, Shape: (num_samples,)
        samples = (samples * weights).sum(dim=-1)

        # Reshape samples to (num_samples, 1, 1, 1, 1) for broadcasting with y_true
        samples = samples.view(num_samples, 1, 1, 1, 1)
        samples = samples.expand(num_samples, B, T, Y, X)

        # Compute the first expectation E_F[|X - y|]
        # Shape: (B, T, Y, X)
        term1 = torch.mean(torch.abs(samples - y_true), dim=0)

        # Compute the second expectation E_F[|X - X'|] using pairwise differences
        samples_flat = samples.view(num_samples)  # Flatten to (num_samples,)
        # Pairwise differences, Shape: (num_samples, num_samples)
        diff = samples_flat.unsqueeze(1) - samples_flat.unsqueeze(0)
        term2 = 0.5 * torch.mean(torch.abs(diff))  # Scalar

        # Compute CRPS for each observation
        crps_values = term1 - term2  # Shape: (B, T, Y, X)

        # Return the average CRPS over all observations
        return crps_values.mean()  # Return scalar average over B, T, Y, X


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
