import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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
