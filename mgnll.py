import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureGaussianNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureGaussianNLLLoss, self).__init__()

    def forward(self, means, std_devs, weights, y_true):
        # Create log likelihood for each Gaussian component
        log_likelihood = (
            -0.5 * (((y_true - means) ** 2) / (std_devs**2))
            - torch.log(std_devs)
            - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        )

        # Compute the weighted log likelihood
        weighted_log_likelihood = log_likelihood + torch.log(weights)

        # Sum over mixture components for the log likelihood of the mixture
        log_likelihood_mixture = torch.logsumexp(weighted_log_likelihood, dim=-1)

        # Negative log likelihood for the Gaussian mixture
        nll = -log_likelihood_mixture

        return nll.mean()
