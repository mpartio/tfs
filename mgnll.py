import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureGaussianNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureGaussianNLLLoss, self).__init__()

    def gaussian_nll_loss(self, means, std_devs, weights, y_true):
        # Clamping std_devs to avoid log(0) and division by zero
        #std_devs = std_devs.clamp(min=1e-6)

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

        return nll

    def forward(self, means, std_devs, weights, y_true):
        # Apply softplus to std_devs for stability
        std_devs = F.softplus(std_devs) + 1e-6  # 1e-6 to ensure numerical stability
        gaussian_loss = self.gaussian_nll_loss(means, std_devs, weights, y_true).mean()
        return gaussian_loss
