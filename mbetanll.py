import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


class MixtureBetaNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureBetaNLLLoss, self).__init__()

    def beta_nll_loss(self, alpha, beta, weights, y_true):
        y_true_c = y_true.clamp(1e-6, 1 - 1e-6)

        log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

        log_likelihood = (alpha - 1) * torch.log(y_true_c) + (beta - 1) * torch.log(
            1 - y_true_c
        )

        # Negative log likelihood for each distribution
        nll = -log_likelihood + log_beta

        # Weighted negative log likelihood
        loss = torch.sum(weights * nll, dim=-1)

        return loss

    def forward(self, alpha, beta, weights, y_true):

        beta_loss = self.beta_nll_loss(alpha, beta, weights, y_true).mean()

        return beta_loss
