import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

# from betanll import BetaNLLLoss


class MixtureBetaNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureBetaNLLLoss, self).__init__()

    def predicted_mean_and_var(self, alpha, beta, weights):
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        weighted_mean = torch.sum(weights * mean, dim=-1, keepdim=True)
        weighted_var = torch.sum(weights * var, dim=-1, keepdim=True)

        # Combine both the internal variance of each Beta distribution
        # (weighted by its corresponding weight) and the variance between
        # the means of the distributions (also weighted by the weights).

        mixture_var = weighted_var + torch.sum(
            weights * (mean - weighted_mean) ** 2, dim=-1, keepdim=True
        )

        return weighted_mean, mixture_var

    def l1_loss(self, predicted_mean, y_true):

        assert predicted_mean.shape == y_true.shape, (
            f"Predicted mean shape: {predicted_mean.shape}, "
            f"True target shape: {y_true.shape}"
        )
        loss = F.l1_loss(predicted_mean, y_true, reduction="mean")

        return loss

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

    def forward(
        self,
        alpha,
        beta,
        weights,
        y_true,
        beta_weight=0.2,
        endpoint_reg_weight=0.1,
        var_reg_weight=8,
    ):

        beta_loss = self.beta_nll_loss(alpha, beta, weights, y_true)

        predicted_mean, predicted_var = self.predicted_mean_and_var(
            alpha, beta, weights
        )

        #        mae_loss = self.l1_loss(predicted_mean, y_true)

        endpoint_reg = torch.mean(
            (predicted_mean - 0) ** 2 * (predicted_mean > 0.1).float()
        ) + torch.mean((predicted_mean - 1) ** 2 * (predicted_mean < 0.9).float())
        var_reg = torch.mean(predicted_var)

        return (
            beta_weight * beta_loss.mean()
            + endpoint_reg_weight * endpoint_reg
            + var_reg_weight * var_reg
        )
