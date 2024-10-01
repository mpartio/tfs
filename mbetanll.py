import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

# from betanll import BetaNLLLoss


class MixtureBetaNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureBetaNLLLoss, self).__init__()

    def l1(self, alpha1, beta1, alpha2, beta2, weights, y_true):

        mean1 = alpha1 / (alpha1 + beta1)
        mean2 = alpha2 / (alpha2 + beta2)
        predicted_mean = (
            weights[:, :, :, :, 0].unsqueeze(-1) * mean1
            + weights[:, :, :, :, 1].unsqueeze(-1) * mean2
        )

        assert predicted_mean.shape == y_true.shape, (
            f"Predicted mean shape: {predicted_mean.shape}, "
            f"True target shape: {y_true.shape}"
        )
        # L1 Loss between the predicted mean and the true target
        l1_loss = F.l1_loss(predicted_mean, y_true, reduction="mean")

        return l1_loss

    def beta_nll(self, alpha, beta, y_true):
        log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_likelihood = (alpha - 1) * torch.log(y_true) + (beta - 1) * torch.log(
            1 - y_true
        )

        return log_likelihood - log_beta

    def forward(self, alpha1, beta1, alpha2, beta2, weights, y_true):

        y_true = y_true.clamp(1e-6, 1 - 1e-6)
        log_loss1 = self.beta_nll(alpha1, beta1, y_true)
        log_loss2 = self.beta_nll(alpha2, beta2, y_true)

        weight1 = weights[..., 0].unsqueeze(-1)
        weight2 = weights[..., 1].unsqueeze(-1)

        loss = -torch.log(
            weight1 * torch.exp(log_loss1) + weight2 * torch.exp(log_loss2)
        )

        assert torch.isnan(loss).sum() == 0, "Loss contains NaN values"

        return loss.mean() + 0.2 * self.l1(
            alpha1, beta1, alpha2, beta2, weights, y_true
        )
