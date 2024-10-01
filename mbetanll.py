import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

# from betanll import BetaNLLLoss


class MixtureBetaNLLLoss(nn.Module):
    def __init__(self):
        super(MixtureBetaNLLLoss, self).__init__()

    def predicted_mean_and_var(self, alpha1, beta1, alpha2, beta2, weights):
        mean1 = alpha1 / (alpha1 + beta1)
        mean2 = alpha2 / (alpha2 + beta2)

        var1 = (alpha1 * beta1) / ((alpha1 + beta1) ** 2 * (alpha1 + beta1 + 1))
        var2 = (alpha2 * beta2) / ((alpha2 + beta2) ** 2 * (alpha2 + beta2 + 1))

        w1 = weights[:, :, :, :, 0].unsqueeze(-1)
        w2 = weights[:, :, :, :, 1].unsqueeze(-1)

        mean = w1 * mean1 + w2 * mean2
        var = w1 * var1 + w2 * var2

        return mean, var

    def l1_loss(self, predicted_mean, y_true):

        assert predicted_mean.shape == y_true.shape, (
            f"Predicted mean shape: {predicted_mean.shape}, "
            f"True target shape: {y_true.shape}"
        )
        loss = F.l1_loss(predicted_mean, y_true, reduction="mean")

        return loss

    def beta_nll(self, alpha, beta, y_true):
        log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_likelihood = (alpha - 1) * torch.log(y_true) + (beta - 1) * torch.log(
            1 - y_true
        )

        return log_likelihood - log_beta

    def beta_nll_loss(self, alpha1, beta1, alpha2, beta2, weights, y_true):
        y_true_c = y_true.clamp(1e-6, 1 - 1e-6)
        log_loss1 = self.beta_nll(alpha1, beta1, y_true_c)
        log_loss2 = self.beta_nll(alpha2, beta2, y_true_c)

        weight1 = weights[..., 0].unsqueeze(-1)
        weight2 = weights[..., 1].unsqueeze(-1)

        loss = -torch.log(
            weight1 * torch.exp(log_loss1) + weight2 * torch.exp(log_loss2)
        )

        return loss

    def forward(self, alpha1, beta1, alpha2, beta2, weights, y_true):

        beta_loss = self.beta_nll_loss(alpha1, beta1, alpha2, beta2, weights, y_true)

        predicted_mean, predicted_var = self.predicted_mean_and_var(alpha1, beta1, alpha2, beta2, weights)

#        mae_loss = self.l1_loss(predicted_mean, y_true)

        endpoint_reg = torch.mean((predicted_mean - 0) ** 2 * (predicted_mean > 0.1).float()) + \
               torch.mean((predicted_mean - 1) ** 2 * (predicted_mean < 0.9).float())
        var_reg = torch.mean(predicted_var)

#        print(0.2*beta_loss.mean().item(), 0.2 * endpoint_reg.item(), 8 * var_reg.item())
        return 0.2 * beta_loss.mean() + 0.2 * endpoint_reg +  8 * var_reg
