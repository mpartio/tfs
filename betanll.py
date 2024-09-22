import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.distributions as dist

class BetaNLLLoss(nn.Module):
    def __init__(self):
        super(BetaNLLLoss, self).__init__()

    def forward(self, alpha, beta, y_true, epsilon=1e-6):
        # Clamp target to avoid issues with log(0) or log(1)
        y_true = torch.clamp(y_true, epsilon, 1 - epsilon)

        # Beta distribution based on the predicted alpha and beta
        beta_dist = dist.Beta(alpha, beta)

        # Compute negative log-likelihood of the target under the predicted distribution
        nll = -beta_dist.log_prob(y_true).mean()

        return nll

