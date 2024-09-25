import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, y_pred_mean, y_pred_std, y_true):
        # Ensure the predicted standard deviation is positive
        y_pred_std = nn.functional.softplus(y_pred_std) + 1e-6

        var = y_pred_std**2  # Standard deviation to variance

        # Gaussian NLL Loss
        loss = 0.5 * torch.log(var) + 0.5 * ((y_true - y_pred_mean) ** 2) / var

        var_reg = 1e-2 * torch.mean(var)

        return loss.mean() + var_reg
