import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    def __init__(self, var_reg_weight=1e-2):
        super(GaussianNLLLoss, self).__init__()
        self.var_reg_weight = var_reg_weight

    def forward(self, y_pred_mean, y_pred_std, y_true, debug=False):
        var = y_pred_std**2  # Standard deviation to variance

        # Gaussian NLL Loss
        loss = 0.5 * torch.log(var) + 0.5 * ((y_true - y_pred_mean) ** 2) / var

        var_reg = self.var_reg_weight * torch.mean(var)

        return loss.mean() + var_reg
