import torch
import torch.nn as nn

class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super(HeteroscedasticLoss, self).__init__()

    def forward(self, y_pred_mean, y_pred_var, y_true):
        # Ensure variance is positive
        var = torch.clamp(y_pred_var, min=1e-6)

        # Heteroscedastic loss
        loss = torch.exp(-var) * torch.abs(y_true - y_pred_mean) + var
#        loss = torch.abs(y_true - y_pred_mean)
        return loss.mean()
