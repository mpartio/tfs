import torch
import torch.nn as nn

def loss_fn(y_true, y_pred):
    return nn.MSELoss(y_true, y_pred)

