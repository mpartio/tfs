import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from reference_layers import (
    ProcessingBlock,
    SEBlock,
    AttentionBlock,
)
from datetime import datetime
from unet import UNet

x_data, y_data = None, None


def read_beta_data(batch_size, input_size):
    global x_data, y_data
    import numpy as np

    if x_data is None:
        data = np.load("../data/train-2k.npz")["arr_0"]
        T, H, W, C = data.shape
        N = (T // 4) * 4
        data = data[:N, ...]
        data = (
            data.reshape(-1, 4, H, W, C).transpose(1, 0, 2, 3, 4).reshape(-1, H, W, C)
        )
        data = np.squeeze(data, axis=-1)

        x_data = np.expand_dims(
            data[
                1::2,
            ],
            axis=1,
        )
        y_data = np.expand_dims(data[::2], axis=1)

    n = torch.randint(0, x_data.shape[0] - input_size, (1,)).item()

    _x_data = x_data[n : n + batch_size]
    _x_data = _x_data[:, :, :input_size, :input_size]

    _y_data = y_data[n : n + batch_size]
    _y_data = _y_data[:, :, :input_size, :input_size]

    return torch.tensor(_x_data), torch.tensor(_y_data)


def generate_beta_data(batch_size, input_size):
    def set_squares(arr, x, y, fill_value, radius=2):
        x_indices = torch.arange(
            max(0, x - radius), min(arr.input_size(-2), x + radius)
        )
        y_indices = torch.arange(
            max(0, y - radius), min(arr.input_size(-1), y + radius)
        )

        xx, yy = torch.meshgrid(x_indices, y_indices, indexing="ij")

        for b in range(arr.input_size(0)):
            arr[b, xx, yy] = fill_value

        return arr

    alpha = torch.tensor(0.2)
    beta = torch.tensor(0.1)

    beta_dist = torch.distributions.Beta(alpha, beta)

    input_field = beta_dist.sample((batch_size, input_size, input_size))

    target = input_field.clone()
    target = target + torch.randn_like(target) * 0.1
    target = torch.clamp(target, 0, 1)

    radius = 7
    for b in range(batch_size):
        for _ in range(6):
            x, y = torch.randint(low=radius, high=input_size - radius, input_size=(2,))

            fill_value = 1 if torch.rand(1).item() > 0.5 else 0
            input_field[b : b + 1] = set_squares(
                input_field[b : b + 1], x, y, fill_value, radius
            )
            target[b : b + 1] = set_squares(
                target[b : b + 1], x + radius, y + radius, fill_value, radius
            )

    assert input_field.shape == (batch_size, input_size, input_size)

    input_field = input_field.unsqueeze(1)
    target = target.unsqueeze(1)

    return input_field, target


def beta_nll_loss(pred_alpha, pred_beta, target):
    """
    Beta negative log likelihood loss

    Args:
        pred_alpha: (batch, 1, h, w) positive parameter
        pred_beta: (batch, 1, h, w) positive parameter
        target: (batch, 1, h, w) values in [0,1]
    """
    # Beta log likelihood:
    # log Beta(x; α, β) = log Γ(α+β) - log Γ(α) - log Γ(β) + (α-1)log(x) + (β-1)log(1-x)
    loss = (
        torch.lgamma(pred_alpha + pred_beta)
        - torch.lgamma(pred_alpha)
        - torch.lgamma(pred_beta)
        + (pred_alpha - 1)
        * torch.log(target + 1e-6)  # add small epsilon to avoid log(0)
        + (pred_beta - 1) * torch.log(1 - target + 1e-6)
    )

    return -loss.mean()  # Negative because we minimize


def local_spatial_beta_loss(pred_alpha, pred_beta, target, window_size=5):
    """
    Beta loss with spatial correlation
    """
    # Create correlation kernel
    center = window_size // 2
    y, x = torch.meshgrid(torch.arange(window_size), torch.arange(window_size))
    coords = torch.stack([y, x], dim=-1).float().to(pred_alpha.device)
    distances = ((coords - coords[center, center]) ** 2).sum(-1)
    kernel = torch.exp(-distances / (2 * 1.0**2))
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, :]

    # Compute local Beta NLL
    nll = (
        -torch.lgamma(pred_alpha + pred_beta)
        + torch.lgamma(pred_alpha)
        + torch.lgamma(pred_beta)
        - (pred_alpha - 1) * torch.log(target + 1e-6)
        - (pred_beta - 1) * torch.log(1 - target + 1e-6)
    )

    # Apply spatial kernel
    weighted_nll = F.conv2d(nll, kernel, padding=center)

    return weighted_nll.mean()


def mae(model, input_field, truth):
    pred_alpha, pred_beta = model(input_field)
    sample = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,)).cpu()
    median = sample.median(0)[0]
    return F.l1_loss(median, truth.cpu()).item()


# Simple UNet-like network that outputs multiple samples
class SimpleNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model_proper = UNet(dim)
        self.model_proper.prediction_head = nn.Identity()

        # Modify final heads to output alpha and beta parameters
        self.alpha_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),  # alpha must be positive
        )
        self.beta_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),  # beta must be positive
        )

    def forward(self, x):
        x = self.model_proper(x)

        alpha = self.alpha_head(x)
        beta = self.beta_head(x)

        return alpha, beta


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 128
batch_size = 128
dim = 32

model = SimpleNet(dim=dim).to(device)
print(model)
print(
    "Number of trainable parameters",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)


input_field, truth = read_beta_data(batch_size=batch_size, input_size=input_size)
input_field = input_field.to(device)
truth = truth.to(device)

val_input_field, val_truth = read_beta_data(
    batch_size=batch_size // 2, input_size=input_size
)
val_input_field = val_input_field.to(device)
val_truth = val_truth.to(device)

# Training loop
n_epochs = 12000
best_l1 = None

start_time = datetime.now()

for epoch in range(n_epochs):
    model.train()

    if epoch > 0:  # and epoch % 2 == 0:
        input_field, truth = read_beta_data(
            batch_size=batch_size, input_size=input_size
        )
        input_field = input_field.to(device)
        truth = truth.to(device)

    # Forward pass
    pred_alpha, pred_beta = model(input_field)

    # Calculate loss

    prb_loss = beta_nll_loss(pred_alpha, pred_beta, truth)
    # prb_loss = local_spatial_beta_loss(pred_alpha, pred_beta, truth)

    loss = prb_loss
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        # Visualize results

        with torch.no_grad():
            pred_alpha, pred_beta = model(val_input_field)

            l1 = mae(model, val_input_field, val_truth)

            if best_l1 is None or l1 < best_l1:
                best_l1 = l1

            stop_time = datetime.now()

            print(
                f"Epoch {epoch+1:05d}, Loss: {loss.item():.4f}, L1: {l1:.4f} Best L1: {best_l1:.4f} Time: {stop_time-start_time}"
            )

            start_time = stop_time

            if epoch < n_epochs - 1:
                continue

            _input_field = val_input_field[0, 0].cpu()
            _truth = val_truth[0, 0].cpu()
            pred_alpha = pred_alpha[0, 0].cpu()
            pred_beta = pred_beta[0, 0].cpu()

            plt.figure(figsize=(20, 10))

            plt.subplot(341)
            plt.imshow(_input_field)
            plt.title("Input")
            plt.colorbar()

            plt.subplot(342)
            plt.imshow(_truth)
            plt.title("Truth")
            plt.colorbar()

            plt.subplot(343)
            pred_mean = pred_alpha / (pred_alpha + pred_beta)

            plt.imshow(pred_mean.cpu())
            plt.title("Predicted Mean")
            plt.colorbar()

            plt.subplot(344)
            # Convert logvar to standard deviation
            pred_var = (pred_alpha * pred_beta) / (
                (pred_alpha + pred_beta) ** 2 * (pred_alpha + pred_beta + 1)
            )
            pred_std = torch.sqrt(pred_var)

            #            pred_std = torch.exp(0.5 * pred_logvar[0, 0]).cpu()
            plt.imshow(pred_std.cpu())
            plt.title("Predicted Std")
            plt.colorbar()

            # Add a random sample from predictions
            sample = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,)).cpu()

            plt.subplot(345)
            plt.imshow(sample[0])
            plt.title("One Random Sample")
            plt.colorbar()

            median = sample.median(0)[0]
            plt.subplot(346)
            plt.imshow(median)
            plt.title("Median of Samples")
            plt.colorbar()

            plt.subplot(347)
            plt.imshow(pred_alpha.cpu())
            plt.title("Predicted Alpha")
            plt.colorbar()

            plt.subplot(348)
            plt.imshow(pred_beta.cpu())
            plt.title("Predicted Beta")
            plt.colorbar()

            data = _truth - _input_field
            cmap = (
                plt.cm.coolwarm
            )  # You can also try 'bwr' or other diverging colormaps
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

            plt.subplot(349)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("True Diff")
            plt.colorbar()

            data = pred_mean.cpu() - _input_field
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            plt.subplot(3, 4, 10)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("Diff of Mean/L1={:.4f}".format(F.l1_loss(pred_mean, _truth)))
            plt.colorbar()

            data = sample[0] - _input_field
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            plt.subplot(3, 4, 11)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("Diff of Sample/L1={:.4f}".format(F.l1_loss(sample[0], _truth)))
            plt.colorbar()

            data = median - _input_field
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            plt.subplot(3, 4, 12)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("Diff of Median/L1={:.4f}".format(l1))
            plt.colorbar()

            plt.tight_layout()

            plt.savefig("epoch_{:04d}.png".format(epoch + 1))
            plt.close()