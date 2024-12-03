import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import beta as stats_beta
from reference_layers import (
    ProcessingBlock,
    SEBlock,
    AttentionBlock,
)
from datetime import datetime, timedelta
from unet import UNet

x_data, y_data = None, None


def convert_delta(dlt: timedelta) -> str:
    minutes, seconds = divmod(int(dlt.total_seconds()), 60)
    return f"{minutes}:{seconds:02}"


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


class MultiHeadAttentionBridge(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable mixing parameter

    def forward(self, x):
        x = self.input_proj(x)

        skip = x

        # Input: [B, C, H, W]
        B, C, H, W = x.shape

        # Reshape to sequence: [B, H*W, C]
        x_seq = x.flatten(2).transpose(1, 2)

        # Apply attention
        out_seq, _ = self.mha(x_seq, x_seq, x_seq)

        # Reshape back: [B, C, H, W]
        out = out_seq.transpose(1, 2).reshape(B, C, H, W)

        return skip + self.gamma * out


class SelfAttentionBridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.query = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(out_channels, out_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.input_proj(x)

        # Input shape: [batch, channels, height, width]
        batch_size, channels, height, width = x.size()

        # Create query, key, value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Calculate attention scores
        attention = F.softmax(torch.bmm(query, key), dim=-1)

        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection with learnable weight
        return self.gamma * out + x


# Simple UNet-like network that outputs multiple samples
class SimpleNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model_proper = UNet(dim)

        # self.model_proper.bridge = SelfAttentionBridge(dim * 8, dim * 16)
        self.model_proper.bridge = MultiHeadAttentionBridge(dim * 8, dim * 16)
        self.model_proper.prediction_head = nn.Identity()

        # Modify final heads to output alpha and beta parameters
        self.alpha_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),
        )
        self.beta_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),
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

torch.set_float32_matmul_precision("medium")

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

T_0 = 8
T_mult = 2
eta_min = 1e-8
annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0, T_mult
)

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
train_loss, val_loss, concentrations, lr, mae = [], [], [], [], []

scaler = torch.amp.GradScaler()

for epoch in range(n_epochs):
    model.train()

    if epoch > 0:  # and epoch % 2 == 0:
        input_field, truth = read_beta_data(
            batch_size=batch_size, input_size=input_size
        )
        input_field = input_field.to(device)
        truth = truth.to(device)

    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        # Forward pass
        pred_alpha, pred_beta = model(input_field)

        # Calculate loss
        prb_loss = beta_nll_loss(pred_alpha, pred_beta, truth)
        # prb_loss = local_spatial_beta_loss(pred_alpha, pred_beta, truth)

        loss = prb_loss
        # Backward pass

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

    train_loss.append(loss.item())
    lr.append(optimizer.param_groups[0]["lr"])

    annealing_scheduler.step(epoch)

    with torch.no_grad():
        pred_alpha, pred_beta = model(val_input_field)

        prb_loss = beta_nll_loss(pred_alpha, pred_beta, val_truth)

        val_loss.append(prb_loss.item())

        concentrations.append(
            (
                torch.median(pred_alpha).cpu().numpy(),
                torch.median(pred_beta).cpu().numpy(),
            ),
        )

        # if len(concentrations) > 200:
        #    concentrations.pop(0)

        samples = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,)).cpu()
        median = samples.median(0)[0]

        sample_l1 = F.l1_loss(samples[0], val_truth.cpu())
        median_l1 = F.l1_loss(median, val_truth.cpu())

        mae.append((sample_l1, median_l1))

    if (epoch + 1) % 500 == 0:
        # Visualize results

        if best_l1 is None or median_l1 < best_l1:
            best_l1 = median_l1

        stop_time = datetime.now()

        print(
            f"Epoch {epoch+1:05d}, Loss: {loss.item():.4f}, L1: {median_l1:.4f} Best L1: {best_l1:.4f} Time: {convert_delta(stop_time-start_time)}"
        )

        start_time = stop_time

        if (epoch + 1) % 2000 != 0:  # epoch < n_epochs - 1:
            continue

        _input_field = val_input_field[0, 0].cpu()
        _truth = val_truth[0, 0].cpu()
        pred_alpha = pred_alpha[0, 0].cpu()
        pred_beta = pred_beta[0, 0].cpu()

        plt.figure(figsize=(20, 10))

        plt.subplot(351)
        plt.imshow(_input_field)
        plt.title("Input")
        plt.colorbar()

        plt.subplot(352)
        plt.imshow(_truth)
        plt.title("Truth")
        plt.colorbar()

        plt.subplot(353)
        pred_mean = pred_alpha / (pred_alpha + pred_beta)

        plt.imshow(pred_mean.cpu())
        plt.title("Predicted Mean")
        plt.colorbar()

        plt.subplot(354)

        pred_var = (pred_alpha * pred_beta) / (
            (pred_alpha + pred_beta) ** 2 * (pred_alpha + pred_beta + 1)
        )
        pred_std = torch.sqrt(pred_var)

        plt.imshow(pred_std.cpu())
        plt.title("Predicted Std")
        plt.colorbar()

        # Add a random sample from predictions
        sample = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,)).cpu()

        plt.subplot(355)
        plt.imshow(sample[0])
        plt.title("One Random Sample")
        plt.colorbar()

        median = sample.median(0)[0]
        plt.subplot(356)
        plt.imshow(median)
        plt.title("Median of Samples")
        plt.colorbar()

        plt.subplot(357)
        plt.imshow(pred_alpha.cpu())
        plt.title("Predicted Alpha")
        plt.colorbar()

        plt.subplot(358)
        plt.imshow(pred_beta.cpu())
        plt.title("Predicted Beta")
        plt.colorbar()

        data = _truth - _input_field
        cmap = plt.cm.coolwarm  # You can also try 'bwr' or other diverging colormaps
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

        plt.subplot(359)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("True Residual")
        plt.colorbar()

        data = pred_mean.cpu() - _input_field
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        plt.subplot(3, 5, 10)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Residual of Mean/L1={:.4f}".format(F.l1_loss(pred_mean, _truth)))
        plt.colorbar()

        data = sample[0] - _input_field
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        plt.subplot(3, 5, 11)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Residual of Sample/L1={:.4f}".format(sample_l1))
        plt.colorbar()

        data = median - _input_field
        norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        plt.subplot(3, 5, 12)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title("Residual of Median/L1={:.4f}".format(median_l1))
        plt.colorbar()

        # Define the range of x values (0 to 1, since it's a Beta distribution)
        x = np.linspace(0, 1, 100)
        y = []
        for alpha, beta in concentrations:
            y.append(stats_beta.pdf(x, alpha, beta))
        y = np.array(y)
        plt.subplot(3, 5, 13)

        plt.plot(x, y.T, color="gray", alpha=0.3, linewidth=0.3)
        plt.plot(x, y[-1], color="blue", linewidth=0.8)
        plt.title(
            f"Distribution latest (α={concentrations[-1][0]:.3f}, β={concentrations[-1][1]:.3f})"
        )

        plt.subplot(3, 5, 14)
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.legend()
        plt.title("Losses")

        ax2 = plt.gca().twinx()
        _mae = np.array(mae).T
        ax2.plot(np.arange(epoch + 1), _mae[0], label="Sample L1")
        ax2.plot(np.arange(epoch + 1), _mae[1], label="Median L1")
        ax2.legend(loc="upper left")

        plt.subplot(3, 5, 15)
        plt.plot(lr, label="LR")
        plt.title("Learning rate")

        plt.tight_layout()
        plt.savefig("epoch_{:05d}.png".format(epoch + 1))
        plt.close()
