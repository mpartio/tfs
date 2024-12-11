import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import platform
import os
import sys
import pywt
from scipy.signal import medfilt2d
from scipy.stats import beta as stats_beta
from datetime import datetime, timedelta
from unet import UNet

x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None
train_no = 0


def calculate_wavelet_snr(prediction, reference=None, wavelet="db2", level=2):
    """
    Calculate SNR using wavelet decomposition, specifically designed for neural network outputs
    with values between 0 and 1 and sharp features.

    Parameters:
    prediction (numpy.ndarray): Predicted field from neural network (values 0-1)
    reference (numpy.ndarray, optional): Ground truth field if available
    wavelet (str): Wavelet type to use (default: 'db2' which preserves edges well)
    level (int): Decomposition level

    Returns:
    dict: Dictionary containing SNR metrics and noise field
    """
    if prediction.ndim != 2:
        raise ValueError("Input must be 2D array: {}".format(prediction.shape))
    if reference is not None and reference.ndim != 2:
        raise ValueError("Reference must be 2D array: {}".format(reference.shape))

    # If we have reference data, we can calculate noise directly
    if reference is not None:
        noise_field = prediction - reference
        _noise_field = noise_field.numpy()

    # If no reference, estimate noise using wavelet decomposition
    else:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(prediction, wavelet, level=level)

        # Get highest frequency details (typically noise)
        cH1, cV1, cD1 = coeffs[1]

        # Estimate noise standard deviation using MAD estimator
        # MAD is more robust to outliers than standard deviation
        noise_std = np.median(np.abs(cD1)) / 0.6745  # 0.6745 is the MAD scaling factor

        # Reconstruct noise field
        coeffs_noise = [np.zeros_like(coeffs[0])]  # Set approximation to zero
        coeffs_noise.extend([(cH1, cV1, cD1)])  # Keep finest details
        coeffs_noise.extend(
            [tuple(np.zeros_like(d) for d in coeff) for coeff in coeffs[2:]]
        )  # Set coarser details to zero

        noise_field = pywt.waverec2(coeffs_noise, wavelet)

        # Normalize noise field to match input scale
        _noise_field = noise_field * (noise_std / np.std(noise_field))

    _prediction = prediction.numpy()

    # Calculate signal power (using smoothed prediction as signal estimate)
    smooth_pred = medfilt2d(_prediction, kernel_size=3)
    signal_power = np.mean(smooth_pred**2)

    # Calculate noise power
    noise_power = np.mean(_noise_field**2)

    # Calculate SNR
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    # Calculate local SNR map to identify problematic regions
    local_noise_power = medfilt2d(_noise_field**2, kernel_size=5)
    local_snr = 10 * np.log10(smooth_pred**2 / (local_noise_power + 1e-9))

    return {
        "snr_db": snr_db,
        "noise_field": _noise_field,
        "local_snr_map": local_snr,
    }


def trimmed_mean(samples, trim_percent=0.2):
    # Sort and remove extreme values
    sorted_samples, _ = torch.sort(samples, dim=0)
    n_samples = samples.shape[0]
    trim = int(n_samples * trim_percent)
    return sorted_samples[trim:-trim].mean(dim=0)


def diagnostics(
    diag, input_field, truth, pred_alpha, pred_beta, snr, iteration, train_no
):

    assert pred_alpha.ndim == 2

    plt.figure(figsize=(28, 12))
    plt.suptitle(
        "{} at train {} iteration {} (host={}, time={})".format(
            sys.argv[0],
            train_no,
            iteration,
            platform.node(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    plt.subplot(361)
    plt.imshow(input_field)
    plt.title("Input")
    plt.colorbar()

    plt.subplot(362)
    plt.imshow(truth)
    plt.title("Truth")
    plt.colorbar()

    plt.subplot(363)
    pred_mean = pred_alpha / (pred_alpha + pred_beta)

    plt.imshow(pred_mean)
    plt.title("Predicted Mean")
    plt.colorbar()

    plt.subplot(364)

    pred_var = (pred_alpha * pred_beta) / (
        (pred_alpha + pred_beta) ** 2 * (pred_alpha + pred_beta + 1)
    )
    pred_std = torch.sqrt(pred_var)

    plt.imshow(pred_std)
    plt.title("Predicted Std")
    plt.colorbar()

    # Add a random sample from predictions
    sample = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,))

    plt.subplot(365)
    plt.imshow(sample[0])
    plt.title("One Random Sample")
    plt.colorbar()

    median = sample.median(0)[0]
    plt.subplot(366)
    plt.imshow(median)
    plt.title("Median of Samples")
    plt.colorbar()

    plt.subplot(367)
    plt.imshow(trimmed_mean(sample, 0.2))
    plt.title("Trimmed Mean of Samples (%=0.2)")
    plt.colorbar()

    plt.subplot(368)
    plt.imshow(pred_alpha)
    plt.title("Predicted Alpha")
    plt.colorbar()

    plt.subplot(369)
    plt.imshow(pred_beta)
    plt.title("Predicted Beta")
    plt.colorbar()

    data = truth - input_field
    cmap = plt.cm.coolwarm  # You can also try 'bwr' or other diverging colormaps
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

    plt.subplot(3, 6, 10)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("True Residual")
    plt.colorbar()

    data = pred_mean - input_field
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 6, 11)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Residual of Mean/L1={:.4f}".format(F.l1_loss(pred_mean, truth)))
    plt.colorbar()

    data = sample[0] - input_field
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 6, 12)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Residual of Sample/L1={:.4f}".format(F.l1_loss(sample[0], truth)))
    plt.colorbar()

    data = median - input_field
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(3, 6, 13)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Residual of Median/L1={:.4f}".format(F.l1_loss(median, truth)))
    plt.colorbar()

    # Define the range of x values (0 to 1, since it's a Beta distribution)
    x = np.linspace(0, 1, 100)
    y = []
    for alpha, beta in diag.concentrations:
        y.append(stats_beta.pdf(x, alpha, beta))
    y = np.array(y)
    plt.subplot(3, 6, 14)

    plt.plot(x, y.T, color="gray", alpha=0.3, linewidth=0.3)
    plt.plot(x, y[-1], color="blue", linewidth=0.8)
    plt.title(
        f"Distribution (latest α={diag.concentrations[-1][0]:.3f}, β={diag.concentrations[-1][1]:.3f})"
    )

    # n = 20
    # n_iterations = len(diag.train_loss)
    # n_iterations = (n_iterations // n) * n  # divisible with n
    # x_labels = torch.arange(1, n_iterations, n)

    # train_loss = (
    #    torch.tensor(diag.train_loss[:n_iterations])
    #    .view(n_iterations // n, -1)
    #    .mean(dim=1)
    # )
    #    val_loss = (
    #        torch.tensor(diag.val_loss[:n_iterations])
    #        .view(n_iterations // 2, -1)
    #        .mean(dim=1)
    #    )

    plt.subplot(3, 6, 15)
    plt.title("Train Loss")
    plt.plot(diag.train_loss, label="Train Loss", color="blue", alpha=0.3)
    plt.plot(
        moving_average(torch.tensor(diag.train_loss), 100),
        label="Moving average",
        color="blue",
    )

    plt.legend(loc="upper left")
    ax2 = plt.gca().twinx()
    ax2.plot(torch.tensor(diag.lr) * 1e6, label="LRx1M", color="green")
    ax2.legend(loc="upper right")

    plt.subplot(3, 6, 16)
    plt.title("Validation Losses")
    plt.plot(
        moving_average(torch.tensor(diag.val_loss), 20), label="Val Loss", color="blue"
    )
    plt.legend(loc="upper left")
    ax2 = plt.gca().twinx()
    _mae = np.array(diag.mae).T
    ax2.plot(
        moving_average(torch.tensor(_mae[0]), 20), label="Sample L1", color="green"
    )
    ax2.plot(moving_average(torch.tensor(_mae[1]), 20), label="Median L1", color="red")
    ax2.legend(loc="upper right")

    plt.subplot(3, 6, 17)
    im = plt.imshow(snr["local_snr_map"], cmap="viridis")
    plt.title("Local SNR Map (dB)")
    plt.colorbar(im)

    snr_db = np.array(diag.snr_db).T

    plt.subplot(3, 6, 18)
    snr_real = torch.tensor(snr_db[0])
    snr_pred = torch.tensor(snr_db[1])
    plt.plot(snr_real, label="Real", color="blue", alpha=0.3)
    plt.plot(
        moving_average(snr_real, 20),
        label="Moving Average",
        color="blue",
    )
    plt.plot(snr_pred, label="Pred", color="orange", alpha=0.3)
    plt.plot(
        moving_average(snr_pred, 20),
        color="orange",
        label="Moving average",
    )
    plt.legend(loc="center right")

    ax2 = plt.gca().twinx()
    ax2.plot(moving_average(snr_real - snr_pred, 20), label="Residual", color="green")
    ax2.legend(loc="upper right")

    plt.title("Signal to Noise Ratio")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/train_{}_iteration_{:05d}.png".format(train_no, iteration))
    plt.close()


def moving_average(arr, window_size):
    """
    Calculate the running mean of a 1D array using a sliding window.

    Parameters:
    - arr (torch.Tensor): 1D input array.
    - window_size (int): The size of the sliding window.

    Returns:
    - torch.Tensor: The running mean array.
    """
    # Ensure input is a 1D tensor
    arr = arr.reshape(1, 1, -1)

    if window_size >= arr.shape[-1]:
        return torch.full((arr.shape[-1],), float("nan"))

    # Create a uniform kernel
    kernel = torch.ones(1, 1, window_size) / window_size

    # Apply 1D convolution
    running_mean = torch.nn.functional.conv1d(
        arr, kernel, padding=0, stride=1
    ).squeeze()

    nan_padding = torch.full((window_size - 1,), float("nan"))

    result = torch.cat((nan_padding, running_mean))

    return result


def convert_delta(dlt: timedelta) -> str:
    minutes, seconds = divmod(int(dlt.total_seconds()), 60)
    return f"{minutes}:{seconds:02}"


def augment_data(x, y):
    # Random flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [2])  # Horizontal flip
        y = torch.flip(y, [2])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [3])  # Vertical flip
        y = torch.flip(y, [3])

    # Random 90-degree rotations
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k, [2, 3])
    y = torch.rot90(y, k, [2, 3])

    return x, y


def partition(tensor, n_single, n_block):
    T, C, H, W = tensor.shape
    total_length = T
    group_size = n_single + n_block

    # Reshape into groups of (n_single + n_block), with padding if necessary
    num_groups = total_length // group_size
    new_length = num_groups * group_size

    padded_tensor = tensor[:new_length]

    # Reshape into groups
    reshaped = padded_tensor.view(-1, group_size, C, H, W)
    # Extract single elements and blocks
    singles = reshaped[:, :n_single]
    blocks = reshaped[:, n_single:]

    assert singles.shape[0] > 0, "Not enough elements"
    return singles, blocks


def read_beta_data(filename, n_x, n_y):
    import numpy as np

    data = np.load(filename)["arr_0"]

    T, H, W, C = data.shape
    N = (T // 4) * 4
    data = data[:N, ...]
    data = (
        data.reshape(-1, 4, H, W, C)
        .transpose(1, 0, 2, 3, 4)
        .reshape(-1, H, W, C)
        .transpose(0, 3, 1, 2)
    )
    data = torch.tensor(data)

    x_data, y_data = partition(data, n_x, n_y)

    return x_data, y_data


def read_train_data(batch_size, input_size, n_x=1, n_y=1):
    global x_train_data, y_train_data
    if x_train_data is None:
        x_train_data, y_train_data = read_beta_data("../data/train-100k.npz", n_x, n_y)

    n = torch.randint(
        0,
        x_train_data.shape[0] - batch_size,
        (1,),
    ).item()

    x_data = x_train_data[n : n + batch_size]
    x_data = x_data[:, :, :input_size, :input_size]

    y_data = y_train_data[n : n + batch_size]
    y_data = y_data[:, :, :input_size, :input_size]

    return x_data, y_data


def read_val_data(batch_size, input_size, shuffle=False, n_x=1, n_y=1):
    global x_val_data, y_val_data
    if x_val_data is None:
        x_val_data, y_val_data = read_beta_data("../data/val-100k.npz", n_x, n_y)

    n = 0
    if shuffle:
        n = torch.randint(0, x_val_data.shape[0] - batch_size, (1,)).item()

    x_data = x_val_data[n : n + batch_size]
    x_data = x_data[:, :, :input_size, :input_size]

    y_data = y_val_data[n : n + batch_size]
    y_data = y_data[:, :, :input_size, :input_size]

    return x_data, y_data


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


def get_lr_schedule(optimizer, warmup_iterations=1000, total_iterations=20000):
    def lr_lambda(current_iteration):
        # Warmup phase
        if current_iteration < warmup_iterations:
            return current_iteration / warmup_iterations

        # Linear decay phase
        else:
            progress = (current_iteration - warmup_iterations) / (
                total_iterations - warmup_iterations
            )
            # Decay from 1.0 to 0.1
            return max(0.1, 1.0 - 0.9 * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


class WindowedAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, window_size=8):
        super().__init__()
        assert (
            channels % num_heads == 0
        ), f"Channel dimension {channels} must be divisible by num_heads {num_heads}"

        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)  # Pre-norm for attention
        self.norm2 = nn.LayerNorm(channels)  # Pre-norm for FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

        self.gamma1 = nn.Parameter(torch.zeros(1))  # for attention
        self.gamma2 = nn.Parameter(torch.zeros(1))  # for ffn
        self.window_size = window_size
        self.scale = channels**-0.5

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # Pre-normalize and prepare for attention
        x_flat = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.permute(0, 3, 1, 2)  # Back to [B, C, H, W]

        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h))

        # Get padded shape
        _, _, Hp, Wp = x_norm.shape

        # Partition into windows and flatten window dim
        x_windows = x_norm.view(
            B,
            C,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
        )
        windows = x_windows.permute(0, 2, 4, 1, 3, 5)
        windows = windows.reshape(-1, C, self.window_size * self.window_size)
        windows = windows.transpose(1, 2)  # [B*num_windows, window_size*window_size, C]

        # Apply attention to each window
        att_out, _ = self.mha(windows, windows, windows)

        # Reverse window partition
        att_out = att_out.transpose(1, 2)  # [B*num_windows, C, window_size*window_size]
        att_out = att_out.reshape(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            C,
            self.window_size,
            self.window_size,
        )

        att_out = att_out.permute(0, 3, 1, 4, 2, 5)
        att_out = att_out.reshape(B, C, Hp, Wp)

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            att_out = att_out[:, :, :H, :W]

        # First residual connection with attention
        x = identity + self.gamma1 * att_out

        # FFN path
        x_ffn = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.ffn(x_ffn)
        x_ffn = x_ffn.permute(0, 3, 1, 2)  # Back to [B, C, H, W]

        # Second residual connection with FFN
        out = x + self.gamma2 * x_ffn

        return out


def roll_forecast_train(model, x, y, n_steps):
    assert y.shape[1] == n_steps

    total_loss = 0

    for step in range(n_steps):
        if x.ndim == 5:
            x = x.squeeze(1)  # Remove "time" -> B, C, H, W

        truth = y[:, step, :, :, :]

        assert (
            x.shape == truth.shape
        ), "x shape does not match y shape: {} vs {}".format(x.shape, truth.shape)
        # Forward pass
        alpha, beta = model(x)

        # Calculate loss
        distribution_loss = beta_nll_loss(alpha, beta, truth)

        total_loss += distribution_loss

        if n_steps > 1:
            sample = torch.distributions.Beta(alpha, beta).sample((10,))[0]
            median = sample.median(0)[0]

            x = median

    return total_loss / n_steps


def roll_forecast_eval(model, x, y, n_steps, diag):
    assert y.shape[1] == n_steps

    total_loss = 0

    alphas, betas, samples = [], [], []
    for step in range(n_steps):
        if x.ndim == 5:
            x = x.squeeze(1)  # Remove "time" -> B, C, H, W

        truth = y[:, step, :, :, :]

        assert (
            x.shape == truth.shape
        ), "x shape does not match y shape: {} vs {}".format(x.shape, truth.shape)
        # Forward pass
        alpha, beta = model(x)

        diag.concentrations.append(
            (
                torch.median(alpha.detach()).cpu().numpy(),
                torch.median(beta.detach()).cpu().numpy(),
            ),
        )

        # Calculate loss
        distribution_loss = beta_nll_loss(alpha, beta, truth)

        total_loss += distribution_loss

        if n_steps > 1:
            sample = torch.distributions.Beta(alpha, beta).sample((10,))[0]
            median = sample.median(0)[0]

            x = median

    # Select last predicted alpha and beta but first of batch
    alpha = alpha[0][-1].squeeze()
    beta = beta[0][-1].squeeze()

    # The same with truth
    _truth = y[0][-1].squeeze()
    samples = torch.distributions.Beta(alpha, beta).sample((10,)).cpu()
    median = samples.median(0)[0]

    sample_l1 = F.l1_loss(samples[0], _truth.cpu())
    median_l1 = F.l1_loss(median, _truth.cpu())

    diag.mae.append((sample_l1, median_l1))

    snr_pred = calculate_wavelet_snr(samples[0].cpu(), None)
    snr_real = calculate_wavelet_snr(_truth.cpu(), None)

    diag.snr_db.append((snr_real["snr_db"], snr_pred["snr_db"]))

    return total_loss / n_steps


class EnhancedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.attention = (
            WindowedAttentionBlock(out_channels) if use_attention else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


class Clamp(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, max=self.max_value)


# Simple UNet-like network that outputs multiple samples
class SimpleNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model_proper = UNet(dim)

        self.model_proper.conv1 = EnhancedConvBlock(1, dim, use_attention=False)
        self.model_proper.conv2 = EnhancedConvBlock(dim, dim * 2, use_attention=False)
        self.model_proper.conv3 = EnhancedConvBlock(
            dim * 2, dim * 4, use_attention=False
        )
        self.model_proper.conv4 = EnhancedConvBlock(
            dim * 4, dim * 8, use_attention=True
        )

        self.model_proper.bridge = MultiHeadAttentionBridge(dim * 8, dim * 16)

        # Modify prediction head to output alpha and beta parameters

        # self.conv5 = EnhancedConvBlock(512, 256, use_attention=True)
        # self.conv6 = EnhancedConvBlock(256, 128, use_attention=True)

        self.model_proper.prediction_head = nn.Identity()

        self.alpha_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),
            Clamp(max_value=40.0),
        )
        self.beta_head = nn.Sequential(
            nn.Conv2d(dim, 1, 5, padding=2),
            nn.Softplus(),
            Clamp(max_value=20.0),
        )

    def forward(self, x):
        assert x.ndim == 4  # B, C, H, W
        x = self.model_proper(x)

        alpha = self.alpha_head(x)
        beta = self.beta_head(x)

        return alpha, beta


class Diagnostics:
    def __init__(self):
        (
            self.train_loss,
            self.val_loss,
            self.concentrations,
            self.lr,
            self.mae,
            self.snr_db,
            self.bnll_loss,
        ) = ([], [], [], [], [], [], [])


def train(n_iterations, n_steps, lr):
    global train_no

    train_no += 1

    print(
        "Start training no={} n_iterations={} n_steps={} lr={}".format(
            train_no, n_iterations, n_steps, lr
        )
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = datetime.now()

    diag = Diagnostics()

    scaler = torch.amp.GradScaler()

    scheduler = get_lr_schedule(
        optimizer, warmup_iterations=n_iterations // 50, total_iterations=n_iterations
    )

    for iteration in range(1, n_iterations + 1):
        model.train()

        input_field, truth = read_train_data(
            batch_size=batch_size, input_size=input_size, n_y=n_steps
        )
        input_field = input_field.to(device)
        truth = truth.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float32):

            train_loss = roll_forecast_train(model, input_field, truth, n_steps)

            diag.bnll_loss.append(train_loss.item())

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        diag.train_loss.append(train_loss.item())
        diag.lr.append(optimizer.param_groups[0]["lr"])

        if iteration % 25 != 0:
            continue

        val_loss = evaluate(model, diag, iteration, n_steps)

        if iteration % 250 == 0:
            stop_time = datetime.now()

            print(
                "Iteration {:05d}/{:05d}, Train Loss: {:.4f}, Val Loss: {:.4f}, L1: {:.4f} Time: {}".format(
                    iteration,
                    n_iterations,
                    train_loss.item(),
                    val_loss.item(),
                    diag.mae[-1][1],
                    convert_delta(stop_time - start_time),
                )
            )

            start_time = datetime.now()


def evaluate(model, diag, iteration, n_steps):
    model.eval()

    with torch.no_grad():
        val_input_field, val_truth = read_val_data(
            batch_size=batch_size * 2, input_size=input_size, shuffle=True, n_y=n_steps
        )

        val_input_field = val_input_field.to(device)
        val_truth = val_truth.to(device)

        val_loss = roll_forecast_eval(model, val_input_field, val_truth, n_steps, diag)

        diag.val_loss.append(val_loss.item())

        if iteration % 1000 == 0:

            input_field, truth = read_val_data(
                batch_size=batch_size * 2, input_size=input_size, n_y=n_steps
            )

            input_field = input_field[0].to(device)
            truth = truth.to(device)

            pred_alpha, pred_beta = model(input_field)
            pred_alpha = pred_alpha.detach()[0].squeeze().cpu()
            pred_beta = pred_beta.detach()[0].squeeze().cpu()

            samples = torch.distributions.Beta(pred_alpha, pred_beta).sample((1,))
            snr_pred = calculate_wavelet_snr(samples[0], None)

            diagnostics(
                diag,
                input_field[0][0].cpu().squeeze(),
                truth[0][-1].cpu().squeeze(),
                pred_alpha,
                pred_beta,
                snr_pred,
                iteration,
                train_no,
            )

    return val_loss


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 128
batch_size = 128
dim = 32

# perceptual_criterion = PerceptualLoss()

model = SimpleNet(dim=dim).to(device)
print(model)
print(
    "Number of trainable parameters: {:,}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)

torch.set_float32_matmul_precision("high")

# First training loop

model_file = f"models/model-train-1.pth"

if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, weights_only=True))
else:
    train(n_iterations=30000, n_steps=1, lr=1e-4)

    torch.save(model.state_dict(), model_file)

    x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None

# Second training loop

train(n_iterations=10000, n_steps=2, lr=1e-5)

torch.save(model.state_dict(), f"models/model-train-{n_steps}.pth")

x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None

# Third

train(n_iterations=5000, n_steps=4, lr=5e-6)

torch.save(model.state_dict(), f"models/model-train-{n_steps}.pth")

print("Done at {}".format(datetime.now()))
