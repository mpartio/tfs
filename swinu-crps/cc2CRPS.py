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
from scipy.stats import beta as stats_beta
from datetime import datetime, timedelta
from swinu import SwinU
from crps import CRPSGaussianLoss
from util import calculate_wavelet_snr

x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None
train_no = 0

import torch
import torch.nn.functional as F


def smooth_data(data: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0):
    """
    Smooths 2D data using Gaussian blur

    Args:
        data: Input tensor of shape (B, C, H, W) or (C, H, W)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
    """
    # Add batch dimension if needed
    if data.dim() == 3:
        data = data.unsqueeze(0)

    # Create Gaussian kernel
    channels = data.size(1)
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size))

    # Fill kernel with Gaussian values
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - center
            dy = y - center
            kernel[0, 0, x, y] = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Apply to all channels
    kernel = kernel.to(data.device)
    smoothed = F.conv2d(data, kernel, padding=center, groups=channels)

    # Ensure output stays in [0,1]
    smoothed = torch.clamp(smoothed, 0, 1)

    return smoothed.squeeze(0) if data.dim() == 3 else smoothed


def diagnostics(diag, input_field, truth, pred, mean, stde, iteration, train_no):

    #    assert pred_alpha.ndim == 2

    plt.figure(figsize=(24, 12))
    plt.suptitle(
        "{} at train {} iteration {} (host={}, time={})".format(
            sys.argv[0],
            train_no,
            iteration,
            platform.node(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    plt.subplot(351)
    plt.imshow(input_field[0])
    plt.title("Input time=T-1")
    plt.colorbar()

    plt.subplot(352)
    plt.imshow(input_field[1])
    plt.title("Input time=T")
    plt.colorbar()

    plt.subplot(353)
    plt.imshow(truth)
    plt.title("Truth")
    plt.colorbar()

    plt.subplot(354)
    plt.imshow(pred)
    plt.title("Prediction")
    plt.colorbar()

    plt.subplot(355)
    plt.imshow(mean)
    plt.title("Mean residual")
    plt.colorbar()

    plt.subplot(356)
    plt.imshow(stde)
    plt.title("Std of residual")
    plt.colorbar()

    # Add a random sample from predictions
    #    sample = torch.distributions.Beta(pred_alpha, pred_beta).sample((10,))

    sample = torch.normal(mean, stde)
    plt.subplot(357)
    plt.imshow(sample.squeeze())
    plt.title("One Random Sample of Residual")
    plt.colorbar()

    data = truth - input_field[-1]
    cmap = plt.cm.coolwarm  # You can also try 'bwr' or other diverging colormaps
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

    plt.subplot(358)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("True Residual")
    plt.colorbar()

    data = pred - input_field[-1]
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.subplot(359)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Residual of Prediction/L1={:.4f}".format(F.l1_loss(pred, truth)))
    plt.colorbar()

    plt.subplot(3, 5, 10)
    plt.title("Losses")
    plt.plot(diag.train_loss, label="Train Loss", color="blue", alpha=0.3)
    plt.plot(
        moving_average(torch.tensor(diag.train_loss), 50),
        label="Train Loss MA",
        color="blue",
    )
    plt.plot(diag.val_loss, label="Val Loss", color="orange", alpha=0.3)
    plt.plot(
        moving_average(torch.tensor(diag.val_loss), 50),
        label="Val Loss MA",
        color="orange",
    )

    plt.legend(loc="upper left")
    ax2 = plt.gca().twinx()
    ax2.plot(torch.tensor(diag.lr) * 1e6, label="LRx1M", color="green")
    ax2.legend(loc="upper right")

    snr_db = np.array(diag.snr_db).T

    plt.subplot(3, 5, 11)
    snr_real = torch.tensor(snr_db[0])
    snr_pred = torch.tensor(snr_db[1])
    plt.plot(snr_real, label="Real", color="blue", alpha=0.3)
    plt.plot(
        moving_average(snr_real, 10),
        label="Moving Average",
        color="blue",
    )
    plt.plot(snr_pred, label="Pred", color="orange", alpha=0.3)
    plt.plot(
        moving_average(snr_pred, 10),
        color="orange",
        label="Moving average",
    )
    plt.legend(loc="center right")

    ax2 = plt.gca().twinx()
    ax2.plot(moving_average(snr_real - snr_pred, 20), label="Residual", color="green")
    ax2.legend(loc="upper right")
    plt.title("Signal to Noise Ratio")

    plt.subplot(3, 5, 12)
    plt.hist(truth.flatten(), bins=20)
    plt.title("Truth histogram")

    plt.subplot(3, 5, 13)
    plt.hist(pred.flatten(), bins=20)
    plt.title("Predicted histogram")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        "figures/{}_train_{}_iteration_{:05d}.png".format(
            platform.node(), train_no, iteration
        )
    )

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
    xshape, yshape = x.shape, y.shape
    if torch.rand(1).item() > 0.6:
        return x, y

    # Random flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [-2])  # Horizontal flip
        y = torch.flip(y, [-2])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, [-1])  # Vertical flip
        y = torch.flip(y, [-1])

    # Random 90-degree rotations
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k, [-2, -1])
    y = torch.rot90(y, k, [-2, -1])

    assert x.shape == xshape, "Invalid y shape after augmentation: {} vs {}".format(
        x.shape, xshape
    )
    assert y.shape == yshape, "Invalid x shape after augmentation: {} vs {}".format(
        y.shape, yshape
    )

    return x, y


def partition(tensor, n_x, n_y):
    S, T, H, W, C = tensor.shape
    group_size = n_x + n_y

    # Reshape into groups of (n_x + n_y), with padding if necessary
    num_groups = T // group_size
    new_length = num_groups * group_size

    padded_tensor = tensor[:, :new_length]

    # Reshape into groups
    reshaped = padded_tensor.reshape(S, -1, group_size, H, W)

    # Merge streams into one dim
    reshaped = reshaped.reshape(S * reshaped.shape[1], group_size, H, W)  # N, G, H, W

    # Extract single elements and blocks
    x = reshaped[:, :n_x]
    y = reshaped[:, n_x:]

    assert x.shape[0] > 0, "Not enough elements"

    ##    S, N, C, H, W = x.shape
    #   x = x.reshape(S * N, C, H, W) # N, C, H, W
    #    S, N, C, H, W = y.shape
    #    y = y.reshape(S * N, C, H, W)

    return x, y


def shuffle_to_hourly_streams(data):
    T, H, W, C = data.shape
    N = (T // 4) * 4
    data = data[:N, ...]

    data = data.reshape(-1, 4, H, W, C).transpose(1, 0, 2, 3, 4)

    return data


def read_beta_data(filename, n_x, n_y):
    import numpy as np

    data = np.load(filename)["arr_0"]

    data = shuffle_to_hourly_streams(data)
    data = torch.tensor(data)

    x_data, y_data = partition(data, n_x, n_y)

    return x_data, y_data


td_index = 0


def read_train_data(batch_size, input_size, n_x=1, n_y=1):
    global x_train_data, y_train_data, td_index
    if x_train_data is None:
        x_train_data, y_train_data = read_beta_data("../data/train-150k.npz", n_x, n_y)

    #    n = torch.randint(
    #        0,
    #        x_train_data.shape[0] - batch_size,
    #        (1,),
    #    ).item()

    x_data = x_train_data[td_index : td_index + batch_size]
    x_data = x_data[:, :, :input_size, :input_size]

    y_data = y_train_data[td_index : td_index + batch_size]
    y_data = y_data[:, :, :input_size, :input_size]

    td_index += batch_size
    if x_data.shape[0] - td_index < batch_size:
        td_index = 0

    # Y data can have multiple times, X not
    if y_data.ndim == 4:
        y_data = y_data.unsqueeze(1)

    assert x_data.ndim == 4, "invalid dimensions for x: {}".x_data.shape
    assert y_data.ndim == 5, "invalid dimensions for y: {}".y_data.shape
    return x_data, y_data


def read_val_data(batch_size, input_size, shuffle=False, n_x=1, n_y=1):
    global x_val_data, y_val_data
    if x_val_data is None:
        x_val_data, y_val_data = read_beta_data("../data/val-150k.npz", n_x, n_y)

    n = 0
    if shuffle:
        n = torch.randint(0, x_val_data.shape[0] - batch_size, (1,)).item()

    x_data = x_val_data[n : n + batch_size]
    x_data = x_data[:, :, :input_size, :input_size]

    y_data = y_val_data[n : n + batch_size]
    y_data = y_data[:, :, :input_size, :input_size]

    if y_data.ndim == 4:
        y_data = y_data.unsqueeze(1)

    assert x_data.ndim == 4, "invalid dimensions for x: {}".x_data.shape
    assert y_data.ndim == 5, "invalid dimensions for y: {}".y_data.shape
    return x_data, y_data


def get_lr_schedule(optimizer, warmup_iterations, total_iterations):
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


def roll_forecast(model, x, y, n_steps):
    assert y.shape[1] == n_steps

    total_loss = []
    means, stdes = [], []

    for step in range(n_steps):
        if x.ndim == 5:
            x = x.squeeze(1)  # Remove "time" -> B, C, H, W

        y_true = y[:, step, :, :, :]

        # X dim: B, C=2, H, W
        # Y dim: B, C=1, H, W
        assert (
            x.shape[-2:] == y_true.shape[-2:]
        ), "x shape does not match y shape: {} vs {}".format(x.shape, y_true.shape)

        assert (
            x.ndim == y_true.ndim
        ), "x and y need to have equal number of dimensions: {} vs {}".format(
            x.shape, y_true.shape
        )
        # Forward pass

        mean, stde = model(x)

        # If multiple x (n_x>1), take the last one as the target for residuals
        last_x = x[:, -1].unsqueeze(1)

        assert (
            last_x.shape == y_true.shape
        ), "last_x and y_true shape mismatch: {} vs {}".format(
            last_x.shape, y_true.shape
        )
        loss = crps_loss(mean, stde, y_true - last_x)

        next_state = last_x + mean
        bound_penalty = torch.mean(F.relu(next_state - 1) + F.relu(-next_state))

        loss = loss + 0.1 * bound_penalty

        total_loss.append(loss)
        means.append(mean.detach())
        stdes.append(stde.detach())

        assert n_steps == 1
        if n_steps > 1:

            x = pred

    return torch.stack(total_loss), means, stdes


class DeltaPredictionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Mean change predictor
        self.delta_mean = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=5, padding=2),
            nn.Tanh(),  # Bound changes to [-1,1] range
        )
        # Uncertainty in the change
        self.delta_std = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=5, padding=2),
            nn.Softplus(),  # Ensure positive std
        )

    def forward(self, x):
        return self.delta_mean(x), self.delta_std(x)


class SimpleNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = SwinU(
            embed_dim=dim,
            patch_size=2,
            num_heads=[4, 8, 16, 16],
        )
        self.backbone.output = nn.Identity()
        self.prediction_head = DeltaPredictionHead(dim)

    def forward(self, x):
        assert x.ndim == 4  # B, C, H, W
        x = self.backbone(x)
        mean, stde = self.prediction_head(x)
        return mean, stde


class Diagnostics:
    def __init__(self):
        (
            self.train_loss,
            self.val_loss,
            self.lr,
            self.mae,
            self.snr_db,
            self.bnll_loss,
        ) = ([], [], [], [], [], [])


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
        optimizer, warmup_iterations=2000, total_iterations=n_iterations
    )

    loss_so_far = 0

    for iteration in range(1, n_iterations + 1):
        model.train()

        input_field, truth = read_train_data(
            batch_size=batch_size, input_size=input_size, n_x=2, n_y=n_steps
        )
        #         input_field, truth = augment_data(input_field, truth)

        input_field = input_field.to(device)
        truth = truth.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):

            train_loss, _, _ = roll_forecast(model, input_field, truth, n_steps)
            train_loss = train_loss.mean()
            loss_so_far += train_loss.item()

            diag.bnll_loss.append(train_loss.item())
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        if iteration % 1000 == 0:
            val_loss = evaluate(model, diag, iteration, n_steps)

            stop_time = datetime.now()
            loss_so_far /= 1000
            diag.train_loss.append(loss_so_far)
            diag.lr.append(optimizer.param_groups[0]["lr"])

            mae = diag.mae[-1] if len(diag.mae) > 0 else float("nan")
            loss_so_far = 0

            if iteration % 3000 == 0:
                print(
                    "Iteration {:05d}/{:05d}, Train Loss: {:.4f}, Val Loss: {:.4f}, L1: {:.4f} Time: {}".format(
                        iteration,
                        n_iterations,
                        diag.train_loss[-1],
                        val_loss.item(),
                        mae,
                        convert_delta(stop_time - start_time),
                    )
                )

            start_time = datetime.now()


def evaluate(model, diag, iteration, n_steps):
    model.eval()

    val_loss = 0

    with torch.no_grad():
        n_val_batches = 1000
        for _ in range(n_val_batches):
            val_input_field, val_truth = read_val_data(
                batch_size=batch_size,
                input_size=input_size,
                shuffle=True,
                n_x=2,
                n_y=n_steps,
            )

            val_input_field = val_input_field.to(device)
            val_truth = val_truth.to(device)

            val_loss += roll_forecast(model, val_input_field, val_truth, n_steps)[
                0
            ].mean()

        val_loss /= n_val_batches
        diag.val_loss.append(val_loss.item())

        input_field, truth = read_val_data(
            batch_size=batch_size, input_size=input_size, n_x=2, n_y=n_steps
        )

        input_field = input_field[0].unsqueeze(0).to(device)  # B, C, H, W
        truth = truth[0].unsqueeze(0).to(device)  # B, T, C, H, W

        _, means, stdes = roll_forecast(model, input_field, truth, n_steps)

        mean = means[0].cpu().squeeze()  # list to first tensor
        stde = stdes[0].cpu().squeeze()

        last_x = (
            input_field[:, -1].cpu().squeeze()
        )  # select last of channels (if n_x > 1)
        truth = truth[:, -1].cpu().squeeze()  # select last of truths (if n_y > 1)

        pred = last_x + mean
        pred = torch.clamp(pred, 0, 1)
        diag.mae.append(F.l1_loss(pred, truth).item())

        snr_pred = calculate_wavelet_snr(pred, None)
        snr_real = calculate_wavelet_snr(truth, None)

        diag.snr_db.append((snr_real["snr_db"], snr_pred["snr_db"]))

        if iteration % 25000 == 0:

            diagnostics(
                diag,
                input_field.cpu().squeeze(),
                truth,
                pred,
                mean,
                stde,
                iteration,
                train_no,
            )

    return val_loss


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 128
batch_size = 32
dim = 96

crps_loss = CRPSGaussianLoss()

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
    train(n_iterations=int(5e5), n_steps=1, lr=1e-4)

    torch.save(model.state_dict(), model_file)

    x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None

# Second training loop

# train(n_iterations=60000, n_steps=2, lr=1e-5)

# torch.save(model.state_dict(), f"models/model-train-{n_steps}.pth")

# x_train_data, y_train_data, x_val_data, y_val_data = None, None, None, None

# Third

# train(n_iterations=30000, n_steps=4, lr=5e-6)

# torch.save(model.state_dict(), f"models/model-train-{n_steps}.pth")

print("Done at {}".format(datetime.now()))
