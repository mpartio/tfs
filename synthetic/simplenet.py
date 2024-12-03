import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Generate synthetic data with gradient pattern
def generate_data(size=32, flip=False):
    # Diagonal gradient from top-left to bottom-right

    x = torch.linspace(0, 1, size)

    if not flip:
        y = torch.linspace(0, 1, size)
    else:
        y = torch.linspace(1, 0, size)

    xx, yy = torch.meshgrid(x, y)

    field = torch.sin(xx - yy)
    field = field + torch.randn_like(field) * 0.001
    input_field = (field - field.min()) / (field.max() - field.min())

    field = torch.cos(xx - yy)
    field = field + torch.randn_like(field) * 0.001
    target_field = (field - field.min()) / (field.max() - field.min())

    return input_field, target_field


def generate_beta_data(batch_size=16, size=32):
    def set_squares(arr, x, y, fill_value):
        x_indices = torch.arange(max(0, x - 1), min(arr.size(-2), x + 3))
        y_indices = torch.arange(max(0, y - 1), min(arr.size(-1), y + 3))

        xx, yy = torch.meshgrid(x_indices, y_indices, indexing="ij")

        for b in range(arr.size(0)):
            arr[b, xx, yy] = fill_value

        return arr

    alpha = torch.tensor(0.9)
    beta = torch.tensor(0.8)

    beta_dist = torch.distributions.Beta(alpha, beta)

    input_field = beta_dist.sample((batch_size, size, size))

    target = input_field.clone()
    target = target + torch.randn_like(target) * 0.01
    target = torch.clamp(target, 0, 1)

    for b in range(batch_size):
        for _ in range(5):
            x, y = torch.randint(low=3, high=size - 4, size=(2,))

            fill_value = 1 if torch.rand(1).item() > 0.5 else 0
            input_field[b : b + 1] = set_squares(
                input_field[b : b + 1], x, y, fill_value
            )
            target[b : b + 1] = set_squares(target[b : b + 1], x + 5, y + 5, fill_value)

    assert input_field.shape == (batch_size, size, size)

    input_field = input_field.unsqueeze(1)
    target = target.unsqueeze(1)

    return input_field, target


def gaussian_nll_loss(pred_mean, pred_logvar, target):
    """
    Regular Gaussian NLL loss where each pixel is independent

    Args:
        pred_mean: (batch, 1, h, w)
        pred_logvar: (batch, 1, h, w)
        target: (batch, 1, h, w)
    """
    var = torch.exp(pred_logvar)

    # Gaussian NLL: 0.5 * (log(var) + (x - μ)²/var)
    loss = 0.5 * (pred_logvar + (target - pred_mean) ** 2 / var)

    return loss.mean()


def local_spatial_gaussian_nll_loss(pred_mean, pred_logvar, target, window_size=5):
    # Create correlation kernel
    center = window_size // 2
    y, x = torch.meshgrid(torch.arange(window_size), torch.arange(window_size))
    coords = torch.stack([y, x], dim=-1).float().to(pred_mean.device)
    distances = ((coords - coords[center, center]) ** 2).sum(-1)
    kernel = torch.exp(-distances / (2 * 1.0**2))  # length_scale=1.0
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, :]  # Add batch and channel dims

    # Convert logvar to var
    var = torch.exp(pred_logvar)

    # Compute weighted squared difference
    squared_diff = (target - pred_mean) ** 2 / var
    weighted_diff = F.conv2d(squared_diff, kernel, padding=center)

    # Weight the log variance term
    weighted_logvar = F.conv2d(pred_logvar, kernel, padding=center)

    # Combine terms
    loss = 0.5 * (weighted_logvar + weighted_diff)

    return loss.mean()


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


# Simple UNet-like network that outputs multiple samples
class SimpleNet(nn.Module):
    def __init__(self, n_samples=32):
        super().__init__()
        self.n_samples = n_samples

        pad_size = 2  # Increase padding size

        # Encoder
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(1, 32, 5),  # Larger kernel
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.enc3 = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(64, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ReflectionPad2d(pad_size), nn.Conv2d(32, n_samples, 5), nn.Sigmoid()
        )

        # Modify final heads to output alpha and beta parameters
        self.alpha_head = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 1, 5),
            nn.Softplus(),  # alpha must be positive
        )
        self.beta_head = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 1, 5),
            nn.Softplus(),  # beta must be positive
        )

    #        self.mean_head = nn.Sequential(
    #            nn.ReflectionPad2d(2), nn.Conv2d(32, 1, 5), nn.Sigmoid()  # For mean
    #        )
    #        self.logvar_head = nn.Sequential(
    #            nn.ReflectionPad2d(2), nn.Conv2d(32, 1, 5)  # No sigmoid for logvar
    #        )

    def forward(self, x):
        # Input shape: (batch, 1, h, w)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x = self.dec1(x3) + x2
        x = self.dec2(x) + x1
        x = self.dec3(x)

        alpha = self.alpha_head(x)
        beta = self.beta_head(x)

        return alpha, beta

        # mean = self.mean_head(x)
        # logvar = self.logvar_head(x)

        # Output shape: (batch, n_samples, h, w)
        # return mean, logvar


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 32
model = SimpleNet(n_samples=32).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

input_field, truth = generate_beta_data(size)
input_field = input_field.to(device)
truth = truth.to(device)

# Training loop
n_epochs = 8000
for epoch in range(n_epochs):
    model.train()

    # Generate random training example
    #    input_field, truth = generate_data(size, flip=(torch.rand(1).item() >= 0.5))

    if epoch > 0 and epoch % 50 == 0:
        input_field, truth = generate_beta_data(size)
        input_field = input_field.to(device)
        truth = truth.to(device)

    # Forward pass
    pred_alpha, pred_beta = model(input_field)

    # Calculate loss

    prb_loss = beta_nll_loss(pred_alpha, pred_beta, truth)
    # prb_loss = local_spatial_beta_loss(pred_alpha, pred_beta, truth)
    # prb_loss = local_spatial_gaussian_nll_loss(
    #    pred_mean, pred_logvar, truth[None, None]
    # )
    # prb_loss = gaussian_nll_loss(pred_mean, pred_logvar, truth[None, None])

    #    mse_loss = F.mse_loss(pred_samples.mean(0), truth)  # Add L2 loss

    loss = prb_loss
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if epoch < 7500:
            continue
        # Visualize results
        _input_field = input_field[0, 0].cpu()
        _truth = truth[0, 0].cpu()
        pred_alpha = pred_alpha[0, 0].cpu()
        pred_beta = pred_beta[0, 0].cpu()

        with torch.no_grad():
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
            concentration = torch.stack([pred_alpha, pred_beta])
            sample = (
                torch.distributions.Beta(pred_alpha, pred_beta)
                .sample((10,))
                .cpu()
            )

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
            plt.title("Diff of Mean")
            plt.colorbar()

            data = sample[0] - _input_field
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            plt.subplot(3, 4, 11)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("Diff of Sample")
            plt.colorbar()

            data = median - _input_field
            norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            plt.subplot(3, 4, 12)
            plt.imshow(data, cmap=cmap, norm=norm)
            plt.title("Diff of Median")
            plt.colorbar()

            plt.tight_layout()

            plt.savefig("epoch_{:04d}.png".format(epoch + 1))
            plt.close()
