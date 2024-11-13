import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from timm.models.layers import DropPath


def pad(x, window_size):
    """Pad the input tensor for the use of window partition."""
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h == 0 and pad_w == 0:
        return x

    x = x.permute(0, 3, 1, 2)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    x = x.permute(0, 2, 3, 1)
    return x


def depad(x, H, W):
    """Depad the input tensor for the use of window partition."""
    B, H_pad, W_pad, C = x.shape
    x = x[:, :H, :W, :]
    return x


# Helper function for window partition and reverse
def window_partition(x, window_size):
    """Split the input into non-overlapping windows."""
    x = pad(x, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = einops.rearrange(x, "b nh ws1 nw ws2 c -> (b nh nw) ws1 ws2 c")
    return windows, H, W


def window_reverse(windows, window_size, H_pad, W_pad, H, W):
    """Reverse the windows into the original image shape."""
    # windows shape: (B * num_windows, window_size ** 2, C)

    # Separate batch dimension and number of windows
    B = windows.shape[0] // (H_pad // window_size * W_pad // window_size)

    nh = H_pad // window_size  # Number of windows along height
    nw = W_pad // window_size  # Number of windows along width

    x = einops.rearrange(
        windows,
        "(b nh nw) (ws1 ws2) c -> b nh ws1 nw ws2 c",
        b=B,
        nh=nh,
        nw=nw,
        ws1=window_size,
        ws2=window_size,
    )

    x = x.reshape(B, H_pad, W_pad, -1)

    x = depad(x, H, W)

    return x


def create_relative_position_index(window_size):
    coords = torch.arange(window_size).to("cuda")
    coords = torch.stack(
        torch.meshgrid([coords, coords], indexing="ij")
    )  # Shape: [2, 4, 4] (rows and columns)

    # Flatten the grid to get patch coordinates
    coords_flatten = coords.flatten(
        1
    )  # Shape: [2, 16] (each column is a patch (row, col))

    # Compute the relative coordinates between patches
    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # Shape: [2, 16, 16]

    # Normalize the relative coordinates to be non-negative by shifting them
    # Shift by window_size - 1 to make all values non-negative
    relative_coords[0] += window_size - 1  # Relative row
    relative_coords[1] += window_size - 1  # Relative col

    # Convert the relative coordinates to a single index
    relative_position_index = (
        relative_coords[0] * (2 * window_size - 1) + relative_coords[1]
    )

    return relative_position_index


class ProbabilisticPredictionHeadWithConv(nn.Module):
    def __init__(self, dim):
        super(ProbabilisticPredictionHeadWithConv, self).__init__()
        # self.mean_head = nn.Conv2d(
        #    in_channels=dim, out_channels=1, kernel_size=5, padding=2
        # )
        self.mean_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        #        self.var_head = nn.Conv2d(
        #            in_channels=dim, out_channels=1, kernel_size=5, padding=2
        #        )
        self.var_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6

        mean = mean.permute(0, 2, 3, 1).view(B, T, H, W, 1)
        var = var.permute(0, 2, 3, 1).view(B, T, H, W, 1)

        return mean, torch.sqrt(var)


class MixtureProbabilisticPredictionHeadWithConv(nn.Module):
    def __init__(self, dim, num_mix):
        super(MixtureProbabilisticPredictionHeadWithConv, self).__init__()
        self.mean_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=dim // 2, out_channels=num_mix, kernel_size=3, padding=1
            ),
            nn.Softplus(),
        )
        self.var_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=dim // 2, out_channels=num_mix, kernel_size=3, padding=1
            ),
            nn.Softplus(),
        )
        self.weights_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=dim // 2, out_channels=num_mix, kernel_size=3, padding=1
            ),
            nn.Softplus(),
        )
        self.num_mix = num_mix

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        mean = self.mean_head(x)
        var = self.var_head(x)
        weights = F.softmax(self.weights_head(x), dim=1) + 1e-6

        mean = mean.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        var = var.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        weights = weights.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)

        return mean, torch.sqrt(var), weights


class MeanPredictionHead(nn.Module):
    def __init__(self):
        super(MeanPredictionHead, self).__init__()
        # self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(1, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        B, T, H, W, C = x.shape
        # x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x = self.linear(x)
        x = self.activation(x)
        return x


class ProbabilisticPredictionHead(nn.Module):
    def __init__(self, dim):
        super(ProbabilisticPredictionHead, self).__init__()
        self.mean_head = nn.Linear(dim, 1)
        self.var_head = nn.Linear(dim, 1)

    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6

        return mean, torch.sqrt(var)


class MultiScaleMixtureBetaPredictionHeadWithConv(nn.Module):
    def __init__(self, dim, num_mix):
        super(MultiScaleMixtureBetaPredictionHeadWithConv, self).__init__()

        self.alpha_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )
        self.beta_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )
        self.weight_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )

        # Heads for refinement predictions
        self.alpha_refine = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )
        self.beta_refine = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )

        self.dropout = nn.Dropout(0.1)
        self.num_mix = num_mix

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = self.dropout(x)

        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)  # Reshape for Conv2d

        # Predict at multiple scales and refine
        # Start with low resolution
        x_down = F.avg_pool2d(x, 4)
        alpha_low = self.alpha_head(x_down)
        beta_low = self.beta_head(x_down)

        # Upsample and refine
        alpha_up = F.interpolate(
            alpha_low, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        beta_up = F.interpolate(
            beta_low, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        # Predict refinement
        alpha_refine = self.alpha_refine(x)
        beta_refine = self.beta_refine(x)

        # Combine
        alpha = alpha_up + alpha_refine
        beta = beta_up + beta_refine

        alpha = F.softplus(alpha) + 1e-6
        beta = F.softplus(beta) + 1e-6
        weights = F.softmax(self.weight_head(x), dim=1) + 1e-6

        alpha = alpha.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        beta = beta.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        weights = weights.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)

        return alpha, beta, weights


class MixtureBetaPredictionHeadWithConv(nn.Module):
    def __init__(self, dim, num_mix):
        super(MixtureBetaPredictionHeadWithConv, self).__init__()

        self.alpha_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )
        self.beta_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )
        self.weight_head = nn.Conv2d(
            in_channels=dim, out_channels=num_mix, kernel_size=3, padding=1
        )

        self.dropout = nn.Dropout(0.1)
        self.num_mix = num_mix

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = self.dropout(x)

        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)  # Reshape for Conv2d

        alpha = F.softplus(self.alpha_head(x)) + 1e-6
        beta = F.softplus(self.beta_head(x)) + 1e-6
        weights = F.softmax(self.weight_head(x), dim=1)

        alpha = alpha.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        beta = beta.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)
        weights = weights.permute(0, 2, 3, 1).view(B, T, H, W, self.num_mix)

        return alpha, beta, weights


class MixtureBetaPredictionHeadLinear(nn.Module):
    def __init__(self, dim, num_mix):
        super(MixtureBetaPredictionHeadLinear, self).__init__()

        self.alpha_head = nn.Linear(dim, num_mix)
        self.beta_head = nn.Linear(dim, num_mix)
        self.weight_head = nn.Linear(dim, num_mix)
        self.num_mix = num_mix

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H * W, C)

        assert (
            torch.max(x) <= 100
        ), "Input to MixtureBetaPredictionHeadLinear is too large {}".format(
            torch.amax(x, dim=(0, 1))
        )
        alpha = F.softplus(self.alpha_head(x)) + 1e-6
        beta = F.softplus(self.beta_head(x)) + 1e-6

        # Predict weights and apply softmax to ensure they sum to 1
        weights = F.softmax(self.weight_head(x), dim=-1)  # Predict w1, w2

        alpha = alpha.view(B, T, H, W, self.num_mix)
        beta = beta.view(B, T, H, W, self.num_mix)
        weights = weights.view(B, T, H, W, self.num_mix)

        return alpha, beta, weights


class MixtureBetaPredictionHead(nn.Module):
    def __init__(self, num_mix):
        super(MixtureBetaPredictionHead, self).__init__()
        self.num_mix = num_mix

    def forward(self, x):
        B, T, H, W, C = x.shape

        n = self.num_mix

        assert C == 3 * n, f"Expected channels to be 3*num_mix, but got {C}"

        alpha = F.softplus(x[:, :, :, :, :n]) + 1e-6
        beta = F.softplus(x[:, :, :, :, n : 2 * n]) + 1e-6
        weights = F.softmax(x[:, :, :, :, 2 * n :], dim=-1)

        assert len(alpha.shape) == 5

        assert (
            alpha.shape == beta.shape == weights.shape
        ), "Invalid shape for output: {} - {} - {}".format(
            alpha.shape, beta.shape, weights.shape
        )

        return alpha, beta, weights


class PatchEmbeddingWithTemporal(nn.Module):
    def __init__(self, patch_size, dim, stride):
        super(PatchEmbeddingWithTemporal, self).__init__()

        # Note: conv2d only works when T=1
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_size,
            stride=stride,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)
        # Apply the convolution to embed patches
        x = self.conv(x)  # (B, dim, H // patch_size[1], W // patch_size[2])
        _, C, D, out_H, out_W = x.shape

        x = x.reshape(B, D, out_H, out_W, C)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim, stride):
        super(PatchEmbedding, self).__init__()

        # Note: conv2d only works when T=1
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_size,
            stride=stride,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Remove the unnecessary time and channel dimension (T and C)
        # Note: this might need to be changed later
        x = x.squeeze(1)  # [:, 0, 1, :, :]  # Shape: (B, H, W)
        # Apply the convolution to embed patches
        x = self.conv(x)  # (B, dim, H // patch_size[1], W // patch_size[2])
        _, _, out_H, out_W = x.shape
        x = x.reshape(B, 1, out_H, out_W, -1)

        return x


class PatchEmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, patch_size, dim):
        super(PatchEmbeddingWithPositionalEncoding, self).__init__()

        # Note: conv2d only works when T=1
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )

        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, dim, 128 // patch_size[-1], 128 // patch_size[-1])
        )
        self.patch_size = patch_size
        self.dim = dim

    def forward(self, x):
        B, _, _, H, W = x.shape
        assert x.shape[1:] == (1, 1, 128, 128)
        # Remove the unnecessary time and channel dimension (T and C)
        # Note: this might need to be changed later
        x = x.squeeze(1)  # [:, 0, 1, :, :]  # Shape: (B, H, W)
        # Apply the convolution to embed patches
        x = self.conv(x)  # (B, dim, H // patch_size[1], W // patch_size[2])
        absolute_pos_embed = self.absolute_pos_embed.expand(B, -1, -1, -1)

        x = x + absolute_pos_embed

        x = x.reshape(B, 1, H // self.patch_size[1], W // self.patch_size[2], -1)
        return x


# Basic multi-head self-attention (MHA)
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        # Linear layers for query, key, and value
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Relative positional bias (simplified)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1, 2 * window_size - 1, num_heads))
        )

        self.relative_position_index = create_relative_position_index(window_size)
        self.window_size = window_size

    def forward(self, x, mask=None):
        B, N, C = x.shape  # B: batch size, N: number of patches in window, C: channels
        q = self.query(x).view(B, N, self.num_heads, C // self.num_heads)
        k = self.key(x).view(B, N, self.num_heads, C // self.num_heads)
        v = self.value(x).view(B, N, self.num_heads, C // self.num_heads)

        # Scaled dot-product attention
        attn = torch.einsum("bnhc,bmhc->bnhm", q, k) * self.scale

        relative_position_index = self.relative_position_index.reshape(-1)
        relative_position_bias_table = self.relative_position_bias_table.view(
            1, -1, self.num_heads
        )

        relative_position_bias = relative_position_bias_table[
            0, relative_position_index
        ]  # .view(49, 49, 6)

        # Broadcast the same positional bias to all batches
        relative_position_bias = relative_position_bias.unsqueeze(0).expand(
            B, -1, -1, -1
        )
        relative_position_bias = relative_position_bias.view(B, N, self.num_heads, N)

        # Add relative positional bias
        attn += relative_position_bias

        if mask is not None:
            mask = mask.expand(B, -1, -1, -1).to(x.device)
            attn = attn.masked_fill(mask == 1, -1e9)

        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bnhm,bmhc->bnhc", attn, v)
        out = out.reshape(B, N, C)

        return self.proj(out)


class GlobalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(GlobalAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, H, W, C = x.shape
        # Assuming input is (batch, num_patches, channels)
        x = x.view(B * T, H * W, C)
        # Apply multihead attention across all patches globally
        x = x.permute(
            1, 0, 2
        )  # Rearrange for attention (sequence length, batch size, channels)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x.permute(1, 0, 2)  # Restore original dimensions

        # Apply LayerNorm and MLP
        x = self.norm1(x)
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, T, H, W, C)
        return x


class DownsampleWithConv(nn.Module):
    def __init__(self, dim):
        super(DownsampleWithConv, self).__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).squeeze(1)
        x = self.conv(x)
        x = x.unsqueeze(1).permute(0, 1, 3, 4, 2)
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        # Downsampling is *not* done with convolutional layers but
        # simply with a linear layer

        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)
        x = pad(x, 2)
        _, H, W, _ = x.shape
        x = x.view(B, T, H, W, C)
        # B, T, H, W, C = x.shape

        assert H % 2 == 0 and W % 2 == 0, "Height and width must be divisible by 2"

        # Split the spatial dimension H and W into half, creating two new dimensions
        x = x.reshape(B, T, H // 2, 2, W // 2, 2, C)

        # Rearrange the tensor so that the halves are next to each others, and the two
        # new dimensions are next to each others.
        x = x.permute(0, 1, 2, 4, 3, 5, 6)

        # Reshape the tensor so that the two new dimensions are concatenated
        x = x.reshape(B, T, H // 2, W // 2, 4 * C)

        # Now spatial resolution is halved, but the number of channels is quadrupled
        # Apply a linear layer to reduce the number of channels back to 2 * dim
        x = self.norm(x)
        x = self.linear(x)

        assert x.shape == (B, T, H // 2, W // 2, 2 * C)
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()

        self.linear1 = nn.Linear(dim, dim * 2, bias=False)
        self.linear2 = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = self.linear1(x)
        x = x.reshape(B, T, H * 2, W * 2, C // 2)
        x = self.norm(x)
        x = self.linear2(x)

        assert x.shape == (B, T, H * 2, W * 2, C // 2)

        return x


class UpsampleWithConv(nn.Module):
    def __init__(self, dim):
        super(UpsampleWithConv, self).__init__()
        self.conv = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).squeeze(1)
        x = self.conv(x)
        x = x.unsqueeze(1).permute(0, 1, 3, 4, 2)
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)
        # How to know how much padding was added?
        x = depad(x, H, W)
        _, H_new, W_new, _ = x.shape
        x = x.view(B, T, H_new, W_new, C)
        return x


class UpsampleWithInterpolation(nn.Module):
    def __init__(self, dim):
        super(UpsampleWithInterpolation, self).__init__()
        # After interpolation, use a regular convolution to adjust channel dimensions
        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)

    def forward(self, x):
        B, T, H, W, C = x.shape
        # Perform bilinear interpolation
        H_new, W_new = H * 2, W * 2
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = F.interpolate(x, size=(H_new, W_new), mode="bilinear", align_corners=False)

        # Follow up with a convolution to adjust channels after upsampling
        x = self.conv(x)

        x = x.reshape(B, T, C // 2, H_new, W_new).permute(0, 1, 3, 4, 2)
        return x


class PatchRecovery(nn.Module):
    def __init__(self, dim, recover_size, patch_size, num_output):
        super(PatchRecovery, self).__init__()

        # Outputs are 2-channel images: mean and variance
        self.num_output = num_output
        self.conv = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=self.num_output,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.recover_size = recover_size

    def forward(self, x):
        # P = number of parameters
        # We don't predict multiple time steps, although multiple time steps
        # are used in the input
        B, P, H, W, C = x.shape
        x = x.reshape(B * P, C, H, W)
        x = self.conv(x)  # (B * C, 2, 128, 128)
        rec_H, rec_W = self.recover_size
        x = x.view(B, P, self.num_output, rec_H, rec_W).permute(0, 1, 3, 4, 2)

        return x


class PatchRecoveryRawWithStride(nn.Module):
    def __init__(self, dim, recover_size, patch_size, num_output):
        super(PatchRecoveryRawWithStride, self).__init__()

        self.stride = (patch_size[0] // 2, patch_size[1] // 2)

        self.conv = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=num_output,
            kernel_size=patch_size,
            stride=self.stride,
        )
        self.recover_size = recover_size
        self.dim = dim
        self.num_output = num_output
        self.patch_size = patch_size

    def forward(self, x):
        # P = number of parameters
        # We don't predict multiple time steps, although multiple time steps
        # are used in the input
        B, P, H, W, C = x.shape
        x = x.reshape(B * P, C, H, W)
        x = self.conv(x)

        output_H = (H - 1) * self.stride[0] + self.patch_size[0]
        output_W = (W - 1) * self.stride[1] + self.patch_size[1]

        # Use unfold to extract all patches at once
        patches = F.unfold(
            x.view(B * P, self.num_output, output_H, output_W),
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=0,
        )  # Shape: (B*P, C*patch_h*patch_w, num_patches)

        patch_h, patch_w = self.patch_size

        # Reshape the patches for accumulation
        patches = patches.view(B * P, self.num_output * patch_h * patch_w, -1)

        # Now reconstruct the full image using fold or tensor reshaping operations
        rec_H, rec_W = self.recover_size

        output_buffer = torch.zeros(B, self.num_output, rec_H, rec_W).to(x.device)
        #        count_buffer = torch.zeros(B, 1, rec_H, rec_W).to(x.device)
        count_buffer = F.fold(
            torch.ones_like(patches),
            output_size=(rec_H, rec_W),
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        # Calculate the output buffer without explicit loops (using efficient torch operations)
        output_buffer = F.fold(
            patches, (rec_H, rec_W), kernel_size=self.patch_size, stride=self.stride
        )

        # Divide by the count buffer (make sure it's computed correctly using similar tensor ops)
        x = output_buffer / count_buffer
        x = x.view(B, P, self.num_output, rec_H, rec_W).permute(0, 1, 3, 4, 2)
        return x


class PatchRecoveryRaw(nn.Module):
    def __init__(self, dim, recover_size, patch_size):
        super(PatchRecoveryRaw, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.recover_size = recover_size
        self.dim = dim

    def forward(self, x):
        # P = number of parameters
        # We don't predict multiple time steps, although multiple time steps
        # are used in the input
        B, P, H, W, C = x.shape
        x = x.reshape(B * P, C, H, W)
        x = self.conv(x)  # (B * C, 2, H, W)
        rec_H, rec_W = self.recover_size
        x = x.view(B, P, self.dim, rec_H, rec_W).permute(0, 1, 3, 4, 2)

        return x


class ConvolvedSkipConnection(nn.Module):
    def __init__(self, dim):
        super(ConvolvedSkipConnection, self).__init__()

        # self.skip_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.x_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, skip, x):
        assert (
            skip.shape == x.shape
        ), "Skip and input tensor must have the same shape: {} vs {}".format(
            skip.shape, x.shape
        )
        B, T, H, W, C = x.shape

        skip = skip.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)

        # skip = self.skip_conv(skip)

        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        x = self.x_conv(x)

        x = self.norm(x)
        x = self.relu(x)

        return x.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)


class GatedSkipConnection(nn.Module):
    def __init__(self, dim, input_resolution):
        super(GatedSkipConnection, self).__init__()
        # Learnable gating coefficient for each pixel
        H, W = input_resolution

        self.gate = nn.Parameter(torch.zeros(1, H, W, 1))  # Initialize the gate to 0
        self.sigmoid = nn.Sigmoid()
        self.input_resolution = input_resolution

    def forward(self, skip, x):
        assert skip.shape == x.shape, "Skip and input tensor must have the same shape"
        B, H, W, C = x.shape
        assert (
            skip.shape[1:3] == self.gate.shape[1:3]
        ), "Skip and gate must have same shape: {} vs {}".format(
            skip.shape[1:3], self.gate.shape[1:3]
        )
        gate_value = self.sigmoid(self.gate).expand(B, H, W, C)

        # Multiply the skip connection by the gate
        gated_skip = gate_value * skip

        return x + gated_skip


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, drop_path):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Layer norm before attention and feed-forward layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        # Layer norm and feed-forward network (MLP)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )

        self.patch_mask = None

        if shift_size > 0:
            self.patch_mask = torch.zeros(1, window_size, window_size, 1, 1)

            for j in range(window_size):
                for i in range(window_size):
                    if j >= window_size - shift_size or i >= window_size - shift_size:
                        self.patch_mask[0, j, i, 0, 0] = 1

            # Repeat the mask for all heads to (almost) match the shape of attn
            # "Almost", because we don't know the batch size yet
            # (1, window_size ** 2, num_heads, window_size ** 2)
            self.patch_mask = self.patch_mask.view(1, window_size**2, 1, 1).repeat(
                1, 1, num_heads, window_size**2
            )

        self.drop_path = nn.Identity() if drop_path is None else DropPath(drop_path)

    #        self.gated_skip = GatedSkipConnection(
    #            dim, (input_resolution[0], input_resolution[1])
    #        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        assert torch.isnan(x).sum() == 0, "NaN values in input tensor"

        skip = x.reshape(B, H * W, C)
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # Cyclically shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x_windows, H_pad, W_pad = window_partition(
            x, self.window_size
        )  # Shape: (B*num_windows, window_size, window_size, C)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # Flatten windows

        # W-MSA/SW-MSA (Window Multi-Head Self Attention)
        attn_windows = self.attn(x_windows, mask=self.patch_mask)

        # Reverse windows to original image
        attn_reversed = window_reverse(
            attn_windows, self.window_size, H_pad, W_pad, H, W
        )

        x = x + attn_reversed
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # x = x.reshape(B, H * W, C)

        # FFN and residual connection
        #       x = self.gated_skip(skip.reshape(B, H, W, C), x)
        x = self.drop_path(x)
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, T, H, W, C)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, input_resolution, num_heads, window_size, shift_size, drop_path_ratio
    ):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Layer norm before attention and feed-forward layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        # Layer norm and feed-forward network (MLP)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )

        self.patch_mask = None

        if shift_size > 0:
            self.patch_mask = torch.zeros(1, window_size, window_size, 1, 1)

            for j in range(window_size):
                for i in range(window_size):
                    if j >= window_size - shift_size or i >= window_size - shift_size:
                        self.patch_mask[0, j, i, 0, 0] = 1

            # Repeat the mask for all heads to (almost) match the shape of attn
            # "Almost", because we don't know the batch size yet
            # (1, window_size ** 2, num_heads, window_size ** 2)
            self.patch_mask = self.patch_mask.view(1, window_size**2, 1, 1).repeat(
                1, 1, num_heads, window_size**2
            )

        self.drop_path = (
            nn.Identity() if drop_path_ratio is None else DropPath(drop_path_ratio)
        )

        self.gated_skip = GatedSkipConnection(
            dim, (input_resolution[0], input_resolution[1])
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        assert torch.isnan(x).sum() == 0, "NaN values in input tensor"

        skip = x.reshape(B, H * W, C)
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # Cyclically shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x_windows, H_pad, W_pad = window_partition(
            x, self.window_size
        )  # Shape: (B*num_windows, window_size, window_size, C)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # Flatten windows

        # W-MSA/SW-MSA (Window Multi-Head Self Attention)
        attn_windows = self.attn(x_windows, mask=self.patch_mask)

        # Reverse windows to original image
        attn_reversed = window_reverse(
            attn_windows, self.window_size, H_pad, W_pad, H, W
        )

        x = x + attn_reversed
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # x = x.reshape(B, H * W, C)

        # FFN and residual connection
        x = self.gated_skip(skip.reshape(B, H, W, C), x)
        x = self.drop_path(x)
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, T, H, W, C)

        return x


class ProcessingLayer(nn.Module):
    def __init__(
        self, depth, num_heads, dim, window_size, drop_path_ratio_list, input_resolution
    ):
        super(ProcessingLayer, self).__init__()

        self.depth = depth
        self.blocks = nn.ModuleList()
        assert (
            len(drop_path_ratio_list) == depth
        ), "Drop path ratios must match depth: {} vs {}".format(
            len(drop_path_ratio_list), depth
        )
        for i in range(depth):
            self.blocks.append(
                TransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2 if i % 2 == 0 else 0,
                    drop_path_ratio=drop_path_ratio_list[i],
                )
            )

    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)

        return x
