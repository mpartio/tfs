import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def pad_tensor(x, window_size):
    """Pad input tensor so that H and W are divisible by window_size."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    padding = (0, 0, 0, pad_w, 0, pad_h)  # (left, right, top, bottom)

    # Apply padding
    x_padded = F.pad(x, padding)
    return x_padded, pad_h, pad_w  # Return padding values for depadding later


def depad_tensor(x_padded, pad_h, pad_w):
    """Remove padding from the input tensor."""
    if pad_h > 0:
        x_padded = x_padded[:, :-pad_h, :, :]
    if pad_w > 0:
        x_padded = x_padded[:, :, :-pad_w, :]
    return x_padded


def window_partition(x, window_size):
    """Split the input feature map into non-overlapping windows"""
    x, pad_h, pad_w = pad_tensor(x, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, pad_h, pad_w


def window_reverse(windows, window_size, H, W, pad_H, pad_W):
    """Reverse the windows into the original image shape"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = depad_tensor(x, pad_H, pad_W)
    return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        # Use a Conv2d layer to halve the spatial dimensions and double the channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=3,  # 3x3 kernel
            stride=2,  # Stride of 2 to downsample by a factor of 2
            padding=1,  # Maintain spatial dimensions while halving
        )
        self.norm = nn.BatchNorm2d(in_channels * 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        B, T, H, W, C = x.shape

        x = x.permute(0, 1, 4, 2, 3).view(B * T, C, H, W)
        # Apply convolution, normalization, and activation
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        # Reshape x back to (B, T, H', W', C) format
        _, out_channels, out_H, out_W = x.shape

        x = x.view(B, T, out_channels, out_H, out_W).permute(0, 1, 3, 4, 2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        # ConvTranspose2d to upsample the feature map and halve the channels
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.norm = nn.BatchNorm2d(in_channels // 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # Reshape to (B*T, C, H, W) to apply ConvTranspose2d
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        # Apply ConvTranspose2d, normalization, and activation
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Get the new height and width after upsampling
        _, out_channels, out_H, out_W = x.shape

        # Reshape back to (B, T, H', W', C') format
        x = x.permute(0, 1, 2, 3).reshape(B, T, out_H, out_W, out_channels)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim, stride):
        super(PatchEmbedding, self).__init__()

        # Create a Conv2d layer to "embed" each patch
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        _, _, out_H, out_W = x.shape
        x = x.reshape(B, T, out_H, out_W, -1)
        return x


class PatchRecovery(nn.Module):
    def __init__(self, patch_size, dim, output_dim=2):
        super(PatchRecovery, self).__init__()

        self.recover = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape

        x = x.permute(0, 1, 4, 2, 3).view(B * T, C, H, W)
        x = self.recover(x)
        x = x.view(B, T, x.shape[1], x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += residual  # Add residual connection
        x = x.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)
        return self.relu(x)


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Multi-head self-attention layer
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout
        )

        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # Dropout for attention and feed-forward outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, H, W, C = x.shape

        # Apply layer normalization before attention (pre-norm)
        x_norm = self.norm1(x)

        x_norm = x_norm.view(B * T, H * W, C)
        # Multi-head self-attention (with residual connection)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)  # Self-attention
        attn_output = attn_output.view(B, T, H, W, C)
        x = x + self.dropout(attn_output)  # Residual connection

        # Apply layer normalization before the feed-forward network (pre-norm)
        x_norm = self.norm2(x)

        # Feed-forward network (with residual connection)
        mlp_output = self.mlp(x_norm)
        x = x + self.dropout(mlp_output)  # Residual connection

        assert x.shape == (B, T, H, W, C)

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Window size (Wh, Ww)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Define qkv projection layers
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1, 2 * window_size - 1, num_heads))
        )

        # Initialize the relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w])
        )  # 2, window_size, window_size
        coords_flatten = torch.flatten(coords, 1)  # 2, window_size*window_size
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords.sum(-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table.view(
            -1, self.num_heads
        )[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.attn_drop(F.softmax(attn, dim=-1))
        else:
            attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, T, H, W, C = x.shape  # input shape [B, T, H, W, C]
        x = x.view(B * T, H, W, C)  # Merge batch and temporal dimensions

        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)  # Partition into windows
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)  # Window attention
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, T, H, W, C)  # Un-merge the batch and temporal dimensions
        return x
