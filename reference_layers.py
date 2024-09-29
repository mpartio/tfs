import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


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
        x = x.view(B, T, x.shape[1], x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
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
        x = x.reshape(B * T, C, H, W)

        # Apply ConvTranspose2d, normalization, and activation
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Get the new height and width after upsampling
        _, out_channels, out_H, out_W = x.shape

        # Reshape back to (B, T, H', W', C') format
        x = x.view(B, T, out_H, out_W, out_channels)

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
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.proj(x)
        _, _, out_H, out_W = x.shape
        x = x.view(B, T, out_H, out_W, -1)
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
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1, 2 * window_size - 1, num_heads))
        )

        # Compute the relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # B_, N, num_heads, head_dim
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
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
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)
        shortcut = x
        x = self.norm1(x)

        # Cyclically shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # Shape: (B*num_windows, window_size, window_size, C)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # Flatten windows

        # Window-based multi-head self-attention
        attn_windows = self.attn(x_windows)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, T, H, W, C)
        return x
