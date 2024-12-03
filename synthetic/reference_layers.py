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
        B, C, H, W = x.shape

        # Apply convolution, normalization, and activation
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
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
        B, C, H, W = x.shape

        # Apply ConvTranspose2d, normalization, and activation
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

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
        B, C, H, W = x.shape
        x = self.proj(x)
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
        B, C, H, W = x.shape
        x = self.recover(x)
        return x


class ProcessingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProcessingBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, C, H, W = x.size()
        y = x.mean(dim=(2, 3))  # Global average pooling
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y))))
        y = y.view(B, C, 1, 1)
        return x * y  # Scale input by channel-wise weights


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(SelfAttentionBlock, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)  # Reshape to (B, H*W, C)
        qkv = self.qkv(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = qkv
        attn = self.softmax(
            (q @ k.transpose(-2, -1)) / (C**0.5)
        )  # Scaled dot-product
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)  # Reshape back
        return self.proj(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)  # Q, K, V projections
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)     # Final projection

    def forward(self, x):
        B, C, H, W = x.size()  # Input shape: [batch, channels, height, width]
       
        # QKV projection (conv2d): Output shape [batch, 3 * channels, height, width]
        qkv = self.qkv(x)

        # Reshape and split into Q, K, V: [batch, channels, height * width]
        qkv = qkv.view(B, 3, C, H * W).permute(1, 0, 2, 3)  # [3, batch, channels, height*width]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Extract Q, K, V

        # Attention scores: [batch, height*width, height*width]
        attn = self.softmax((q @ k.transpose(-2, -1)) / (C ** 0.5))  # Scaled dot product

        # Apply attention to values: [batch, channels, height * width]
        x = (attn @ v).view(B, C, H, W)  # Reshape back to [batch, channels, height, width]

        # Final projection
        x = self.proj(x)
        return x
