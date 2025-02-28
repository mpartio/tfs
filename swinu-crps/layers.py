import torch
import torch.nn as nn
import torch.nn.functional as F
from swin import SwinTransformerBlock
from einops import rearrange
from math import sqrt


class SkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Main feature transform
        self.transform = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x_encoder, x_decoder):
        # x_encoder: [B, L, C]
        # x_decoder: [B, L, C]
        # noise_embedding: [B, noise_dim]

        # Transform encoder features
        x_skip = self.transform(x_encoder)  # [B, L, C]

        # Combine
        x_combined = x_skip

        return x_decoder + x_combined


class SqueezeExciteBlock(nn.Module):
    def __init__(self, dim, reduction=16, noise_dim=128):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=True)

        if noise_dim is not None:
            self.noise_proj = nn.Linear(noise_dim, dim)

    def forward(self, x, noise_embedding=None):

        if x.ndim == 4:
            # x shape: (batch, channels, height, width)
            B, C, H, W = x.size()

            # Squeeze: global average pooling over spatial dimensions
            y = x.view(B, C, -1).mean(dim=2)  # shape: (batch, channels)

        else:
            B, S, Ft = x.size()
            y = x.mean(dim=1)

        if noise_embedding is not None:
            y = y + self.noise_proj(noise_embedding)

        # Excitation: two-layer MLP with a bottleneck
        y = F.relu(self.fc1(y), inplace=True)  # shape: (batch, channels//reduction)
        y = torch.sigmoid(self.fc2(y))  # shape: (batch, channels)

        # Reshape to (batch, channels, 1, 1) for scaling
        if x.ndim == 4:
            y = y.view(B, C, 1, 1)
        else:
            y = y.view(B, 1, Ft)

        return x * y


class PatchMerge(nn.Module):
    def __init__(self, input_resolution, dim, time_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        self.time_dim = time_dim

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        T = self.time_dim

        assert L == T * H * W, f"input feature has wrong size: {L} != {T * H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, T, H * W, C)

        # Process each time step separately
        output_time_steps = []

        for t in range(T):
            xt = x[:, t, :, :]  # B H*W C
            xt = xt.reshape(B, H, W, C)

            x0 = xt[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = xt[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = xt[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = xt[:, 1::2, 1::2, :]  # B H/2 W/2 C
            xt = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            xt = self.norm(xt)
            xt = self.reduction(xt)
            xt = xt.reshape(B, (H // 2) * (W // 2), 2 * C)  # [B, H/2*W/2, 2*C]
            output_time_steps.append(xt)

        # Recombine time steps
        x = torch.cat(
            [step.unsqueeze(1) for step in output_time_steps], dim=1
        )  # [B, T, H/2*W/2, 2*C]
        x = x.reshape(B, T * (H // 2) * (W // 2), 2 * C)  # [B, T*H/2*W/2, 2*C]
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size: {L} != {H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, dim, patch_size, stride, in_channels=1):
        super().__init__()
        # Add +1 to in_channels for timestep
        self.proj = nn.Conv2d(
            in_channels=in_channels + 1,  # +1 for timestep
            out_channels=dim,
            kernel_size=patch_size,
            stride=stride,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, timestep):
        B, C, H, W = x.shape
        # Create timestep channel - same value across spatial dimensions
        time_channel = torch.full(
            (B, 1, H, W), timestep, device=x.device, dtype=x.dtype
        )
        # Concatenate along channel dimension
        x = torch.cat([x, time_channel], dim=1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B H*W C
        x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)  # Upscale by factor of 2
        self.norm = nn.LayerNorm(dim // dim_scale)
        self.channel_reduction = nn.Conv2d(
            dim, dim // dim_scale, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Reshape for Conv2D processing
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Apply PixelShuffle upsampling
        with torch.amp.autocast("cuda", enabled=False):
            x = self.expand(x.float())
            x = self.pixel_shuffle(x)
            x = self.channel_reduction(x)

        # Reshape back to (B, L, C) for LayerNorm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous()

        # Apply normalization
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.expand = nn.Conv2d(dim, dim * dim_scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output_dim = dim // 2  # Output channel count after shuffle
        self.norm = nn.LayerNorm(self.output_dim)
        self.refinement = nn.Conv2d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Expand channels and apply PixelShuffle
        with torch.amp.autocast("cuda", enabled=False):
            x = self.expand(x.float())

        x = self.pixel_shuffle(x)  # Upscale resolution by 4x

        # Convert back to (B, L, C) for normalization
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous()  # (B, L, C)

        # Apply normalization
        x = self.norm(x)

        # Reshape back for refinement
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.refinement(x)  # Final refinement step

        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, window_size, input_resolution):
        super(BasicBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    input_resolution=input_resolution,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
