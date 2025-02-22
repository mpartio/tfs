import torch
import torch.nn as nn
import torch.nn.functional as F
from swin import SwinTransformerBlock
from einops import rearrange


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

        assert L == H * W, "input feature has wrong size"
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
