import torch
import torch.nn as nn
import torch.nn.functional as F
from swinu_l_cond import SwinTransformerBlock, ConditionalLayerNorm
from einops import rearrange


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, noise_dim=128):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = ConditionalLayerNorm(4 * dim, noise_dim)

    def forward(self, x, noise_embedding):
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

        x = self.norm(x, noise_embedding)
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
    def __init__(self, input_resolution, dim, dim_scale=2, noise_dim=128):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)  # Upscale by factor of 2
        self.norm = ConditionalLayerNorm(dim // dim_scale, noise_dim)
        self.channel_reduction = nn.Conv2d(
            dim, dim // dim_scale, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x, noise_embedding):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Reshape for Conv2D processing
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Apply PixelShuffle upsampling
        x = self.expand(x)
        x = self.pixel_shuffle(x)

        x = self.channel_reduction(x)

        # Reshape back to (B, L, C) for LayerNorm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        # Apply normalization
        x = self.norm(x, noise_embedding)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, noise_dim=128):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.expand = nn.Conv2d(dim, dim * dim_scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output_dim = dim // 2  # Output channel count after shuffle
        self.norm = ConditionalLayerNorm(self.output_dim, noise_dim)
        self.refinement = nn.Conv2d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, noise_embedding):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Expand channels and apply PixelShuffle
        x = self.expand(x)  # Increase feature channels
        x = self.pixel_shuffle(x)  # Upscale resolution by 4x

        # Convert back to (B, L, C) for normalization
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)  # (B, L, C)

        # Apply normalization
        x = self.norm(x, noise_embedding)

        # Reshape back for refinement
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.refinement(x)  # Final refinement step
        #        x = self.final_projection(x)

        return x


class NoisySkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Main feature transform
        self.transform = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # Project noise embedding
        self.noise_project = nn.Linear(128, dim)  # 128 is noise_dim

        # Separate path to process noise
        self.noise_transform = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x_encoder, x_decoder, noise_embedding):
        # x_encoder: [B, L, C]
        # x_decoder: [B, L, C]
        # noise_embedding: [B, noise_dim]

        # Transform encoder features
        x_skip = self.transform(x_encoder)  # [B, L, C]

        # Project and transform noise
        noise_proj = self.noise_project(noise_embedding)  # [B, C]
        noise_proj = noise_proj.unsqueeze(1).expand(
            -1, x_encoder.shape[1], -1
        )  # [B, L, C]
        noise_contribution = self.noise_transform(noise_proj)

        # Combine
        x_combined = x_skip + noise_contribution

        return x_decoder + x_combined


class NoiseProcessor(nn.Module):
    def __init__(self, noise_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = noise_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, noise_dim),
        )
        self.norm = nn.LayerNorm(noise_dim)

    def forward(self, noise):
        processed = self.mlp(noise)
        return self.norm(processed)


class BasicBlock(nn.Module):
    def __init__(
        self, dim, num_blocks, num_heads, window_size, noise_dim, input_resolution
    ):
        super(BasicBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    noise_dim=noise_dim,
                    input_resolution=input_resolution,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                )
            )

    def forward(self, x, noise_embedding):
        for layer in self.layers:
            x = layer(x, noise_embedding)
        return x
