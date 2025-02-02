import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
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
