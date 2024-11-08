import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from timm.models.layers import DropPath
from layers import (
    ConvolvedSkipConnection,
    SwinTransformerBlock,
    MixtureProbabilisticPredictionHeadWithConv
)
from reference_layers import (
    Downsample,
    PatchEmbedding,
    PatchRecovery,
    ResidualBlock,
    Upsample,
    ViTBlock,
)


class CloudCastV2(nn.Module):
    def __init__(self, patch_size, dim, stride=None, num_mix=2):
        super(CloudCastV2, self).__init__()
        if stride is None:
            stride = patch_size

        self.patch_embed = PatchEmbedding(patch_size, dim, stride)

        self.downsample1 = Downsample(dim)

        self.residual_block1 = ResidualBlock(dim * 2)

        self.transformer_block1 = nn.ModuleList()

        for i in range(2):
            self.transformer_block1.append(
                SwinTransformerBlock(
                    dim=dim * 2,
                    window_size=5,
                    shift_size=((i % 2) * 2),
                    num_heads=4,
                    drop_path=0.05,
                )
            )

        self.downsample2 = Downsample(dim * 2)

        self.residual_block2 = ResidualBlock(dim * 4)

        self.transformer_block2 = nn.ModuleList()

        for i in range(4):
            self.transformer_block2.append(
                SwinTransformerBlock(
                    dim=dim * 4,
                    window_size=5,
                    shift_size=((i % 2) * 2),
                    num_heads=8,
                    drop_path=0.1,
                )
            )

        self.conv_skip3 = ConvolvedSkipConnection(dim * 4)

        self.upsample2 = Upsample(dim * 4)

        self.residual_block3 = ResidualBlock(dim * 2)

        self.transformer_block3 = nn.ModuleList()

        for i in range(4):
            self.transformer_block3.append(
                SwinTransformerBlock(
                    dim=dim * 2,
                    window_size=5,
                    shift_size=((i % 2) * 2),
                    num_heads=8,
                    drop_path=0.1,
                )
            )

        self.conv_skip2 = ConvolvedSkipConnection(dim * 2)

        self.upsample1 = Upsample(dim * 2)

        self.residual_block4 = ResidualBlock(dim)

        self.transformer_block4 = nn.ModuleList()

        for i in range(2):
            self.transformer_block4.append(
                SwinTransformerBlock(
                    dim=dim,
                    window_size=5,
                    shift_size=((i % 2) * 2),
                    num_heads=4,
                    drop_path=0.1,
                )
            )

        self.conv_skip1 = ConvolvedSkipConnection(dim)

        self.patch_recover = PatchRecovery(
            dim=dim,
            output_dim=dim,
            patch_size=patch_size,
        )

        self.prediction_head = MixtureProbabilisticPredictionHeadWithConv(dim, num_mix)

        self.num_mix = num_mix

    def forward(self, x):
        # Reproject input data to latent space
        # From (B, T, H, W, C) to (B, T, H // patch, W // patch, C * 2)

        x = self.patch_embed(x)

        # Store the tensor for skip connection
        skip1 = x

        # Downsample from (B, 32, 32, C) to (B, 16, 16, C*2)
        x = self.downsample1(x)

        x = self.residual_block1(x)

        for block in self.transformer_block1:
            x = block(x)

        skip2 = x

        # Downsample from (B, 16, 16, C*2) to (B, 8, 8, C*4)
        x = self.downsample2(x)

        skip3 = x

        x = self.residual_block2(x)

        for block in self.transformer_block2:
            x = block(x)

        x = self.conv_skip3(skip3, x)

        # Upsample from (B, 8, 8, C*4) to (B, 16, 16, C*2)
        x = self.upsample2(x)

        x = self.residual_block3(x)

        for block in self.transformer_block3:
            x = block(x)

        x = self.conv_skip2(skip2, x)

        # Upsample from (B, 16, 16, 384) to (B, 32, 32, 192)
        x = self.upsample1(x)

        x = self.residual_block4(x)

        for block in self.transformer_block4:
            x = block(x)

        x = self.conv_skip1(skip1, x)

        x = self.patch_recover(x)

        mean, stde, weights = self.prediction_head(x)

        return mean, stde, weights
