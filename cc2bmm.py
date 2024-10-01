import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from timm.models.layers import DropPath
from layers import (
    #    PatchEmbedding,
    #    PatchEmbeddingWithTemporal,
    #    PatchEmbeddingWithPositionalEncoding,
    #    WindowAttention,
    #    Downsample,
    #    Upsample,
    #    DownsampleWithConv,
    #    UpsampleWithConv,
    #    UpsampleWithInterpolation,
    #    PatchRecovery,
    #    PatchRecoveryRaw,
    #    PatchRecoveryRawWithStride,
    ConvolvedSkipConnection,
    #    TransformerBlock,
    #    ProcessingLayer,
    #    GlobalAttentionLayer,
    #   ProbabilisticPredictionHeadWithConv,
    #   ProbabilisticPredictionHead,
    #   MeanPredictionHead,
    MixtureBetaPredictionHead,
    MixtureBetaPredictionHeadLinear,
    MixtureBetaPredictionHeadWithConv,
)
from reference_layers import (
    Downsample,
    PatchEmbedding,
    PatchRecovery,
    #   ResidualBlock,
    Upsample,
    ViTBlock,
)


class CloudCastV2(nn.Module):
    def __init__(self, patch_size, dim, stride=None):
        super(CloudCastV2, self).__init__()
        if stride is None:
            stride = patch_size

        self.patch_embed = PatchEmbedding(patch_size, dim, stride)

        self.downsample1 = Downsample(dim)

        self.transformer_block1 = nn.ModuleList()

        for _ in range(1):
            self.transformer_block1.append(
                ViTBlock(dim=dim * 2, num_heads=4, mlp_dim=1024)
            )

        self.downsample2 = Downsample(dim * 2)

        self.transformer_block2 = nn.ModuleList()

        for _ in range(1):
            self.transformer_block2.append(
                ViTBlock(dim=dim * 4, num_heads=4, mlp_dim=2048)
            )

        self.upsample2 = Upsample(dim * 4)

        self.conv_skip2 = ConvolvedSkipConnection(dim * 2)

        self.upsample1 = Upsample(dim * 2)

        self.conv_skip1 = ConvolvedSkipConnection(dim)

        self.patch_recover = PatchRecovery(
            dim=dim,
            output_dim=dim,# 6, #2,
            patch_size=patch_size,
        )

        self.prediction_head = MixtureBetaPredictionHeadWithConv(dim)

    def forward(self, x):
        # Reproject input data to latent space
        # From (B, T, H, W, C) to (B, T, H // patch, W // patch, C * 2)

        x = self.patch_embed(x)

        # Store the tensor for skip connection
        skip1 = x

        # Downsample from (B, 32, 32, C) to (B, 16, 16, C*2)
        x = self.downsample1(x)

        for block in self.transformer_block1:
            x = block(x)

        skip2 = x

        # Downsample from (B, 16, 16, C*2) to (B, 8, 8, C*4)
        x = self.downsample2(x)

        for block in self.transformer_block2:
            x = block(x)

        # Upsample from (B, 8, 8, C*4) to (B, 16, 16, C*2)
        x = self.upsample2(x)

        x = self.conv_skip2(skip2, x)

        # Upsample from (B, 16, 16, 384) to (B, 32, 32, 192)
        x = self.upsample1(x)

        x = self.conv_skip1(skip1, x)

        x = self.patch_recover(x)

        assert torch.isnan(x).sum() == 0, "NaN detected in patch recovery output"

        alpha1, beta1, alpha2, beta2, weights = self.prediction_head(x)

        return alpha1, beta1, alpha2, beta2, weights
