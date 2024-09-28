import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from timm.models.layers import DropPath
from layers import (
    PatchEmbedding,
    PatchEmbeddingWithTemporal,
    PatchEmbeddingWithPositionalEncoding,
    WindowAttention,
    Downsample,
    Upsample,
    DownsampleWithConv,
    UpsampleWithConv,
    UpsampleWithInterpolation,
    PatchRecovery,
    PatchRecoveryRaw,
    PatchRecoveryRawWithStride,
    GatedSkipConnection,
    TransformerBlock,
    ProcessingLayer,
    GlobalAttentionLayer,
)

class CloudCastV2(nn.Module):
    def __init__(self, patch_size, dim, stride=None):
        super(CloudCastV2, self).__init__()
        if stride is None:
            stride = patch_size

        self.patch_embed = PatchEmbedding(patch_size, dim, stride)
        # dpr_list = np.linspace(0, 0.2, 8)  # from Pangu
        dpr_list_shallow = [0.01, 0.05]
        dpr_list_deep = [0.1, 0.15]
        depths = [2, 2, 2, 2]
        input_resolution = int((128 - patch_size[0]) / stride[0] + 1)
        input_resolution_a = (input_resolution, input_resolution)
        input_resolution_b = (input_resolution // 2, input_resolution // 2)

        self.encoder1 = ProcessingLayer(
            depth=depths[0],
            dim=dim,
            num_heads=6,
            window_size=7,
            drop_path_ratio_list=dpr_list_shallow,
            input_resolution=input_resolution_a,
        )
        self.encoder2 = ProcessingLayer(
            depth=depths[1],
            dim=dim * 2,
            num_heads=12,
            window_size=7,
            drop_path_ratio_list=dpr_list_deep,
            input_resolution=input_resolution_b,
        )
        self.decoder1 = ProcessingLayer(
            depth=depths[2],
            dim=dim * 2,
            num_heads=12,
            window_size=7,
            drop_path_ratio_list=dpr_list_deep,
            input_resolution=input_resolution_b,
        )
        self.decoder2 = ProcessingLayer(
            depth=depths[3],
            dim=dim,
            num_heads=6,
            window_size=7,
            drop_path_ratio_list=dpr_list_shallow,
            input_resolution=input_resolution_a,
        )

        # self.downsample = DownsampleWithConv(dim)
        self.downsample = Downsample(dim)
        # self.upsample = UpsampleWithInterpolation(dim * 2)
        self.upsample = UpsampleWithConv(dim * 2)
        # self.upsample = Upsample(dim * 2)

        # self.patch_recover = PatchRecoveryRawWithStride(
        #    dim, recover_size=(128, 128), patch_size=patch_size, num_output=2
        # )

        self.patch_recover = PatchRecoveryRaw(
            dim,  # * 2
            recover_size=(128, 128),
            patch_size=patch_size,
        )

        self.loss_type = None
        self.patch_size = patch_size

        self.gated_skip = GatedSkipConnection(dim, input_resolution_a)

        self.mean_head = nn.Conv2d(dim, 1, kernel_size=1)
        self.var_head = nn.Conv2d(dim, 1, kernel_size=1)
        self.var_dropout = nn.Dropout(p=0.1)

        self.global_attn = GlobalAttentionLayer(dim, 6)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._initialize_variance_head()

    def forward(self, x):
        # Reproject input data to latent space
        # From (B, 1, 1, 128, 128) to (B, 1, 32, 32, 192)
        x = self.patch_embed(x)

        # Store the tensor for skip connection
        skip = x

        # assert x.shape[1:] == (
        #    1,
        #    16,
        #    16,
        #    192,
        # ), f"Invalid shape after patch embedding: {x.shape}"

        # Encode 1, keep dimensions the same
        x = self.encoder1(x)

        # Downsample from (B, 32, 32, 192) to (B, 16, 16, 384)
        x = self.downsample(x)

        # x = self.encoder2(x)

        # Encoder finished, now decode
        # x = self.decoder1(x)

        # Upsample from (B, 16, 16, 384) to (B, 32, 32, 192)
        x = self.upsample(x)

        x = self.decoder2(x)

        x = self.norm1(x)
        attn_out = self.global_attn(x)

        # Add skip connection: (B, 32, 32, 384)
        # x = self.gated_skip(skip, x + attn_out)
        x = self.gated_skip(skip.squeeze(1), (x + attn_out).squeeze(1))
        x = x.unsqueeze(1)
        x = self.norm2(x)

        x = self.patch_recover(x)

        if self.loss_type == "gaussian_nll" or self.loss_type == "crps":
            B, P, H, W, C = x.shape
            x = x.reshape(B * P, H, W, C).permute(0, 3, 1, 2)
            mean = self.mean_head(x)
            variance_logits = self.var_head(x)
            variance_logits = self.var_dropout(variance_logits)
            var = F.softplus(variance_logits) + 1e-6
            stde = torch.sqrt(var)

            return mean, stde

        elif self.loss_type == "hete":
            B, P, H, W, C = x.shape
            x = x.reshape(B * P, H, W, C).permute(0, 3, 1, 2)
            mean = self.mean_head(x)
            variance_logits = self.var_head(x)
            var = F.softplus(variance_logits) + 1e-6

            return mean, var

        elif self.loss_type == "beta_nll":
            alpha = F.softplus(x[..., 0]) + 1e-6
            beta = F.softplus(x[..., 1]) + 1e-6
            return alpha, beta

        elif self.loss_type == "mse" or self.loss_type == "mae":
            return x.permute(0, 1, 4, 2, 3)

        raise Exception("Invalid loss function: {}".format(self.loss_type))

    def _initialize_variance_head(self):
        nn.init.xavier_normal_(self.var_head.weight, gain=0.01)
        nn.init.constant_(self.var_head.bias, 0.1)
