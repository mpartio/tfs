import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from layers import ConvolvedSkipConnection, MeanPredictionHead, pad, depad
from reference_layers import (
    Downsample,
    Upsample,
    PatchEmbedding,
    PatchRecovery,
    ViTBlock,
)


def pad_tensor(x, size=4):
    B, T, H, W, C = x.shape

    x = x.view(B * T, H, W, C)
    x = pad(x, size)
    _, H, W, _ = x.shape
    x = x.view(B, T, H, W, C)

    return x


def depad_tensor(x, newH, newW):
    B, T, H, W, C = x.shape

    x = x.view(B * T, H, W, C)
    x = depad(x, newH, newW)
    _, H, W, _ = x.shape
    x = x.view(B, T, H, W, C)

    return x


class CloudCastV2(nn.Module):
    def __init__(self, patch_size, dim, stride=None):
        super(CloudCastV2, self).__init__()
        if stride is None:
            stride = patch_size

        self.patch_embed = PatchEmbedding(patch_size, dim, stride)
        # dpr_list = np.linspace(0, 0.2, 8)  # from Pangu
        # dpr_list_shallow = [0.01, 0.05]
        # dpr_list_deep = [0.1, 0.15]
        # depths = [2, 2, 2, 2]
        # input_resolution = int((128 - patch_size[0]) / stride[0] + 1)
        # input_resolution_a = (input_resolution, input_resolution)
        # input_resolution_b = (input_resolution // 2, input_resolution // 2)

        # self.encoder1 = ProcessingLayer(
        #    depth=depths[0],
        #    dim=dim,
        #    num_heads=6,
        #    window_size=7,
        #    drop_path_ratio_list=dpr_list_shallow,
        #    input_resolution=input_resolution_a,
        # )
        # self.encoder2 = ProcessingLayer(
        #    depth=depths[1],
        #    dim=dim * 2,
        #    num_heads=12,
        #    window_size=7,
        #    drop_path_ratio_list=dpr_list_deep,
        #    input_resolution=input_resolution_b,
        # )
        # self.decoder1 = ProcessingLayer(
        #    depth=depths[2],
        #    dim=dim * 2,
        #    num_heads=12,
        #    window_size=7,
        #    drop_path_ratio_list=dpr_list_deep,
        #    input_resolution=input_resolution_b,
        # )
        # self.decoder2 = ProcessingLayer(
        #    depth=depths[3],
        #    dim=dim,
        #    num_heads=6,
        #    window_size=7,
        #    drop_path_ratio_list=dpr_list_shallow,
        #    input_resolution=input_resolution_a,
        # )

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
                ViTBlock(dim=dim * 4, num_heads=4, mlp_dim=1024)
            )

        self.upsample2 = Upsample(dim * 4)

        self.conv_skip2 = ConvolvedSkipConnection(dim * 2)

        self.upsample1 = Upsample(dim * 2)

        self.conv_skip1 = ConvolvedSkipConnection(dim)

        self.patch_recover = PatchRecovery(
            dim=dim,
            output_dim=1,
            patch_size=patch_size,
        )

    # self.gated_skip = GatedSkipConnection(dim, input_resolution_a)

    #        self.prediction_head = ProbabilisticPredictionHeadWithConv(dim//2)
    #        self.prediction_head = ProbabilisticPredictionHead(2)
    # self.mean_head = nn.Conv2d(dim, 1, kernel_size=1)
    # self.var_head = nn.Conv2d(dim, 1, kernel_size=1)

    # self.global_attn = GlobalAttentionLayer(dim, 6)

    def forward(self, x):
        # Reproject input data to latent space
        # From (B, 1, 128, 128, C) to (B, 1, 32, 32, C)

        _, _, origH, origW, _ = x.shape

        x = pad_tensor(x)
        x = self.patch_embed(x)

        # Store the tensor for skip connection
        skip1 = x
        skip1 = pad_tensor(skip1)

        # Encode 1, keep dimensions the same
        # x = self.encoder1(x)

        # Downsample from (B, 32, 32, C) to (B, 16, 16, C*2)
        x = self.downsample1(x)

        for block in self.transformer_block1:
            x = block(x)

        skip2 = x
        skip2 = pad_tensor(skip2)

        # Downsample from (B, 16, 16, C*2) to (B, 8, 8, C*4)
        x = self.downsample2(x)

        for block in self.transformer_block2:
            x = block(x)

        # Upsample from (B, 8, 8, C*4) to (B, 16, 16, C*2)
        x = self.upsample2(x)
        x = self.conv_skip2(skip2, x)

        # Upsample from (B, 16, 16, 384) to (B, 32, 32, 192)
        x = self.upsample1(x)

        # x = self.decoder2(x)

        # x = self.norm1(x)
        # attn_out = self.global_attn(x)

        x = self.conv_skip1(skip1, x)

        # Add skip connection: (B, 32, 32, 384)
        # x = self.gated_skip(skip, x + attn_out)
        # x = self.gated_skip(skip.squeeze(1), (x + attn_out).squeeze(1))
        # x = x.unsqueeze(1)
        # x = self.norm2(x)

        x = self.patch_recover(x)
        x = depad_tensor(x, origH, origW)

        return x
