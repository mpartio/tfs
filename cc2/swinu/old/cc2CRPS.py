import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    PatchEmbedding,
    PatchMerging,
    PatchExpand,
    SkipConnection,
    BasicBlock,
    FinalPatchExpand_X4,
    SqueezeExciteBlock,
)
import lightning as L
import config
import os


class MultiHeadAttentionBridge(nn.Module):
    def __init__(self, in_dim, bridge_dim, n_layers=1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, bridge_dim)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "mha": nn.MultiheadAttention(
                            bridge_dim, num_heads=8, dropout=0.1
                        ),
                        "norm1": nn.LayerNorm(bridge_dim),
                        "norm2": nn.LayerNorm(bridge_dim),
                        "ff": nn.Sequential(
                            nn.Linear(bridge_dim, bridge_dim * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(bridge_dim * 2, bridge_dim),
                        ),
                    }
                )
            )

        self.output_norm = nn.LayerNorm(bridge_dim)
        self.gate_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x shape: [B, N_patches, C]
        B, N, C = x.shape

        # Project features
        x = self.input_norm(x)
        x = self.input_proj(x)

        # Main identity for deep residual
        main_identity = x

        for layer in self.layers:
            # Attention
            normed_x = layer["norm1"](x)
            attn_out = layer["mha"](normed_x)
            x = x + attn_out

            # FFN
            ffn_out = layer["ff"](layer["norm2"](x))
            x = x + ffn_out

        gate = torch.sigmoid(self.gate_param)
        x = x + gate * main_identity

        return self.output_norm(x)


class cc2CRPS(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(cc2CRPS, self).__init__()

        dim = config.hidden_dim

        self.patch_embed = PatchEmbedding(
            in_channels=config.num_input_channels, dim=dim, patch_size=2, stride=2
        )

        input_h, input_w = config.input_resolution

        # Encoder
        self.encoder1 = BasicBlock(
            dim=dim,
            num_heads=config.num_heads[0],
            window_size=config.window_size,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
            num_blocks=config.num_blocks[0],
        )

        self.skip1 = SkipConnection(dim)

        self.downsample1 = PatchMerging(
            dim=dim,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
        )

        self.encoder2 = BasicBlock(
            dim=dim * 2,
            num_heads=config.num_heads[1],
            window_size=config.window_size,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
            num_blocks=config.num_blocks[1],
        )

        self.skip2 = SkipConnection(dim * 2)

        self.downsample2 = PatchMerging(
            dim=dim * 2,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
        )

        self.encoder3 = BasicBlock(
            dim=dim * 4,
            num_heads=config.num_heads[2],
            window_size=config.window_size,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
            num_blocks=config.num_blocks[2],
        )

        self.skip3 = SkipConnection(dim * 4)

        # Attention Bridge

        if config.num_layers:
            self.bridge = SqueezeExciteBlock(dim=dim * 4, noise_dim=None)
        else:
            self.bridge = lambda x: x

        # Decoder (mirroring encoder)
        self.decoder3 = BasicBlock(
            dim=dim * 4,
            num_heads=config.num_heads[3],
            window_size=config.window_size,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
            num_blocks=config.num_blocks[3],
        )

        # Upsample layers
        self.upsample2 = PatchExpand(
            dim=dim * 4,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
        )

        self.decoder2 = BasicBlock(
            dim=dim * 2,
            num_heads=config.num_heads[4],
            window_size=config.window_size,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
            num_blocks=config.num_blocks[4],
        )

        # Upsample layers
        self.upsample1 = PatchExpand(
            dim=dim * 2,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
        )

        self.decoder1 = BasicBlock(
            dim=dim,
            num_heads=config.num_heads[5],
            window_size=config.window_size,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
            num_blocks=config.num_blocks[5],
        )

        self.final_expand = FinalPatchExpand_X4(
            dim=dim,
            dim_scale=2,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
        )

        self.prediction_head = nn.Sequential(
            # Tanh for delta prediction
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Tanh(),
        )

    def _forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.downsample1(x1))
        x3 = self.encoder3(self.downsample2(x2))

        # Bridge
        x = self.bridge(x3)

        # Decoder
        x = self.decoder3(x)
        x = self.skip3(x3, x)
        x = self.decoder2(self.upsample2(x))
        x = self.skip2(x2, x)
        x = self.decoder1(self.upsample1(x))
        x = self.skip1(x1, x)

        x = self.final_expand(x)
        x = self.prediction_head(x)

        return x

    def soft_clamp(self, x, min_val, max_val, T=5):
        return min_val + (max_val - min_val) * torch.sigmoid(T * x)

    def members_vec(self, x, last_state, timestep):
        B, C, H, W = x.shape

        x = self.patch_embed(x, timestep)
        #        features = torch.clone(x).detach()

        # Get tendencies for all members at once
        #        tendency = self._forward(features)
        tendency = self._forward(x)

        # Calculate allowed deltas for all members
        max_allowed_delta = 1.0 - last_state
        min_allowed_delta = 0.0 - last_state
        tendency = self.soft_clamp(tendency, min_allowed_delta, max_allowed_delta)

        # Reshape back to separate batch and member dimensions
        tendency = tendency.view(B, 1, H, W)

        return tendency

    def forward(self, x, last_state, timestep):
        assert (
            type(timestep) == int and timestep >= 1
        ), f"timestep must be integer larger than zero, got: {timestep}"

        assert x.ndim == 5, "expected input shape is: B, T, C, H, W, got: {}".format(
            x.shape
        )

        B, T, C, H, W = x.shape

        x = x.reshape(B, T * C, H, W)

        timestep = 1 if timestep == 1 else 2

        return self.members_vec(x, last_state, timestep)


if __name__ == "__main__":
    model = cc2CRPS(config.get_config())
    print(model)
    print(
        "Number of trainable parameters: {:,}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    x = torch.randn(1, 3, 2, 128, 128)
    deltas, predictions = model(x, 1)

    print("tendencies:", deltas.shape)
    print("predictions:", predictions.shape)
