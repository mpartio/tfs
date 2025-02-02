import torch
import torch.nn as nn
import torch.nn.functional as F
from swinu_l_cond import (
    SwinTransformerBlock,
    PatchExpand,
    FinalPatchExpand_X4,
    ConditionalLayerNorm,
)
from layers import PatchEmbedding, PatchMerging, NoisySkipConnection
import lightning as L
import config


class NoiseProcessor(nn.Module):
    def __init__(self, noise_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = noise_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),  # AIFS-CRPS uses GELU
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


class MultiHeadAttentionBridge(nn.Module):
    def __init__(self, in_dim, bridge_dim, noise_dim=128, n_layers=1):
        super().__init__()
        self.input_norm = ConditionalLayerNorm(in_dim, noise_dim)
        self.input_proj = nn.Linear(in_dim, bridge_dim)

        # Add noise processing
        self.noise_proj = nn.Linear(noise_dim, bridge_dim)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "mha": nn.MultiheadAttention(
                            bridge_dim, num_heads=8, dropout=0.1, batch_first=True
                        ),
                        "norm1": ConditionalLayerNorm(bridge_dim, noise_dim),
                        "norm2": ConditionalLayerNorm(bridge_dim, noise_dim),
                        "ff": nn.Sequential(
                            nn.Linear(bridge_dim, bridge_dim * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(bridge_dim * 2, bridge_dim),
                        ),
                    }
                )
            )

        self.output_norm = ConditionalLayerNorm(bridge_dim, noise_dim)

    def forward(self, x, noise_embedding):
        # x shape: [B, N_patches, C]
        B, N, C = x.shape

        # Project features
        x = self.input_norm(x, noise_embedding)
        x = self.input_proj(x)

        # Process noise and expand to match sequence length
        noise = self.noise_proj(noise_embedding)  # [B, bridge_dim]
        noise = noise.unsqueeze(1).expand(
            -1, x.shape[1], -1
        )  # [B, N_patches, bridge_dim]

        # Add noise to input features
        x = x + noise

        # Main identity for deep residual
        main_identity = x

        # Prevent NaNs
        #        x = F.layer_norm(x, (x.shape[-1],))

        for layer in self.layers:
            # Attention
            normed_x = layer["norm1"](x, noise_embedding)
            attn_out, _ = layer["mha"](normed_x, normed_x, normed_x)
            x = x + attn_out

            # FFN
            ffn_out = layer["ff"](layer["norm2"](x, noise_embedding))
            x = x + ffn_out

        x = x + main_identity

        return self.output_norm(x, noise_embedding)


class cc2CRPS(nn.Module):
    def __init__(
        self,
        config,
        noise_dim=128,
    ):
        super(cc2CRPS, self).__init__()

        dim = config.hidden_dim

        self.patch_embed = PatchEmbedding(
            in_channels=2, dim=dim, patch_size=2, stride=2
        )

        input_h, input_w = config.input_resolution

        # Encoder
        self.encoder1 = BasicBlock(
            dim=dim,
            num_heads=config.num_heads[0],
            window_size=config.window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
            num_blocks=config.num_blocks[0],
        )

        self.skip1 = NoisySkipConnection(dim)

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
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
            num_blocks=config.num_blocks[1],
        )

        self.skip2 = NoisySkipConnection(dim * 2)

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
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
            num_blocks=config.num_blocks[2],
        )

        self.skip3 = NoisySkipConnection(dim * 4)

        # Attention Bridge
        self.bridge = MultiHeadAttentionBridge(
            in_dim=dim * 4, bridge_dim=dim * 4, n_layers=config.num_layers
        )

        # Decoder (mirroring encoder)
        self.decoder3 = BasicBlock(
            dim=dim * 4,
            num_heads=config.num_heads[3],
            window_size=config.window_size,
            noise_dim=noise_dim,
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
            noise_dim=noise_dim,
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
            noise_dim=noise_dim,
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

        # Add noise processing
        self.noise_dim = noise_dim
        self.noise_processor = NoiseProcessor(self.noise_dim)
        self.n_members = config.num_members

    def _forward(self, x, noise_embedding):
        # Encoder
        x1 = self.encoder1(x, noise_embedding)

        x2 = self.encoder2(self.downsample1(x1), noise_embedding)
        x3 = self.encoder3(self.downsample2(x2), noise_embedding)

        # Bridge
        x = self.bridge(x3, noise_embedding)

        # Decoder
        x = self.decoder3(x, noise_embedding)
        x = self.skip3(x3, x, noise_embedding)
        x = self.decoder2(self.upsample2(x), noise_embedding)
        x = self.skip2(x2, x, noise_embedding)
        x = self.decoder1(self.upsample1(x), noise_embedding)
        x = self.skip1(x1, x, noise_embedding)

        x = self.final_expand(x)
        x = self.prediction_head(x)
        return x

    def forward(self, x, timestep):
        def soft_clamp(x, min_val, max_val, T=5):
            return min_val + (max_val - min_val) * torch.sigmoid(T * x)

        assert (
            type(timestep) == int and timestep >= 1
        ), f"timestep must be integer larger than zero, got: {timestep}"
        assert x.ndim == 5, "expected input shape is: B, M, C, H, W, got: {}".format(
            x.shape
        )

        B, M, C, H, W = x.shape
        timestep = 1 if timestep == 1 else 2
        assert (
            M == self.n_members
        ), "input members does not match configuration: {} vs {}".format(
            M, self.n_members
        )

        # Reshape to treat batch and members as one batch dimension
        xm = x.reshape(B * M, C, H, W)

        last_state = torch.clone(xm[:, -1:, :, :]).detach()  # B*M, 1, H, W

        xm = self.patch_embed(xm, timestep)
        features = torch.clone(xm).detach()

        # Generate noise for all members at once
        noise = torch.randn(B * M, self.noise_dim, device=x.device)
        noise_embed = self.noise_processor(noise)

        # Get tendencies for all members at once
        tendency = self._forward(features, noise_embed)

        # Calculate allowed deltas for all members
        max_allowed_delta = 1.0 - last_state
        min_allowed_delta = 0.0 - last_state
        delta = soft_clamp(tendency, min_allowed_delta, max_allowed_delta)

        # Reshape back to separate batch and member dimensions
        delta = delta.view(B, M, 1, H, W)

        predictions = x[:, :, -1:, ...] + delta
        #        predictions = torch.clamp(predictions, 0, 1)

        return delta, predictions


if __name__ == "__main__":
    model = cc2CRPS(config.get_config())
    print(model)
    print(
        "Number of trainable parameters: {:,}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    x = torch.randn(1, 2, 128, 128)
    deltas, predictions = model(x, 1)

    print("tendencies:", deltas.shape)
    print("predictions:", predictions.shape)
