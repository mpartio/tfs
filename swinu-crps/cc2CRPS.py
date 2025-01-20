import torch
import torch.nn as nn
import torch.nn.functional as F
from swinu_l_cond import (
    SwinTransformerBlock,
    PatchEmbedding,
    PatchMerging,
    PatchExpand,
    FinalPatchExpand_X4,
)
import lightning as L


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
    def __init__(self, in_dim, bridge_dim, n_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, bridge_dim)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "mha": nn.MultiheadAttention(
                            bridge_dim, num_heads=8, dropout=0.05, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(bridge_dim),
                        "norm2": nn.LayerNorm(bridge_dim),
                        "ff": nn.Sequential(
                            nn.Linear(bridge_dim, bridge_dim),
                            nn.GELU(),
                            nn.Linear(bridge_dim, bridge_dim),
                        ),
                    }
                )
            )

    def forward(self, x):
        # x shape: [B, N_patches, C]
        B, N, C = x.shape
        # Project features

        x = self.input_proj(x)

        # Prevent NaNs
        x = F.layer_norm(x, (x.shape[-1],))

        for layer in self.layers:
            # Attention
            attn_out, _ = layer["mha"](x, x, x)
            x = layer["norm1"](x + attn_out)

            # FFN
            ffn_out = layer["ff"](x)
            x = layer["norm2"](x + ffn_out)

        return x


class cc2CRPS(nn.Module):
    def __init__(
        self,
        dim,
        n_members=3,
        n_layers=4,
        input_resolution=(128, 128),
        noise_dim=128,
        window_size=8,
        num_heads=[8, 8, 8, 8, 8, 8],
    ):
        super(cc2CRPS, self).__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=2, dim=dim, patch_size=2, stride=2
        )

        input_h, input_w = input_resolution

        # Encoder
        self.encoder1 = BasicBlock(
            dim=dim,
            num_heads=num_heads[0],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
            num_blocks=2,
        )

        self.downsample1 = PatchMerging(
            dim=dim,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
        )

        self.encoder2 = BasicBlock(
            dim=dim * 2,
            num_heads=num_heads[1],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
            num_blocks=2,
        )

        self.downsample2 = PatchMerging(
            dim=dim * 2,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
        )

        self.encoder3 = BasicBlock(
            dim=dim * 4,
            num_heads=num_heads[2],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
            num_blocks=6,
        )

        # Attention Bridge (like AIFS-CRPS)
        self.bridge = MultiHeadAttentionBridge(
            in_dim=dim * 4, bridge_dim=dim * 8, n_layers=n_layers
        )

        # Decoder (mirroring encoder)
        self.decoder3 = BasicBlock(
            dim=dim * 8,
            num_heads=num_heads[3],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
            num_blocks=6,
        )

        # Upsample layers
        self.upsample2 = PatchExpand(
            dim=dim * 8,
            input_resolution=(
                input_h // 8,
                input_w // 8,
            ),
        )

        self.decoder2 = BasicBlock(
            dim=dim * 4,
            num_heads=num_heads[4],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
            num_blocks=2,
        )

        # Upsample layers
        self.upsample1 = PatchExpand(
            dim=dim * 4,
            input_resolution=(
                input_h // 4,
                input_w // 4,
            ),
        )

        self.decoder1 = BasicBlock(
            dim=dim * 2,
            num_heads=num_heads[5],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
            num_blocks=2,
        )

        self.final_expand = FinalPatchExpand_X4(
            dim=dim * 2,
            dim_scale=2,
            input_resolution=(
                input_h // 2,
                input_w // 2,
            ),
        )

        self.prediction_head = nn.Sequential(
            # Tanh for delta prediction
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Tanh(),
        )

        # Add noise processing
        self.noise_dim = noise_dim
        self.noise_processor = NoiseProcessor(self.noise_dim)
        self.n_members = n_members

    def _forward(self, x, noise_embedding):
        # Encoder
        x = self.encoder1(x, noise_embedding)
        x = self.encoder2(self.downsample1(x), noise_embedding)
        x = self.encoder3(self.downsample2(x), noise_embedding)

        # Bridge
        x = self.bridge(x)

        # Decoder
        x = self.decoder3(x, noise_embedding)
        x = self.decoder2(self.upsample2(x), noise_embedding)
        x = self.decoder1(self.upsample1(x), noise_embedding)

        x = self.final_expand(x)
        x = self.prediction_head(x)
        return x

    def forward(self, x):
        assert x.ndim == 4  # B, C, H, W
        B, C, H, W = x.shape

        deltas = []

        last_state = (
            torch.clone(x[:, -1, :, :]).detach().unsqueeze(1).unsqueeze(1)
        )  # B, 1, 1, H, W

        x = self.patch_embed(x)

        # generate predictions
        for _ in range(self.n_members):

            features = torch.clone(x).detach()
            # Generate and process noise
            noise = torch.randn(B, self.noise_dim, device=x.device)
            noise_embed = self.noise_processor(noise)

            # Get prediction
            tendency = self._forward(features, noise_embed)

            # tendency = self.prediction_head(features)
            deltas.append(tendency)

        # Stack all deltas
        deltas = torch.stack(deltas, dim=1)  # Shape: [batch, n_members, C, H, W]

        max_allowed_delta = 1.0 - last_state  # How much room to increase
        min_allowed_delta = 0.0 - last_state  # How much room to decrease
        deltas = torch.clamp(deltas, min_allowed_delta, max_allowed_delta)

        predictions = last_state + deltas  # Add current state to all deltas

        predictions = torch.clamp(predictions, 0, 1)

        assert predictions.shape == (
            B,
            self.n_members,
            1,
            H,
            W,
        ), "predictions shape invalid: {}".format(predictions.shape)
        return deltas, predictions


if __name__ == "__main__":
    model = cc2CRPS(
        dim=128,
        n_members=3,
        n_layers=4,
        input_resolution=(128, 128),
        noise_dim=128,
        window_size=4,
    )
    print(model)
    print(
        "Number of trainable parameters: {:,}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    x = torch.randn(1, 2, 128, 128)
    deltas, predictions = model(x)

    print("tendencies:", deltas.shape)
    print("predictions:", predictions.shape)
