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

        # self.mha = nn.MultiheadAttention(bridge_dim, num_heads=8)

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


#    def forward(self, x):
#        # x shape: [B, N_patches, C]
#        B, N, C = x.shape
#        # Project features
#
#        x = self.input_proj(x)  # [B, N_patches, bridge_dim]
#
#        x = F.layer_norm(x, (x.shape[-1],))
#
#        # Prepare for attention (already in right format)
#        x = x.transpose(0, 1)  # [N_patches, B, bridge_dim]
#
#        # Apply attention
#        x, _ = self.mha(x, x, x)
#
#        if torch.isnan(x).sum() > 1:
#            print("Bridge : NaNs after mha")
#            print(
#                "After projection stats:",
#                "min:",
#                x.min().item(),
#                "max:",
#                x.max().item(),
#                "mean:",
#                x.mean().item(),
#                "std:",
#                x.std().item(),
#            )
#            sys.exit(1)
#        # Back to batch-first
#        x = x.transpose(0, 1)  # [B, N_patches, bridge_dim]
#
#        return x


class cc2CRPS(nn.Module):
    def __init__(
        self,
        dim,
        n_members=3,
        n_layers=4,
        input_resolution=(128, 128),
        noise_dim=128,
        window_size=4,
        num_heads=[8, 8, 8, 8],
    ):
        super(cc2CRPS, self).__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=2, dim=dim, patch_size=2, stride=2
        )

        # Encoder
        self.encoder1 = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads[0],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_resolution[0] // 2,
                input_resolution[1] // 2,
            ),
        )

        # Downsample layers between blocks
        self.downsample1 = PatchMerging(
            dim=dim,
            input_resolution=(
                input_resolution[0] // 2,
                input_resolution[1] // 2,
            ),
        )

        self.encoder2 = SwinTransformerBlock(
            dim=dim * 2,
            num_heads=num_heads[1],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_resolution[0] // 4,
                input_resolution[1] // 4,
            ),
        )

        # Attention Bridge (like AIFS-CRPS)
        self.bridge = MultiHeadAttentionBridge(
            in_dim=dim * 2, bridge_dim=dim * 4, n_layers=n_layers
        )

        # Decoder (mirroring encoder)
        self.decoder2 = SwinTransformerBlock(
            dim=dim * 4,
            num_heads=num_heads[2],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_resolution[0] // 4,
                input_resolution[1] // 4,
            ),
        )

        # Upsample layers
        self.upsample1 = PatchExpand(
            dim=dim * 4,
            input_resolution=(
                input_resolution[0] // 4,
                input_resolution[1] // 4,
            ),
        )

        self.decoder1 = SwinTransformerBlock(
            dim=dim * 2,
            num_heads=num_heads[3],
            window_size=window_size,
            noise_dim=noise_dim,
            input_resolution=(
                input_resolution[0] // 2,
                input_resolution[1] // 2,
            ),
        )

        self.final_expand = FinalPatchExpand_X4(
            dim=dim * 2,
            dim_scale=2,
            input_resolution=(
                input_resolution[0] // 2,
                input_resolution[1] // 2,
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
        x = self.encoder1(x, noise_embedding)
        x = self.downsample1(x)
        x = self.encoder2(x, noise_embedding)
        x = self.bridge(x)
        x = self.decoder2(x, noise_embedding)
        x = self.upsample1(x)
        x = self.decoder1(x, noise_embedding)
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
