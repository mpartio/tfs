import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pgu_ens.layers import (
    FeedForward,
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    EncoderBlock,
    DecoderBlock,
    NoiseProcessor,
    ConditionalLayerNorm,
    pad_tensors,
    pad_tensor,
    depad_tensor,
    get_padded_size,
)
from types import SimpleNamespace
from typing import Optional
from torch.utils.checkpoint import checkpoint


class cc2CRPS(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        config = SimpleNamespace(**config)

        self.model_class = "pgu_ens"
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_dim
        self.num_members = config.num_members
        self.noise_dim = config.noise_dim

        self.real_input_resolution = config.input_resolution

        input_resolution = get_padded_size(
            config.input_resolution[0], config.input_resolution[1], self.patch_size, 1
        )

        self.h_patches = input_resolution[0] // self.patch_size
        self.w_patches = input_resolution[1] // self.patch_size
        self.num_patches = self.h_patches * self.w_patches

        # Noise processor
        self.noise_processor = NoiseProcessor(config.noise_dim, self.embed_dim)

        # Patch embedding for converting images to tokens
        self.patch_embed = PatchEmbed(
            input_resolution=input_resolution,
            patch_size=self.patch_size,
            data_channels=len(config.prognostic_params),
            forcing_channels=len(config.forcing_params)
            + len(config.static_forcing_params),
            embed_dim=self.embed_dim,
        )

        # Spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Input layer norm and dropout
        self.norm_input = ConditionalLayerNorm(self.embed_dim, config.noise_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        # Transformer encoder blocks
        self.encoder1 = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    noise_dim=self.noise_dim,
                )
                for _ in range(config.encoder1_depth)
            ]
        )

        self.input_resolution_halved = (
            input_resolution[0] // self.patch_size,
            input_resolution[1] // self.patch_size,
        )
        self.downsample = PatchMerge(
            self.input_resolution_halved, self.embed_dim, time_dim=config.history_length
        )

        self.encoder2 = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    noise_dim=self.noise_dim,
                )
                for _ in range(config.encoder2_depth)
            ]
        )

        self.decoder1 = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    noise_dim=self.noise_dim,
                )
                for _ in range(config.decoder1_depth)
            ]
        )

        self.upsample = PatchExpand(
            self.embed_dim * 2, self.embed_dim * 2, scale_factor=2
        )

        self.decoder2 = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    noise_dim=self.noise_dim,
                )
                for _ in range(config.decoder2_depth)
            ]
        )

        # Final norm and output projection
        self.norm_final = ConditionalLayerNorm(self.embed_dim * 2, config.noise_dim)

        # Patch expansion for upsampling to original resolution

        self.patch_expand = PatchExpand(
            self.embed_dim * 2, self.embed_dim // 4, scale_factor=1
        )

        self.final_expand = nn.Sequential(
            nn.Linear(
                self.embed_dim // 4, self.patch_size**2 * len(config.prognostic_params)
            ),
            nn.Tanh(),
        )

        # Step identification embeddings
        self.step_id_embeddings = nn.Parameter(torch.randn(2, self.embed_dim * 2))
        nn.init.normal_(self.step_id_embeddings, std=0.02)

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.add_skip_connection = config.add_skip_connection

        if config.add_skip_connection:
            self.skip_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)
            self.skip_fusion = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)

        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.use_scheduled_sampling = config.use_scheduled_sampling

        self.refinement_head = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, data, forcing, noise_embedding):
        """Encode input sequence into latent representation"""
        x = self.patch_embed(data, forcing)  # [B, T, patches, embed_dim]
        B, T, P, D = x.shape

        # Add positional embedding to each patch
        for t in range(T):
            x[:, t] = x[:, t] + self.pos_embed

        # Apply input normalization and dropout
        x = x.reshape(B, T * P, D)
        x = self.norm_input(x, noise_embedding)
        x = self.dropout(x)

        # Pass through encoder blocks
        if self.use_gradient_checkpointing:
            for block in self.encoder1:
                x = checkpoint(block, x, noise_embedding, use_reentrant=False)
        else:
            for block in self.encoder1:
                x = block(x, noise_embedding)

        if self.add_skip_connection:
            skip = x.clone()  # skip connection, B, T*P, D
            skip = skip.reshape(B, T, -1, D)
        else:
            skip = None

        # Downsample
        x = self.downsample(x)

        # Pass through encoder blocks
        if self.use_gradient_checkpointing:
            for block in self.encoder2:
                x = checkpoint(block, x, noise_embedding, use_reentrant=False)
        else:
            for block in self.encoder2:
                x = block(x, noise_embedding)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, -1, D * 2)

        return x, skip

    def decode(
        self,
        encoded: torch.tensor,
        step: int,
        noise_embedding: torch.tensor,
        skip: Optional[torch.tensor],
    ):
        B, T, P, D = encoded.shape
        outputs = []

        # Initial input is the encoded sequence
        decoder_input = encoded

        # Keep track of the latest state
        latest_state = encoded[:, -1:, :, :]  # Just the last time step

        encoded_flat = encoded.reshape(B, -1, D)

        # Reshape for decoder
        decoder_in = decoder_input.reshape(B, -1, D)

        # Determine step id (0 for first step using ground truth, 1 for subsequent steps)
        step_id = 0
        if not self.use_scheduled_sampling and step > 0:
            step_id = 1

        # Get appropriate step embedding
        step_embedding = self.step_id_embeddings[step_id]  # [D]

        # Add step embedding to decoder input
        # For first token in each sequence (acts as a "step type" token)
        # We'll add it to the last P tokens which represent our current state
        decoder_in_with_id = decoder_in.clone()
        decoder_in_with_id[:, -P:] = decoder_in[:, -P:] + step_embedding.unsqueeze(
            0
        ).unsqueeze(1)

        # Process through decoder blocks
        x = decoder_in_with_id

        if self.use_gradient_checkpointing:
            for block in self.decoder1:
                x = checkpoint(
                    block, x, encoded_flat, noise_embedding, use_reentrant=False
                )
        else:
            for block in self.decoder1:
                x = block(x, encoded_flat, noise_embedding)

        # Get the delta prediction
        delta_pred1 = x[:, -P:].reshape(B, 1, P, D)

        new_H, new_W = self.input_resolution_halved
        new_H = new_H // 2  # 2 = num_times
        new_W = new_W // 2  # 2 = num_times

        upsampled_delta, P_new, D_new = self.upsample(
            delta_pred1.reshape(B, -1, D), H=new_H, W=new_W
        )

        P_new, D_new = upsampled_delta.shape[1], upsampled_delta.shape[2]

        upsampled_delta = upsampled_delta.reshape(B, 1, P_new, D_new)
        P_new, D_new = upsampled_delta.shape[2], upsampled_delta.shape[3]
        x2 = upsampled_delta.reshape(B, -1, D_new)

        if self.add_skip_connection:
            skip_token = skip[:, -1, :, :]  # shape: [B, num_tokens, embed_dim]
            assert skip_token.ndim == 3
            skip_proj = self.skip_proj(skip_token)  # [B, num_tokens, embed_dim*2]
            x2 = torch.cat([x2, skip_proj], dim=-1)  # [B, num_tokens, embed_dim*4]
            x2 = self.skip_fusion(x2)

        if self.use_gradient_checkpointing:
            for block in self.decoder2:
                x2 = checkpoint(
                    block, x2, encoded_flat, noise_embedding, use_reentrant=False
                )
        else:
            for block in self.decoder2:
                x2 = block(x2, encoded_flat, noise_embedding)

        delta_pred2 = x2.reshape(B, 1, P_new, D_new)

        return delta_pred2

    def project_to_image(self, x, noise_embedding):
        """Project latent representation back to image space"""
        B, T, P, D = x.shape

        # Apply final norm
        x = self.norm_final(x, noise_embedding)

        # Process each time step
        outputs = []
        for t in range(T):
            # Expand patches back to image resolution
            h_patches, w_patches = self.input_resolution_halved
            expanded, h_new, w_new = self.patch_expand(x[:, t], h_patches, w_patches)

            # Project to output channels and reshape to image format
            output = self.final_expand(
                expanded
            )  # [B, h_new*w_new, patch_size*patch_size*output_channels]

            output = output.reshape(
                B, h_new, w_new, self.patch_size, self.patch_size, 1
            )

            output = output.permute(0, 5, 1, 3, 2, 4).reshape(
                B, -1, h_new * self.patch_size, w_new * self.patch_size
            )

            outputs.append(output)

        # Stack time steps
        outputs = torch.stack(outputs, dim=1)  # [B, T, C, H, W]

        return outputs

    def forward(self, data, forcing, step):
        assert (
            data.ndim == 6
        ), "Input data tensor shape should be [B, M, T, C, H, W], is: {}".format(
            data.shape
        )
        assert (
            forcing.ndim == 6
        ), "Forcing tensor shape should be [B, M, T, C, H, W], is: {}".format(
            forcing.shape
        )

        assert forcing.shape[1] == data.shape[1]

        data, padding_info = pad_tensor(data, self.patch_size, 1)
        forcing, _ = pad_tensor(forcing, self.patch_size, 1)

        # Expand data and add member dimension
        B, M, T, C_data, H, W = data.shape
        data = data.reshape(B * M, T, C_data, H, W)

        C_forcing = forcing.shape[2]
        forcing = forcing.reshape(B * M, forcing.shape[2], forcing.shape[3], H, W)

        # Generate noise for all members at once
        noise = torch.randn(B * M, self.noise_dim, device=data[0].device)
        noise_embed = self.noise_processor(noise)

        encoded, skip = self.encode(data, forcing, noise_embed)

        decoded = self.decode(encoded, step, noise_embed, skip)

        output = self.project_to_image(decoded, noise_embed)

        output_ref = self.refinement_head(output.squeeze(2))
        output = output + output_ref.unsqueeze(2)

        output = depad_tensor(output, padding_info)  # B * M, T, C, H, W

        H, W = padding_info["original_size"]

        output = output.reshape(B, M, 1, C_data, H, W)

        assert list(output.shape[-2:]) == list(
            self.real_input_resolution
        ), "Output shape {} does not match real input resolution {}".format(
            output.shape[-2:], self.real_input_resolution
        )
        return output


if __name__ == "__main__":

    conf = config.get_config()

    conf.input_resolution = (268, 238)
    # Sample data of shape (batch_size, times, channels, height, width)
    sample_data = (torch.randn(1, 2, 1, 268, 238), torch.randn(1, 2, 1, 268, 238))
    sample_forcing = (torch.randn(1, 2, 9, 268, 238), torch.randn(1, 2, 9, 268, 238))

    # Create the model
    model = cc2Pangu(conf)

    print(model)
    print(
        "Number of trainable parameters: {:,}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    # Forward pass
    output = model(sample_data, sample_forcing, 1)

    print(f"Input shape: {sample_data[0].shape} and {sample_forcing[0].shape}")
    print(f"Output shape: {output.shape}")
