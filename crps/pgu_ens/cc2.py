import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers import (
    FeedForward,
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    EncoderBlock,
    DecoderBlock,
    NoiseProcessor,
    ConditionalLayerNorm,
    pad_tensors,
    depad_tensor,
    get_padded_size,
)
import config


class cc2CRPS(nn.Module):
    def __init__(
        self,
        config,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    ):
        super().__init__()

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
            data_channels=config.num_data_channels,
            forcing_channels=config.num_forcing_channels,
            embed_dim=self.embed_dim,
        )

        # Spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Variable embedding (optional, for multi-variable forecasting)
        self.var_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Input layer norm and dropout
        self.norm_input = ConditionalLayerNorm(self.embed_dim, config.noise_dim)
        self.dropout = nn.Dropout(drop_rate)

        # Transformer encoder blocks
        self.encoder1 = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(config.encoder_depth)
            ]
        )

        self.input_resolution_halved = (
            input_resolution[0] // self.patch_size,
            input_resolution[1] // self.patch_size,
        )
        self.downsample = PatchMerge(
            self.input_resolution_halved, self.embed_dim, time_dim=2
        )

        self.encoder2 = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(config.encoder_depth)
            ]
        )

        self.decoder1 = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(config.decoder_depth)
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
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(config.decoder_depth)
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
                self.embed_dim // 4, self.patch_size**2 * config.num_data_channels
            ),
            nn.Tanh(),
        )

        # Step identification embeddings
        self.step_id_embeddings = nn.Parameter(torch.randn(2, self.embed_dim * 2))
        nn.init.normal_(self.step_id_embeddings, std=0.02)

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        x = self.patch_embed(data[0], forcing[0])  # [B, T, patches, embed_dim]
        B, T, P, D = x.shape

        # Add positional embedding to each patch
        for t in range(T):
            x[:, t] = x[:, t] + self.pos_embed

        # Apply input normalization and dropout
        x = x.reshape(B, T * P, D)
        x = self.norm_input(x, noise_embedding)
        x = self.dropout(x)

        # Pass through encoder blocks
        for block in self.encoder1:
            x = block(x, noise_embedding)

        # Downsample
        x = self.downsample(x)

        # Pass through encoder blocks
        for block in self.encoder2:
            x = block(x, noise_embedding)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, -1, D * 2)
        return x

    def decode(self, encoded, target_len, noise_embedding):
        B, T, P, D = encoded.shape
        outputs = []

        # Initial input is the encoded sequence
        decoder_input = encoded

        # Keep track of the latest state
        latest_state = encoded[:, -1:, :, :]  # Just the last time step

        encoded_flat = encoded.reshape(B, -1, D)

        # Decode one step at a time
        for t in range(target_len):
            # Reshape for decoder
            decoder_in = decoder_input.reshape(B, -1, D)

            # Determine step id (0 for first step using ground truth, 1 for subsequent steps)
            step_id = 0 if t == 0 else 1

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

            for block in self.decoder2:
                x2 = block(x2, encoded_flat, noise_embedding)

            delta_pred2 = x2.reshape(B, 1, P_new, D_new)

            # If latest_state doesn't match the upsampled dimensions, we need to upsample it too
            if latest_state.shape[2] != P_new:
                latest_state_upsampled, _, _ = self.upsample(
                    latest_state.reshape(B, -1, D),
                    H=new_H,
                    W=new_W,
                )
                latest_state_upsampled = latest_state_upsampled.reshape(
                    B, 1, P_new, D_new
                )
                latest_state = latest_state_upsampled

            # Add delta to latest state to get new state
            new_state = latest_state + delta_pred2

            # Add new state to outputs
            outputs.append(new_state)

            # Update latest state
            latest_state = new_state

            if t == 0:
                # First iteration after upsampling
                decoder_input = new_state
            else:
                decoder_input = torch.cat([decoder_input, new_state], dim=1)

        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)

        return outputs

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

    def forward(self, data, forcing, target_len):
        assert (
            data[0].ndim == 5
        ), "Input data tensor shape should be [B, T, C, H, W], is: {}".format(
            data[0].shape
        )

        data, padding_info = pad_tensors(data, self.patch_size, 1)
        forcing, _ = pad_tensors(forcing, self.patch_size, 1)

        # Expand data and add member dimension
        B, T, _, H, W = data[0].shape
        C_data = data[0].shape[2]
        C_forcing = forcing[0].shape[2]

        M = self.num_members

        x, y = data
        xf, yf = forcing

        x = x.unsqueeze(1).expand(B, M, T, C_data, H, W).reshape(B * M, T, C_data, H, W)
        y = y.unsqueeze(1).expand(B, M, T, C_data, H, W).reshape(B * M, T, C_data, H, W)
        xf = (
            xf.unsqueeze(1)
            .expand(B, M, T, C_forcing, H, W)
            .reshape(B * M, T, C_forcing, H, W)
        )
        yf = (
            yf.unsqueeze(1)
            .expand(B, M, T, C_forcing, H, W)
            .reshape(B * M, T, C_forcing, H, W)
        )

        # Generate noise for all members at once
        noise = torch.randn(B * M, self.noise_dim, device=data[0].device)
        noise_embed = self.noise_processor(noise)

        data = (x, y)
        forcing = (xf, yf)

        encoded = self.encode(data, forcing, noise_embed)

        decoded = self.decode(encoded, target_len, noise_embed)

        output = self.project_to_image(decoded, noise_embed)
        output = depad_tensor(output, padding_info)
        H, W = padding_info["original_size"]

        output = output.reshape(B, M, target_len, C_data, H, W)

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
