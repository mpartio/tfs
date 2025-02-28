import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from layers import FeedForward
import config


class PatchEmbed(nn.Module):
    """Convert image data into patches"""

    def __init__(
        self,
        img_size,
        patch_size,
        data_channels,
        forcing_channels,
        embed_dim,
    ):
        super().__init__()
        assert type(img_size) in (list, tuple)
        self.img_size = img_size
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (
            self.img_size[1] // self.patch_size[1]
        )

        # Separate projections for data and forcings
        self.data_proj = nn.Conv2d(
            data_channels,
            embed_dim // 2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.forcing_proj = nn.Conv2d(
            forcing_channels,
            embed_dim // 2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Optional: fusion layer to combine the embeddings
        self.fusion = nn.Linear(embed_dim, embed_dim)

    def forward(self, data, forcing):
        B, T, C_data, H, W = data.shape
        _, _, C_forcing, _, _ = forcing.shape

        # Process each time step
        embeddings = []
        for t in range(T):
            # Get embeddings for data and forcings
            data_emb = (
                self.data_proj(data[:, t]).flatten(2).transpose(1, 2)
            )  # [B, patches, embed_dim//2]
            forcing_emb = (
                self.forcing_proj(forcing[:, t]).flatten(2).transpose(1, 2)
            )  # [B, patches, embed_dim//2]

            # Concatenate along embedding dimension
            combined_emb = torch.cat(
                [data_emb, forcing_emb], dim=2
            )  # [B, patches, embed_dim]

            # Optional: apply fusion layer
            combined_emb = self.fusion(combined_emb)

            embeddings.append(combined_emb)

        # Stack time steps
        embeddings = torch.stack(embeddings, dim=1)  # [B, T, patches, embed_dim]

        return embeddings


class EncoderBlock(nn.Module):
    """Standard Transformer encoder block"""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention"""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

    def forward(self, x, context, self_attention_mask=None):
        # Self-attention (without mask for now to avoid shape issues)
        x_norm1 = self.norm1(x)
        self_attn, _ = self.self_attn(x_norm1, x_norm1, x_norm1)
        x = x + self_attn

        # Cross-attention to encoder outputs
        x_norm2 = self.norm2(x)
        cross_attn, _ = self.cross_attn(x_norm2, context, context)
        x = x + cross_attn

        # Feedforward
        x = x + self.mlp(self.norm3(x))

        return x


class PatchExpand(nn.Module):
    """Expand patches to higher resolution"""

    def __init__(self, input_dim, output_dim, scale_factor=2):
        super().__init__()
        self.dim = input_dim
        self.expand = nn.Linear(input_dim, output_dim * scale_factor * scale_factor)
        self.scale_factor = scale_factor

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = self.expand(x)  # B, H*W, C*scale_factor^2

        # Reshape to spatial format for upsampling
        x = x.view(B, H, W, -1)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.scale_factor,
            p2=self.scale_factor,
        )
        x = rearrange(x, "b h w c -> b (h w) c")

        return x, H * self.scale_factor, W * self.scale_factor


class cc2Pangu(nn.Module):
    """Pangu-Weather inspired model for autoregressive weather forecasting"""

    def __init__(
        self,
        config,
        output_channels=1,
        encoder_depth=8,
        decoder_depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    ):
        super().__init__()

        self.patch_size = 4
        self.embed_dim = config.hidden_dim
        self.h_patches = self.w_patches = config.input_resolution[0] // self.patch_size
        self.num_patches = self.h_patches * self.w_patches

        # Patch embedding for converting images to tokens
        self.patch_embed = PatchEmbed(
            img_size=config.input_resolution,
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
        self.norm_input = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(encoder_depth)
            ]
        )

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(decoder_depth)
            ]
        )

        # Final norm and output projection
        self.norm_final = nn.LayerNorm(self.embed_dim)

        # Patch expansion for upsampling to original resolution
        self.patch_expand = PatchExpand(
            self.embed_dim, self.embed_dim // 4, scale_factor=1
        )

        self.final_expand = nn.Sequential(
            nn.Linear(self.embed_dim // 4, self.patch_size**2 * output_channels),
            nn.Tanh(),
        )

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

    def _prepare_decoder_input(self, B, seq_len):
        """Create an initial decoder input (start token or zeros)"""
        decoder_input = torch.zeros(
            B, seq_len, self.num_patches, self.embed_dim, device=self.pos_embed.device
        )
        # Add positional embedding to each time step
        decoder_input = decoder_input + self.pos_embed.unsqueeze(1)
        return decoder_input

    def _create_causal_mask(self, seq_len):
        """Create a causal mask for autoregressive generation"""
        # Create mask that prevents attending to future time steps
        mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=self.pos_embed.device
        )
        mask = torch.triu(
            mask, diagonal=1
        )  # Upper triangular part (excluding diagonal)
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def encode(self, data, forcing):
        """Encode input sequence into latent representation"""
        x = self.patch_embed(data[0], forcing[0])  # [B, T, patches, embed_dim]

        B, T, P, D = x.shape

        # Add positional embedding to each patch
        for t in range(T):
            x[:, t] = x[:, t] + self.pos_embed

        # Apply input normalization and dropout
        x = x.reshape(B, T * P, D)
        x = self.norm_input(x)
        x = self.dropout(x)

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, P, D)

        return x

    def decode_autoregressive(self, encoded, target_len):
        B, T, P, D = encoded.shape
        outputs = []

        # Initial input is the encoded sequence
        decoder_input = encoded

        # Keep track of the latest state
        latest_state = encoded[:, -1:, :, :]  # Just the last time step

        # Decode one step at a time
        for t in range(target_len):
            # Reshape for decoder
            decoder_in = decoder_input.reshape(B, -1, D)
            encoded_flat = encoded.reshape(B, -1, D)

            # Process through decoder blocks
            x = decoder_in
            for block in self.decoder_blocks:
                x = block(x, encoded_flat)

            # Get the delta prediction
            delta_pred = x[:, -P:].reshape(B, 1, P, D)

            # Add delta to latest state to get new state
            new_state = latest_state + delta_pred

            # Add new state to outputs
            outputs.append(new_state)

            # Update latest state
            latest_state = new_state

            # Update decoder input for next step
            decoder_input = torch.cat([decoder_input, new_state], dim=1)

        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def project_to_image(self, x):
        """Project latent representation back to image space"""
        B, T, P, D = x.shape

        # Apply final norm
        x = self.norm_final(x)

        # Process each time step
        outputs = []
        for t in range(T):
            # Expand patches back to image resolution
            h_patches = w_patches = int(math.sqrt(P))
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

        # Encode input sequence
        encoded = self.encode(
            data,
            forcing,
        )

        decoded = self.decode_autoregressive(encoded, target_len)

        # Project back to image space
        output = self.project_to_image(decoded)

        return output


if __name__ == "__main__":

    # Sample data of shape (batch_size, times, channels, height, width)
    sample_data = (torch.randn(1, 2, 1, 128, 128), torch.randn(1, 2, 1, 128, 128))
    sample_forcing = (torch.randn(1, 2, 9, 128, 128), torch.randn(1, 2, 9, 128, 128))

    # Create the model
    model = cc2Pangu(config.get_config())

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
