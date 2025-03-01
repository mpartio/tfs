import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from layers import FeedForward, PatchMerge
import config


def get_padded_size(H, W, patch_size=4, num_merges=1):
    # Calculate required factor for divisibility
    required_factor = patch_size * (2**num_merges)

    # Calculate target dimensions (must be divisible by required_factor)
    target_h = ((H + required_factor - 1) // required_factor) * required_factor
    target_w = ((W + required_factor - 1) // required_factor) * required_factor

    return target_h, target_w


def pad_tensor(tensor, patch_size=4, num_merges=1):
    H, W = tensor.shape[-2:]

    target_h, target_w = get_padded_size(H, W, patch_size, num_merges)

    # Calculate padding needed
    pad_h = target_h - H
    pad_w = target_w - W

    # Create padding configuration (left, right, top, bottom)
    pad_left = 0
    pad_right = pad_w
    pad_top = 0
    pad_bottom = pad_h

    # Apply padding
    if pad_h > 0 or pad_w > 0:
        padded_tensor = F.pad(
            tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )
    else:
        padded_tensor = tensor

    # Store padding info for later de-padding
    padding_info = {
        "original_size": (H, W),
        "padded_size": (target_h, target_w),
        "pad_h": pad_h,
        "pad_w": pad_w,
    }

    return padded_tensor, padding_info


def pad_tensors(tensors, patch_size=4, num_merges=1):
    padded_list = []
    for t in tensors:
        padded, pad_info = pad_tensor(t, patch_size, num_merges)
        padded_list.append(padded)

    return padded_list, pad_info


def depad_tensor(tensor, padding_info):
    if padding_info["pad_h"] == 0 and padding_info["pad_w"] == 0:
        return tensor

    H, W = tensor.shape[-2:]
    original_h, original_w = padding_info["original_size"]

    # Handle case where tensor has been processed and dimensions changed
    # Scale the original dimensions to match current tensor's scale
    scale_h = H / padding_info["padded_size"][0]
    scale_w = W / padding_info["padded_size"][1]

    target_h = int(original_h * scale_h)
    target_w = int(original_w * scale_w)

    # Extract the unpadded region
    depadded_tensor = tensor[:, :, :target_h, :target_w]

    return depadded_tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        data_channels,
        forcing_channels,
        embed_dim,
    ):
        super().__init__()
        assert type(input_resolution) in (list, tuple)
        self.input_resolution = input_resolution
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.num_patches = (self.input_resolution[0] // self.patch_size[0]) * (
            self.input_resolution[1] // self.patch_size[1]
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

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop):
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

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop):
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

    def forward(self, x, context):
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
        self.expand = nn.Linear(input_dim, output_dim * scale_factor**2)
        self.scale_factor = scale_factor

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size: {} vs {}".format(L, H * W)

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
        encoder_depth=2,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    ):
        super().__init__()

        self.patch_size = 4
        self.embed_dim = config.hidden_dim

        input_resolution = get_padded_size(
            config.input_resolution[0], config.input_resolution[1], self.patch_size, 1
        )

        if config.input_resolution != input_resolution:
            print(
                f"Input resolution changed from {config.input_resolution} to {input_resolution}"
            )
        self.h_patches = input_resolution[0] // self.patch_size
        self.w_patches = input_resolution[1] // self.patch_size
        self.num_patches = self.h_patches * self.w_patches

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
        self.norm_input = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        # Transformer encoder blocks
        self.encoder1 = nn.ModuleList(
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
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(encoder_depth)
            ]
        )

        self.decoder1 = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.upsample = PatchExpand(
            self.embed_dim * 2, self.embed_dim * 2, scale_factor=2
        )

        self.decoder2 = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.embed_dim * 2,
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
        self.norm_final = nn.LayerNorm(self.embed_dim * 2)

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
        for block in self.encoder1:
            x = block(x)

        # Downsample
        x = self.downsample(x)

        # Pass through encoder blocks
        for block in self.encoder2:
            x = block(x)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, -1, D * 2)
        return x

    def decode(self, encoded, target_len):
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

            # Process through decoder blocks
            x = decoder_in
            for block in self.decoder1:
                x = block(x, encoded_flat)

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
                x2 = block(x2, encoded_flat)

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

    def project_to_image(self, x):
        """Project latent representation back to image space"""
        B, T, P, D = x.shape

        # Apply final norm
        x = self.norm_final(x)

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

        data, padding_info = pad_tensors(data, 4, 1)
        forcing, _ = pad_tensors(forcing, 4, 1)

        print(f"Data shape: {data[0].shape}")
        encoded = self.encode(
            data,
            forcing,
        )

        decoded = self.decode(encoded, target_len)

        output = self.project_to_image(decoded)

        output = depad_tensor(output, padding_info)

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
