import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pgu.layers import (
    FeedForward,
    OverlapPatchEmbed,
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    EncoderBlock,
    DecoderBlock,
    SwinEncoderBlock,
    DualStem,
    get_padded_size,
    pad_tensors,
    pad_tensor,
    depad_tensor,
)
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint


class cc2CRPS(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = SimpleNamespace(**config)

        self.model_class = "pgu"
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_dim

        self.real_input_resolution = config.input_resolution

        input_resolution = get_padded_size(
            config.input_resolution[0], config.input_resolution[1], self.patch_size, 1
        )

        self.overlap_patch_embed = config.overlap_patch_embed

        self.num_prognostic = len(config.prognostic_params)
        self.num_forcings = len(config.forcing_params) + len(
            config.static_forcing_params
        )

        self.stem_ch = 32

        if config.overlap_patch_embed:
            self.patch_embed = OverlapPatchEmbed(
                input_resolution=input_resolution,
                patch_size=self.patch_size,
                stride=self.patch_size // 2,
                stem_ch=self.stem_ch,
                embed_dim=self.embed_dim,
            )

            in_ch = self.num_prognostic + self.num_forcings
            self.stem = DualStem(
                self.num_prognostic, self.num_forcings, stem_ch=self.stem_ch
            )

        else:
            self.patch_embed = PatchEmbed(
                input_resolution=input_resolution,
                patch_size=self.patch_size,
                data_channels=self.num_prognostic,
                forcing_channels=self.num_forcings,
                embed_dim=self.embed_dim,
            )

        self.h_patches = self.patch_embed.h_patches
        self.w_patches = self.patch_embed.w_patches
        self.num_patches = self.patch_embed.num_patches

        # Spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Input layer norm and dropout
        self.norm_input = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        self.use_swin_encoder = config.use_swin_encoder

        # Transformer encoder blocks
        if self.use_swin_encoder:
            self.encoder1 = nn.ModuleList(
                [
                    SwinEncoderBlock(
                        dim=self.embed_dim,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=True,
                        drop=config.drop_rate,
                        attn_drop=config.attn_drop_rate,
                        drop_path_rate=config.drop_path_rate,
                        window_size=8,
                        shift_size=0 if i % 2 == 0 else 4,
                        H=self.h_patches,
                        W=self.w_patches,
                        T=config.history_length,
                    )
                    for i in range(config.encoder1_depth)
                ]
            )

        else:
            self.encoder1 = nn.ModuleList(
                [
                    EncoderBlock(
                        dim=self.embed_dim,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=True,
                        drop=config.drop_rate,
                        attn_drop=config.attn_drop_rate,
                        drop_path_rate=config.drop_path_rate,
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

        if self.use_swin_encoder:
            self.encoder2 = nn.ModuleList(
                [
                    SwinEncoderBlock(
                        dim=self.embed_dim * 2,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=True,
                        drop=config.drop_rate,
                        attn_drop=config.attn_drop_rate,
                        drop_path_rate=config.drop_path_rate,
                        window_size=8,
                        shift_size=0 if i % 2 == 0 else 4,
                        H=self.h_patches // 2,
                        W=self.w_patches // 2,
                        T=config.history_length,
                    )
                    for i in range(config.encoder2_depth)
                ]
            )
        else:
            self.encoder2 = nn.ModuleList(
                [
                    EncoderBlock(
                        dim=self.embed_dim * 2,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        qkv_bias=True,
                        drop=config.drop_rate,
                        attn_drop=config.attn_drop_rate,
                        drop_path_rate=config.drop_path_rate,
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
                    drop_path_rate=config.drop_path_rate,
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
                    drop_path_rate=config.drop_path_rate,
                )
                for _ in range(config.decoder2_depth)
            ]
        )

        # Final norm and output projection
        self.norm_final = nn.LayerNorm(self.embed_dim * 2)

        # Patch expansion for upsampling to original resolution

        self.patch_expand = PatchExpand(
            self.embed_dim * 2, self.embed_dim // 4, scale_factor=1
        )

        self.use_ste = config.use_ste

        if self.use_ste:
            self.final_expand = nn.Linear(
                self.embed_dim // 4, self.patch_size**2 * len(config.prognostic_params)
            )
        else:
            self.final_expand = nn.Sequential(
                nn.Linear(
                    self.embed_dim // 4,
                    self.patch_size**2 * len(config.prognostic_params),
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

        self.add_refinement_head = config.add_refinement_head

        if self.add_refinement_head:
            self.refinement_head = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, 3, padding=1),
            )
        else:
            self.refinement_head = nn.Identity()

        self.autoregressive_mode = config.autoregressive_mode

        if self.autoregressive_mode is False:
            self.max_step = 12
            self.step_embedding_direct = nn.Embedding(
                self.max_step + 1, self.embed_dim * 2
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_embedding(self, x, forcing):
        if self.overlap_patch_embed:
            x_stem, f_stem = self.stem(x, forcing)
            x_tokens, f_future = self.patch_embed(x_stem, f_stem)

        else:
            x_tokens, f_future = self.patch_embed(
                x, forcing
            )  # [B, T, patches, embed_dim]

        B, T, P, D = x_tokens.shape

        # Add positional embedding to each patch
        for t in range(T):
            x_tokens[:, t] = x_tokens[:, t] + self.pos_embed

        return x_tokens, f_future

    def encode(self, x):
        """Encode input sequence into latent representation"""
        B, T, P, D = x.shape

        # Apply input normalization and dropout
        x = x.reshape(B, T * P, D)
        x = self.norm_input(x)
        x = self.dropout(x)

        # Pass through encoder blocks
        if self.use_gradient_checkpointing:
            for block in self.encoder1:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.encoder1:
                x = block(x)

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
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.encoder2:
                x = block(x)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, -1, D * 2)

        return x, skip

    def decode(self, encoded, step, skip, f_future=None):
        B, T, P, D = encoded.shape
        outputs = []

        # Initial input is the encoded sequence
        decoder_input = encoded

        # Keep track of the latest state
        latest_state = encoded[:, -1:, :, :]  # Just the last time step

        encoded_flat = encoded.reshape(B, -1, D)
        decoder_in = decoder_input.reshape(B, -1, D)

        # Determine step id (0 for first step using ground truth, 1 for subsequent steps)
        if self.autoregressive_mode:
            step_id = 0
            if not self.use_scheduled_sampling and step > 0:
                step_id = 1
            # Get appropriate step embedding
            step_embedding = self.step_id_embeddings[step_id]  # [D]
        else:
            step_id = step + 1
            if step_id > self.max_step:
                step_id = self.max_step
            step_tensor = torch.tensor([step_id], device=encoded.device)
            step_embedding = self.step_embedding_direct(step_tensor).squeeze(0)

        # Add step embedding to decoder input
        # For first token in each sequence (acts as a "step type" token)
        # We'll add it to the last P tokens which represent our current state
        decoder_in_with_id = decoder_in.clone()
        decoder_in_with_id[:, -P:] = decoder_in[:, -P:] + step_embedding.unsqueeze(
            0
        ).unsqueeze(1)

        # Process through decoder blocks
        x = decoder_in_with_id

        # Process through decoder blocks
        if self.use_gradient_checkpointing:
            for block in self.decoder1:
                x = checkpoint(block, x, encoded_flat, use_reentrant=False)
        else:
            for block in self.decoder1:
                x = block(x, encoded_flat)

        # Get the delta prediction
        delta_pred1 = x[:, -P:].reshape(B, 1, P, D)

        new_H, new_W = self.input_resolution_halved
        new_H = new_H // 2  # 2 = num_times
        new_W = new_W // 2  # 2 = num_times

        with torch.amp.autocast("cuda", enabled=False):
            upsampled_delta, P_new, D_new = self.upsample(
                delta_pred1.reshape(B, -1, D).float(), H=new_H, W=new_W
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
                x2 = checkpoint(block, x2, encoded_flat, use_reentrant=False)
        else:
            for block in self.decoder2:
                x2 = block(x2, encoded_flat)

        delta_pred2 = x2.reshape(B, 1, P_new, D_new)

        return delta_pred2

    def project_to_image(self, x):
        """Project latent representation back to image space"""
        B, T, P, D = x.shape
        assert T == 1
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

    def forward(self, data, forcing, step):
        assert (
            data.ndim == 5
        ), "Input data tensor shape should be [B, T, C, H, W], is: {}".format(
            data.shape
        )

        data, padding_info = pad_tensor(data, self.patch_size, 1)
        forcing, _ = pad_tensor(forcing, self.patch_size, 1)

        x, f_future = self.patch_embedding(data, forcing)
        x, skip = self.encode(x)
        x = self.decode(x, step, skip, f_future)

        output = self.project_to_image(x)
        if self.add_refinement_head:
            output_ref = self.refinement_head(output.squeeze(2))
            output = output + output_ref.unsqueeze(2)

        output = depad_tensor(output, padding_info)

        assert list(output.shape[-2:]) == list(
            self.real_input_resolution
        ), f"Output shape {output.shape[-2:]} does not match real input resolution {self.real_input_resolution}"
        return output
