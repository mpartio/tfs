import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
from timm.models.layers import DropPath


def get_padded_size(H, W, patch_size, num_merges=1):
    # Calculate required factor for divisibility
    required_factor = patch_size * (2**num_merges)

    # Calculate target dimensions (must be divisible by required_factor)
    target_h = ((H + required_factor - 1) // required_factor) * required_factor
    target_w = ((W + required_factor - 1) // required_factor) * required_factor

    return target_h, target_w


def pad_tensor(tensor, patch_size, num_merges=1):
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


def pad_tensors(tensors, patch_size, num_merges=1):
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
    depadded_tensor = tensor[:, :, :, :target_h, :target_w]

    return depadded_tensor


class EncoderBlock(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate
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
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention"""

    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate
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
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path3 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, context):
        # Self-attention (without mask for now to avoid shape issues)
        x_norm1 = self.norm1(x)
        self_attn, _ = self.self_attn(x_norm1, x_norm1, x_norm1)

        x = x + self.drop_path1(self_attn)

        # Cross-attention to encoder outputs
        x_norm2 = self.norm2(x)

        cross_attn, _ = self.cross_attn(x_norm2, context, context)

        x = x + self.drop_path2(cross_attn)

        # Feedforward
        x = x + self.drop_path3(self.mlp(self.norm3(x)))

        return x


class PatchExpand(nn.Module):
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


class SkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Main feature transform
        self.transform = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x_encoder, x_decoder):
        # x_encoder: [B, L, C]
        # x_decoder: [B, L, C]
        # noise_embedding: [B, noise_dim]

        # Transform encoder features
        x_skip = self.transform(x_encoder)  # [B, L, C]

        # Combine
        x_combined = x_skip

        return x_decoder + x_combined


class PatchMerge(nn.Module):
    def __init__(self, input_resolution, dim, time_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        self.time_dim = time_dim

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        T = self.time_dim

        assert L == T * H * W, f"input feature has wrong size: {L} != {T * H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, T, H * W, C)

        # Process each time step separately
        output_time_steps = []

        for t in range(T):
            xt = x[:, t, :, :]  # B H*W C
            xt = xt.reshape(B, H, W, C)

            x0 = xt[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = xt[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = xt[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = xt[:, 1::2, 1::2, :]  # B H/2 W/2 C
            xt = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            xt = self.norm(xt)
            xt = self.reduction(xt)
            xt = xt.reshape(B, (H // 2) * (W // 2), 2 * C)  # [B, H/2*W/2, 2*C]
            output_time_steps.append(xt)

        # Recombine time steps
        x = torch.cat(
            [step.unsqueeze(1) for step in output_time_steps], dim=1
        )  # [B, T, H/2*W/2, 2*C]
        x = x.reshape(B, T * (H // 2) * (W // 2), 2 * C)  # [B, T*H/2*W/2, 2*C]
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size: {L} != {H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class DualStem(nn.Module):
    """
    Apply ConvStem separately to prognostic and forcing inputs.
    Outputs stemmed feature maps for both, still at full resolution.
    """

    def __init__(self, prog_ch, forcing_ch, stem_ch):
        super().__init__()
        self.stem_prognostic = nn.Sequential(
            nn.Conv2d(prog_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stem_forcing = nn.Sequential(
            nn.Conv2d(forcing_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, forcing):
        """
        x: [B, T, C_prog, H, W]
        forcing: [B, T_f, C_force, H, W]
        returns:
            x_stem: [B, T, stem_ch, H, W]
            f_stem: [B, T_f, stem_ch, H, W]
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x_stem = self.stem_prognostic(x).reshape(B, T, -1, H, W)

        Bf, Tf, Cf, H, W = forcing.shape
        f = forcing.reshape(Bf * Tf, Cf, H, W)
        f_stem = self.stem_forcing(f).reshape(Bf, Tf, -1, H, W)

        return x_stem, f_stem


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        stride,
        stem_ch,
        embed_dim,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.proj_data = nn.Conv2d(
            stem_ch, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.proj_forcing = nn.Conv2d(
            stem_ch, embed_dim, kernel_size=patch_size, stride=stride
        )

        H, W = input_resolution
        # number of tokens in each dim
        self.h_patches = (H - patch_size) // stride + 1
        self.w_patches = (W - patch_size) // stride + 1
        self.num_patches = self.h_patches * self.w_patches

        self.fuse = nn.Linear(2 * embed_dim, embed_dim)

    def _proj(self, x, is_data):  # x: [B, T, C, H, W] -> [B, T, P, D]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        proj = self.proj_data if is_data else self.proj_forcing
        x = proj(x)  # [B*T, D, h_t, w_t]
        x = x.flatten(2).transpose(1, 2)  # [B*T, P, D]
        x = x.reshape(B, T, -1, x.shape[-1])  # [B, T, P, D]
        return x

    def forward(self, x_stem, f_stem):
        x_tok = self._proj(x_stem, is_data=True)  # [B, T,   P, D]
        f_tok = self._proj(f_stem, is_data=False)  # [B, T_f, P, D]

        B, T, P, D = x_tok.shape
        Tf = f_tok.shape[1]

        if Tf >= T:
            f_hist = f_tok[:, :T]  # forcings aligned with history
            f_future = f_tok[:, T : T + 1]  # the "future" forcing (shape [B, 1, P, D])
        else:
            # should not ever get here
            repeat = (T + Tf - 1) // Tf
            f_hist = f_tok.repeat(1, repeat, 1, 1)[:, :T]
            f_future = None

        fused = torch.cat([x_tok, f_hist], dim=3)  # [B, T, P, 2D]
        fused = self.fuse(fused)  # [B, T, P, D]

        return fused, f_future


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

        self.h_patches = input_resolution[0] // self.patch_size[0]
        self.w_patches = input_resolution[1] // self.patch_size[1]
        self.num_patches = self.h_patches * self.w_patches

    def forward(self, data, forcing):
        assert data.ndim == 5, "x dims should be B, T, C, H, W, found: {}".format(
            data.shape
        )
        assert (
            forcing.ndim == 5
        ), "forcing dims should be B, T, C, H, W, found: {}".format(forcing.shape)
        B, T, C_data, H, W = data.shape
        _, T_forcing, C_forcing, _, _ = forcing.shape

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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwinEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path_rate,
        window_size,
        shift_size,
        H,
        W,
        T,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.H = H
        self.W = W
        self.T = T

        assert 0 <= self.shift_size < self.window_size, "shift must be < window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

    def _partition_windows(self, x4):
        """
        x4: [B*, Hpad, Wpad, C]
        returns windows: [Bwin, Ws*Ws, C], plus Hpad, Wpad to reverse later
        """
        B_, H, W, C = x4.shape
        ws = self.window_size
        assert (
            H % ws == 0 and W % ws == 0
        ), f"Input ({H},{W}) not divisible by window_size {ws}"
        nH, nW = H // ws, W // ws

        # [B, H, W, C] -> [B, nH, ws, nW, ws, C]
        xw = x4.view(B_, nH, ws, nW, ws, C)
        # [B, nH, nW, ws, ws, C]
        xw = xw.permute(0, 1, 3, 2, 4, 5).contiguous()
        # flatten windows
        xw = xw.view(B_ * nH * nW, ws * ws, C)
        return xw, nH, nW

    def _reverse_windows(self, xw, nH, nW):
        """
        xw: [Bwin, Ws*Ws, C] -> [B*, Hpad, Wpad, C]
        """
        ws = self.window_size
        Bwin, L, C = xw.shape
        B_ = Bwin // (nH * nW)
        xw = xw.view(B_, nH, nW, ws, ws, C)
        # [B, nH, ws, nW, ws, C] -> [B, H, W, C]
        x4 = xw.permute(0, 1, 3, 2, 4, 5).contiguous().view(B_, nH * ws, nW * ws, C)
        return x4

    def forward(self, x):
        """
        x: [B, T*P, C] where P = H*W tokens
        """
        B, N, C = x.shape
        H, W, T = self.H, self.W, self.T
        P = H * W

        assert N % P == 0, f"Expected N multiple of P=H*W={P}, got N={N}"
        assert (N // P) == T, f"N={N} corresponds to T={N//P}, expected T={T}"

        x_attn_in = x  # save residuals
        y = self.norm1(x)

        # [B, T*P, C] -> [B, T, P, C] -> [B*T, P, C] -> [B*T, H, W, C]
        y = y.view(B, T, P, C).reshape(B * T, P, C).view(B * T, H, W, C)

        # cyclic shift (on spatial dims)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # pad H/W to multiples of window_size if needed
        ws = self.window_size
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))  # pad W then H
            Hpad, Wpad = H + pad_h, W + pad_w
        else:
            Hpad, Wpad = H, W

        # partition windows -> [Bwin, ws*ws, C]
        yw, nH, nW = self._partition_windows(y)

        # local attention per window
        # attn expects [Bwin, L, C] and returns same (batch_first=True)
        attn_out, _ = self.attn(yw, yw, yw)

        # merge windows -> [B*T, Hpad, Wpad, C]
        y = self._reverse_windows(attn_out, nH, nW)

        # remove pad
        if (Hpad != H) or (Wpad != W):
            y = y[:, :H, :W, :]

        # reverse cyclic shift
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # [B*T, H, W, C] -> [B, T, P, C] -> [B, T*P, C]
        y = y.reshape(B, T, P, C).reshape(B, T * P, C)

        # residual + MLP
        x = x + x_attn_in + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
