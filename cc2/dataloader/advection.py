"""Semi-Lagrangian advection of a 2D scalar field by a wind field.

Used to build the `advected-persistence` forcing channel: at each forecast lead
`k`, we backward-advect the current cloud state by k * (wind * dt) pixels and
hand the result to the model as an additional forcing channel. This gives the
model a "free persistence baseline" at every lead so it can specialise on
cloud-process residuals (genesis/lysis/lifecycle) instead of re-learning transport
implicitly from u/v.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def advect_semi_lagrangian(
    field: torch.Tensor,
    u_pix: torch.Tensor,
    v_pix: torch.Tensor,
    n_steps: int = 1,
) -> torch.Tensor:
    """Backward semi-Lagrangian advection of `field` by (u_pix, v_pix) for
    `n_steps` forward-in-time steps. Wind held constant across the trajectory
    (cheap; sufficient for the first iteration — improved transport requires
    sub-stepping which we can add if the channel proves useful).

    Args:
        field   : [B, 1, H, W] — the scalar field to advect (e.g. cloud cover).
        u_pix   : [B, H, W]    — east-west pixel displacement per time step
                                 (positive = +column).
        v_pix   : [B, H, W]    — north-south pixel displacement per time step
                                 (positive = +row; caller handles
                                 geographic-vs-image axis flip).
        n_steps : forward-in-time advection steps; field returned is at lead
                  `n_steps`.

    Returns:
        [B, 1, H, W] — advected field at lead = n_steps.
    """
    B, _, H, W = field.shape
    device = field.device
    dtype = field.dtype

    iy, ix = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    # Backward trajectory: pixel (i, j) at lead n_steps was at
    # (i - v*n_steps, j - u*n_steps) at t=0.
    src_x = ix.unsqueeze(0) - float(n_steps) * u_pix  # [B, H, W]
    src_y = iy.unsqueeze(0) - float(n_steps) * v_pix  # [B, H, W]

    norm_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(H - 1, 1) - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, H, W, 2]

    return F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def winds_to_pixel_per_step(
    u_ms: torch.Tensor,
    v_ms: torch.Tensor,
    dt_seconds: float = 3 * 3600.0,
    grid_spacing_m: float = 5000.0,
    flip_v_for_image_origin: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert m/s wind components to pixel/timestep displacements.

    The MEPS+NWCSAF zarr uses a top-left grid origin (image convention), so a
    positive geographic v (northward) corresponds to DECREASING row index.
    Caller passes `flip_v_for_image_origin=True` to apply this flip; only set
    it False if the input data is already in geographic (bottom-left) order.
    """
    scale = dt_seconds / grid_spacing_m
    u_pix = u_ms * scale
    v_pix = v_ms * scale
    if flip_v_for_image_origin:
        v_pix = -v_pix
    return u_pix, v_pix
