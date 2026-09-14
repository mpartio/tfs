"""Architecture-level acceptance tests for the swinu backbone fixes.

These are the standing P0 gates from the 2026-09-14 architecture review:

  1. frame-0 sensitivity   — the first history frame must influence the output
  2. frame-order sensitivity — swapping the two history frames must change the output
  3. residual identity     — encoder block with zeroed attn/mlp scales must be identity
                             (guards against the doubled-residual bug, 2x + attn)
  4. shift-window leakage  — a shifted block must not leak information between
                             opposite image edges through torch.roll wraparound

Run standalone (no pytest needed):

    python3 cc2/tests/test_swinu_architecture.py

Exit code 0 = all pass. Each test prints PASS/FAIL with the measured numbers.
Run them again on every trained checkpoint (frame-0 sensitivity can be trained
back into a near-dead state even when the architecture allows it).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from swinu.cc2 import cc2model  # noqa: E402
from swinu.layers import SwinEncoderBlock  # noqa: E402

SMALL_CONFIG = dict(
    patch_size=4,
    hidden_dim=32,
    num_heads=4,
    mlp_ratio=2.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    window_size=4,
    window_size_deep=4,
    encoder1_depth=2,
    encoder2_depth=2,
    decoder1_depth=2,
    decoder2_depth=2,
    input_resolution=[32, 32],
    prognostic_params=["tcc"],
    forcing_params=["f1", "f2"],
    static_forcing_params=["s1"],
    history_length=2,
    use_gradient_checkpointing=False,
    use_scheduled_sampling=False,
    preprocessor=None,
)


def _build_model(seed=0):
    torch.manual_seed(seed)
    return cc2model(dict(SMALL_CONFIG)).eval()


def _forward(model, data, forcing):
    with torch.no_grad():
        return model(data, forcing, 0)


def test_frame0_sensitivity():
    """Perturbing history frame 0 must change the output.

    Two modes:
    - init (default): threshold 1e-5 relative to the frame-1 delta — a wiring
      check. At init the DWConvResidual3D LayerScale (ls_init=1e-2/1e-3)
      deliberately keeps block outputs near zero, so sensitivity is small but
      must be far above fp noise (~1e-7 relative).
    - trained (CC2_ARCH_TEST_TRAINED=1): threshold 1% — the usage gate from the
      verification protocol. A structurally live path can be trained into a
      near-dead state; run this against every trained checkpoint.
    """
    trained = os.environ.get("CC2_ARCH_TEST_TRAINED", "0") == "1"
    rel_threshold = 0.01 if trained else 1e-5
    model = _build_model()
    torch.manual_seed(1)
    data = torch.rand(1, 2, 1, 32, 32)
    forcing = torch.rand(1, 3, 3, 32, 32)

    base = _forward(model, data, forcing)

    d0, f0 = data.clone(), forcing.clone()
    d0[:, 0] = torch.rand_like(d0[:, 0]) * 100
    f0[:, 0] = torch.rand_like(f0[:, 0]) * 100
    delta_frame0 = (_forward(model, d0, f0) - base).abs().max().item()

    d1 = data.clone()
    d1[:, 1] = torch.rand_like(d1[:, 1]) * 100
    delta_frame1 = (_forward(model, d1, forcing) - base).abs().max().item()

    ok = delta_frame1 > 0 and delta_frame0 >= rel_threshold * delta_frame1
    print(
        f"[{'PASS' if ok else 'FAIL'}] frame0_sensitivity "
        f"({'trained' if trained else 'init'} mode): "
        f"|d(frame0)|={delta_frame0:.3e} vs |d(frame1)|={delta_frame1:.3e} "
        f"(need >= {rel_threshold:.0e} of frame1)"
    )
    return ok


def test_frame_order_sensitivity():
    """Swapping the two history frames (same content, reversed order) must
    change the output — otherwise the model pools frames without seeing
    motion direction."""
    model = _build_model()
    torch.manual_seed(2)
    data = torch.rand(1, 2, 1, 32, 32)
    forcing = torch.rand(1, 3, 3, 32, 32)

    base = _forward(model, data, forcing)

    d_swap = data.flip(dims=[1])
    f_swap = forcing.clone()
    f_swap[:, :2] = forcing[:, :2].flip(dims=[1])
    delta = (_forward(model, d_swap, f_swap) - base).abs().max().item()

    scale = base.abs().max().item()
    ok = delta > 1e-4 * max(scale, 1.0)
    print(
        f"[{'PASS' if ok else 'FAIL'}] frame_order_sensitivity: "
        f"|d(swap)|={delta:.3e} (output scale {scale:.3e})"
    )
    return ok


def test_residual_identity():
    """With gamma_attn = gamma_mlp = 0 and no drop-path, an encoder block must
    be exactly the identity. The doubled-residual bug makes it 2x instead."""
    torch.manual_seed(3)
    block = SwinEncoderBlock(
        dim=32, num_heads=4, mlp_ratio=2.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
        window_size=4, shift_size=0, H=8, W=8, T=1,
    ).eval()
    with torch.no_grad():
        block.gamma_attn.zero_()
        block.gamma_mlp.zero_()
        x = torch.randn(2, 64, 32)
        out = block(x)

    max_diff = (out - x).abs().max().item()
    ok = max_diff == 0.0
    print(
        f"[{'PASS' if ok else 'FAIL'}] residual_identity: "
        f"max|out - in|={max_diff:.3e} (need exactly 0; doubled residual gives |x|-scale)"
    )
    return ok


def test_shift_no_edge_leakage():
    """In a shifted block, an impulse at the top-left corner must not influence
    output at the bottom rows: those only share a window with the top rows
    through torch.roll wraparound, which the shift attention mask must block."""
    torch.manual_seed(4)
    H = W = 8
    block = SwinEncoderBlock(
        dim=32, num_heads=4, mlp_ratio=2.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
        window_size=4, shift_size=2, H=H, W=W, T=1,
    ).eval()

    x = torch.zeros(1, H * W, 32)
    x_imp = x.clone()
    # Impulse at token (row 0, col 0). Must vary across channels: LayerNorm
    # annihilates a constant vector, which would hide the impulse from attention.
    x_imp[0, 0, :] = torch.randn(32) * 10.0

    with torch.no_grad():
        diff = block(x_imp) - block(x)

    diff = diff.view(H, W, 32).abs()
    # rows >= H/2 are >= 4 rows away from the impulse: unreachable through a
    # 4-wide window unless the roll wraps them into the impulse's window.
    far_leak = diff[H // 2:, :, :].max().item()
    near_response = diff[: H // 2, :, :].max().item()

    ok = near_response > 0 and far_leak <= 1e-6 * max(near_response, 1.0)
    print(
        f"[{'PASS' if ok else 'FAIL'}] shift_no_edge_leakage: "
        f"far-edge leak={far_leak:.3e} vs near response={near_response:.3e}"
    )
    return ok


def main():
    results = [
        test_frame0_sensitivity(),
        test_frame_order_sensitivity(),
        test_residual_identity(),
        test_shift_no_edge_leakage(),
    ]
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} tests passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
