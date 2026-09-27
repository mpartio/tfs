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

IMPORTANT: with CC2_ARCH_TEST_TRAINED=1 alone, tests 1-2 still run on a freshly
INITIALIZED small synthetic model (SMALL_CONFIG) -- that flag only relaxes the
pass threshold, it does not load any checkpoint. To actually exercise a trained
checkpoint's weights, ALSO set CC2_ARCH_TEST_CKPT (and CC2_ARCH_TEST_CONFIG,
the run's config.yaml, for the data-side params not saved in the checkpoint's
own hparams) -- see _load_checkpoint_model(). Discovered 2026-09-27 while
verifying principal-biography/1: the flag-only invocation silently re-tested
the same untrained toy model every time, regardless of which checkpoint was
nominally "being verified".

    CC2_ARCH_TEST_TRAINED=1 \\
    CC2_ARCH_TEST_CKPT=/data/tfs/runs/<run>/1/checkpoints/best.ckpt \\
    CC2_ARCH_TEST_CONFIG=/data/tfs/runs/<run>/1/config.yaml \\
    python3 cc2/tests/test_swinu_architecture.py
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

# init_args names that go straight from a checkpoint's saved hyper_parameters into
# the model config dict (cc2model reads these via SimpleNamespace attribute access;
# see swinu/cc2.py:cc2model.__init__). Data-side names (prognostic/forcing/static
# forcing params, input_resolution) are NOT here: they live on the datamodule, not
# the LightningModule, so they are read from config.yaml's `data:` block instead
# (see _load_checkpoint_model).
_MODEL_HPARAM_KEYS = [
    "patch_size", "hidden_dim", "num_heads", "mlp_ratio", "drop_rate",
    "attn_drop_rate", "drop_path_rate", "window_size", "window_size_deep",
    "encoder1_depth", "encoder2_depth", "decoder1_depth", "decoder2_depth",
    "history_length", "use_gradient_checkpointing", "use_scheduled_sampling",
    "use_flow_matching", "direct_prediction",
]

# CERRA / NWCSAF common grid used by every config in this campaign.
_DEFAULT_INPUT_RESOLUTION = [535, 475]


def _build_model(seed=0):
    torch.manual_seed(seed)
    return cc2model(dict(SMALL_CONFIG)).eval()


def _load_checkpoint_model(ckpt_path, config_path):
    """Build a full-size cc2model from a run's config.yaml + load its trained
    checkpoint's weights. Returns (model, prognostic_params, forcing_params,
    static_forcing_params, input_resolution) so callers can build correctly
    shaped test tensors.
    """
    import yaml

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    data_cfg = cfg["data"]
    model_init_args = cfg["model"]["init_args"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_hparams = ckpt.get("hyper_parameters", {})

    # Start from ALL of config.yaml's model init_args -- by definition these were
    # sufficient to build the model at train time for whatever code era produced
    # this checkpoint (unlike a hardcoded key allowlist, which broke across eras:
    # e.g. trusting-radar's Dec-2025 cc2model reads config.use_deep_refinement_head,
    # a key the current-era allowlist didn't carry). Extra keys cc2model doesn't
    # read are harmless (SimpleNamespace attribute access; unused ones are just
    # never touched). Then let the checkpoint's own saved hparams override, since
    # those are the more authoritative per-instance record.
    real_config = dict(model_init_args)
    real_config.update({k: v for k, v in ckpt_hparams.items() if k in model_init_args})
    real_config.setdefault("use_flow_matching", False)
    real_config.setdefault("direct_prediction", False)
    # Historical code-era default (commit 8014683 removed this option, message
    # "default: true") -- not saved in older checkpoints' hparams because it was
    # never a cc2module.__init__ parameter, only a cc2model-internal default.
    # Historical cc2model-internal-only toggles, never saved to older checkpoints'
    # hparams (they were cc2model-config-dict defaults, not cc2module.__init__
    # parameters). Defaults taken verbatim from each removal commit's own message.
    real_config.setdefault("use_deep_refinement_head", True)   # 8014683
    real_config.setdefault("use_hard_skip", False)              # c7c6640
    real_config.setdefault("use_residual_adapter_head", False)  # 2a78956
    real_config.setdefault("use_high_pass_filter", False)       # 6725a35
    real_config.setdefault("use_residual_io_adapter", False)    # b98b727
    # autoregressive_mode gates a REAL architecture difference (step_embedding_direct
    # only exists when False -- swinu/cc2.py "if self.autoregressive_mode is False").
    # The removal commit's stated default (908c79d, "default: false") is the LATER,
    # direct-prediction-era default and is wrong for AR-era checkpoints (rollout_length
    # > 1 with scheduled sampling, no direct_prediction flag) -- those trained as AR.
    # Infer from direct_prediction instead of a single blanket constant.
    real_config.setdefault("autoregressive_mode", not real_config.get("direct_prediction", False))

    prognostic_params = data_cfg["prognostic_params"]
    forcing_params = data_cfg["forcing_params"]
    static_forcing_params = data_cfg["static_forcing_params"]
    input_resolution = data_cfg.get("input_resolution", _DEFAULT_INPUT_RESOLUTION)

    real_config["prognostic_params"] = prognostic_params
    real_config["forcing_params"] = forcing_params
    real_config["static_forcing_params"] = static_forcing_params
    real_config["input_resolution"] = input_resolution
    real_config.setdefault("preprocessor", None)

    model = cc2model(real_config)

    state_dict = ckpt["state_dict"]
    # Lightning saves the wrapped submodule under "model." (self.model = cc2model(...)).
    prefix = "model."
    stripped = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {ckpt_path} did not load cleanly against config "
            f"{config_path}: {len(missing)} missing key(s), "
            f"{len(unexpected)} unexpected key(s). First few missing: "
            f"{missing[:5]}. First few unexpected: {unexpected[:5]}. "
            "This means the config's model init_args do not match the "
            "checkpoint's actual architecture -- do not proceed with a "
            "partially-loaded model."
        )
    model.eval()
    return model, prognostic_params, forcing_params, static_forcing_params, input_resolution


def _checkpoint_env():
    ckpt_path = os.environ.get("CC2_ARCH_TEST_CKPT")
    config_path = os.environ.get("CC2_ARCH_TEST_CONFIG")
    if not ckpt_path:
        return None
    if not config_path:
        raise SystemExit(
            "CC2_ARCH_TEST_CKPT is set but CC2_ARCH_TEST_CONFIG is not -- both "
            "are required to build a correctly-shaped real model (data-side "
            "params live in config.yaml, not the checkpoint's own hparams)."
        )
    return ckpt_path, config_path


def _forward(model, data, forcing, step=0):
    with torch.no_grad():
        return model(data, forcing, step)


def test_frame0_sensitivity():
    """Perturbing history frame 0 must change the output.

    Two axes, independent of each other:
    - threshold: init (default, 1e-5) vs CC2_ARCH_TEST_TRAINED=1 (1%) -- the
      usage gate from the verification protocol.
    - model: small synthetic init (default) vs a real trained checkpoint, when
      CC2_ARCH_TEST_CKPT/CC2_ARCH_TEST_CONFIG are set. The threshold flag alone
      does NOT load a checkpoint -- see the module docstring.
    """
    trained = os.environ.get("CC2_ARCH_TEST_TRAINED", "0") == "1"
    rel_threshold = 0.01 if trained else 1e-5
    ckpt_env = _checkpoint_env()

    if ckpt_env:
        ckpt_path, config_path = ckpt_env
        model, prog, forc, static, res = _load_checkpoint_model(ckpt_path, config_path)
        h, w = res
        n_prog, n_forc = len(prog), len(forc) + len(static)
        torch.manual_seed(1)
        data = torch.rand(1, 2, n_prog, h, w)
        forcing = torch.rand(1, 3, n_forc, h, w)
        model_desc = f"CHECKPOINT {os.path.basename(ckpt_path)}"
    else:
        model = _build_model()
        torch.manual_seed(1)
        data = torch.rand(1, 2, 1, 32, 32)
        forcing = torch.rand(1, 3, 3, 32, 32)
        model_desc = "small synthetic init"

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
        f"({'trained' if trained else 'init'} threshold, {model_desc}): "
        f"|d(frame0)|={delta_frame0:.3e} vs |d(frame1)|={delta_frame1:.3e} "
        f"(need >= {rel_threshold:.0e} of frame1)"
    )
    return ok


def test_frame_order_sensitivity():
    """Swapping the two history frames (same content, reversed order) must
    change the output — otherwise the model pools frames without seeing
    motion direction. Uses a real checkpoint if CC2_ARCH_TEST_CKPT/
    CC2_ARCH_TEST_CONFIG are set, else the small synthetic init model."""
    ckpt_env = _checkpoint_env()
    if ckpt_env:
        ckpt_path, config_path = ckpt_env
        model, prog, forc, static, res = _load_checkpoint_model(ckpt_path, config_path)
        h, w = res
        n_prog, n_forc = len(prog), len(forc) + len(static)
        torch.manual_seed(2)
        data = torch.rand(1, 2, n_prog, h, w)
        forcing = torch.rand(1, 3, n_forc, h, w)
        model_desc = f"CHECKPOINT {os.path.basename(ckpt_path)}"
    else:
        model = _build_model()
        torch.manual_seed(2)
        data = torch.rand(1, 2, 1, 32, 32)
        forcing = torch.rand(1, 3, 3, 32, 32)
        model_desc = "small synthetic init"

    base = _forward(model, data, forcing)

    d_swap = data.flip(dims=[1])
    f_swap = forcing.clone()
    f_swap[:, :2] = forcing[:, :2].flip(dims=[1])
    delta = (_forward(model, d_swap, f_swap) - base).abs().max().item()

    scale = base.abs().max().item()
    ok = delta > 1e-4 * max(scale, 1.0)
    print(
        f"[{'PASS' if ok else 'FAIL'}] frame_order_sensitivity ({model_desc}): "
        f"|d(swap)|={delta:.3e} (output scale {scale:.3e})"
    )
    return ok


def test_residual_identity():
    """With gamma_attn = gamma_mlp = 0 and no drop-path, an encoder block must
    be exactly the identity. The doubled-residual bug makes it 2x instead.
    Pure architecture-code check -- does not depend on trained weights, so no
    checkpoint-loading path here (a zeroed-gamma block is the same regardless
    of what the rest of the network learned)."""
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
    through torch.roll wraparound, which the shift attention mask must block.
    Pure architecture-code check, same rationale as test_residual_identity."""
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
    ckpt_env = _checkpoint_env()
    if ckpt_env:
        print(f"Running against real checkpoint: {ckpt_env[0]}")
        print(f"                   config:       {ckpt_env[1]}\n")
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
