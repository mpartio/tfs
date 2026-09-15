"""Time DWConvResidual3D variants at the real config's tensor shapes.

Settles whether the depthwise-Conv3d temporal mixing was responsible for the
stall of job 22035530, and confirms the separable replacement is not itself
slow. Run on ONE GPU inside the training container:

    python3 cc2/tests/bench_dwres.py

Reports forward+backward ms/iter for:
  separable  - the shipped implementation (temporal tap by slicing + Conv2d)
  conv3d     - the previous implementation (depthwise Conv3d)
  spatial2d  - Conv2d only, no temporal mixing (the pre-fix baseline speed)

A large separable/spatial2d ratio means the temporal tap itself is expensive.
A large conv3d/spatial2d ratio means the 3D kernel was the problem.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from swinu.layers import DWConvResidual3D  # noqa: E402

# Real shapes from etc/arch/config-cerra-root-fixed.yaml (hidden_dim 256,
# patch_size 4, 475x535 padded to 480x536 -> 120x134 patches).
CASES = [
    ("encoder1", 256, 120, 134, 2, 2.0),
    ("encoder2", 512, 60, 67, 2, 1.0),
]


class Conv3dVariant(nn.Module):
    """The previous implementation: true depthwise Conv3d over (T, h, w)."""

    def __init__(self, C, grid_hw, time_dim, expand=2.0, dilation=1, ls_init=1e-2):
        super().__init__()
        self.C, self.T = C, time_dim
        self.h, self.w = grid_hw
        hidden = max(1, int(C * expand))
        kt = 3 if time_dim > 1 else 1
        self.dw = nn.Conv3d(C, C, (kt, 3, 3), padding=(kt // 2, dilation, dilation),
                            dilation=(1, dilation, dilation), groups=C, bias=True)
        self.pw1 = nn.Conv3d(C, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, C, 1, bias=True)
        self.ls = nn.Parameter(torch.ones(C) * ls_init)

    def forward(self, x):
        B, L, C = x.shape
        x3 = x.view(B, self.T, self.h, self.w, C).permute(0, 4, 1, 2, 3).contiguous()
        y = self.dw(x3)
        y = self.pw2(self.act(self.pw1(y)))
        y = y * self.ls.view(1, -1, 1, 1, 1) + x3
        return y.permute(0, 2, 3, 4, 1).contiguous().view(B, L, C)


class Spatial2dVariant(nn.Module):
    """Pre-fix baseline: per-slice Conv2d, no temporal mixing at all."""

    def __init__(self, C, grid_hw, time_dim, expand=2.0, dilation=1, ls_init=1e-2):
        super().__init__()
        self.C, self.T = C, time_dim
        self.h, self.w = grid_hw
        hidden = max(1, int(C * expand))
        self.dw = nn.Conv2d(C, C, 3, padding=dilation, dilation=dilation,
                            groups=C, bias=True)
        self.pw1 = nn.Conv2d(C, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, C, 1, bias=True)
        self.ls = nn.Parameter(torch.ones(C) * ls_init)

    def forward(self, x):
        B, L, C = x.shape
        x2 = x.view(B, self.T, self.h, self.w, C).permute(0, 1, 4, 2, 3)
        x2 = x2.reshape(B * self.T, C, self.h, self.w)
        y = self.dw(x2)
        y = self.pw2(self.act(self.pw1(y)))
        y = y * self.ls.view(1, -1, 1, 1) + x2
        return (y.view(B, self.T, C, self.h, self.w)
                 .permute(0, 1, 3, 4, 2).contiguous().view(B, L, C))


def timeit(mod, x, iters=20, warmup=5):
    """Forward+backward ms/iter. Warmup covers MIOpen solver search."""
    mod = mod.cuda()
    x = x.cuda().requires_grad_(True)
    for _ in range(warmup):
        mod(x).sum().backward()
        mod.zero_grad(set_to_none=True)
        x.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mod(x).sum().backward()
        mod.zero_grad(set_to_none=True)
        x.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    if not torch.cuda.is_available():
        print("No GPU visible — run this inside the container on an allocated node.")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    print(f"{'case':<10} {'variant':<11} {'ms/iter':>10} {'vs spatial2d':>13}")
    print("-" * 48)

    for name, C, h, w, T, expand in CASES:
        x = torch.randn(1, T * h * w, C)
        base = None
        for label, cls in (("spatial2d", Spatial2dVariant),
                           ("separable", DWConvResidual3D),
                           ("conv3d", Conv3dVariant)):
            torch.manual_seed(0)
            mod = cls(C, (h, w), time_dim=T, expand=expand)
            try:
                ms = timeit(mod, x)
            except RuntimeError as e:
                print(f"{name:<10} {label:<11} {'FAILED':>10}   {type(e).__name__}: {e}")
                continue
            if base is None:
                base = ms
            print(f"{name:<10} {label:<11} {ms:10.2f} {ms / base:12.1f}x")
            del mod
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
