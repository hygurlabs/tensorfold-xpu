"""Prompt rows as int8 for the 4-bit weights' int8 kernel (``int4_gemm_w4a8``): opt-in, a precision the default path keeps.

Each row is quantised on its own (symmetric, scale = its largest magnitude / 127) and the kernel sums integers, so a row's
result depends on that row alone: it has the same bits at any block size, in any position of a block and beside any padding
(tests/xpu/test_a8.py). What changes is the value: 8-bit activations are about 1e-2 of a row's largest output away from the
fp16 ones. The kernel runs at 180-238 TOPS, about 1.7x the fp16 activations' 125-142 TFLOPs, whatever the shape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ENABLED = False
ONLY: set | None = None   # (N, K) shapes that run as int8; None: every one
CLIP = 1.0                # a row's scale is CLIP * its largest magnitude / 127: values past it saturate
PAD = 1024            # rows of the buffers: whole 1,024-row blocks, so any block plan of a call fits
_zeros: dict = {}


def enable(on: bool = True, only: set | None = None) -> None:
    global ENABLED, ONLY
    ENABLED, ONLY = bool(on), only


MLP = {(17408, 5120), (5120, 17408)}     # gate and up, down: 70% of a token's FLOPs


def configure(level: str) -> None:
    """``off``; ``mlp``: the MLP's three projections; ``all``: every projection of the 4-bit layers."""

    if level not in ("off", "mlp", "all"):
        raise ValueError(f"--prefill-int8 takes off, mlp or all, not {level!r}")
    enable(level != "off", MLP if level == "mlp" else None)


def runs(n: int, k: int) -> bool:
    return ENABLED and (ONLY is None or (n, k) in ONLY)


@triton.jit
def _quantize(X, Q, S, K, CLIP, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * K
    top = tl.zeros((BLOCK,), tl.float32)
    for c in range(0, K, BLOCK):
        col = c + tl.arange(0, BLOCK)
        x = tl.load(X + base + col) if EVEN else tl.load(X + base + col, mask=col < K, other=0.0)
        top = tl.maximum(top, tl.abs(x.to(tl.float32)))
    scale = tl.maximum(tl.max(top, axis=0) * CLIP, 1e-6) / 127.0
    tl.store(S + row, scale)
    for c in range(0, K, BLOCK):
        col = c + tl.arange(0, BLOCK)
        x = (tl.load(X + base + col) if EVEN else tl.load(X + base + col, mask=col < K, other=0.0)).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.floor(x / scale + 0.5), -127.0), 127.0).to(tl.int8)
        if EVEN:
            tl.store(Q + base + col, q)
        else:
            tl.store(Q + base + col, q, mask=col < K)


def quantized(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 rows, fp32 scales (rows, 1)) of ``x`` (rows, K), both padded to whole ``PAD`` rows. Kept on ``x``: the
    projections that read the same rows (q, k, v; gate, up; qkv, z) quantise them once."""

    cached = getattr(x, "_a8", None)
    if cached is not None:
        return cached
    rows, k = x.shape
    padded = -(-rows // PAD) * PAD
    q = torch.empty((padded, k), dtype=torch.int8, device=x.device)
    s = torch.empty((padded, 1), dtype=torch.float32, device=x.device)
    x = x.contiguous()
    block = 1024
    _quantize[(rows,)](x, q, s, k, CLIP, BLOCK=block, EVEN=k % block == 0, num_warps=8)
    try:
        x._a8 = (q, s)
    except AttributeError:
        pass
    return q, s


def zero_points(device, rows: int) -> torch.Tensor:
    """(rows, 1) int32 zeros: the kernel's activation zero points, symmetric here."""

    z = _zeros.get(device)
    if z is None or z.shape[0] < rows:
        z = _zeros[device] = torch.zeros((max(rows, 2 * PAD), 1), dtype=torch.int32, device=device)
    return z[:rows]
