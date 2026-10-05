"""The decode glue on an Intel XPU: the rows that feed a 4-bit projection are written as fp16, where it reads them.

The int4 kernel takes fp16 rows; the CUDA glue writes bf16, so every projection paid two conversion launches (in and
out, about 5 us each, 11% of a round). These four kernels do the CUDA glue's arithmetic and store fp16 straight into a
buffer padded to the kernel's 16 rows, so ``Int4`` reads them as they lie. The group sums the CUDA kernels also write
are for its FP8 path and are not computed here. Each program reads only its own row: a row's bits do not depend on
the window.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .gptq import ROWS


def rows_buffer(rows: int, width: int, device) -> torch.Tensor:
    """(rows, width) fp16 view of a buffer padded to whole ``ROWS`` (a window) or to whole 1,024 rows (a prompt chunk):
    ``Int4`` reads the padding where it lies, whatever block size its shape's prompt calls use."""

    pad = ROWS if rows <= ROWS else 1024
    return torch.empty((-(-rows // pad) * pad, width), dtype=torch.float16, device=device)[:rows]


@triton.jit
def _add_rmsnorm(X, R, W, H, Y, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    if HAS_R:
        r = tl.load(R + row * D + offs, mask=ok, other=0.0).to(tl.float32)
        h = (x + r).to(tl.bfloat16)
        tl.store(H + row * D + offs, h, mask=ok)
        x = h.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, (x * inv * w).to(tl.float16), mask=ok)


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor | None, w: torch.Tensor, eps: float):
    """h = x + r (bf16, the residual stream), y = rmsnorm(h) * w (fp16). Returns (h, y, None)."""

    rows, d = x.shape
    h = torch.empty_like(x) if r is not None else x
    y = rows_buffer(rows, d, x.device)
    _add_rmsnorm[(rows,)](x, r if r is not None else x, w, h, y, eps, D=d, BLOCK=triton.next_power_of_2(d),
                          HAS_R=r is not None, num_warps=8)
    return h, y, None


@triton.jit
def _gated_norm(Yr, Z, W, OUT, eps, VH: tl.constexpr, DV: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    offs = (row * VH + h) * DV + tl.arange(0, DV)
    y = tl.load(Yr + offs).to(tl.float32)
    z = tl.load(Z + offs).to(tl.float32)
    w = tl.load(W + tl.arange(0, DV)).to(tl.float32)
    yn = y * (1.0 / tl.sqrt(tl.sum(y * y, axis=0) / DV + eps)) * w
    tl.store(OUT + offs, (z * tl.sigmoid(z) * yn).to(tl.float16))


def gated_norm(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float):
    """y, z (W, VH, DV) -> silu(z) * rmsnorm(y) * w as (W, VH*DV) fp16."""

    rows, vh, dv = y.shape
    out = rows_buffer(rows, vh * dv, y.device)
    _gated_norm[(rows, vh)](y, z, w, out, eps, VH=vh, DV=dv, num_warps=1)
    return out, None


@triton.jit
def _swiglu(GATE, UP, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < N
    g = tl.load(GATE + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(UP + row * N + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + row * N + offs, (g * tl.sigmoid(g) * u).to(tl.float16), mask=ok)


def swiglu(gate: torch.Tensor, up: torch.Tensor):
    rows, n = gate.shape
    out = rows_buffer(rows, n, gate.device)
    block = 1024
    _swiglu[(rows, triton.cdiv(n, block))](gate, up, out, N=n, BLOCK=block, num_warps=4)
    return out, None


@triton.jit
def _gate_mul(O, QG, OUT, H: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    o = tl.load(O + (row * H + h) * D + d).to(tl.float32)
    g = tl.load(QG + (row * H + h) * 2 * D + D + d).to(tl.float32)
    tl.store(OUT + (row * H + h) * D + d, (o * tl.sigmoid(g)).to(tl.float16))


def gate_mul(o: torch.Tensor, qg: torch.Tensor, *, heads: int, head_dim: int):
    """Attention output (W, H, D) times sigmoid(gate) from the [q | gate] rows -> (W, H*D) fp16."""

    rows = o.shape[0]
    out = rows_buffer(rows, heads * head_dim, o.device)
    _gate_mul[(rows, heads)](o, qg, out, H=heads, D=head_dim, num_warps=2)
    return out, None
