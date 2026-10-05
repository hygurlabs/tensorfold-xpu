"""An unquantized projection on an Intel XPU with one fixed order of sums: a row's bits depend on neither the other rows nor the run."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROWS = 16        # rows a program holds; every row meets the same instructions whatever the row count
OUTPUTS = 64
BLOCK = 64
SPLIT_BELOW = 4096   # a projection with fewer outputs than this spreads its sum over K across programs
SPLITS = (80, 40, 20, 16, 8, 4, 2)      # parts of the sum over K, the most that divide it: one dot a program


@triton.jit
def _dense(X, W, OUT, M, N, K, RT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.program_id(1) * RT + tl.arange(0, RT)
    k = tl.arange(0, BK)
    acc = tl.zeros((RT, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rows[:, None] * K + k0 + k[None, :], mask=rows[:, None] < M, other=0.0).to(tl.float16)
        w = tl.load(W + cols[:, None] * K + k0 + k[None, :], mask=cols[:, None] < N, other=0.0).to(tl.float16)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _partial(X, W, P, M, N, K, KS, RT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One slice of K: the same fp16 products summed in fp32, kept apart so the slices add in a fixed order."""

    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.program_id(1) * RT + tl.arange(0, RT)
    part = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.zeros((RT, BN), tl.float32)
    for k0 in range(part * KS, part * KS + KS, BK):
        x = tl.load(X + rows[:, None] * K + k0 + k[None, :], mask=rows[:, None] < M, other=0.0).to(tl.float16)
        w = tl.load(W + cols[:, None] * K + k0 + k[None, :], mask=cols[:, None] < N, other=0.0).to(tl.float16)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(P + (part * M + rows[:, None]) * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _fold(P, OUT, M, N, S: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for part in range(S):
        acc = acc + tl.load(P + (part * M + row) * N + cols, mask=cols < N, other=0.0)
    tl.store(OUT + row * N + cols, acc.to(tl.bfloat16), mask=cols < N)


def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x (M, K) bf16 times weight (N, K) fp16 or bf16, transposed: (M, N) bf16 from fp16 products summed in fp32."""

    rows, k = x.shape
    n = weight.shape[0]
    if k % BLOCK or weight.shape[1] != k or not (x.is_contiguous() and weight.is_contiguous()):
        raise ValueError("dense projection: contiguous operands and an input width that is a multiple of 64")
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    split = next((s for s in SPLITS if n < SPLIT_BELOW and k % (s * BLOCK) == 0), 1)
    if rows and split > 1:
        parts = torch.empty((split, rows, n), dtype=torch.float32, device=x.device)
        _partial[(triton.cdiv(n, OUTPUTS), triton.cdiv(rows, ROWS), split)](x, weight, parts, rows, n, k, k // split, RT=ROWS,
                                                                            BN=OUTPUTS, BK=BLOCK, num_warps=4)
        _fold[(triton.cdiv(n, OUTPUTS), rows)](parts, out, rows, n, S=split, BN=OUTPUTS, num_warps=1)
    elif rows:
        _dense[(triton.cdiv(n, OUTPUTS), triton.cdiv(rows, ROWS))](x, weight, out, rows, n, k, RT=ROWS, BN=OUTPUTS, BK=BLOCK,
                                                                   num_warps=4)
    return out
