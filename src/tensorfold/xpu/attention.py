"""Tree attention's shared-chunk kernel for an Intel XPU: IGC crashes on its masked row-indexed store, so tiles go out whole."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import _fold, _tile

ROWS = 16     # query rows a program holds (the CUDA kernel's tile)


@triton.jit
def _shared(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, TO, TB, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, GR: tl.constexpr):
    """The CUDA kernel's arithmetic; its fp32 tile goes to TO whole and TB says where each row belongs (-1: none)."""

    prog = tl.program_id(0)
    item = prog // HK
    hk = prog % HK
    s = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    code = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + s * 4)
    rows = tl.load(STREAM + s * 4 + 1)
    p = tl.load(STREAM + s * 4 + 2)
    groups = ((p + rows + CH - 1) // CH - tl.load(STREAM + s * 4 + 3)) // (GR - 1)
    whole = code >= 0
    chunk = tl.where(whole, code * GR, groups * GR - 1 - code)
    n = tl.where(whole, GR, 1)
    slot = tl.where(whole, code, chunk - groups * (GR - 1))
    if ((chunk + n) * CH <= p) & (code < groups):
        koff = tl.multiple_of(tl.load(OFF + s * 2), 8)
        voff = tl.multiple_of(tl.load(OFF + s * 2 + 1), 8)
        rr = first + tl.arange(0, 16)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        gm = tl.full((16,), float("-inf"), tl.float32)
        gl = tl.zeros((16,), tl.float32)
        go = tl.zeros((16, D), tl.float32)
        m, l, o = gm, gl, go
        for c in range(n):
            key = (chunk + c) * CH + tl.arange(0, 64)
            m = tl.full((16,), float("-inf"), tl.float32)
            l = tl.zeros((16,), tl.float32)
            o = tl.zeros((16, D), tl.float32)
            for t in range(CH // 64):
                ki = key + t * 64
                kk = tl.load(KC + koff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                vv = tl.load(VC + voff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                m, l, o = _tile(q, kk, vv, m, l, o, ki < p, SCALE)
            gm, gl, go = _fold(gm, gl, go, m, l, o)
        if not whole:
            gm, gl, go = m, l, o
        base = (slot * W + node) * H + head
        local = tl.arange(0, 16)
        tl.store(TO + (prog * 16 + local[:, None]) * D + d[None, :], go)
        tl.store(TB + prog * 16 + local, tl.where(ok, base, -1))
        tl.store(PM + base, gm, mask=ok)
        tl.store(PL + base, gl, mask=ok)


class _Launcher:
    """Stands in for ``attention._shared``: same call, the tile placed by a gather after the kernel."""

    def __getitem__(self, grid):
        programs = int(grid[0])

        def launch(q, kc, vc, offs, streams, items, po, pm, pl, w, **kwargs):
            d = kwargs["D"]
            tile = torch.empty((programs * ROWS, d), dtype=torch.float32, device=q.device)
            where = torch.full((programs * ROWS,), -1, dtype=torch.int32, device=q.device)
            _shared[grid](q, kc, vc, offs, streams, items, po, pm, pl, w, tile, where, **kwargs)
            valid = where >= 0
            po.view(-1, d)[where[valid].long()] = tile[valid]

        return launch


shared = _Launcher()
