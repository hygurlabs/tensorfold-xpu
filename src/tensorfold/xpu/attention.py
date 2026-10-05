"""Tree attention's shared-chunk kernel for an Intel XPU: IGC crashes on its masked row-indexed store, so tiles go out whole."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import _fold, _merge as _cuda_merge, _tail as cuda_tail, _tile  # noqa: F401  (cuda_tail: the bits' reference)
from tensorfold.cuda.kernels.prefill_attention import _attend, _check

ROWS = 16     # query rows a program holds (the CUDA kernel's tile)
TAIL_TILE = 32
SHARED_TILE = 32
WARPS = 2     # a tile costs 20 us on two warps and 36 on four (measured, one program, 4,096 keys)
PROMPT_ROWS = 128   # prompt rows a program holds; the bits do not depend on it (measured: one checksum for 32..128)
PROMPT_KEYS = 32    # keys a prompt tile folds at once, by absolute position: the bits' definition on this backend


@triton.jit
def _shared(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, TO, TB, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, GR: tl.constexpr, TS: tl.constexpr):
    """The CUDA kernel's arithmetic, in TS-key tiles; its fp32 tile goes to TO whole and TB says where each row belongs (-1: none)."""

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
        d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D), 8), D)
        rs = tl.minimum(rr, rows * G - 1)                     # no mask on the load (see _tail): the padded rows repeat the last
        q = tl.load(Q + ((start + rs // G)[:, None] * H + (hk * G + rs % G)[:, None]) * D + d[None, :]).to(tl.bfloat16)
        gm = tl.full((16,), float("-inf"), tl.float32)
        gl = tl.zeros((16,), tl.float32)
        go = tl.zeros((16, D), tl.float32)
        m, l, o = gm, gl, go
        for c in range(n):
            key = (chunk + c) * CH + tl.arange(0, TS)
            m = tl.full((16,), float("-inf"), tl.float32)
            l = tl.zeros((16,), tl.float32)
            o = tl.zeros((16, D), tl.float32)
            for t in range(CH // TS):
                ki = key + t * TS
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


@triton.jit
def _tail(Q, ORIGIN, OFF, STREAM, ROWS, PATHS, DEPTHS, PO, PM, PL, W, KNO, VNO, H: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, MAXD: tl.constexpr, SCALE: tl.constexpr,
          GR: tl.constexpr, RT: tl.constexpr, TT: tl.constexpr):
    """The CUDA kernel's tail: the chunk's keys, committed then the node's path, in TT-key tiles anchored at the chunk
    whatever p is (a tile cut at p would make a row depend on how many tokens were committed before its round). The
    two sources are chosen on the loaded values, not on the addresses: a per-key address choice runs 5x slower
    (measured). TT is 32, not CUDA's 64: the 64-key tile spills registers here and runs 3.7x slower (measured).

    Keys and values are element offsets from ORIGIN (KNO and VNO: the window's rows), one row address a key."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    p = tl.load(STREAM + s * 4 + 2)
    nch = (p + tl.load(STREAM + s * 4 + 1) + CH - 1) // CH
    groups = (nch - tl.load(STREAM + s * 4 + 3)) // (GR - 1)
    chunk = p // CH + tl.program_id(2)
    if chunk < nch:
        koff = tl.load(OFF + s * 2)
        voff = tl.load(OFF + s * 2 + 1)
        gg = tl.arange(0, RT)
        d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D), 8), D)
        qrow = tl.minimum(gg, G - 1)       # no mask on the load: a masked q slows every tile of the loop 6x (measured)
        q = tl.load(Q + (node * H + hk * G + qrow[:, None]) * D + d[None, :]).to(tl.bfloat16)
        depth = tl.load(DEPTHS + node)
        m = tl.full((RT,), float("-inf"), tl.float32)
        l = tl.zeros((RT,), tl.float32)
        o = tl.zeros((RT, D), tl.float32)
        lane = tl.arange(0, TT)
        end = tl.minimum((chunk + 1) * CH, p + depth)               # the keys of this chunk: committed, then the node's path
        tiles = (tl.maximum(end - chunk * CH, 0) + TT - 1) // TT
        for t in range(tiles):
            logical = chunk * CH + t * TT + lane                    # tiles are anchored at the chunk, whatever p is
            have = logical < end
            committed = logical < p
            crow = tl.maximum(tl.minimum(logical, p - 1), 0)
            slot = tl.minimum(tl.maximum(logical - p, 0), depth - 1)
            path_node = tl.load(PATHS + node * MAXD + slot)
            ckr = tl.multiple_of(koff + (crow * HK + hk) * D, 8)
            cvr = tl.multiple_of(voff + (crow * HK + hk) * D, 8)
            pkr = tl.multiple_of(KNO + (path_node * HK + hk) * D, 8)
            pvr = tl.multiple_of(VNO + (path_node * HK + hk) * D, 8)
            kk = tl.where(committed[:, None], tl.load(ORIGIN + ckr[:, None] + d[None, :]), tl.load(ORIGIN + pkr[:, None] + d[None, :]))
            vv = tl.where(committed[:, None], tl.load(ORIGIN + cvr[:, None] + d[None, :]), tl.load(ORIGIN + pvr[:, None] + d[None, :]))
            m, l, o = _tile(q, kk.to(tl.bfloat16), vv.to(tl.bfloat16), m, l, o, have, SCALE)
        base = ((chunk - groups * (GR - 1)) * W + node) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _place(TILE, WHERE, PO, D: tl.constexpr):
    """Row i of the tile goes to row WHERE[i] of PO, or nowhere when that is negative: no host round trip."""

    i = tl.program_id(0)
    dest = tl.load(WHERE + i)
    d = tl.arange(0, D)
    tl.store(PO + dest.to(tl.int64) * D + d, tl.load(TILE + i.to(tl.int64) * D + d), mask=(d >= 0) & (dest >= 0))


class _Launcher:
    """Stands in for ``attention._shared``: same call, the tile placed by a gather after the kernel."""

    def __getitem__(self, grid):
        programs = int(grid[0])

        def launch(q, kc, vc, offs, streams, items, po, pm, pl, w, **kwargs):
            d = kwargs["D"]
            tile = torch.empty((programs * ROWS, d), dtype=torch.float32, device=q.device)
            where = torch.full((programs * ROWS,), -1, dtype=torch.int32, device=q.device)
            kwargs.pop("num_warps", None)
            kwargs.pop("num_stages", None)
            _shared[grid](q, kc, vc, offs, streams, items, po, pm, pl, w, tile, where, enable_fp_fusion=False, num_warps=WARPS,
                          num_stages=1, TS=SHARED_TILE, **kwargs)
            _place[(programs * ROWS,)](tile, where, po, D=d, num_warps=1)

        return launch


shared = _Launcher()


class _TailLauncher:
    """Stands in for ``attention._tail``: the same call, the window's rows turned into offsets from the caches' origin."""

    def __getitem__(self, grid):
        def launch(q, kn, vn, kc, vc, offs, streams, rows, paths, depths, po, pm, pl, w, **kwargs):
            if kc.data_ptr() != vc.data_ptr():
                raise ValueError("the tail takes one origin for the keys and the values")
            kno, vno = ((t.data_ptr() - kc.data_ptr()) // 2 for t in (kn, vn))
            if (kn.data_ptr() - kc.data_ptr()) % 16 or (vn.data_ptr() - kc.data_ptr()) % 16:
                raise ValueError("the window's rows must sit a multiple of 16 bytes from the origin")
            kwargs.pop("num_warps", None)
            kwargs.pop("num_stages", None)
            _tail[grid](q, kc, offs, streams, rows, paths, depths, po, pm, pl, w, kno, vno, RT=ROWS, TT=TAIL_TILE, num_warps=WARPS, num_stages=1,
                        enable_fp_fusion=False, **kwargs)

        return launch


tail = _TailLauncher()


def prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """Prompt attention (the CUDA module's kernel) in tiles that suit an Arc GPU: 3x faster, chunking still changes no bit."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    out = torch.empty_like(q) if out is None else out
    _attend[(triton.cdiv(w, PROMPT_ROWS), h)](q, k_cache, v_cache, out, p0, w, H=h, HK=hk, D=d, BM=PROMPT_ROWS,
                                              BN=PROMPT_KEYS, SCALE=scale, num_warps=16, num_stages=1)
    return out


class _MergeLauncher:
    """Stands in for ``attention._merge``: the same call with no fused multiply-add, as the two kernels that feed it."""

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            _cuda_merge[grid](*args, enable_fp_fusion=False, **kwargs)

        return launch


merge = _MergeLauncher()
