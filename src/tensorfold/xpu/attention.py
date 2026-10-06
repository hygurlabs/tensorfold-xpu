"""Tree attention's shared-chunk kernel for an Intel XPU: IGC crashes on its masked row-indexed store, so tiles go out whole."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import _fold, _merge as _cuda_merge, _tail as cuda_tail, _tile  # noqa: F401  (cuda_tail: the bits' reference)
from tensorfold.cuda.kernels.prefill_attention import _check, _tile as _prompt_tile

ROWS = 16     # query rows a program holds (the CUDA kernel's tile)
TAIL_TILE = 32
TAIL_COLUMNS = 64   # columns of q and the keys a tail program reads at a time
TAIL_SPLIT = 4      # programs that share a tail tile's output columns
SHARED_TILE = 32
SHARED_SPLIT = 4    # programs that share a shared-chunk tile's output columns
WARPS = 2     # a tile costs 20 us on two warps and 36 on four (measured, one program, 4,096 keys)
PROMPT_ROWS = 128   # prompt rows a program holds; the bits do not depend on it
PROMPT_KEYS = 16    # keys a prompt tile folds at once, by absolute position: the bits' definition on this backend
PROMPT_WARPS = 16


@triton.jit(do_not_specialize=["W"])
def _shared(Q, KC, VC, OFF, STREAM, ITEMS, PO, PM, PL, W, TO, TB, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
            G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, GR: tl.constexpr, TS: tl.constexpr, DC: tl.constexpr,
            NS: tl.constexpr):
    """The CUDA kernel's arithmetic, in TS-key tiles; its fp32 tile goes to TO whole and TB says where each row belongs (-1: none).

    Like ``_tail``, NS programs share an item's output columns and read q and the keys DC columns at a time."""

    z = tl.program_id(0)
    part = z % NS
    prog = z // NS
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
        DS: tl.constexpr = D // NS
        koff = tl.multiple_of(tl.load(OFF + s * 2), 8)
        voff = tl.multiple_of(tl.load(OFF + s * 2 + 1), 8)
        rr = first + tl.arange(0, 16)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        dc = tl.max_contiguous(tl.multiple_of(tl.arange(0, DC), 8), DC)
        ds = part * DS + tl.max_contiguous(tl.multiple_of(tl.arange(0, DS), 8), DS)
        rs = tl.minimum(rr, rows * G - 1)                     # no mask on the load (see _tail): the padded rows repeat the last
        qbase = ((start + rs // G) * H + (hk * G + rs % G)) * D
        gm = tl.full((16,), float("-inf"), tl.float32)
        gl = tl.zeros((16,), tl.float32)
        go = tl.zeros((16, DS), tl.float32)
        m, l, o = gm, gl, go
        for c in range(n):
            key = (chunk + c) * CH + tl.arange(0, TS)
            m = tl.full((16,), float("-inf"), tl.float32)
            l = tl.zeros((16,), tl.float32)
            o = tl.zeros((16, DS), tl.float32)
            for t in range(CH // TS):
                ki = key + t * TS
                acc = tl.zeros((16, TS), tl.float32)
                for k0 in tl.static_range(D // DC):
                    qc = tl.load(Q + qbase[:, None] + (k0 * DC + dc)[None, :]).to(tl.bfloat16)
                    kk = tl.load(KC + koff + (ki[:, None] * HK + hk) * D + (k0 * DC + dc)[None, :]).to(tl.bfloat16)
                    acc = tl.dot(qc, tl.trans(kk), acc)
                valid = ki < p
                scores = tl.where(valid[None, :], acc * SCALE, float("-inf"))
                tile_m = tl.max(scores, 1)
                active = tile_m != float("-inf")
                next_m = tl.where(active, tl.maximum(m, tile_m), m)
                alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
                pr = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
                vv = tl.load(VC + voff + (ki[:, None] * HK + hk) * D + ds[None, :]).to(tl.bfloat16)
                o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), vv)
                l = l * alpha + tl.sum(pr, 1)
                m = next_m
            gm, gl, go = _fold(gm, gl, go, m, l, o)
        if not whole:
            gm, gl, go = m, l, o
        base = (slot * W + node) * H + head
        local = tl.arange(0, 16)
        tl.store(TO + (prog * 16 + local[:, None]) * D + ds[None, :], go)
        if part == 0:
            tl.store(TB + prog * 16 + local, tl.where(ok, base, -1))
            tl.store(PM + base, gm, mask=ok)
            tl.store(PL + base, gl, mask=ok)


@triton.jit
def _update(acc, vv, valid, m, l, o, SCALE: tl.constexpr):
    """One tile's online-softmax step from its scores ``acc`` (the CUDA ``_tile``'s arithmetic) and values ``vv``."""

    scores = tl.where(valid[None, :], acc * SCALE, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    pr = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    return next_m, l * alpha + tl.sum(pr, 1), o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), vv.to(tl.bfloat16))


@triton.jit(do_not_specialize=["W", "KNO", "VNO"])
def _tail(Q, ORIGIN, OFF, STREAM, ROWS, PATHS, DEPTHS, PO, PM, PL, W, KNO, VNO, H: tl.constexpr, HK: tl.constexpr,
          D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, MAXD: tl.constexpr, SCALE: tl.constexpr,
          GR: tl.constexpr, RT: tl.constexpr, TT: tl.constexpr, DC: tl.constexpr, NS: tl.constexpr):
    """The CUDA kernel's tail: the chunk's keys, committed then the node's path, in TT-key tiles anchored at the chunk
    whatever p is (a tile cut at p would make a row depend on how many tokens were committed before its round).

    Programs (node, kv head, chunk * NS + part): each holds D / NS of the output columns and reads q and the keys DC
    columns at a time, which keeps the tile in registers; the bits are those of one program holding all of D. Tiles
    of committed keys only load them unmasked; the few that hold the path choose the source on the loaded values, and
    never read a committed row that does not exist. Keys and values are element offsets from ORIGIN."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    z = tl.program_id(2)
    part = z % NS
    s = tl.load(ROWS + node)
    p = tl.load(STREAM + s * 4 + 2)
    nch = (p + tl.load(STREAM + s * 4 + 1) + CH - 1) // CH
    groups = (nch - tl.load(STREAM + s * 4 + 3)) // (GR - 1)
    chunk = p // CH + z // NS
    if chunk < nch:
        DS: tl.constexpr = D // NS
        koff = tl.load(OFF + s * 2)
        voff = tl.load(OFF + s * 2 + 1)
        gg = tl.arange(0, RT)
        dc = tl.max_contiguous(tl.multiple_of(tl.arange(0, DC), 8), DC)
        ds = part * DS + tl.max_contiguous(tl.multiple_of(tl.arange(0, DS), 8), DS)
        qbase = (node * H + hk * G + tl.minimum(gg, G - 1)) * D        # no mask on the load: padded rows repeat the last
        depth = tl.load(DEPTHS + node)
        m = tl.full((RT,), float("-inf"), tl.float32)
        l = tl.zeros((RT,), tl.float32)
        o = tl.zeros((RT, DS), tl.float32)
        lane = tl.arange(0, TT)
        end = tl.minimum((chunk + 1) * CH, p + depth)
        tiles = (tl.maximum(end - chunk * CH, 0) + TT - 1) // TT
        whole = tl.minimum(tl.maximum(p - chunk * CH, 0) // TT, tiles)           # tiles of committed keys only
        every = lane < TT
        for t in range(whole):
            row = chunk * CH + t * TT + lane
            ckr = tl.multiple_of(koff + (row * HK + hk) * D, 8)
            cvr = tl.multiple_of(voff + (row * HK + hk) * D, 8)
            acc = tl.zeros((RT, TT), tl.float32)
            for c in tl.static_range(D // DC):
                qc = tl.load(Q + qbase[:, None] + (c * DC + dc)[None, :]).to(tl.bfloat16)
                kc = tl.load(ORIGIN + ckr[:, None] + (c * DC + dc)[None, :])
                acc = tl.dot(qc, tl.trans(kc.to(tl.bfloat16)), acc)
            m, l, o = _update(acc, tl.load(ORIGIN + cvr[:, None] + ds[None, :]), every, m, l, o, SCALE)
        for t in range(whole, tiles):
            logical = chunk * CH + t * TT + lane
            valid = logical < end
            committed = logical < p
            crow = tl.maximum(tl.minimum(logical, p - 1), 0)
            slot = tl.minimum(tl.maximum(logical - p, 0), depth - 1)
            path_node = tl.load(PATHS + node * MAXD + slot)
            ckr = tl.multiple_of(koff + (crow * HK + hk) * D, 8)
            cvr = tl.multiple_of(voff + (crow * HK + hk) * D, 8)
            pkr = tl.multiple_of(KNO + (path_node * HK + hk) * D, 8)
            pvr = tl.multiple_of(VNO + (path_node * HK + hk) * D, 8)
            acc = tl.zeros((RT, TT), tl.float32)
            for c in tl.static_range(D // DC):
                qc = tl.load(Q + qbase[:, None] + (c * DC + dc)[None, :]).to(tl.bfloat16)
                kc = tl.where(committed[:, None], tl.load(ORIGIN + ckr[:, None] + (c * DC + dc)[None, :], mask=committed[:, None], other=0.0),
                              tl.load(ORIGIN + pkr[:, None] + (c * DC + dc)[None, :]))
                acc = tl.dot(qc, tl.trans(kc.to(tl.bfloat16)), acc)
            vv = tl.where(committed[:, None], tl.load(ORIGIN + cvr[:, None] + ds[None, :], mask=committed[:, None], other=0.0),
                          tl.load(ORIGIN + pvr[:, None] + ds[None, :]))
            m, l, o = _update(acc, vv, valid, m, l, o, SCALE)
        base = ((chunk - groups * (GR - 1)) * W + node) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + ds[None, :], o, mask=gg[:, None] < G)
        if part == 0:
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
            _shared[(programs * SHARED_SPLIT,)](q, kc, vc, offs, streams, items, po, pm, pl, w, tile, where, enable_fp_fusion=False,
                                                num_warps=WARPS, num_stages=1, TS=SHARED_TILE, DC=TAIL_COLUMNS, NS=SHARED_SPLIT, **kwargs)
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
            _tail[(grid[0], grid[1], grid[2] * TAIL_SPLIT)](q, kc, offs, streams, rows, paths, depths, po, pm, pl, w, kno, vno, RT=ROWS,
                                                            TT=TAIL_TILE, DC=TAIL_COLUMNS, NS=TAIL_SPLIT, num_warps=WARPS,
                                                            num_stages=1, enable_fp_fusion=False, **kwargs)

        return launch


tail = _TailLauncher()


@triton.jit(do_not_specialize=["p0", "W", "L"])
def _prompt(Q, K, V, OUT, p0, W, L, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            SCALE: tl.constexpr):
    """The CUDA prompt kernel's fold (keys in tiles of BN by absolute position), its tiles moved as 2D blocks."""

    block = tl.program_id(0)
    head = tl.program_id(1)
    hk = head // (H // HK)
    dq = tl.make_tensor_descriptor(Q + head * D, shape=[W, D], strides=[H * D, 1], block_shape=[BM, D])
    dk = tl.make_tensor_descriptor(K + hk * D, shape=[L, D], strides=[HK * D, 1], block_shape=[BN, D])
    dv = tl.make_tensor_descriptor(V + hk * D, shape=[L, D], strides=[HK * D, 1], block_shape=[BN, D])
    do = tl.make_tensor_descriptor(OUT + head * D, shape=[W, D], strides=[H * D, 1], block_shape=[BM, D])
    pos = p0 + block * BM + tl.arange(0, BM)
    q = dq.load([block * BM, 0])
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    full = (p0 + block * BM + 1) // BN                  # tiles every row of the block sees whole
    last_pos = p0 + tl.minimum(block * BM + BM, W) - 1
    for t in range(0, full):
        m, l, o = _prompt_tile(q, dk.load([t * BN, 0]), dv.load([t * BN, 0]), m, l, o, (t * BN + tl.arange(0, BN))[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        m, l, o = _prompt_tile(q, dk.load([t * BN, 0]), dv.load([t * BN, 0]), m, l, o, (t * BN + tl.arange(0, BN))[None, :] <= pos[:, None], SCALE)
    do.store([block * BM, 0], (o / l[:, None]).to(tl.bfloat16))


def prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """Prompt attention, the CUDA module's fold with tiles that suit an Arc GPU: chunking changes no bit."""

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    out = torch.empty_like(q) if out is None else out
    _prompt[(triton.cdiv(w, PROMPT_ROWS), h)](q, k_cache, v_cache, out, p0, w, p0 + w, H=h, HK=hk, D=d, BM=PROMPT_ROWS, BN=PROMPT_KEYS,
                                               SCALE=scale, num_warps=PROMPT_WARPS, num_stages=2, grf_mode="256")
    return out


class _MergeLauncher:
    """Stands in for ``attention._merge``: the same call with no fused multiply-add, as the two kernels that feed it."""

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            _cuda_merge[grid](*args, enable_fp_fusion=False, **kwargs)

        return launch


merge = _MergeLauncher()
