"""Gated delta rule on an Intel XPU in Triton: one serial step shared by chains, trees and replays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

BV = 4           # value rows a program holds; a row's bits never depend on it (measured: one checksum from 4 to 32)
WARPS = 1


@triton.jit
def _step(s, k, v, g, beta):
    """One node from its parent's state ``s`` (BV, DK): decay, read the memory, write the delta."""

    s = s * g
    mem = tl.sum(s * k[None, :], axis=1)
    delta = (v - mem) * beta
    return s + k[None, :] * delta[:, None]


@triton.jit
def _inputs(Q, K, V, G, BETA, node, kh, head, rows, ok, dk, HK: tl.constexpr, HV: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr):
    q = tl.load(Q + (node * HK + kh) * DK + dk).to(tl.float32)
    k = tl.load(K + (node * HK + kh) * DK + dk).to(tl.float32)
    v = tl.load(V + (node * HV + head) * DV + rows, mask=ok, other=0.0).to(tl.float32)
    return q, k, v, tl.load(G + node * HV + head), tl.load(BETA + node * HV + head)


@triton.jit
def _walk(Q, K, V, G, BETA, STATE, FINAL, SLOTS, ENTRIES, Y, W, STARTS, TABLE, ORIGIN, SLOT_STRIDE, HK: tl.constexpr,
          HV: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr, BV: tl.constexpr, CHAIN: tl.constexpr,
          HAS_FINAL: tl.constexpr, MULTI: tl.constexpr):
    """Each step's inputs are loaded while the step before computes: the chain is one load latency a step otherwise.

    ``MULTI``: program z is stream z, its entries STARTS[z]..STARTS[z + 1], its state at ORIGIN + TABLE[z] (elements),
    its slots a SLOT_STRIDE-element region of SLOTS: each stream walks as it would alone."""

    vb, head = tl.program_id(0), tl.program_id(1)
    if MULTI:
        stream = tl.program_id(2)
        first = tl.load(STARTS + stream)
        count = tl.load(STARTS + stream + 1) - first
        state_ptr = ORIGIN + tl.load(TABLE + stream)
        slots_ptr = SLOTS + stream.to(tl.int64) * SLOT_STRIDE
    else:
        first = tl.program_id(0) * 0
        count = W
        state_ptr = STATE
        slots_ptr = SLOTS
    rows = vb * BV + tl.arange(0, BV)
    ok = rows < DV
    dk = tl.arange(0, DK)
    kh = head // (HV // HK)
    init = tl.load(state_ptr + (head * DV + rows[:, None]) * DK + dk[None, :], mask=ok[:, None], other=0.0)
    cur = init
    if CHAIN:
        node_n = tl.program_id(0) * 0
        source_n = node_n
        dest_n = node_n
    else:
        node_n = tl.load(ENTRIES + 3 * first)
        source_n = tl.load(ENTRIES + 3 * first + 1)
        dest_n = tl.load(ENTRIES + 3 * first + 2)
    q_n, k_n, v_n, g_n, beta_n = _inputs(Q, K, V, G, BETA, node_n, kh, head, rows, ok, dk, HK, HV, DV, DK)
    for e in range(0, count):
        node, source, dest = node_n, source_n, dest_n
        q, k, v, g, beta = q_n, k_n, v_n, g_n, beta_n
        ahead = tl.minimum(e + 1, count - 1)
        if CHAIN:
            node_n = ahead
        else:
            node_n = tl.load(ENTRIES + 3 * (first + ahead))
            source_n = tl.load(ENTRIES + 3 * (first + ahead) + 1)
            dest_n = tl.load(ENTRIES + 3 * (first + ahead) + 2)
        q_n, k_n, v_n, g_n, beta_n = _inputs(Q, K, V, G, BETA, node_n, kh, head, rows, ok, dk, HK, HV, DV, DK)
        if CHAIN:
            s = tl.where(e == 0, init, cur)
        else:
            slot = tl.load(slots_ptr + ((tl.maximum(source, 0) * HV + head) * DV + rows[:, None]) * DK + dk[None, :],
                           mask=ok[:, None] & (source >= 0), other=0.0)
            s = tl.where(source == -1, init, tl.where(source >= 0, slot, cur))
        s = _step(s, k, v, g, beta)
        y = tl.sum(s * q[None, :], axis=1)
        tl.store(Y + (node * HV + head) * DV + rows, y.to(tl.bfloat16), mask=ok)
        if not CHAIN:
            tl.store(slots_ptr + ((tl.maximum(dest, 0) * HV + head) * DV + rows[:, None]) * DK + dk[None, :], s,
                     mask=ok[:, None] & (dest >= 0))
        cur = s
    if HAS_FINAL:
        tl.store(FINAL + (head * DV + rows[:, None]) * DK + dk[None, :], cur, mask=ok[:, None])


@triton.jit
def _replay(K, V, G, BETA, STATE, OUT, ROWS, COUNT, HK: tl.constexpr, HV: tl.constexpr, DV: tl.constexpr,
            DK: tl.constexpr, BV: tl.constexpr):
    vb, head = tl.program_id(0), tl.program_id(1)
    rows = vb * BV + tl.arange(0, BV)
    ok = rows < DV
    dk = tl.arange(0, DK)
    kh = head // (HV // HK)
    at = (head * DV + rows[:, None]) * DK + dk[None, :]
    cur = tl.load(STATE + at, mask=ok[:, None], other=0.0)
    count = tl.load(COUNT)
    node_n = tl.load(ROWS)
    k_n = tl.load(K + (node_n * HK + kh) * DK + dk).to(tl.float32)
    v_n = tl.load(V + (node_n * HV + head) * DV + rows, mask=ok, other=0.0).to(tl.float32)
    g_n = tl.load(G + node_n * HV + head)
    beta_n = tl.load(BETA + node_n * HV + head)
    for i in range(0, count):
        k, v, g, beta = k_n, v_n, g_n, beta_n
        node_n = tl.load(ROWS + tl.minimum(i + 1, count - 1))
        k_n = tl.load(K + (node_n * HK + kh) * DK + dk).to(tl.float32)
        v_n = tl.load(V + (node_n * HV + head) * DV + rows, mask=ok, other=0.0).to(tl.float32)
        g_n = tl.load(G + node_n * HV + head)
        beta_n = tl.load(BETA + node_n * HV + head)
        cur = _step(cur, k, v, g, beta)
    tl.store(OUT + at, cur, mask=ok[:, None])


def _dims(q: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int]:
    nodes, hk, dk = q.shape
    return nodes, hk, v.shape[1], v.shape[2], dk


def _check(*tensors: torch.Tensor) -> None:
    if any(t.device.type != "xpu" or not t.is_contiguous() for t in tensors):
        raise ValueError("the XPU gated delta rule takes contiguous XPU tensors")


def chain(q, k, v, g, beta, state, final):
    """Prefill chain from ``state`` (read only) to ``final``; each row is the serial step, whatever the chunking."""

    _check(q, k, v, g, beta, state, final)
    nodes, hk, hv, dv, dk = _dims(q, v)
    y = torch.empty((nodes, hv, dv), dtype=torch.bfloat16, device=q.device)
    if nodes:
        _walk[(triton.cdiv(dv, BV), hv)](q, k, v, g, beta, state, final, state, state, y, nodes, state, state, state, 0,
                                         HK=hk, HV=hv, DV=dv, DK=dk, BV=BV, CHAIN=True, HAS_FINAL=True, MULTI=False,
                                         num_warps=WARPS, enable_fp_fusion=False)
    else:
        final.copy_(state)
    return y


# the plan helpers are pure Python and torch: the CUDA module's own are kept
def tree(q, k, v, g, beta, p, state=None, table=None, pending=None, final=None):
    """Outputs (W, Hv, Dv) bf16 for a window's nodes, each from its ancestors' states; no state is changed. One stream's
    ``state``, or a ``table`` of every stream's state address (one device int64 a stream) with ``p.starts``' row ranges."""

    if pending is not None or final is not None:
        raise NotImplementedError("the XPU tree takes no pending rows or final state yet")
    if (state is None) == (table is None):
        raise ValueError("pass one stream's state or a table of every stream's")
    _check(q, k, v, g, beta, p.entries)
    nodes, hk, hv, dv, dk = _dims(q, v)
    y = torch.empty((nodes, hv, dv), dtype=torch.bfloat16, device=q.device)
    slots_each = max(p.slots, 1)
    if state is not None:
        _check(state)
        slots = torch.empty((slots_each, hv, dv, dk), dtype=torch.float32, device=q.device)
        _walk[(triton.cdiv(dv, BV), hv)](q, k, v, g, beta, state, state, slots, p.entries, y, nodes, state, state, state, 0,
                                         HK=hk, HV=hv, DV=dv, DK=dk, BV=BV, CHAIN=False, HAS_FINAL=False, MULTI=False,
                                         num_warps=WARPS, enable_fp_fusion=False)
        return y
    streams = int(p.starts.shape[0]) - 1
    origin = _origin(q.device)
    at = table.reshape(-1).div(4, rounding_mode="floor")                  # byte offsets from the origin to fp32 elements
    slots = torch.empty((streams * slots_each, hv, dv, dk), dtype=torch.float32, device=q.device)
    _walk[(triton.cdiv(dv, BV), hv, streams)](q, k, v, g, beta, origin, origin, slots, p.entries, y, nodes, p.starts, at,
                                              origin, slots_each * hv * dv * dk, HK=hk, HV=hv, DV=dv, DK=dk, BV=BV,
                                              CHAIN=False, HAS_FINAL=False, MULTI=True, num_warps=WARPS,
                                              enable_fp_fusion=False)
    return y


def _origin(device) -> torch.Tensor:
    """The fixed fp32 tensor every state's address is measured from (the attention's own origin, viewed as fp32)."""

    from tensorfold.cuda.kernels import attention as tree_attention

    return tree_attention.base(device).view(torch.float32)


@dataclass
class Table:
    """What the CUDA kernel reads through a table of device addresses, held as tensors here."""

    k: Sequence[torch.Tensor]
    v: Sequence[torch.Tensor]
    g: Sequence[torch.Tensor]
    beta: Sequence[torch.Tensor]
    states: Sequence[Sequence[torch.Tensor]]


def replay_table(k, v, g, beta, states) -> Table:
    layers = len(k)
    if not (len(v) == len(g) == len(beta) == layers) or any(len(s) != layers for s in states):
        raise ValueError("one k, v, g, beta per layer and one state per stream and layer")
    return Table(k, v, g, beta, states)


def to_device(values, dtype, device):
    """A ``Table`` stays as it is; a list of integers goes to the device."""

    if isinstance(values, Table):
        return values
    return torch.tensor(values, dtype=dtype).to(device)


def pointers(tensors: Sequence[torch.Tensor]) -> list[int]:
    """Byte offsets from the fixed origin, the XPU's pointer table: its addresses do not fit a signed 64-bit integer."""

    if any(t.device.type != "xpu" or not t.is_contiguous() for t in tensors):
        raise ValueError("pointer tables take contiguous XPU tensors")
    origin = _origin(tensors[0].device).data_ptr()
    return [t.data_ptr() - origin for t in tensors]


def replay(table: Table, layers: int, streams: int, rows: torch.Tensor, counts: torch.Tensor, k0: torch.Tensor,
           v0: torch.Tensor, *, in_place: bool = False):
    """States after each stream's accepted rows: a new (streams, layers, Hv, Dv, DK) tensor, or over the states."""

    hk, hv, dv = k0.shape[1], v0.shape[1], v0.shape[2]
    dk = k0.shape[2]
    out = None if in_place else torch.empty((streams, layers, hv, dv, dk), dtype=torch.float32, device=rows.device)
    for s in range(streams):
        for layer in range(layers):
            state = table.states[s][layer]
            _check(table.k[layer], table.v[layer], table.g[layer], table.beta[layer], state)
            target = state if in_place else out[s, layer]
            _replay[(triton.cdiv(dv, BV), hv)](table.k[layer], table.v[layer], table.g[layer], table.beta[layer], state,
                                               target, rows[s], counts[s:s + 1], HK=hk, HV=hv, DV=dv, DK=dk, BV=BV,
                                               num_warps=WARPS, enable_fp_fusion=False)
    return None if in_place else out
