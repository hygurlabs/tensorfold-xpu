"""The XPU gated delta rule: a serial step whose bits do not depend on the window, the chunking or the kernel."""

import pytest

torch = pytest.importorskip("torch")

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("no Intel XPU", allow_module_level=True)
pytest.importorskip("triton")

from tensorfold.cuda.kernels import gdn as base  # noqa: E402
from tensorfold.xpu import gdn  # noqa: E402

HK, HV, DK, DV = 2, 4, 128, 40          # DV is not a multiple of the 16 rows a program holds: the tail is masked


def _inputs(rows, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, HK, DK, generator=g_).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(rows, HK, DK, generator=g_), dim=-1).bfloat16()
    v = torch.randn(rows, HV, DV, generator=g_).bfloat16()
    g = torch.rand(rows, HV, generator=g_) * 0.4 + 0.6
    beta = torch.rand(rows, HV, generator=g_)
    state = torch.randn(HV, DV, DK, generator=g_) * 0.1
    return [t.to("xpu") for t in (q, k, v, g, beta, state)]


def _plan(parents):
    entries, starts, slots, rows = base.plan_host([parents])
    device = "xpu"
    return base.Plan(torch.tensor(entries, dtype=torch.int32, device=device).view(-1, 3),
                     torch.tensor(starts, dtype=torch.int32, device=device), slots, rows)


def _reference(q, k, v, g, beta, state):
    q, k, v, g, beta, s = (t.float().cpu() for t in (q, k, v, g, beta, state))
    s = s.clone()
    out = torch.empty(q.shape[0], HV, DV)
    for t in range(q.shape[0]):
        for h in range(HV):
            kh = h // (HV // HK)
            s[h] = s[h] * g[t, h]
            delta = (v[t, h] - s[h] @ k[t, kh]) * beta[t, h]
            s[h] = s[h] + delta[:, None] * k[t, kh][None, :]
            out[t, h] = s[h] @ q[t, kh]
    return out, s


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def test_chain_matches_a_float32_reference():
    q, k, v, g, beta, state = _inputs(12)
    final = torch.empty_like(state)
    out = gdn.chain(q, k, v, g, beta, state, final)
    ref, ref_final = _reference(q, k, v, g, beta, state)
    assert ((out.float().cpu() - ref).abs().max() / ref.abs().max()) < 8e-3
    assert ((final.cpu() - ref_final).abs().max() / ref_final.abs().max()) < 1e-4


def test_chunking_a_chain_keeps_every_bit():
    q, k, v, g, beta, state = _inputs(20)
    whole_final = torch.empty_like(state)
    whole = gdn.chain(q, k, v, g, beta, state, whole_final)
    for cut in (1, 7, 16):
        mid, last = torch.empty_like(state), torch.empty_like(state)
        a = gdn.chain(q[:cut], k[:cut], v[:cut], g[:cut], beta[:cut], state, mid)
        b = gdn.chain(q[cut:].contiguous(), k[cut:].contiguous(), v[cut:].contiguous(), g[cut:].contiguous(),
                      beta[cut:].contiguous(), mid, last)
        assert torch.equal(_bits(torch.cat([a, b])), _bits(whole))
        assert torch.equal(_bits(last), _bits(whole_final))


def test_a_window_of_a_chain_equals_the_prefill_chain():
    q, k, v, g, beta, state = _inputs(9)
    saved = state.clone()
    window = gdn.tree(q, k, v, g, beta, _plan([-1] + list(range(8))), state=state)
    chained = gdn.chain(q, k, v, g, beta, state, torch.empty_like(state))
    assert torch.equal(_bits(window), _bits(chained))
    assert torch.equal(state, saved)                              # a verify never touches the committed state


def test_each_node_of_a_tree_equals_the_chain_along_its_path():
    parents = [-1, 0, 0, 1, 1, 2, 4, 6]
    q, k, v, g, beta, state = _inputs(len(parents))
    out = gdn.tree(q, k, v, g, beta, _plan(parents), state=state)
    for node in range(len(parents)):
        path = [node]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        path.reverse()
        idx = torch.tensor(path, device="xpu")
        alone = gdn.chain(q[idx].contiguous(), k[idx].contiguous(), v[idx].contiguous(), g[idx].contiguous(),
                          beta[idx].contiguous(), state, torch.empty_like(state))
        assert torch.equal(_bits(out[node]), _bits(alone[-1])), f"node {node}"


def test_replaying_the_accepted_rows_equals_the_chain_state():
    q, k, v, g, beta, state = _inputs(10)
    path = [0, 1, 3, 4, 7]
    idx = torch.tensor(path, device="xpu")
    final = torch.empty_like(state)
    gdn.chain(q[idx].contiguous(), k[idx].contiguous(), v[idx].contiguous(), g[idx].contiguous(),
              beta[idx].contiguous(), state, final)
    table = gdn.replay_table([k], [v], [g], [beta], [[state.clone()]])
    rows = torch.tensor([path + [0] * 5], dtype=torch.int32, device="xpu")
    counts = torch.tensor([len(path)], dtype=torch.int32, device="xpu")
    out = gdn.replay(table, 1, 1, rows, counts, k, v)
    assert torch.equal(_bits(out[0, 0]), _bits(final))
    inplace = state.clone()
    gdn.replay(gdn.replay_table([k], [v], [g], [beta], [[inplace]]), 1, 1, rows, counts, k, v, in_place=True)
    assert torch.equal(_bits(inplace), _bits(final))


PARENT_SETS = ([-1, 0, 1, 2], [-1, 0, 0, 1, 1, 2], [-1], [-1, 0, 0, 0, 3, 3, 1])


def _multi(windows, order):
    """The streams' windows in one call over a table of their states' addresses, in the stream order ``order``."""

    inputs = [_inputs(len(parents), seed=11 + i) for i, parents in enumerate(windows)]
    entries, starts, slots, rows = base.plan_host([windows[i] for i in order])
    plan = base.Plan(torch.tensor(entries, dtype=torch.int32, device="xpu").view(-1, 3),
                     torch.tensor(starts, dtype=torch.int32, device="xpu"), slots, rows)
    cat = lambda n: torch.cat([inputs[i][n] for i in order])
    table = torch.tensor(gdn.pointers([inputs[i][5] for i in order]), dtype=torch.int64, device="xpu")
    got = gdn.tree(cat(0), cat(1), cat(2), cat(3), cat(4), plan, table=table)
    return inputs, got


def test_streams_walked_together_equal_each_walked_alone():
    inputs, got = _multi(PARENT_SETS, range(len(PARENT_SETS)))
    row = 0
    for parents, (q, k, v, g, beta, state) in zip(PARENT_SETS, inputs):
        alone = gdn.tree(q, k, v, g, beta, _plan(parents), state=state)
        assert torch.equal(got[row:row + len(parents)].view(torch.int16), alone.view(torch.int16))
        row += len(parents)


def test_a_stream_has_the_same_bits_in_any_place_among_the_others():
    _, a = _multi(PARENT_SETS, [0, 1, 2, 3])
    _, b = _multi(PARENT_SETS, [3, 1, 0, 2])
    sizes = [len(p) for p in PARENT_SETS]
    where = lambda order, i: sum(sizes[j] for j in order[:order.index(i)])
    for i, n in enumerate(sizes):
        x, y = where([0, 1, 2, 3], i), where([3, 1, 0, 2], i)
        assert torch.equal(a[x:x + n].view(torch.int16), b[y:y + n].view(torch.int16))
