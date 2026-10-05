"""Tree attention (Triton) on an Intel XPU against a float32 reference: committed keys, then each node's own path."""

import pytest

torch = pytest.importorskip("torch")

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("no Intel XPU", allow_module_level=True)
pytest.importorskip("triton")

from tensorfold import xpu  # noqa: E402

xpu.install()          # the engine's modules call torch.cuda; this file runs only where an XPU answers

from tensorfold.cuda.kernels import attention as ta  # noqa: E402

H, HK, D = 24, 4, 256


def _reference(q, kc, vc, k_nodes, v_nodes, parents, ctx, scale):
    out = torch.empty_like(q, dtype=torch.float32)
    for node in range(len(parents)):
        path = [node]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        path.reverse()
        keys = torch.cat([kc[:ctx].float(), k_nodes[path].float()])            # (n, HK, D)
        vals = torch.cat([vc[:ctx].float(), v_nodes[path].float()])
        for h in range(H):
            kh = h // (H // HK)
            att = torch.softmax((keys[:, kh] @ q[node, h].float()) * scale, dim=0)
            out[node, h] = att @ vals[:, kh]
    return out


@pytest.mark.parametrize("ctx", [0, 37, 600, 2100, 4300])
@pytest.mark.parametrize("parents", [[-1], [-1, 0, 1, 2], [-1, 0, 0, 1, 1, 2, 4]])
def test_matches_a_float32_reference(ctx, parents):
    g = torch.Generator().manual_seed(ctx + len(parents))
    w = len(parents)
    cap = max(ctx, 1) + 8
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    q, k_nodes, v_nodes, kc, vc = mk(w, H, D), mk(w, HK, D), mk(w, HK, D), mk(cap, HK, D), mk(cap, HK, D)
    scale = D ** -0.5
    offs = torch.tensor(ta.offsets([(kc, vc)], "xpu"), dtype=torch.int64).view(1, 2).to("xpu")
    plan = ta.plan([parents], [ctx], H // HK, "xpu")
    out = ta.attention(q, k_nodes, v_nodes, offs, plan, scale=scale).float().cpu()
    ref = _reference(q.cpu(), kc.cpu(), vc.cpu(), k_nodes.cpu(), v_nodes.cpu(), parents, ctx, scale)
    assert ((out - ref).abs().max() / ref.abs().max()) < 1.5e-2


def test_a_row_does_not_depend_on_its_window():
    g = torch.Generator().manual_seed(5)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    ctx = 600
    kc, vc = mk(ctx + 8, HK, D), mk(ctx + 8, HK, D)
    q, k_nodes, v_nodes = mk(5, H, D), mk(5, HK, D), mk(5, HK, D)
    offs = torch.tensor(ta.offsets([(kc, vc)], "xpu"), dtype=torch.int64).view(1, 2).to("xpu")
    whole = ta.attention(q, k_nodes, v_nodes, offs, ta.plan([[-1, 0, 1, 2, 3]], [ctx], H // HK, "xpu"), scale=0.0625)
    alone = ta.attention(q[:1].contiguous(), k_nodes[:1].contiguous(), v_nodes[:1].contiguous(), offs,
                         ta.plan([[-1]], [ctx], H // HK, "xpu"), scale=0.0625)
    assert torch.equal(whole[0].view(torch.int16), alone[0].view(torch.int16))


@pytest.mark.parametrize("ctx", [600, 2100, 4300, 9000])
def test_a_row_does_not_depend_on_its_window_at_any_context(ctx):
    """Every row of a branching window equals the same row alone: the whole-group and chunk paths of the long contexts."""

    g = torch.Generator().manual_seed(ctx)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    parents = [-1, 0, 0, 1, 2, 4, 4, 3, 7, 7, 1, 10, 10, 6, 13, 0]
    w = len(parents)
    kc, vc = mk(ctx + 20, HK, D), mk(ctx + 20, HK, D)
    q, k_nodes, v_nodes = mk(w, H, D), mk(w, HK, D), mk(w, HK, D)
    offs = torch.tensor(ta.offsets([(kc, vc)], "xpu"), dtype=torch.int64).view(1, 2).to("xpu")
    whole = ta.attention(q, k_nodes, v_nodes, offs, ta.plan([parents], [ctx], H // HK, "xpu"), scale=0.0625)
    for node in range(w):
        path = [node]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        path.reverse()
        local = [-1] + list(range(len(path) - 1))
        idx = torch.tensor(path, device="xpu")
        alone = ta.attention(q[idx].contiguous(), k_nodes[idx].contiguous(), v_nodes[idx].contiguous(), offs,
                             ta.plan([local], [ctx], H // HK, "xpu"), scale=0.0625)
        assert torch.equal(whole[node].view(torch.int16), alone[-1].view(torch.int16)), f"node {node} at context {ctx}"


@pytest.mark.parametrize("ctx", [600, 4300, 9000])
@pytest.mark.parametrize("behind", [1, 3, 31, 33, 70])
def test_a_row_does_not_depend_on_how_many_tokens_were_committed_before_its_round(ctx, behind):
    """The same keys, committed or still on the path: serial decoding and a wide round must give one row."""

    g = torch.Generator().manual_seed(ctx + behind)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    kc, vc = mk(ctx + 1, HK, D), mk(ctx + 1, HK, D)
    q = mk(behind + 1, H, D)
    offs = torch.tensor(ta.offsets([(kc, vc)], "xpu"), dtype=torch.int64).view(1, 2).to("xpu")
    alone = ta.attention(q[-1:].contiguous(), kc[ctx:ctx + 1].contiguous(), vc[ctx:ctx + 1].contiguous(), offs,
                         ta.plan([[-1]], [ctx], H // HK, "xpu"), scale=0.0625)
    chain = ta.attention(q, kc[ctx - behind:].contiguous(), vc[ctx - behind:].contiguous(), offs,
                         ta.plan([[-1] + list(range(behind))], [ctx - behind], H // HK, "xpu"), scale=0.0625)
    assert torch.equal(alone[0].view(torch.int16), chain[-1].view(torch.int16))
