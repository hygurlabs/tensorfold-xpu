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


@pytest.mark.parametrize("ctx", [0, 37, 600])
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


@pytest.mark.parametrize("ctx", [0, 5, 70, 511, 512, 600])
def test_the_tail_visiting_fewer_tiles_changes_no_bit(ctx):
    from tensorfold.xpu import attention as xpu_attention

    g = torch.Generator().manual_seed(ctx)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    parents = [-1, 0, 0, 1, 2, 4]
    w = len(parents)
    kc, vc = mk(max(ctx, 1) + 8, HK, D), mk(max(ctx, 1) + 8, HK, D)
    q, k_nodes, v_nodes = mk(w, H, D), mk(w, HK, D), mk(w, HK, D)
    offs = torch.tensor(ta.offsets([(kc, vc)], "xpu"), dtype=torch.int64).view(1, 2).to("xpu")
    plan = ta.plan([parents], [ctx], H // HK, "xpu")
    fast = ta.attention(q, k_nodes, v_nodes, offs, plan, scale=0.0625).view(torch.int16)
    ta._tail = xpu_attention.cuda_tail
    try:
        reference = ta.attention(q, k_nodes, v_nodes, offs, plan, scale=0.0625).view(torch.int16)
    finally:
        ta._tail = xpu_attention._tail
    assert torch.equal(fast, reference)
