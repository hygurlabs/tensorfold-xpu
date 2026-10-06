"""Prompt attention on an Intel XPU: against a float32 reference, and a row keeps its bits whatever the chunking."""

import pytest

torch = pytest.importorskip("torch")

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("no Intel XPU", allow_module_level=True)
pytest.importorskip("triton")

from tensorfold import xpu  # noqa: E402

xpu.install()

from tensorfold.xpu.attention import prefill  # noqa: E402

H, HK, D = 24, 4, 256


def _operands(total, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.5).bfloat16().to("xpu")
    return mk(total, H, D), mk(total, HK, D), mk(total, HK, D)


def test_matches_a_float32_reference():
    total, p0 = 200, 70
    q, kc, vc = _operands(total)
    out = prefill(q[p0:].contiguous(), kc, vc, p0, scale=0.0625).float().cpu()
    keys = kc.float().cpu().repeat_interleave(H // HK, dim=1)
    vals = vc.float().cpu().repeat_interleave(H // HK, dim=1)
    att = torch.einsum("whd,shd->hws", q[p0:].float().cpu(), keys) * 0.0625
    mask = torch.arange(total)[None, :] > (p0 + torch.arange(total - p0))[:, None]
    ref = torch.einsum("hws,shd->whd", torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1), vals)
    assert ((out - ref).abs().max() / ref.abs().max()) < 1.5e-2


@pytest.mark.parametrize("cut", [1, 31, 32, 33, 64, 100, 127])
def test_chunking_the_prompt_changes_no_bit(cut):
    total = 260
    q, kc, vc = _operands(total, seed=cut)
    whole = prefill(q, kc, vc, 0, scale=0.0625).view(torch.int16)
    first = prefill(q[:cut].contiguous(), kc, vc, 0, scale=0.0625)
    second = prefill(q[cut:].contiguous(), kc, vc, cut, scale=0.0625)
    assert torch.equal(torch.cat([first, second]).view(torch.int16), whole)


@pytest.mark.parametrize("p0,rows", [(0, 300), (70, 200), (1000, 400)])
def test_the_kernel_gives_the_bits_of_the_cuda_modules_triton_kernel(p0, rows):
    import triton

    from tensorfold.cuda.kernels.prefill_attention import _attend

    q, kc, vc = _operands(p0 + rows, seed=p0)
    mine = prefill(q[p0:].contiguous(), kc, vc, p0, scale=0.0625)
    ref = torch.empty_like(mine)
    _attend[(triton.cdiv(rows, 64), H)](q[p0:].contiguous(), kc, vc, ref, p0, rows, H=H, HK=HK, D=D, BM=64, BN=16, SCALE=0.0625,
                                        num_warps=8, num_stages=1)
    assert torch.equal(mine.view(torch.int16), ref.view(torch.int16))
