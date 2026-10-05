import pytest
import torch

pytest.importorskip("vllm_xpu_kernels")
if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("needs an Intel XPU", allow_module_level=True)

from tensorfold.xpu import a8, gptq


@pytest.fixture(autouse=True)
def int8_prompts():
    a8.enable(True)
    yield
    a8.enable(False)


def _layer(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), dtype=torch.int32, generator=g).to("xpu")
    scales = (torch.rand(k // 128, n, generator=g) * 0.02).half().to("xpu")
    zero = torch.tensor([gptq.ZERO_POINT], dtype=torch.int8, device="xpu")
    return gptq.Int4(words, scales, zero, 128), g


def test_the_quantizer_rounds_each_row_by_its_own_largest_magnitude():
    x = torch.randn(7, 1536, device="xpu").half() * torch.arange(1, 8, device="xpu")[:, None]
    q, s = a8.quantized(x)
    assert q.shape[0] % a8.PAD == 0 and q.dtype == torch.int8 and s.shape == (q.shape[0], 1)
    assert torch.equal(q[:7].abs().amax(1), torch.full((7,), 127, dtype=torch.int8, device="xpu"))
    back = q[:7].float() * s[:7]
    assert ((back - x.float()).abs().amax(1) <= s[:7, 0] * 0.5001).all()


def test_the_int8_projection_is_close_to_the_fp16_one():
    layer, g = _layer(1024, 5120, 1)
    x = (torch.randn(300, 5120, generator=g) * 0.5).half().to("xpu")
    a8.enable(False)
    want = layer.prefill(x).float()
    a8.enable(True)
    got = layer.prefill(x).float()
    assert ((got - want).abs().mean() / want.abs().mean()) < 2e-2


@pytest.mark.parametrize("shape", [(17408, 5120), (5120, 17408), (1024, 5120), (5120, 6144)])
def test_int8_prompt_rows_have_the_same_bits_whatever_the_chunking(shape):
    n, k = shape
    layer, g = _layer(n, k, n + k)
    x = torch.randn(2600, k, generator=g).half().to("xpu")
    whole = layer.prefill(x).view(torch.int16)
    for cuts in ((1024, 2048), (700, 1500), (37, 1400, 2000), (128, 640)):
        edges = (0, *cuts, 2600)
        parts = torch.cat([layer.prefill(x[a:b].contiguous()) for a, b in zip(edges, edges[1:])])
        assert torch.equal(parts.view(torch.int16), whole), f"chunks cut at {cuts}"
    padded = torch.full((1024, k), float("nan"), dtype=torch.float16, device="xpu")
    padded[:90] = x[1000:1090]
    assert torch.equal(layer.prefill(padded)[:90].view(torch.int16), whole[1000:1090])
