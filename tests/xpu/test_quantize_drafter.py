import pytest
import torch

pytest.importorskip("vllm_xpu_kernels")
if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("needs an Intel XPU", allow_module_level=True)

from tensorfold.xpu import quantize_drafter as qd
from tensorfold.xpu.gptq import Int4, ZERO_POINT


def _int4(words, scales):
    zero = torch.tensor([ZERO_POINT], dtype=torch.int8, device="xpu")
    return Int4(words.t().contiguous(), scales.contiguous(), zero, qd.GROUP)


def test_the_pack_is_what_the_kernel_reads():
    g = torch.Generator().manual_seed(0)
    weight = (torch.randn(384, 512, generator=g) * 0.05).to("xpu")
    words, scales = qd.gptq(weight, None)
    x = (torch.randn(16, 512, generator=g) * 0.5).bfloat16().to("xpu")
    got = _int4(words, scales)(x).float()
    ref = x.float() @ qd.dequantize(words, scales).t()
    assert ((got - ref).abs().max() / ref.abs().max()) < 1e-2


def test_a_calibrated_rounding_beats_round_to_nearest_on_the_rows_it_was_calibrated_on():
    g = torch.Generator().manual_seed(1)
    weight = (torch.randn(256, 512, generator=g) * 0.05).to("xpu")
    mixing = torch.randn(512, 512, generator=g)
    rows = (torch.randn(4096, 512, generator=g) @ mixing * 0.1)           # correlated inputs: the Hessian is not diagonal
    h = qd.hessian(rows, "xpu")
    x = rows.to("xpu")
    exact = x @ weight.t()
    error = lambda words, scales: float(((x @ qd.dequantize(words, scales).t()) - exact).pow(2).mean())
    assert error(*qd.gptq(weight, h)) < 0.9 * error(*qd.gptq(weight, None))
