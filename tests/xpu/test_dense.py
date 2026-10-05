"""The XPU dense projection: a row's bits do not depend on the row count or on the run."""

import pytest

torch = pytest.importorskip("torch")

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("no Intel XPU", allow_module_level=True)
pytest.importorskip("triton")

from tensorfold.xpu.dense import matmul  # noqa: E402


def _operands(rows, n=200, k=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=g).bfloat16().to("xpu")
    w = (torch.randn(n, k, generator=g) * 0.05).half().to("xpu")
    return x, w


def test_matches_a_float32_reference():
    x, w = _operands(7)
    ref = x.float().cpu() @ w.float().cpu().T
    out = matmul(x, w).float().cpu()
    assert ((out - ref).abs().max() / ref.abs().max()) < 8e-3


def test_a_row_keeps_its_bits_whatever_the_row_count():
    x, w = _operands(40)
    whole = matmul(x, w).view(torch.int16)
    for rows in (1, 2, 5, 16, 17, 33):
        assert torch.equal(matmul(x[:rows].contiguous(), w).view(torch.int16), whole[:rows])


def test_the_same_call_gives_the_same_bits_every_time():
    x, w = _operands(108, n=48, k=5120, seed=3)
    first = matmul(x, w).view(torch.int16)
    for _ in range(25):
        assert torch.equal(matmul(x, w).view(torch.int16), first)


def test_a_large_output_count_keeps_the_same_properties():
    x, w = _operands(20, n=5000, k=1024, seed=4)               # beyond the split threshold: one program a slice of outputs
    whole = matmul(x, w).view(torch.int16)
    assert torch.equal(matmul(x[:3].contiguous(), w).view(torch.int16), whole[:3])
    ref = x.float().cpu() @ w.float().cpu().T
    assert ((matmul(x, w).float().cpu() - ref).abs().max() / ref.abs().max()) < 8e-3
