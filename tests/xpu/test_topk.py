import pytest
import torch

if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    pytest.skip("needs an Intel XPU", allow_module_level=True)

from tensorfold.xpu import topk as xtopk


@pytest.mark.parametrize("shape,k", [((3, 248320), 20), ((15, 98592), 16), ((2, 5000), 64), ((1, 2049), 5), ((4, 6000), 512)])
@pytest.mark.parametrize("ordered", [True, False])
def test_two_stage_topk_gives_the_stock_result(shape, k, ordered):
    g = torch.Generator().manual_seed(shape[1])
    x = torch.randperm(shape[0] * shape[1], generator=g).reshape(shape).float().to("xpu")      # distinct values: no ties
    values, ids = xtopk.topk(x, k, -1, True, ordered)
    want_values, want_ids = xtopk._original(x, k, -1, True, True)
    if not ordered:
        order = values.argsort(dim=-1, descending=True)
        values, ids = values.gather(-1, order), ids.gather(-1, order)
    assert torch.equal(values, want_values) and torch.equal(ids, want_ids)
    assert torch.equal(x.gather(-1, ids), values)


def test_narrow_and_other_inputs_go_to_the_stock_kernel():
    x = torch.randn(4, 1000, device="xpu")
    assert torch.equal(xtopk.topk(x, 8)[1], xtopk._original(x, 8)[1])
    assert torch.equal(xtopk.topk(x, 8, largest=False)[1], xtopk._original(x, 8, largest=False)[1])


def test_a_wide_k_keeps_the_stock_kernel():
    x = torch.randn(2, 5000, device="xpu")
    assert torch.equal(xtopk.topk(x, 2048)[1], xtopk._original(x, 2048)[1])
