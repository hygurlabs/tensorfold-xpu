"""GPTQ 4-bit on an Intel XPU: the format check and admission bytes (any host), the int4 projection (an XPU)."""

import json

import pytest

torch = pytest.importorskip("torch")

from tensorfold.xpu import gptq  # noqa: E402


def _config(tmp_path, **block):
    base = {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False, "sym": True}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {**base, **block}}))
    return tmp_path


def test_reads_the_gptq_block(tmp_path):
    assert gptq.quant_config(_config(tmp_path))["group_size"] == 128


def test_other_formats_are_not_gptq(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    assert gptq.quant_config(tmp_path) is None


@pytest.mark.parametrize("change", [{"bits": 8}, {"sym": False}, {"desc_act": True}, {"group_size": 48}, {"group_size": 0}])
def test_refuses_packs_it_cannot_read(tmp_path, change):
    with pytest.raises(ValueError, match="symmetric 4-bit GPTQ"):
        gptq.require_supported(gptq.quant_config(_config(tmp_path, **change)))


def test_admission_counts_what_loads():
    transform = gptq.weight_transform(".")
    info = {"dtype": "I32", "shape": [640, 17408]}
    assert transform("model.language_model.layers.0.mlp.gate_proj.qweight", info) == (640 * 17408 * 4, 0)
    for dropped in ("x.qzeros", "x.g_idx", "mtp.fc.weight", "model.visual.blocks.0.weight"):
        assert transform(dropped, info) == (0, 0)
    head = {"dtype": "F16", "shape": [10, 4]}
    assert transform("lm_head.weight", head) == (160, 0)                          # a drafter copy is counted
    assert gptq.weight_transform(".", one_gpu=True)("lm_head.weight", head) == (80, 0)


def _xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("no Intel XPU")
    pytest.importorskip("vllm_xpu_kernels")


def _pack(k, n, gs=128):
    """A random symmetric GPTQ projection: its words, scales and the float weights they stand for."""

    q = torch.randint(0, 16, (k, n), dtype=torch.int32)
    scales = (torch.rand(k // gs, n) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    words = (q.view(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(dim=1, dtype=torch.int64).to(torch.int32)
    return words, scales, (q.float() - 8.0) * scales.float().repeat_interleave(gs, dim=0)


def _linear(k=1024, n=512):
    _xpu()
    words, scales, weight = _pack(k, n)
    zero = torch.tensor([gptq.ZERO_POINT], dtype=torch.int8, device="xpu")
    return gptq.Int4(words.t().contiguous().to("xpu"), scales.to("xpu"), zero, 128), weight


def test_projection_matches_a_dequantized_reference():
    layer, weight = _linear()
    x = torch.randn(5, 1024, dtype=torch.bfloat16)
    out = layer(x.to("xpu")).float().cpu()
    ref = x.float() @ weight
    assert out.dtype == torch.float32 and ((out - ref).abs().max() / ref.abs().max()) < 8e-3


def test_rows_of_one_call_keep_the_bits_of_a_single_row():
    layer, _ = _linear()
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device="xpu")
    whole = layer(x).view(torch.int16)
    for rows in (1, 2, 5, 8):
        assert torch.equal(layer(x[:rows].contiguous()).view(torch.int16), whole[:rows])


def test_more_rows_than_a_call_takes_split_without_changing_bits():
    layer, _ = _linear()
    x = torch.randn(20, 1024, dtype=torch.bfloat16, device="xpu")
    split = torch.cat([layer(x[r:r + 8].contiguous()) for r in range(0, 20, 8)])
    assert torch.equal(layer(x).view(torch.int16), split.view(torch.int16))


def test_prompt_rows_keep_their_bits_whatever_the_chunking():
    layer, _ = _linear()
    x = torch.randn(300, 1024, dtype=torch.bfloat16, device="xpu")
    whole = layer.prefill(x).view(torch.int16)
    for cut in (1, 37, 128, 129, 250):
        parts = torch.cat([layer.prefill(x[:cut].contiguous()), layer.prefill(x[cut:].contiguous())])
        assert torch.equal(parts.view(torch.int16), whole), cut


SHAPES = {"qkv": (5120, 10240), "z": (5120, 6144), "out": (6144, 5120), "q": (5120, 12288), "kv": (5120, 1024),
          "gate": (5120, 17408), "down": (17408, 5120)}


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_row_count_the_engine_uses_is_bit_repeatable(name):
    """The kernel is not repeatable for every row count (192 rows and more on several shapes): the engine stays below."""

    _xpu()
    k, n = SHAPES[name]
    words, scales, _ = _pack(k, n)
    layer = gptq.Int4(words.t().contiguous().to("xpu"), scales.to("xpu"), torch.tensor([8], dtype=torch.int8, device="xpu"), 128)
    for rows in (1, 2, 8, 9, 64, gptq.BLOCK):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="xpu")
        call = layer.prefill if rows == gptq.BLOCK else layer
        first = call(x).view(torch.int16)
        assert all(torch.equal(call(x).view(torch.int16), first) for _ in range(25)), f"{name} at {rows} rows"
