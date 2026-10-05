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
    x = torch.randn(16, 1024, dtype=torch.bfloat16, device="xpu")
    whole = layer(x).view(torch.int16)
    for rows in (1, 2, 5, 8, 9, 12):
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
    for rows in (1, 2, 8, 9, 12, gptq.ROWS, gptq.ROWS + 1, 64, gptq.BLOCK):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="xpu")
        call = layer.prefill if rows in (64, gptq.BLOCK) else layer
        first = call(x).view(torch.int16)
        assert all(torch.equal(call(x).view(torch.int16), first) for _ in range(25)), f"{name} at {rows} rows"


def test_padding_rows_do_not_reach_the_real_ones():
    """The padded rows hold whatever the buffer had, even NaN: a row's bits are the same."""

    layer, _ = _linear()
    x = torch.randn(5, 1024, dtype=torch.bfloat16, device="xpu")
    reference = layer(x).to(torch.bfloat16).view(torch.int16)
    x16 = torch.full((16, 1024), float("nan"), dtype=torch.float16, device="xpu")
    x16[:5].copy_(x)
    out = torch.ops._xpu_C.int4_gemm_w4a16(x16, layer.words.t(), None, layer.scales, layer.zero, layer.gs, None)[:5]
    assert torch.equal(out.to(torch.bfloat16).view(torch.int16), reference)


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_a_padded_glue_buffer_is_read_where_it_lies_with_the_same_bits(rows):
    """What the glue kernels write (fp16 into a buffer padded to 16 rows, the padding unset) needs no conversion launch."""

    from tensorfold.xpu import glue

    layer, _ = _linear()
    x = torch.randn(rows, 1024, dtype=torch.bfloat16, device="xpu")
    buffer = glue.rows_buffer(rows, 1024, "xpu")
    buffer.copy_(x)
    full = buffer.as_strided((16, 1024), (1024, 1))
    full[rows:] = float("nan")                                  # whatever the allocator left
    assert torch.equal(layer(buffer).view(torch.int16), layer(x).view(torch.int16))
    assert layer(buffer).dtype == torch.float16


def test_the_glue_kernels_do_the_cuda_glues_arithmetic_in_fp16():
    from tensorfold.families.qwen3_5.cuda import glue as cuda_glue
    from tensorfold.xpu import glue

    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 5120, generator=g).bfloat16().to("xpu")
    r = torch.randn(3, 5120, generator=g).bfloat16().to("xpu")
    w = (torch.randn(5120, generator=g) * 0.1 + 1).bfloat16().to("xpu")
    h, y, none = glue.add_rmsnorm(x, r, w, 1e-6)
    h0, y0, _ = cuda_glue.add_rmsnorm(x, r, w, 1e-6)
    assert none is None and torch.equal(h, h0) and y.dtype == torch.float16
    assert ((y.float() - y0.float()).abs().max() / y0.float().abs().max()) < 1e-2
    gate, up = (torch.randn(3, 17408, generator=g).bfloat16().to("xpu") for _ in range(2))
    act = glue.swiglu(gate, up)[0]
    assert act.dtype == torch.float16 and ((act.float() - cuda_glue.swiglu(gate, up)[0].float()).abs().max() < 1e-1)
