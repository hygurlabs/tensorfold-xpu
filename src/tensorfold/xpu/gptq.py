"""GPTQ 4-bit checkpoints on an Intel XPU: the format check, admission bytes, the loader and the int4 projection."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

ZERO_POINT = 8        # a symmetric pack stores 7, read as 7 + 1
ROWS = 16             # rows of every decode and verify call, padded to this many: one kernel class, one read of the weights
DECODE_ROWS = ROWS    # ``16 * --parallel``: one class for every decode call of a server, a lone stream's included
CLASSES = (16, 32, 48, 64)   # the decode sizes whose bits do not depend on a row's place or its neighbours (tests/xpu)
BLOCK = 128           # rows of every prompt call of a shape not listed below: the kernel is bit-repeatable here (192 and 256 are not)
# Rows of the prompt calls of the wide projections, by (N, K). Within one of these sets a row has the same bits at any block
# size, in any position of the block and beside any padding (tests/xpu/test_gptq.py), and the kernel runs at 125-142 TFLOPs
# against 85-110 at 128 rows. Not listed: the MLP's down projection, whose bits depend on the row's place in a block beyond
# 128 rows, and every shape of another model.
# The MLP's down projection: a row's bits depend on its place in the block (at 127 TFLOPs, against 89 at 128 rows), but they
# are the same for the same place of the same grid. Its blocks sit on a fixed grid of absolute positions, so a row has one
# place whatever the chunking; the chunk that starts mid-block pads the front (``anchor``).
ANCHORED = {(5120, 17408): 1024}
PROMPT_BLOCKS = {(17408, 5120): (512, 1024), (10240, 5120): (512, 1024), (6144, 5120): (512, 1024),
                 (5120, 6144): (512, 1024), (12288, 5120): (512, 1024), (1024, 5120): (1024,)}


_anchor = [0]


@contextmanager
def anchor(position: int):
    """The absolute position of the first row of the prompt calls made inside: where an anchored shape's grid starts."""

    before, _anchor[0] = _anchor[0], int(position)
    try:
        yield
    finally:
        _anchor[0] = before


def set_decode_rows(streams: int) -> None:
    """Every decode call of this process takes ``16 * streams`` rows, so a stream's bits are the same alone or with others."""

    global DECODE_ROWS
    rows = ROWS * int(streams)
    if rows not in CLASSES:
        raise ValueError(f"--parallel on the XPU backend takes 1 to {CLASSES[-1] // ROWS} streams, not {streams}")
    DECODE_ROWS = rows


def quant_config(model_dir: str | Path) -> dict | None:
    """The checkpoint's GPTQ block, or None for any other format."""

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    block = raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config")
    if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "gptq":
        return block
    return None


def require_supported(block: dict) -> None:
    """Refuse a GPTQ pack this backend cannot read, before any weight loads."""

    group = int(block.get("group_size", 0))
    if int(block.get("bits", 0)) != 4 or block.get("sym") is False or block.get("desc_act") or group % 32 or group <= 0:
        raise ValueError("the XPU backend reads symmetric 4-bit GPTQ without act-order, in groups that are a multiple of "
                         f"32 (this checkpoint: bits={block.get('bits')}, sym={block.get('sym')}, "
                         f"desc_act={block.get('desc_act')}, group_size={block.get('group_size')})")


def weight_transform(model_dir, *, one_gpu: bool = False):
    """Admission bytes of each checkpoint tensor as loaded: words and scales kept, zero points and indices dropped."""

    from tensorfold.cuda.geometry import size

    def transform(name, info):
        if ".visual." in name or name.startswith(("model.visual.", "vision_tower", "mtp.")) or ".mtp." in name:
            return 0, 0
        if name.endswith((".g_idx", ".qzeros")):
            return 0, 0
        return size(info, name) * (2 if "lm_head." in name and not one_gpu else 1), 0
    return transform


def _kernels() -> None:
    import vllm_xpu_kernels._C  # noqa: F401  registers the ops used below
    import vllm_xpu_kernels._xpu_C  # noqa: F401


@dataclass
class Int4:
    """A GPTQ projection on ``int4_gemm_w4a16``: fp16 activations, fp32 sums, at most ``ROWS`` rows a call.

    A call returns what the kernel returns, fp16, for the glue kernels that read it; ``out`` bf16 converts it, for the rows
    torch ops or the attention read as bf16."""

    words: torch.Tensor    # (N, K/8) int32: the checkpoint's qweight, transposed
    scales: torch.Tensor   # (K/gs, N) fp16
    zero: torch.Tensor     # (1,) int8
    gs: int
    layout: str = "gptq"
    bits: int = 4
    fast: bool = False     # no FP8 prompt kernel on this backend
    out: torch.dtype = torch.float16
    rows: int = 0          # decode call size; 0: ``DECODE_ROWS``

    @property
    def n(self) -> int:
        return int(self.words.shape[0])

    @property
    def k(self) -> int:
        return int(self.words.shape[1]) * 8

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.words, self.scales, self.zero))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Decode and verify rows in calls of exactly ``ROWS`` rows: a row has the same bits at any width of window."""

        return self._blocks(x, (self.rows or DECODE_ROWS,), self.out)

    def _blocks(self, x: torch.Tensor, sizes: tuple[int, ...], out: torch.dtype) -> torch.Tensor:
        """Rows in calls of the ``sizes`` (ascending), the last one padded with whatever the buffer holds: rows do not meet."""

        _kernels()
        rows = x.shape[0]
        blocks, left = [], rows
        while left > sizes[-1]:
            blocks.append(sizes[-1])
            left -= sizes[-1]
        if left:
            blocks.append(next(b for b in sizes if b >= left))
        padded = sum(blocks)
        if x.dtype == torch.float16 and x.is_contiguous() and (
                padded == rows or (x.storage_offset() == 0 and x.untyped_storage().nbytes() >= padded * self.k * 2)):
            x16 = x if padded == rows else x.as_strided((padded, self.k), (self.k, 1))   # the glue's padded buffer, as it lies
        else:
            x16 = torch.empty((padded, self.k), dtype=torch.float16, device=x.device)
            x16[:rows].copy_(x)                                  # the conversion; the padding rows are never read back
        parts, r0 = [], 0
        for size in blocks:
            parts.append(torch.ops._xpu_C.int4_gemm_w4a16(x16[r0:r0 + size], self.words.t(), None, self.scales, self.zero,
                                                          self.gs, None))
            r0 += size
        y = (parts[0] if len(parts) == 1 else torch.cat(parts))[:rows]
        return y if out == torch.float16 else y.to(out)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Prompt rows in calls of the shape's block sizes: any chunking gives a row the same bits."""

        from . import a8

        if a8.runs(self.n, self.k):
            return self._prefill8(x)
        cell = ANCHORED.get((self.n, self.k))
        if cell:
            return self._on_grid(x, cell)
        return self._blocks(x, PROMPT_BLOCKS.get((self.n, self.k), (BLOCK,)), self.out)

    def _prefill8(self, x: torch.Tensor) -> torch.Tensor:
        """Prompt rows as int8 on ``int4_gemm_w4a8`` (``a8``): integer sums, so any block plan gives a row the same bits."""

        from . import a8

        _kernels()
        rows = x.shape[0]
        q, scale = a8.quantized(x)
        sizes = (1024,) if self.n == 1024 else (512, 1024)      # the k/v projection has one class from 1,024 rows
        blocks, left = [], rows
        while left > sizes[-1]:
            blocks.append(sizes[-1])
            left -= sizes[-1]
        if left:
            blocks.append(next(b for b in sizes if b >= left))
        zp = a8.zero_points(x.device, sum(blocks))
        parts, r0 = [], 0
        for size in blocks:
            parts.append(torch.ops._xpu_C.int4_gemm_w4a8(q[r0:r0 + size], scale[r0:r0 + size], zp[r0:r0 + size],
                                                         self.words.t(), self.scales, self.zero, self.gs, None, None))
            r0 += size
        y = (parts[0] if len(parts) == 1 else torch.cat(parts))[:rows]
        return y if self.out == torch.float16 else y.to(self.out)

    def _on_grid(self, x: torch.Tensor, cell: int) -> torch.Tensor:
        """Rows in calls of ``cell`` rows on the grid of absolute positions: a block's unused rows are padding."""

        _kernels()
        rows, at = x.shape[0], _anchor[0] % cell
        x16 = x if x.dtype == torch.float16 and x.is_contiguous() else x.to(torch.float16).contiguous()
        parts, r0 = [], 0
        while r0 < rows:
            take = min(rows - r0, cell - at)
            if at == 0 and x16.untyped_storage().nbytes() >= (x16.storage_offset() // self.k + r0 + cell) * self.k * 2:
                block = x16.as_strided((cell, self.k), (self.k, 1), x16.storage_offset() + r0 * self.k)   # as it lies
            else:
                block = torch.empty((cell, self.k), dtype=torch.float16, device=x.device)
                block[at:at + take].copy_(x16[r0:r0 + take])
            parts.append(torch.ops._xpu_C.int4_gemm_w4a16(block, self.words.t(), None, self.scales, self.zero, self.gs,
                                                          None)[at:at + take])
            r0, at = r0 + take, 0
        y = parts[0] if len(parts) == 1 else torch.cat(parts)
        return y if self.out == torch.float16 else y.to(self.out)


@dataclass
class Dense:
    """A projection the pack leaves unquantized (head, GDN gates), fp16 as stored."""

    weight: torch.Tensor   # (N, K)
    layout: str = "dense"
    fast: bool = False

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from .dense import matmul

        return matmul(x.contiguous(), self.weight)

    prefill = __call__


class _Reader:
    """Checkpoint tensors read one at a time, so the weights never sit in device memory twice."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from safetensors import safe_open

        self.device, self._open, self._handles = device, safe_open, {}
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            self.where = {k: model_dir / f for k, f in json.loads(index.read_text())["weight_map"].items()}
        else:
            self.where = {}
            for path in sorted(model_dir.glob("*.safetensors")):
                with safe_open(str(path), "pt") as f:
                    self.where.update({k: path for k in f.keys()})
        self.where = {k: v for k, v in self.where.items() if not _skipped(k)}

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def pop(self, name: str) -> torch.Tensor:
        path = self.where.pop(name)
        handle = self._handles.get(path) or self._handles.setdefault(path, self._open(str(path), "pt"))
        return handle.get_tensor(name).to(self.device)

    def left(self) -> list[str]:
        return list(self.where)


def _skipped(name: str) -> bool:
    return ".visual." in name or name.startswith(("model.visual.", "vision_tower", "mtp.")) or ".mtp." in name


def load_gptq(model_dir: str | Path, device: str = "xpu"):
    """The ``Weights`` of a GPTQ Qwen3.8 dense checkpoint, projections as ``Int4``, the rest as stored."""

    from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, QLinear, Weights

    model_dir = Path(model_dir)
    block = quant_config(model_dir)
    require_supported(block)
    gs = int(block["group_size"])
    cfg = Config.read(model_dir)
    t = _Reader(model_dir, device)
    zero = torch.tensor([ZERO_POINT], dtype=torch.int8, device=device)

    def get(name: str) -> torch.Tensor:
        return t.pop(name)

    def int4(name: str, out: torch.dtype = torch.float16) -> Int4:
        words = get(name + ".qweight")
        scales = get(name + ".scales").contiguous()
        get(name + ".qzeros")                  # symmetric: every zero point is 8, checked by the tests
        if name + ".g_idx" in t:
            get(name + ".g_idx")               # act-order is refused above: the indices are the plain group runs
        return Int4(words.t().contiguous(), scales, zero, gs, out=out)

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter writes it."""

        w = get(name)
        return (w.float() + 1.0).to(w.dtype).contiguous()

    def dense(name: str) -> Dense:
        return Dense(get(name + ".weight").contiguous())

    base = "model.language_model."
    layers = []
    for i in range(cfg.layers):
        p = f"{base}layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            q = p + "linear_attn."
            gdn = GDN(qkv=int4(q + "in_proj_qkv", torch.bfloat16), z=int4(q + "in_proj_z"), b=dense(q + "in_proj_b"), a=dense(q + "in_proj_a"),
                      out=int4(q + "out_proj"), conv=get(q + "conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=get(q + "A_log").float().contiguous(), dt_bias=get(q + "dt_bias").float().contiguous(),
                      norm=get(q + "norm.weight").contiguous())
        else:
            q = p + "self_attn."
            attn = Attention(q=int4(q + "q_proj"), k=int4(q + "k_proj"), v=int4(q + "v_proj", torch.bfloat16), o=int4(q + "o_proj"),
                             q_norm=norm(q + "q_norm.weight"), k_norm=norm(q + "k_norm.weight"))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=norm(p + "input_layernorm.weight"),
                            post_norm=norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn,
                            gate=int4(p + "mlp.gate_proj"), up=int4(p + "mlp.up_proj"), down=int4(p + "mlp.down_proj")))
    embed = QLinear(get(base + "embed_tokens.weight").contiguous(), None, None, layout="dense", bits=0, gs=0)
    w = Weights(config=cfg, embed=embed, layers=layers, norm=norm(base + "norm.weight"), head=dense("lm_head"), quant="gptq")
    half = cfg.rope_dims // 2
    w.inv_freq = (cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).to(torch.float32).to(device)
    if t.left():
        raise ValueError(f"unused checkpoint tensors: {t.left()[:5]} ...")
    return w
