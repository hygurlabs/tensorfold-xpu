"""Answers the CUDA engine's ``torch.cuda`` calls with ``torch.xpu``, so no CUDA module is edited."""

from __future__ import annotations

import types

TREE_CHUNK = 128         # keys of a tree-attention chunk: the last, partial chunk is a sequential tail, so a short one is faster
PROMPT_CHUNK = 1024      # rows of a prompt chunk: the largest block of the wide projections

# torch.xpu has each of these under the same name, with the same meaning
_SAME = ("synchronize", "empty_cache", "Event", "Stream", "current_stream", "stream", "set_device", "current_device",
         "device", "device_count", "memory_allocated", "memory_reserved", "max_memory_allocated", "reset_peak_memory_stats",
         "mem_get_info", "get_device_name", "get_device_properties", "graph", "graph_pool_handle", "manual_seed")
_ACTIVE = False


def available() -> bool:
    """Whether an Intel XPU is present."""

    try:
        import torch

        return bool(torch.xpu.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def active() -> bool:
    """Whether this process runs the engine on an XPU (``install`` was called)."""

    return _ACTIVE


def install() -> None:
    """Point ``torch.cuda`` at ``torch.xpu``; call once, before any module of the CUDA engine runs."""

    global _ACTIVE
    import torch

    if _ACTIVE:
        return
    if not available():
        raise RuntimeError("no Intel XPU found: install a torch build with XPU support and the Intel GPU driver")
    for name in _SAME:
        setattr(torch.cuda, name, getattr(torch.xpu, name))
    torch.cuda.is_available = torch.xpu.is_available
    torch.cuda.CUDAGraph = torch.xpu.XPUGraph
    # no NVIDIA chip: every capability test in the engine reads "not a known GPU", never an NVIDIA architecture
    torch.cuda.get_device_capability = lambda device=None: (0, 0)
    memory = types.SimpleNamespace(**{n: getattr(torch.xpu.memory, n) for n in dir(torch.xpu.memory) if not n.startswith("__")})
    memory._set_allocator_settings = lambda settings: None    # expandable segments come from PYTORCH_ALLOC_CONF here
    torch.cuda.memory = memory
    from tensorfold.cuda import build

    def no_extension(*args, **kwargs):
        raise RuntimeError("this kernel is a CUDA extension (nvcc); the XPU backend has its own for it")

    build.refuse_old_gpu = lambda need=None: None             # the floor is an NVIDIA compute capability
    build.load = no_extension
    from tensorfold.cuda.kernels import gdn as cuda_gdn
    from tensorfold.xpu import gdn

    for name in ("chain", "tree", "replay", "replay_table", "to_device", "pointers"):    # the CUDA extension's entry points
        setattr(cuda_gdn, name, getattr(gdn, name))
    from tensorfold.cuda.kernels import attention as tree_attention
    from tensorfold.xpu import attention as xpu_attention

    tree_attention.CHUNK, tree_attention.SPAN = TREE_CHUNK, TREE_CHUNK * tree_attention.GROUP
    tree_attention._shared = xpu_attention.shared            # IGC crashes on the CUDA kernel's masked row-indexed store
    tree_attention._tail = xpu_attention.tail                # the same arithmetic, without the empty tiles
    tree_attention._merge = xpu_attention.merge             # no fused multiply-add in any of the three, so the folds agree
    from tensorfold.families.qwen3_5.cuda import affine_memory, dflash2
    from tensorfold.xpu import dflash

    dflash2.DFlash2 = dflash.DFlash2                         # the drafter in bf16: no packed 4-bit lane kernels here
    affine_memory.draft_weights = dflash.draft_weights
    from tensorfold.xpu import topk

    topk.install()                                           # the stock large-slice kernel compiles for 38 s on first use
    from tensorfold.families.qwen3_5.cuda import glue as cuda_glue
    from tensorfold.xpu import glue

    for name in ("add_rmsnorm", "swiglu", "gated_norm", "gate_mul"):    # the rows a 4-bit projection reads, written as fp16
        setattr(cuda_glue, name, getattr(glue, name))
    from tensorfold.cuda.kernels import prefill_attention

    prefill_attention.attention = xpu_attention.prefill                                  # the Triton kernel, tiled for an Arc GPU
    from tensorfold.families.qwen3_5.cuda import prefill as cuda_prefill
    from tensorfold.xpu import gptq

    def chunks(start: int, end: int, size: int = 0) -> list[tuple[int, int]]:
        """Chunks that end on the 1,024-row grid of absolute positions: each is one call of the largest block of a wide
        projection and one block of the down projection's grid, nothing to join; the first and last may be short."""

        edges = [start, *range((start // PROMPT_CHUNK + 1) * PROMPT_CHUNK, end, PROMPT_CHUNK), end]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]

    chunk = cuda_prefill.prefill_chunk

    def prefill_chunk(w, tokens, st, **kwargs):
        with gptq.anchor(st.pos):
            return chunk(w, tokens, st, **kwargs)

    cuda_prefill.prefill_chunk = prefill_chunk
    cuda_prefill.chunks = chunks
    from tensorfold.families.qwen3_5.cuda import multi

    multi.BATCH = False             # queued prompts' steps do not share a forward: the down projection's grid is per prompt
    _ACTIVE = True
