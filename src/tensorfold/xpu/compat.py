"""Answers the CUDA engine's ``torch.cuda`` calls with ``torch.xpu``, so no CUDA module is edited."""

from __future__ import annotations

import types

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
    from tensorfold.cuda.kernels import prefill_attention

    prefill_attention.attention = prefill_attention.triton_attention                    # the Triton kernel defines its bits
    _ACTIVE = True
