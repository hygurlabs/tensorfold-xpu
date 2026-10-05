"""Intel XPU backend: the CUDA engine on an Arc GPU through torch.xpu, with GPTQ 4-bit weights."""

from .compat import active, available, install

__all__ = ("active", "available", "install")
