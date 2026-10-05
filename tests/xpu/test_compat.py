"""``tensorfold.xpu.install`` points torch.cuda at torch.xpu; run in a child process, since it changes torch.cuda."""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

CHILD = """
import torch
from tensorfold import xpu
xpu.install()
assert xpu.active() and torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == torch.xpu.get_device_name(0)
assert torch.cuda.get_device_capability() == (0, 0)
assert torch.cuda.CUDAGraph is torch.xpu.XPUGraph
torch.cuda.memory._set_allocator_settings("expandable_segments:True")
from tensorfold.cuda import build
build.refuse_old_gpu((12, 0))
try:
    build.load("x", ["x.cu"])
except RuntimeError as exc:
    assert "CUDA extension" in str(exc)
else:
    raise SystemExit("an nvcc build was attempted")
"""


def test_cuda_calls_are_answered_by_the_xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("no Intel XPU")
    done = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_without_an_xpu_install_says_so():
    from tensorfold import xpu

    if xpu.available():
        pytest.skip("this host has an XPU")
    with pytest.raises(RuntimeError, match="no Intel XPU"):
        xpu.install()
