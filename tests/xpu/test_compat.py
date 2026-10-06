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


SHORT_CHILD = """
import torch
from tensorfold import xpu
xpu.install()
from tensorfold.cuda.streams import PrefixCache
from tensorfold.families.qwen3_5.cuda import decode, forward
from tensorfold.xpu import gptq

st = object.__new__(forward.State)
st.pos, st.limit, st.rope_delta, st.room, st.conv, st.rec, st.kv = 0, 0, 0, None, [], [], []
assert st.short is False
st.short = True
assert decode.clone_state(st).short is True
short_state, long_state = decode.clone_state(st), object.__new__(forward.State)
long_state.short = False
prompt = list(range(300))
cache = PrefixCache(8)
cache.add(prompt[:100], short_state, None)
cache.add(prompt[:50], long_state, None)
assert cache.longest(prompt)[1] is short_state                       # a short prompt resumes a short prompt's state
longer = list(range(gptq.SHORT_PROMPT + 100))
assert cache.longest(longer)[1] is long_state                        # a long one, a long prompt's
assert PrefixCache(8).longest(longer) is None
"""


def test_a_cached_state_resumes_only_a_prompt_of_its_own_kind():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("no Intel XPU")
    done = subprocess.run([sys.executable, "-c", SHORT_CHILD], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
