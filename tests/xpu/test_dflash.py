"""DFlash2 on an Intel XPU: the drafter's byte count, and (with the weights at hand) output equal to the serial reference."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from tensorfold.xpu import dflash  # noqa: E402


def test_the_drafters_bytes_are_those_stored(tmp_path):
    save_file({"layers.0.mlp.gate_proj.weight": torch.zeros(64, 128, dtype=torch.bfloat16),
               "norm.weight": torch.zeros(128, dtype=torch.bfloat16)}, str(tmp_path / "model.safetensors"))
    held = dflash.draft_weights(tmp_path)
    assert held.resident == 64 * 128 * 2 + 128 * 2 and held.staging == 0


MODEL, DRAFTER = os.environ.get("TENSORFOLD_XPU_TEST_MODEL"), os.environ.get("TENSORFOLD_XPU_TEST_DRAFTER")


@pytest.mark.skipif(not (MODEL and DRAFTER), reason="set TENSORFOLD_XPU_TEST_MODEL and TENSORFOLD_XPU_TEST_DRAFTER")
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_drafted_replies_equal_serial_ones(temperature):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("no Intel XPU")
    from tokenizers import Tokenizer

    from tensorfold import xpu
    from tensorfold.engine.exact_sampling import Sampling

    xpu.install()
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    engine = Qwen27Engine(Path(MODEL), Path(DRAFTER), max_rows=16, tree_rows=16, allow_copy=True, streams=1, context=4096)
    ids = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json")).encode(
        "Write a Python class LRUCache with get and put methods, using an OrderedDict.").ids

    def reply(draft):
        out = []
        engine.cache.__init__(4)
        engine.generate(list(ids), 64, Sampling(seed=7, temperature=temperature, top_k=20 if temperature else 1, top_p=0.95),
                        lambda t: (out.extend(t), len(out) >= 64)[1], draft=draft)
        return out[:64]

    assert reply(True) == reply(False)
