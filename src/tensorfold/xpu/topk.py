"""``torch.topk`` on an Intel XPU without its large-slice kernel.

The first call of that kernel in a process compiles for about 38 s (measured: slices of 4,096 and more; 3,072 and below
take 0.04 s). A top-k over a vocabulary is done in two stages instead: the top k of each 2,048-wide block, then the top k
of the blocks' candidates, recursively while they are still wider than a block (k up to 512; wider k keeps the stock kernel). A row's result depends on that row only.
"""

from __future__ import annotations

import torch

BLOCK = 2048
_original = torch.topk


def topk(input: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    if (input.device.type != "xpu" or not largest or not input.is_floating_point() or dim not in (-1, input.ndim - 1)
            or input.shape[-1] <= BLOCK or k > BLOCK // 4):   # a wider k: no shrink per stage
        return _original(input, k, dim, largest, sorted)
    n = input.shape[-1]
    blocks = -(-n // BLOCK)
    wide = torch.nn.functional.pad(input, (0, blocks * BLOCK - n), value=float("-inf")).reshape(-1, blocks, BLOCK)
    values, local = _original(wide, k, -1, True, False)
    ids = (local + torch.arange(blocks, device=input.device)[None, :, None] * BLOCK).reshape(-1, blocks * k)
    values, pick = topk(values.reshape(-1, blocks * k), k, -1, True, sorted)
    ids = ids.gather(-1, pick)
    shape = (*input.shape[:-1], k)
    return torch.return_types.topk((values.reshape(shape), ids.reshape(shape)))


def install() -> None:
    torch.topk = topk
    torch.Tensor.topk = lambda self, k, dim=-1, largest=True, sorted=True: topk(self, k, dim, largest, sorted)
