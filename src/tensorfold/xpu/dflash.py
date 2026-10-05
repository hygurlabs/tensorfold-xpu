"""DFlash2 drafting on an Intel XPU: the drafter's reference path in bf16, since the target verifies every token."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from tensorfold.families.qwen3_5.cuda import dflash2 as cuda_dflash2
from tensorfold.families.qwen3_5.cuda.glue import embedding

from .gptq import ROWS

SELECTOR_CODEBOOKS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")


def draft_weights(draft_dir):
    """The drafter's bytes as this backend holds them: every tensor in bf16 as stored, nothing packed."""

    from tensorfold.cuda.capacity import Weights, estimate_weights
    from tensorfold.cuda.geometry import size

    held = estimate_weights(draft_dir, lambda name, info: (size(info, name), 0))
    return Weights(held.resident, 0, held.mapped)


class DFlash2(cuda_dflash2.DFlash2):
    """Proposals come from the unpacked bf16 drafter; a rounding that changes a proposal changes speed, never output."""

    def __init__(self, draft_dir, target, bits: int = 4, block: int = 16, fast: bool = True, rank: int = 0,
                 world: int = 1):
        if world != 1:
            raise ValueError("the XPU backend serves one GPU")
        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        self.rank, self.world, self.block = 0, 1, block
        self.hidden = int(cfg["hidden_size"])
        self.head_dim = int(cfg["head_dim"])
        self.heads = int(cfg["num_attention_heads"])
        self.kv_heads = int(cfg["num_key_value_heads"])
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.mask_id = int(cfg["dflash_config"]["mask_token_id"])
        self.trained = int(cfg["dflash_config"].get("block_size", 8))
        self.group_size = int(cfg["dflash_config"]["conv_group_size"])
        self.layers = int(cfg["num_hidden_layers"])
        self.window = int(cfg["sliding_window"]) - 1
        self.is_causal = bool(cfg.get("is_causal", True))
        self.target_embed = target.embed
        self.device = target.norm.device
        self.weights: dict = {}
        packed: dict = {}
        with safe_open(str(path / "model.safetensors"), "pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith((".qweight", ".scales")):       # a pack from ``quantize_drafter``: 4-bit projections
                    packed[name] = tensor.to(self.device)
                    continue
                self.weights[name] = tensor.float().numpy().copy() if name in SELECTOR_CODEBOOKS else tensor.to(self.device)
        self.inv_freq = (1.0 / self.theta ** (torch.arange(self.head_dim // 2, device=self.device, dtype=torch.float32)
                                              * 2 / self.head_dim))
        vocab = target.config.vocab
        self.vocab_spans = tuple((a, min(b, vocab)) for a, b in ((0, 98304), (248032, 248320)) if a < vocab)
        self.head_ids = torch.cat([torch.arange(a, b, device=self.device) for a, b in self.vocab_spans])
        self.head_cols = self.sub_rows = self.sub_parts = None
        self.head_rows = [target.head.weight[a:b] for a, b in self.vocab_spans]      # views of the target's head: no copy
        self.q4: dict = {}
        if packed:
            from .gptq import Int4, ZERO_POINT
            from .quantize_drafter import GROUP

            zero = torch.tensor([ZERO_POINT], dtype=torch.int8, device=self.device)
            for name in [n for n in packed if n.endswith(".qweight")]:
                base = name[:-len(".qweight")]
                self.q4[base + ".weight"] = Int4(packed.pop(name).t().contiguous(), packed.pop(base + ".scales").contiguous(),
                                                 zero, GROUP)
        self.fast = False
        self.heads_local, self.kv_local = self.heads, self.kv_heads
        torch.cuda.empty_cache()
        self.kc = [None] * self.layers
        self.vc = [None] * self.layers
        self.context_len = 0
        self.context_end = 0

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        if name in self.q4:
            proj = self.q4[name]
            return proj(x) if x.shape[0] <= ROWS else proj.prefill(x)      # the drafter is not held to the verify bits
        return F.linear(x, self.weights[name])

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """The reference path of the CUDA module, its head read from the target's own rows."""

        out: list = [None] * len(snaps)
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live or max_nodes < 1:
            return out
        length = min(block or self.block, max_nodes + 1)
        tokens = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                              dtype=torch.int32, device=self.device)
        x = embedding(tokens, self.target_embed)
        hs = []
        for j, i in enumerate(live):
            self.restore(snaps[i])
            y = x[j * length:(j + 1) * length]
            for layer in range(self.layers):
                y = self._layer(layer, y)
            hs.append(cuda_dflash2._norm(y[1:], self.weights["norm.weight"], self.eps))
        h = torch.cat(hs)
        projected = self._lin(h, "candidate_selector.hidden_projection.weight").float()
        logits = torch.cat([self._head(h, rows) for rows in self.head_rows], dim=1)
        values, local_ids = torch.topk(logits.float(), k=16, dim=-1, sorted=False)
        global_ids = self.head_ids[local_ids]
        shared = [global_ids, torch.cat((values, projected), dim=1), None]
        for j, i in enumerate(live):
            out[i] = (shared, j * (length - 1), length - 1, int(pendings[i]))
        return out

    @staticmethod
    def _head(h: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        from .dense import matmul

        return matmul(h.contiguous(), rows)
