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

from . import gptq

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
                                                 zero, GROUP, out=torch.bfloat16)
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
            return proj(x) if x.shape[0] <= gptq.DECODE_ROWS else proj.prefill(x)      # the drafter is not held to the verify bits
        return F.linear(x, self.weights[name])

    def _attention(self, layer: int, x: torch.Tensor, ctx: list, length: int, positions: torch.Tensor) -> torch.Tensor:
        """The reference attention over several streams' blocks (``length`` rows each): the projections, norms and rotations
        run once over every row, the scores stay a stream at a time (each has its own context)."""

        w = self.weights
        prefix = f"layers.{layer}.self_attn."
        q = self._lin(x, prefix + "q_proj.weight").view(-1, self.heads, self.head_dim)
        kn = self._lin(x, prefix + "k_proj.weight").view(-1, self.kv_heads, self.head_dim)
        vn = self._lin(x, prefix + "v_proj.weight").view(-1, self.kv_heads, self.head_dim)
        q = cuda_dflash2._rope(cuda_dflash2._norm(q, w[prefix + "q_norm.weight"], self.eps).transpose(0, 1),
                               positions, self.inv_freq)
        kn = cuda_dflash2._rope(cuda_dflash2._norm(kn, w[prefix + "k_norm.weight"], self.eps).transpose(0, 1),
                                positions, self.inv_freq)
        vn = vn.transpose(0, 1)
        qidx = torch.arange(length, device=self.device)[:, None]
        outputs = []
        for j, snap in enumerate(ctx):
            rows = slice(j * length, (j + 1) * length)
            kc, vc = snap[0][layer], snap[1][layer]
            keys = torch.cat((kc, kn[:, rows]), dim=1).unsqueeze(0).repeat_interleave(self.heads // self.kv_heads, dim=1)
            values = torch.cat((vc, vn[:, rows]), dim=1).unsqueeze(0).repeat_interleave(self.heads // self.kv_heads, dim=1)
            s = kc.shape[1]
            kidx = torch.arange(s + length, device=self.device)[None, :]
            allowed = (kidx < s) & (s + qidx - kidx < self.window + 1)
            block_allowed = kidx >= s
            if self.is_causal:
                block_allowed = block_allowed & (kidx <= s + qidx)
            out = F.scaled_dot_product_attention(q[:, rows].unsqueeze(0), keys, values,
                                                 attn_mask=(allowed | block_allowed)[None, None],
                                                 scale=self.head_dim ** -0.5)
            outputs.append(out.squeeze(0).transpose(0, 1).reshape(length, self.heads * self.head_dim))
        return self._lin(torch.cat(outputs) if len(outputs) > 1 else outputs[0], prefix + "o_proj.weight")

    def _layer(self, i: int, x: torch.Tensor, ctx: list, length: int, positions: torch.Tensor) -> torch.Tensor:
        """One layer over several streams' blocks, on the fused kernels the CUDA fast path uses: the convolutions and their
        residual adds in one launch each, the norms in one. Same roundings as the reference layer."""

        w = self.weights
        base = f"layers.{i}."
        normed = F.rms_norm(x, (self.hidden,), w[base + "input_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "attention_conv.kernel_projection.weight")
        conv = w[base + "attention_conv.base_kernel"]
        attended = self._attention(i, cuda_dflash2._dconv(normed, dyn, conv, 0, self.group_size, seg=length), ctx, length,
                                   positions)
        x = cuda_dflash2._dconv(attended, dyn, conv, 1, self.group_size, x, seg=length)
        normed = F.rms_norm(x, (self.hidden,), w[base + "post_attention_layernorm.weight"], self.eps)
        dyn = self._lin(normed, base + "mlp_conv.kernel_projection.weight")
        conv = w[base + "mlp_conv.base_kernel"]
        h = cuda_dflash2._dconv(normed, dyn, conv, 0, self.group_size, seg=length)
        mlp = self._lin(F.silu(self._lin(h, base + "mlp.gate_proj.weight")) * self._lin(h, base + "mlp.up_proj.weight"),
                        base + "mlp.down_proj.weight")
        return cuda_dflash2._dconv(mlp, dyn, conv, 1, self.group_size, x, seg=length)

    def _starts(self, starts: list[int], length: int) -> torch.Tensor:
        """Each stream's ``length`` absolute positions from its start, in one vector."""

        first = torch.tensor(starts, dtype=torch.int64, device=self.device)[:, None]
        return (first + torch.arange(length, device=self.device)).reshape(-1)

    @torch.no_grad()
    def add_taps_streams(self, snaps: list, taps: list[torch.Tensor]) -> list:
        """``add_taps`` for several streams, each projection run once over all their rows; returns the new contexts."""

        sizes = [t.shape[0] for t in taps]
        projected = cuda_dflash2._norm(self._lin(torch.cat(taps) if len(taps) > 1 else taps[0], "fc.weight"),
                                       self.weights["hidden_norm.weight"], self.eps)
        pos = torch.cat([torch.arange(snap[3], snap[3] + n, device=self.device) for snap, n in zip(snaps, sizes)])
        kcs, vcs = [list(snap[0]) for snap in snaps], [list(snap[1]) for snap in snaps]
        for layer in range(self.layers):
            prefix = f"layers.{layer}.self_attn."
            k = self._lin(projected, prefix + "k_proj.weight").view(-1, self.kv_heads, self.head_dim)
            v = self._lin(projected, prefix + "v_proj.weight").view(-1, self.kv_heads, self.head_dim)
            k = cuda_dflash2._rope(cuda_dflash2._norm(k, self.weights[prefix + "k_norm.weight"], self.eps).transpose(0, 1),
                                   pos, self.inv_freq)
            v = v.transpose(0, 1)
            at = 0
            for j, n in enumerate(sizes):
                for cache, fresh in ((kcs[j], k), (vcs[j], v)):
                    old = cache[layer]
                    rows = fresh[:, at:at + n]
                    cache[layer] = (rows if old is None else torch.cat((old, rows), dim=1))[:, -self.window:].contiguous()
                at += n
        return [(kc, vc, min(self.window, snap[2] + n), snap[3] + n) for kc, vc, snap, n in zip(kcs, vcs, snaps, sizes)]

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """The reference path of the CUDA module, its head read from the target's own rows; every stream's block in one pass."""

        out: list = [None] * len(snaps)
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live or max_nodes < 1:
            return out
        length = min(block or self.block, max_nodes + 1)
        tokens = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                              dtype=torch.int32, device=self.device)
        x = embedding(tokens, self.target_embed)
        ctx = [snaps[i] for i in live]
        positions = self._starts([snap[3] for snap in ctx], length)
        for layer in range(self.layers):
            x = self._layer(layer, x, ctx, length, positions)
        h = cuda_dflash2._norm(x.view(len(live), length, -1)[:, 1:].reshape(-1, self.hidden),
                               self.weights["norm.weight"], self.eps)
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
