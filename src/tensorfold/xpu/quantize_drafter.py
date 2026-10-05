"""Quantise the DFlash2 drafter to the 4-bit pack the XPU int4 kernel reads: GPTQ, symmetric, groups of 128.

The drafter need not be exact (the target verifies every token), so this is about acceptance: the rounding is
calibrated on the rows each projection really sees (``capture`` of the drafter's inputs), not on the weights alone.

    python -m tensorfold.xpu.quantize_drafter DRAFT_DIR CALIBRATION_DIR OUT_DIR
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch

GROUP = 128
ZERO = 8                 # q in 0..15, value (q - 8) * scale: what ``Int4`` hands the kernel
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def pack(q: torch.Tensor) -> torch.Tensor:
    """(K, N) values 0..15 -> (K/8, N) int32, the GPTQ layout: row k lives in nibble k % 8 of word k // 8."""

    k, n = q.shape
    words = torch.zeros((k // 8, n), dtype=torch.int64, device=q.device)
    q = q.to(torch.int64).view(k // 8, 8, n)
    for j in range(8):
        words |= q[:, j] << (4 * j)
    return ((words + 2 ** 31) % 2 ** 32 - 2 ** 31).to(torch.int32)


def dequantize(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(K/8, N) words and (K/GROUP, N) scales -> (N, K) float32 weights."""

    k8, n = qweight.shape
    nibbles = torch.stack([(qweight.to(torch.int64) >> (4 * j)) & 0xF for j in range(8)], dim=1).reshape(k8 * 8, n)
    return ((nibbles - ZERO).float() * scales.float().repeat_interleave(GROUP, 0)).t().contiguous()


def hessian(rows: torch.Tensor, device: str, chunk: int = 2048) -> torch.Tensor:
    """X^T X / n in float64 on the host, from the calibration rows."""

    k = rows.shape[1]
    h = torch.zeros((k, k), dtype=torch.float32, device=device)
    for i in range(0, rows.shape[0], chunk):
        x = rows[i:i + chunk].to(device).float()
        h += x.t() @ x
    return (h / rows.shape[0]).cpu().double()


def gptq(weight: torch.Tensor, h: torch.Tensor | None, damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise (N, K) ``weight``: error of each column carried onto the later ones through the inverse Hessian ``h``
    (K, K, host float64). Without ``h`` this is round-to-nearest. Returns the packed words and the fp16 scales."""

    n, k = weight.shape
    if k % GROUP:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    device = weight.device
    w = weight.float().clone()
    hinv = None
    if h is not None:
        h = h.clone()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w[:, dead.to(device)] = 0
        h.diagonal().add_(damp * h.diagonal().mean())
        hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float().to(device)
    q = torch.empty((k, n), dtype=torch.uint8, device=device)
    scales = torch.empty((k // GROUP, n), dtype=torch.float32, device=device)
    for g in range(k // GROUP):
        lo, hi = g * GROUP, (g + 1) * GROUP
        block = w[:, lo:hi].clone()
        scale = (2 * block.abs().amax(dim=1) / 15).clamp(min=1e-8)
        scales[g] = scale
        err = torch.zeros_like(block)
        for i in range(GROUP):
            col = block[:, i]
            qi = (torch.round(col / scale) + ZERO).clamp(0, 15)
            q[lo + i] = qi.to(torch.uint8)
            if hinv is not None:
                e = (col - (qi - ZERO) * scale) / hinv[lo + i, lo + i]
                block[:, i:] -= e[:, None] * hinv[lo + i, lo + i:hi][None, :]
                err[:, i] = e
        if hinv is not None and hi < k:
            w[:, hi:] -= err @ hinv[lo:hi, hi:]
    return pack(q), scales.half()


def quantised_names(names) -> list[str]:
    """The drafter's big projections: attention, MLP and the tap projection. The rest stays as stored."""

    return [n for n in names if n.endswith(".weight") and (n == "fc.weight" or
            any(f".{p}." in n for p in PROJECTIONS))]


def main(draft_dir: str, calibration_dir: str, out_dir: str, device: str = "xpu", rtn: bool = False) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(draft_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, dst / f.name)
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(src / "model.safetensors"), "pt") as f:
        names = list(f.keys())
        todo = set(quantised_names(names))
        for name in names:
            if name not in todo:
                out[name] = f.get_tensor(name)
        for i, name in enumerate(sorted(todo)):
            t0 = time.perf_counter()
            weight = f.get_tensor(name).to(device)
            calib = Path(calibration_dir) / f"{name}.pt"
            h = None if rtn or not calib.exists() else hessian(torch.load(calib), device)
            words, scales = gptq(weight, h)
            base = name[:-len(".weight")]
            out[base + ".qweight"], out[base + ".scales"] = words.cpu(), scales.cpu()
            err = (dequantize(words, scales) - weight.float()).pow(2).sum() / weight.float().pow(2).sum()
            print(f"{i + 1}/{len(todo)} {name} {tuple(weight.shape)} {'GPTQ' if h is not None else 'RTN'} "
                  f"erreur relative {float(err):.2e} {time.perf_counter() - t0:.0f} s", flush=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt", "quantization": "gptq-int4-sym-g128"})
    config = json.loads((dst / "config.json").read_text())
    config["xpu_int4"] = {"bits": 4, "group_size": GROUP, "sym": True}
    (dst / "config.json").write_text(json.dumps(config, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4], rtn="--rtn" in sys.argv)
