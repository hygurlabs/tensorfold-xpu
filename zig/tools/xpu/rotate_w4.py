"""Checkpoint GPTQ int4 symétrique g128 pour l'étude W4A4 du backend XPU : chaque projection quantifiée du checkpoint GPTQ
de référence est refaite depuis les poids bf16 d'origine, éventuellement tournée (W·H par blocs de 128 le long de K,
Hadamard de Sylvester normé ; le moteur tourne les activations de même, XPU_A8_ROT=1), puis arrondie au plus près
(RTN : échelle 2·max|w| / 15 par groupe de 128, zéro 8, comme le GPTQ symétrique que le moteur lit). Les autres
tenseurs (plongement, normes, portes a/b, convolution, tête, MTP, vision) sont repris tels quels du checkpoint GPTQ.

    python rotate_w4.py BF16_DIR GPTQ_DIR SORTIE_DIR rot|norot
"""
import json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

bf16_dir, gptq_dir, out_dir, mode = sys.argv[1:5]
rot = mode == "rot"
os.makedirs(out_dir, exist_ok=True)

def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5

H = hadamard(128)
bidx = json.load(open(f"{bf16_dir}/model.safetensors.index.json"))["weight_map"]
gidx = json.load(open(f"{gptq_dir}/model.safetensors.index.json"))["weight_map"]

def bf16(name):
    with safe_open(f"{bf16_dir}/{bidx[name]}", "pt") as f:
        return f.get_tensor(name)

def quantize(w):
    """w [N, K] fp32 → qweight int32 [K/8, N], scales fp16 [K/128, N]."""
    n, k = w.shape
    if rot:
        w = (w.view(n, k // 128, 128) @ H).view(n, k)
    g = w.view(n, k // 128, 128)
    s = (2.0 * g.abs().amax(2, keepdim=True) / 15.0).clamp_min(1e-10)
    q = (torch.round(g / s) + 8).clamp(0, 15).to(torch.int32).view(n, k)       # [N, K]
    q = q.t().contiguous().view(k // 8, 8, n)                                  # mot k/8, quartet k % 8
    qw = torch.zeros(k // 8, n, dtype=torch.int32)
    for j in range(8):
        qw |= q[:, j, :] << (4 * j)
    return qw, s.squeeze(2).t().contiguous().to(torch.float16)

for shard in sorted(set(gidx.values())):
    names = [n for n, f in gidx.items() if f == shard]
    out = {}
    with safe_open(f"{gptq_dir}/{shard}", "pt") as f:
        for n in names:
            if n.endswith(".scales") and n[:-7] + ".qweight" in gidx:
                continue                                                     # refaite avec qweight
            if n.endswith(".qweight"):
                base = n[:-8]
                src = base.replace("model.language_model.", "model.language_model.") + ".weight"
                if src not in bidx:
                    raise KeyError(f"{src} absent des poids bf16")
                qw, sc = quantize(bf16(src).float())
                ref = f.get_tensor(n)
                assert qw.shape == ref.shape, (n, qw.shape, ref.shape)
                out[n] = qw
                out[base + ".scales"] = sc
            else:
                out[n] = f.get_tensor(n)
    save_file(out, f"{out_dir}/{shard}", metadata={"format": "pt"})
    print(shard, len(out), flush=True)

for fn in os.listdir(gptq_dir):
    if not fn.endswith(".safetensors") and os.path.isfile(f"{gptq_dir}/{fn}"):
        shutil.copy(f"{gptq_dir}/{fn}", f"{out_dir}/{fn}")
print("fait", out_dir)
