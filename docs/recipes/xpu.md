# Intel Arc GPUs (XPU)

TensorFold's CUDA engine on an Intel Arc GPU, through `torch.xpu`: `--backend xpu`. It serves GPTQ 4-bit checkpoints
(symmetric, groups of 32 or a multiple, no act-order) of the Qwen3.8 27B family, with the DFlash2 drafter. The output
contract is the CUDA backend's: a drafted reply equals this server's own serial decoding, a resumed prompt equals a fresh
one, token for token, greedy and sampled.

Measured on one Arc Pro B70 (32 GB). One stream, no vision, no MTP head.

## Run

    vllm/vllm-openai-xpu image (torch 2.13.0+xpu, Triton 3.7.2, vllm-xpu-kernels 0.1.12.3)

    tensorfold serve <GPTQ Qwen3.8-27B dir> --backend xpu --drafter <DFlash2 dir>

Keep two directories across restarts, or each start recompiles (the warm-up below takes 15 s instead of 4): Triton's (`~/.triton`) and the
driver's (`~/.cache/neo_compiler_cache`). The server warms up before it listens (`tensorfold.xpu.warm`: a prompt past one
block through serial, drafted and sampled decoding, 4 s with the caches, 15 s without); the first request then runs at
full speed.

The drafter can be quantised to the same 4-bit pack (GPTQ, calibrated on its own inputs; about 5 minutes on the card):

    python -m tensorfold.xpu.quantize_drafter <DFlash2 dir> <calibration dir> <out dir>

Over eight prompts the drafter takes 4.66 tokens a round in bf16 and 4.66 in GPTQ, 4.59 in round-to-nearest; the round's
draft stage drops from 11.8 to 8.4 ms. Its vocabulary head, the target's rows it reads, is rounded to the same 4-bit pack at
load (0.7 s): proposals only, the target still verifies every token.

## Speed

| | |
|---|---|
| serial decoding | 38 ms a token (26 tok/s) |
| drafted decoding, short context | 76-86 tok/s (4.4 tokens a round) |
| drafted, 9,000-token context | 67 tok/s |
| prompt, 20 / 128 / 300 tokens | 48 / 98 / 245 ms |
| prompt, 2,000 tokens | 1,810 tok/s |
| prompt, 4,000 / 9,000 / 14,000 tokens | 1,751 / 1,635 / 1,530 tok/s |

The same card with vLLM-XPU's MTP (the DFlash2 architecture is not supported there): 45 to 61 tok/s drafted, 33.8 serial.

## Several streams

`--parallel 2` to `4` serves that many requests at once. Every decode call of the server then takes `16 * parallel` rows
(a lone stream's too), so a stream's tokens are the same alone or beside others: concurrent == solo == serial, tested in
`tests/xpu` and with `tools/bench_concurrent.py --alone --serial` against a running server.

Aggregate decode speed, drafted, 192 tokens a stream:

| streams | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| tok/s | 83 | 119 | 125 | 146 |

Over HTTP, `bench_concurrent.py` with the code and chat prompts at four streams gives 113 to 157 tok/s. Four streams of
60,000-token prompts run to the end and equal their solo runs.

## Prompt precision

`--prefill-int8 off|mlp|all` (default off) runs the prompt matmuls on the 4-bit weights' int8 kernel, one scale a row.
Integer sums keep every property of the default path (any chunking, block size, position or padding gives a row the same
bits; drafted == serial and resumed == fresh hold, tested in `tests/xpu/test_a8.py`), but the activations lose precision.
Measured on 16,376 positions of real text (the same prompts through both paths, teacher-forced):

| | prompt tok/s (2,000) | loss, nats a token | perplexity | best token agrees | KL |
|---|---|---|---|---|---|
| off (fp16 activations) | 1,778 | 0.8062 | 2.239 | | |
| `mlp` (gate, up, down) | 2,212 | 0.8199 (+1.7%) | 2.270 | 95.5% | 0.028 |
| `all` | 2,464 | 0.8351 (+3.6%) | 2.305 | 93.5% | 0.058 |

Clipping a row's scale below its largest magnitude is far worse (0.7: +21% loss; 0.5: +150%): the model has large
activation outliers.

## What differs from the CUDA backend

- `compat.install()` answers the engine's `torch.cuda` calls with `torch.xpu` and replaces the CUDA kernels: the gated delta
  rule, tree attention (`_shared`, `_tail`, `_merge`), prompt attention, the dense projections, the glue, the drafter.
- The 4-bit projections run on `int4_gemm_w4a16` (vllm-xpu-kernels). Its bits depend on the row count and, for some
  shapes, on a row's place in a block, and it is not repeatable at 192 or 256 rows. So: decode and verify calls are padded
  to exactly 16 rows (one class, the weights read once); prompt calls use block sizes whose bits are the same for every
  size of a set (512 and 1,024 rows for the wide shapes; 1,024 for k/v), and the MLP's down projection a grid of
  absolute positions. `tests/xpu/test_gptq.py` states each property on the real shapes.
- A prompt of 384 tokens or fewer runs its wide projections in blocks of 64 or 128 rows instead of 512 or 1,024 (a 20-token
  prompt takes 48 ms, not 250). Those sizes give a row one class of bits, which differs from the large blocks', so a state
  kept by the prompt cache resumes only a prompt of the same kind: a short prompt never resumes a long one's state, nor the
  reverse (the prefix is then computed again).
- Triton kernels that must agree with each other run without fused multiply-add (`enable_fp_fusion=False`): the three
  tree-attention kernels, the gated delta rule.
- The dense projections of the gates and the head use a Triton kernel with one fixed order of sums (`F.linear` is not
  repeatable on this stack).
- Activations stay fp16 from the glue to the int4 kernel and back (`xpu/glue.py`), padded to the kernel's rows: about 740
  conversion launches a round fewer. `qkv` and `v` stay bf16 (the convolution state and the attention read them so).
- `torch.topk` is replaced by a two-stage top-k: the stock kernel compiles for 38 s on its first use.
- A prompt is cut at multiples of 1,024 absolute positions, and tree attention folds 128-key chunks (the last, partial
  chunk is a sequential tail). The tree-attention kernels share each tile's output columns over four programs, which keeps
  the tile in registers.
