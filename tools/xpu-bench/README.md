# XPU measurements (Arc Pro B70)

The numbers in `docs/recipes/xpu.md` ("Against vLLM") come from these files, measured on one Arc Pro B70 (32 GB) on
2026-10-05 and 2026-10-06, Qwen3.8-27B GPTQ 4-bit, standard library Python clients against a running server.

- `bench_openai.py`: decode speed through `/v1/completions`, 192 tokens a stream, `code` and `chat` prompts, temperature 0
  and 1, 1 to 4 streams. TensorFold runs also check that every stream equals its solo and serial run (`alone`, `serial`).
- `workflow_sim.py`: an office workload. `warm`: a ~20-page file read once, then 4 colleagues ask it 4 different questions
  at once. `cold`: 4 colleagues open 4 different files at once. Every answer is checked for the facts it needs (`facts_ok`).

| file | engine |
|---|---|
| `refs/ref_p1_3d20551.json` | TensorFold XPU at `3d20551`, one stream |
| `refs/ref_p4_3d20551.json` | TensorFold XPU at `3d20551`, `--parallel 4` |
| `refs/ref_p4_2170f4c.json` | TensorFold XPU at `2170f4c`, `--parallel 4` |
| `refs/vllm_plain.json` | vLLM-XPU, no speculative decoding |
| `refs/vllm_v31mtp.json` | vLLM v0.31.0, MTP with 4 drafts |
| `refs/wf_tf2_*.json` | TensorFold XPU, `workflow_sim.py` |
| `refs/wf_vllm_v31_*.json` | vLLM v0.31.0 + MTP4, `workflow_sim.py` |
