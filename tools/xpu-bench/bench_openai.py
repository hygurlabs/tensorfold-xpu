#!/usr/bin/env python3
"""Throughput benchmark for an OpenAI-compatible server (vLLM, TensorFold, ...).

Measures, through /v1/completions with streaming:
  prefill      time to first token for prompts of N tokens (unique prompts: no prefix-cache hit)
  decode       tokens/s after the first token, for C concurrent requests
Output is one JSON object per scenario on stdout, then a markdown table on stderr.
Standard library only, so it runs unchanged on any box.

  python3 bench_openai.py --url http://127.0.0.1:8000 --model qwen38-27b \
      --prefill 1024,4096,8192,14000 --concurrency 1,2,4 --decode-tokens 256
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = ("alpha bravo cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow "
         "nimbus onyx pebble quartz ridge saffron timber umbra velvet willow xenon yarrow zephyr").split()


TOKENS_PER_WORD = 1.3  # replaced by calibrate() with the server's own tokenizer


def make_prompt(approx_tokens: int, rng: random.Random) -> str:
    # The exact count comes back in usage.prompt_tokens.
    n = max(1, int(approx_tokens / TOKENS_PER_WORD))
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


def calibrate(url: str, model: str, rng: random.Random) -> float:
    """Tokens per word of make_prompt's text, measured on the server being benchmarked."""
    global TOKENS_PER_WORD
    r = one_request(url, model, make_prompt(300, rng), 1)
    TOKENS_PER_WORD = r["prompt_tokens"] / max(1, int(300 / TOKENS_PER_WORD))
    return TOKENS_PER_WORD


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: int = 1800,
                chat: bool = False) -> dict:
    """chat=False: raw /v1/completions, output length forced (ignore_eos). chat=True: /v1/chat/completions
    with thinking off and the model's own stopping point, for prompts where speculation matters."""
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    if chat:
        payload.update(messages=[{"role": "user", "content": prompt}],
                       chat_template_kwargs={"enable_thinking": False})
    else:
        payload.update(prompt=prompt, ignore_eos=True)
    body = json.dumps(payload).encode()
    path = "/v1/chat/completions" if chat else "/v1/completions"
    req = urllib.request.Request(url + path, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    last = t0
    usage = {}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
    with resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev["choices"][0] if ev.get("choices") else {}
            if ch.get("text") or ch.get("delta", {}).get("content"):
                last = time.perf_counter()
                if first is None:
                    first = last
    if first is None:
        raise RuntimeError("no token received")
    out_tokens = usage.get("completion_tokens", 0)
    decode_s = max(last - first, 1e-9)
    return {
        "ttft_s": first - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": out_tokens,
        "decode_tps": (out_tokens - 1) / decode_s if out_tokens > 1 else 0.0,
        "total_s": last - t0,
    }


# Natural prompts: speculative decoding (MTP) accepts far more tokens on prose and code than on the
# random words of make_prompt, so a speculation result must come from these.
NATURAL_PROMPTS = {
    "french_prose": "Rédige un texte détaillé d'au moins 600 mots expliquant le fonctionnement d'un moteur "
                    "à combustion à quatre temps, avec des exemples concrets.",
    "python_code": "Écris en Python une classe LRUCache complète, avec docstrings, gestion des erreurs et une "
                   "suite de tests pytest.",
}


def run_natural(args):
    rows = []
    for name, prompt in NATURAL_PROMPTS.items():
        reps = [one_request(args.url, args.model, prompt, args.decode_tokens, chat=True) for _ in range(args.reps)]
        rows.append({"scenario": "natural", "prompt": name, "decode_tps": round(statistics.median(
            r["decode_tps"] for r in reps), 1), "completion_tokens": reps[0]["completion_tokens"],
            "ttft_s": round(statistics.median(r["ttft_s"] for r in reps), 3), "reps": len(reps)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def run_prefill(args, sizes, rng):
    rows = []
    for size in sizes:
        reps = []
        for _ in range(args.reps):
            r = one_request(args.url, args.model, make_prompt(size, rng), 1)
            reps.append(r)
        pt = reps[0]["prompt_tokens"]
        ttft = statistics.median(r["ttft_s"] for r in reps)
        rows.append({"scenario": "prefill", "prompt_tokens": pt, "ttft_s": round(ttft, 3),
                     "prefill_tps": round(pt / ttft, 1), "reps": len(reps)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def run_decode(args, levels, rng):
    rows = []
    for c in levels:
        results, errors = [], []

        def work():
            try:
                results.append(one_request(args.url, args.model, make_prompt(args.decode_prompt, rng),
                                           args.decode_tokens))
            except Exception as e:  # noqa: BLE001 - reported in the row
                errors.append(str(e))

        t0 = time.perf_counter()
        threads = [threading.Thread(target=work) for _ in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        if not results:
            rows.append({"scenario": "decode", "concurrency": c, "error": errors[:1]})
            print(json.dumps(rows[-1]), flush=True)
            continue
        per = [r["decode_tps"] for r in results]
        total_out = sum(r["completion_tokens"] for r in results)
        rows.append({"scenario": "decode", "concurrency": c, "per_request_tps": round(statistics.mean(per), 1),
                     "aggregate_tps": round(sum(per), 1), "ttft_s": round(max(r["ttft_s"] for r in results), 3),
                     "completion_tokens": total_out, "wall_s": round(wall, 1), "errors": len(errors)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", default="1024,4096,8192", help="comma list of prompt sizes in tokens, '' to skip")
    ap.add_argument("--concurrency", default="1,2,4", help="comma list, '' to skip")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--decode-prompt", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2, help="repetitions per prefill size (median reported)")
    ap.add_argument("--natural", action="store_true", help="also run the prose and code prompts (chat endpoint)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for _ in range(args.warmup):  # compiles kernels, fills allocator pools
        one_request(args.url, args.model, make_prompt(512, rng), 32)
    print(f"# tokens per word: {calibrate(args.url, args.model, rng):.2f}", file=sys.stderr)

    rows = []
    sizes = [int(x) for x in args.prefill.split(",") if x]
    levels = [int(x) for x in args.concurrency.split(",") if x]
    if sizes:
        rows += run_prefill(args, sizes, rng)
    if levels:
        rows += run_decode(args, levels, rng)
    if args.natural:
        rows += run_natural(args)

    print("\n| scenario | detail | result |\n|---|---|---|", file=sys.stderr)
    for r in rows:
        if r["scenario"] == "prefill":
            print(f"| prefill | {r['prompt_tokens']} tok | {r['prefill_tps']} tok/s, TTFT {r['ttft_s']} s |",
                  file=sys.stderr)
        elif r["scenario"] == "natural":
            print(f"| natural | {r['prompt']} | {r['decode_tps']} tok/s ({r['completion_tokens']} tok) |",
                  file=sys.stderr)
        elif "error" in r:
            print(f"| decode | {r['concurrency']} req | ERROR {r['error']} |", file=sys.stderr)
        else:
            print(f"| decode | {r['concurrency']} req | {r['aggregate_tps']} tok/s total, "
                  f"{r['per_request_tps']} /req |", file=sys.stderr)


if __name__ == "__main__":
    main()
