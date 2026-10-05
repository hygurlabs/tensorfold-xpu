"""Start-up warm-up: the first request meets compiled kernels, not a compiler."""

from __future__ import annotations


def warm(engine, prompt_tokens: int = 140, new_tokens: int = 24) -> None:
    """Run a prompt past one 128-row block through serial, drafted and sampled decoding, then forget it.

    Every projection shape, window width and sampling path is used once, so Triton's binaries and the driver's
    per-shape programs exist before a request arrives; the first request otherwise runs at a third of its speed."""

    from tensorfold.engine.exact_sampling import Sampling

    ids = [1000 + (i * 7 + 11) % 5000 for i in range(prompt_tokens)]
    greedy = Sampling(seed=1, temperature=0.0, top_k=1, top_p=1.0)
    sampled = Sampling(seed=2, temperature=0.8, top_k=20, top_p=0.95)
    runs = [(greedy, False)]
    if getattr(engine, "draft", None) is not None:
        runs += [(greedy, True), (sampled, True)]
    for sampling, draft in runs:
        engine.generate(list(ids), new_tokens, sampling, lambda tokens: None, draft=draft, stop_eos=False)
    while engine.cache.evict():                    # the warm-up prompt is not a prefix anyone will ask for
        pass
