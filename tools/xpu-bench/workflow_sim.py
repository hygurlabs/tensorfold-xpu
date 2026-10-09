"""Simulated office workload on an OpenAI-compatible server: a ~20-page synthetic file read once, then several colleagues
asking it questions at the same time. Standard library only (Python 3.11+).

  python3 workflow_sim.py http://127.0.0.1:8080 MODEL --scenario warm|cold --users 4 --pages 20 --output out.json

``warm``: the file is read by a first request alone (cold prefill), then ``--users`` colleagues ask a different question
each about the same file at once (the file is the system message, so a server with prefix caching can reuse it).
``cold``: ``--users`` colleagues each open a different file at once (the worst case: every prompt is a cold prefill).
Per request: time to first token, time to the whole answer, tokens, decode tok/s. Per phase: wall time.
Every file is invented (a made-up company, invoices, VAT lines); the 6 facts a question needs are in the text, and an
answer is checked for them (``facts_ok``), so a fast engine that answers wrongly shows.
"""

import argparse
import json
import re
import random
import statistics
import threading
import time
import urllib.request

QUESTIONS = [
    ("Résume le dossier en cinq points.", ()),
    ("Quelles sont les échéances de TVA du dossier et leurs montants ?", ("{vat1}", "{vat2}")),
    ("Quel est le montant hors taxes de la facture {look_no} ?", ("{look}",)),
    ("Quelle facture est restée impayée, et de quel montant ?", ("{unpaid_no}", "{unpaid}")),
    ("Rédige un courriel de relance poli pour la facture impayée.", ("{unpaid_no}",)),
    ("Quelle est la date limite de dépôt de la déclaration annuelle ?", ("{deadline}",)),
]
CLAUSES = ["Les parties conviennent que les délais de paiement ne peuvent excéder trente jours à compter de la réception de la facture.",
           "Toute modification du présent accord doit faire l'objet d'un avenant signé par les deux parties.",
           "Le prestataire s'engage à conserver les pièces justificatives pendant la durée légale de conservation.",
           "Les montants s'entendent hors taxes, sauf mention contraire expressément indiquée dans la facture.",
           "En cas de désaccord sur une écriture, la société en informe le cabinet dans un délai de quinze jours.",
           "La société reste seule responsable de l'exactitude des informations transmises au cabinet comptable."]
SUPPLIERS = ["Atelier Nordique", "Menuiserie Belvédère", "Transports Lumière", "Papeteries du Canal", "Services Horizon"]


def money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")


def dossier(seed: int, pages: int) -> tuple[str, dict]:
    """A made-up accounting file of about ``pages`` pages (500 words each), every invoice listed once, and the facts a
    question asks for."""

    r = random.Random(seed)
    count = pages * 11
    invoices = [(f"F-{seed}{i:03d}", r.choice(SUPPLIERS), r.randrange(180, 9800) + r.random()) for i in range(1, count + 1)]
    unpaid_no, _, unpaid = invoices[r.randrange(5, count - 5)]
    look_no, look_supplier, look = invoices[r.randrange(5, count - 5)]
    facts = {"look_no": look_no, "look": money(look), "supplier": look_supplier, "unpaid_no": unpaid_no, "unpaid": money(unpaid),
             "vat1": f"{r.randrange(2000, 9000)},00", "vat2": f"{r.randrange(2000, 9000)},00",
             "deadline": f"{r.randrange(1, 28):02d}/05/2026"}
    parts = [f"DOSSIER COMPTABLE N° {seed} — SOCIÉTÉ EXEMPLE {seed} SARL (données inventées)\n",
             f"Échéance TVA du premier trimestre : {facts['vat1']} euros. Échéance TVA du deuxième trimestre : {facts['vat2']} euros. "
             f"Déclaration annuelle à déposer au plus tard le {facts['deadline']}.\n"]
    for i, (no, sup, amount) in enumerate(invoices):
        paid = "IMPAYÉE" if no == unpaid_no else "payée"
        parts.append(f"Facture {no} du fournisseur {sup} : montant {money(amount)} euros hors taxes, TVA à 20 %, statut {paid}. "
                     + r.choice(CLAUSES) + f" Écriture {i + 1}, journal des achats, pièce rattachée au classeur {r.randrange(1, 40)}.\n")
    return "".join(parts), facts


def _flat(text: str) -> str:
    """Spaces of every kind out, a decimal point as a comma: ``5 394,00`` and ``5394.00`` are the same figure."""

    return re.sub(r"[\s\u00a0\u202f]", "", text).replace(".", ",")


def ask(base: str, model: str, system: str, question: str, facts: tuple, max_tokens: int, out: dict, key) -> None:
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0, first, last, text, usage = time.perf_counter(), None, None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or (choice.get("delta") or {}).get("reasoning_content")
                    if piece:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        text.append(piece)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        out[key] = {"error": f"{type(exc).__name__}: {exc}"}
        return
    answer = "".join(text)
    tokens = usage.get("completion_tokens", 0)
    out[key] = {"ttft_s": round(first - t0, 2) if first else None, "total_s": round((last or t0) - t0, 2), "tokens": tokens,
                "prompt_tokens": usage.get("prompt_tokens"),
                "decode_tps": round((tokens - 1) / (last - first), 1) if first and last and last > first and tokens > 1 else None,
                "facts_ok": all(_flat(f) in _flat(answer) for f in facts), "answer_head": answer[:120]}


def phase(base, model, jobs, max_tokens) -> tuple[list[dict], float]:
    out: dict = {}
    threads = [threading.Thread(target=ask, args=(base, model, s, q, f, max_tokens, out, i)) for i, (s, q, f) in enumerate(jobs)]
    t0 = time.perf_counter()
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [out[i] for i in range(len(jobs))], round(time.perf_counter() - t0, 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--scenario", choices=("warm", "cold"), default="warm")
    p.add_argument("--users", type=int, default=4)
    p.add_argument("--pages", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output")
    a = p.parse_args()
    runs = []
    for rep in range(a.reps):
        seed = a.seed + 100 * rep                      # a new file each repetition, so no run reads a cached one
        record: dict = {"rep": rep, "scenario": a.scenario}
        if a.scenario == "warm":
            text, facts = dossier(seed, a.pages)
            q = lambda i: (lambda s, f: (s.format(**facts), tuple(x.format(**facts) for x in f)))(*QUESTIONS[i])  # noqa: E731
            first, wall = phase(a.base, a.model, [(text, *q(0))], a.max_tokens)
            record["read"] = {"wall_s": wall, "requests": first}
            jobs = [(text, *q(1 + i % (len(QUESTIONS) - 1))) for i in range(a.users)]
        else:
            jobs = []
            for i in range(a.users):
                text, facts = dossier(seed + i, a.pages)
                s, f = QUESTIONS[1 + i % (len(QUESTIONS) - 1)]
                jobs.append((text, s.format(**facts), tuple(x.format(**facts) for x in f)))
        results, wall = phase(a.base, a.model, jobs, a.max_tokens)
        record["colleagues"] = {"wall_s": wall, "requests": results}
        runs.append(record)
        ok = [r for r in results if "error" not in r]
        print(f"rep {rep}: prompt {results[0].get('prompt_tokens')} tokens | colleagues wall {wall} s | first token "
              f"{[r.get('ttft_s') for r in results]} | answer {[r.get('total_s') for r in results]} s | "
              f"decode {[r.get('decode_tps') for r in results]} tok/s | facts ok {sum(r.get('facts_ok', False) for r in ok)}/{len(results)}",
              flush=True)
    if a.output:
        json.dump(runs, open(a.output, "w"), indent=1, ensure_ascii=False)
    walls = [r["colleagues"]["wall_s"] for r in runs]
    print(f"median wall of the colleagues' phase: {statistics.median(walls)} s over {len(runs)} runs")


if __name__ == "__main__":
    main()
