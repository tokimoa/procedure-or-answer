"""exp21 — MIXED task: defeats the "tautology" objection with a compositional prediction.

A task that needs BOTH a non-recomputable FACT (a product's unit price, stored only
in memory) AND a PROCEDURE (compute total = qty*price*(1+tax)). The test query gives
the entity and a NEW quantity but NOT the price — the price must be retrieved.

If the dichotomy were a mere tautology ("answer matters iff non-recomputable"), it
could not predict a decomposition. It predicts a NON-obvious one:
  - corrupt the FACT (price in memory)      -> HURTS (price is non-recomputable)
  - corrupt the METHOD (computation shown)   -> HURTS (model follows the method)
  - corrupt only the ANSWER (final total)    -> HARMLESS (recomputed from fact+method)

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp21_mixed.py \
      --models llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b --n 24 --k 3 --seeds 3
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend
from experiments.models import MODELS
from experiments.memory import solve as mem_solve

CONDS = ["nomem", "worked_clean", "fact_corrupt", "method_corrupt", "answer_corrupt"]
TAX = 10


def make(n, seed):
    # Composite task: needs a non-recomputable FACT (unit price, only in memory) AND a
    # PROCEDURE (total = qty * price). Simple multiplication -> headroom once the price is
    # available; the test omits the price so it must be retrieved. Method-corruption uses a
    # coherent wrong operation (addition), which models can actually follow.
    rng = np.random.default_rng(seed + 555)
    items = []
    for _ in range(n):
        eid = int(rng.integers(1000, 9999))
        price = int(rng.integers(120, 900))        # the FACT (only in memory)
        q0 = int(rng.integers(3, 9))               # exemplar quantity
        q1 = int(rng.integers(3, 9))               # test quantity
        while q1 == q0:
            q1 = int(rng.integers(3, 9))
        ex_total = q0 * price
        gold = q1 * price
        question = (f"Product #{eid}: a customer orders {q1} units. What is the total "
                    f"cost in JPY (integer)?")
        clean = (f"Product #{eid}: unit price is JPY {price}. For {q0} units: "
                 f"total = {q0} * {price} = {ex_total}.\nANSWER: {ex_total}")
        wprice = price + int(rng.choice([-40, -25, 25, 40, 60]))
        fact_bad = (f"Product #{eid}: unit price is JPY {wprice}. For {q0} units: "
                    f"total = {q0} * {wprice} = {q0 * wprice}.\nANSWER: {q0 * wprice}")
        # WRONG METHOD: addition instead of multiplication (a coherent wrong operation)
        method_bad = (f"Product #{eid}: unit price is JPY {price}. For {q0} units: "
                      f"total = {q0} + {price} = {q0 + price}.\nANSWER: {q0 + price}")
        bad = ex_total + int(rng.choice([-30, -20, 20, 30]))
        ans_bad = clean.replace(f"ANSWER: {ex_total}", f"ANSWER: {bad}")
        items.append({"eid": eid, "gold": gold, "question": question,
                      "worked_clean": clean, "fact_corrupt": fact_bad,
                      "method_corrupt": method_bad, "answer_corrupt": ans_bad})
    return items


def acc(be, kb, cond, k):
    if cond == "nomem":
        mem = None
    else:
        mem = [{"problem": it["question"], "sol": it[cond]} for it in kb]
        embs = be.embed([m["problem"] for m in mem])
    corr = []
    for it in kb:
        if mem is None:
            ex = []
        else:
            qe = be.embed([it["question"]])[0]
            sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
            ex = [mem[i] for i in np.argsort(-sims)[:k]]
        a, _, _ = mem_solve(be, it["question"], ex)
        corr.append(1.0 if a == it["gold"] else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", default="results/exp21_mixed.json")
    args = ap.parse_args()

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for s in range(args.seeds):
            kb = make(args.n, s)
            row = {"model": key, "seed": s}
            for c in CONDS:
                row[c] = acc(be, kb, c, args.k)
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 78)
    print(" exp21 — MIXED (fact + procedure): what part of memory matters?")
    print("=" * 78)
    print(f"{'model':<16}" + "".join(f"{c:>15}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>15.2f}" for c in CONDS))
    print("=" * 78)
    print("NON-tautological if fact_corrupt & method_corrupt HURT but answer_corrupt is HARMLESS")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
