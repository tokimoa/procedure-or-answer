"""exp15 — the BOUNDARY: on a KNOWLEDGE-lookup task, does the answer matter?

Procedural tasks: memory's value is the procedure, answer irrelevant (exp13).
Prediction/contrast: on a task where the memory must supply an arbitrary FACT the
model cannot recompute, the ANSWER is the content — so answer-corruption should HURT
and answer_only should HELP (opposite of procedural tasks).

Synthetic KB: Product #E has warranty V months (V random, not derivable). Test asks
V for entities that ARE in memory; retrieval by embedding should surface the fact.
Conditions: nomem / answer_only / worked_clean / worked_ans_bad (fact corrupted).

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp15_knowledge.py \
      --models phi4-mini,ministral3-3b,granite4.1-8b --n 24 --test 20 --k 3 --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend, mx_seed
from experiments.models import MODELS
from experiments.memory import solve as mem_solve

CONDS = ["nomem", "answer_only", "worked_clean", "worked_ans_bad"]


def make_kb(n, seed):
    rng = np.random.default_rng(seed + 4242)
    items = []
    for _ in range(n):
        eid = int(rng.integers(1000, 9999))
        val = int(rng.integers(6, 60))
        q = (f"What is the warranty period, in months, of Product #{eid}? "
             f"End with EXACTLY: ANSWER: <integer>")
        items.append({"eid": eid, "gold": val, "problem": q,
                      "worked": f"Product #{eid} record: the warranty period is {val} months.\nANSWER: {val}",
                      "answer_only": f"Product #{eid}: {val}"})
    return items


def mem_items(kb, cond, rng):
    out = []
    for it in kb:
        if cond == "answer_only":
            sol = it["answer_only"]
        elif cond == "worked_ans_bad":
            wrong = it["gold"] + int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
            # rebuild the record with the wrong fact (a str.replace on the value
            # string could also mangle the entity id if it contains that substring)
            sol = (f"Product #{it['eid']} record: the warranty period is {wrong} "
                   f"months.\nANSWER: {wrong}")
        else:  # worked_clean
            sol = it["worked"]
        out.append({"problem": it["problem"], "sol": sol})
    return out


def acc(be, kb, cond, k, seed):
    rng = np.random.default_rng(seed)
    mem = None if cond == "nomem" else mem_items(kb, cond, rng)
    embs = None if mem is None else be.embed([m["problem"] for m in mem])
    corr = []
    for it in kb:  # test = the same entities (they must be in memory)
        if mem is None:
            ex = []
        else:
            qe = be.embed([it["problem"]])[0]
            sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
            ex = [mem[i] for i in np.argsort(-sims)[:k]]
        ans, _, _ = mem_solve(be, it["problem"], ex)
        corr.append(1.0 if ans == it["gold"] else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--n", type=int, default=24, help="KB size (= test size)")
    ap.add_argument("--test", type=int, default=20)  # unused; test == kb entities
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", default="results/exp15_knowledge.json")
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
            mx_seed(s)
            kb = make_kb(args.n, s)
            row = {"model": key, "seed": s}
            for c in CONDS:
                row[c] = acc(be, kb, c, args.k, s)
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)

    print("\n" + "=" * 64)
    print(" exp15 — KNOWLEDGE task: does the answer matter? (mean over seeds)")
    print("=" * 64)
    print(f"{'model':<16}" + "".join(f"{c:>15}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>15.2f}" for c in CONDS))
    print("=" * 64)
    print("BOUNDARY confirmed if here answer_only ≈ worked_clean >> worked_ans_bad")
    print("(opposite of procedural tasks) -> the answer matters for knowledge lookup.")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
