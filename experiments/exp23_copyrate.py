"""exp23 — copy vs. recompute (direct mechanism test).

Places the TEST item itself in memory with a CORRUPTED answer W, then asks: does the
model output W (copied from memory) or the correct answer C (recomputed / recalled)?
  - Procedural (arith_hard): W = wrong final answer on the item's own worked solution.
    Prediction: copy-rate ~ 0 (the model recomputes C, ignoring the planted W).
  - Knowledge (KB lookup): W = wrong fact for the queried entity.
    Prediction: copy-rate high (the fact is non-recomputable, so the model reports W).
This directly separates "recompute" from "ignore memory": procedural memory clearly
HELPS elsewhere (worked > no-mem), yet here the planted wrong answer is not copied.

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp23_copyrate.py \
      --models llama3.2-3b,phi4-mini,granite4.1-8b --n 30 --seeds 3
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend, mx_seed
from experiments.models import MODELS
from experiments.memory import solve as mem_solve
from experiments.exp18_procedure import _arith_hard_item
from experiments.exp15_knowledge import make_kb


def procedural_trial(be, rng):
    q, gold, correct, _, _ = _arith_hard_item(rng)
    W = gold + int(rng.choice([-3, -2, 2, 3]))          # planted wrong answer
    bad_sol = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {W}", correct)
    ans, _, _ = mem_solve(be, q, [{"problem": q, "sol": bad_sol}])
    return ans, gold, W


def knowledge_trial(be, kb, it, rng):
    others = [x["gold"] for x in kb if x["gold"] != it["gold"]]
    W = int(rng.choice(others)) if others else it["gold"] + 1   # planted wrong fact
    # rebuild the record with the wrong fact (avoid str.replace collisions)
    sol = (f"Product #{it['eid']} record: the warranty period is {W} months.\n"
           f"ANSWER: {W}")
    ans, _, _ = mem_solve(be, it["problem"], [{"problem": it["problem"], "sol": sol}])
    return ans, it["gold"], W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,granite4.1-8b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default="results/exp23_copyrate.json")
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
            rng = np.random.default_rng(s + 313)
            pc = pr = kc = kk = 0  # proc: copied W / recomputed C; know: copied W / correct
            for _ in range(args.n):
                a, C, W = procedural_trial(be, rng)
                pc += (a == W); pr += (a == C)
            kb = make_kb(args.n, s)
            for it in kb:
                a, C, W = knowledge_trial(be, kb, it, rng)
                kc += (a == W); kk += (a == C)
            row = {"model": key, "seed": s,
                   "proc_copy": pc / args.n, "proc_recompute": pr / args.n,
                   "know_copy": kc / len(kb), "know_correct": kk / len(kb)}
            records.append(row)
            print(f"  seed{s}: proc copy={row['proc_copy']:.2f} recompute={row['proc_recompute']:.2f}"
                  f" | know copy={row['know_copy']:.2f} correct={row['know_correct']:.2f}", flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 64)
    print(" exp23 — copy-rate: P(output == planted wrong answer W)")
    print("=" * 64)
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"  {key:<14} procedural copy={np.mean([r['proc_copy'] for r in rs]):.2f}"
              f"  knowledge copy={np.mean([r['know_copy'] for r in rs]):.2f}")
    print("=" * 64)
    print("mechanism confirmed if procedural copy-rate ~0 (recompute) << knowledge copy-rate")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
