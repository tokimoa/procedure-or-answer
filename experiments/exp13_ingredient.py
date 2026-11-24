"""exp13 — what is the ACTIVE INGREDIENT of memory: procedure or answer?

A pilot pollution sweep showed that corrupting stored answers barely matters. Hypothesis: the benefit
comes from the DEMONSTRATED PROCEDURE (worked reasoning), not the answer. Test by
comparing exemplar formats at matched retrieval:
  - nomem          : no exemplars
  - answer_only    : "ANSWER: <gold>" (correct answer, NO reasoning)
  - worked_clean   : full model reasoning + correct answer
  - worked_ans_bad : full reasoning, but the ANSWER corrupted (procedure intact)

Prediction: worked_clean ≈ worked_ans_bad  >  answer_only ≈ nomem
=> procedure is the active ingredient; answer correctness is not.

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp13_ingredient.py \
      --models phi4-mini,ministral3-3b,granite4.1-8b --tasks arith_hard,invoice --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend, mx_seed
from experiments.models import MODELS
from experiments.pools import POOLS, build_clean_pool, corrupt, acc_with

CONDS = ["nomem", "answer_only", "worked_clean", "worked_ans_bad"]


def answer_only(pool):
    return [{"problem": it["problem"], "gold": it["gold"], "sol": f"ANSWER: {it['gold']}"}
            for it in pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default="results/exp13_ingredient.json")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for task in tasks:
            pf = POOLS[task]
            for s in range(args.seeds):
                mx_seed(s)
                test = pf(args.test, np.random.default_rng(s + 7777))
                clean = build_clean_pool(be, pf, args.pool, s, "model")
                r = lambda: np.random.default_rng(s + 1)
                row = {"model": key, "task": task, "seed": s,
                       "nomem": acc_with(be, None, test, 0, r()),
                       "answer_only": acc_with(be, answer_only(clean), test, args.k, r()),
                       "worked_clean": acc_with(be, clean, test, args.k, r()),
                       "worked_ans_bad": acc_with(be, corrupt(clean, 1.0, np.random.default_rng(s)),
                                                  test, args.k, r()),
                       "pool": len(clean)}
                records.append(row)
                print(f"  {task:<11} seed{s}: " +
                      " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)

    print("\n" + "=" * 70)
    print(" exp13 — active ingredient (mean over seeds)")
    print("=" * 70)
    print(f"{'model/task':<24}" + "".join(f"{c:>15}" for c in CONDS))
    for task in tasks:
        for key in [m.strip() for m in args.models.split(",")]:
            rs = [r for r in records if r["model"] == key and r["task"] == task]
            if not rs:
                continue
            print(f"{key+'/'+task:<24}" + "".join(f"{np.mean([r[c] for r in rs]):>15.2f}" for c in CONDS))
    print("=" * 70)
    print("procedure is the active ingredient if worked_* > answer_only ≈ nomem")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
