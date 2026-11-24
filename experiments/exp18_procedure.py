"""exp18 — does corrupting the METHOD hurt (unlike corrupting the answer)?

Completes the dichotomy's procedural leg. Hand-templated exemplars (same style
across conditions, so the contrast isolates the method vs the answer):
  - nomem
  - worked_correct       : correct method + correct answer
  - worked_ans_corrupt   : correct method + WRONG answer   (pools.corrupt)
  - worked_wrong_method   : WRONG method (e.g. + instead of *) + the wrong answer it yields

Prediction: worked_correct ≈ worked_ans_corrupt  >  worked_wrong_method.
(For edge-competence models that LEARN the method from exemplars; strong models
that already know it recompute and are unaffected.)

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp18_procedure.py \
      --models phi4-mini,ministral3-3b,granite4.1-8b --seeds 3
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

CONDS = ["nomem", "worked_correct", "worked_ans_corrupt", "worked_wrong_method"]


def _arith_hard_item(rng):
    P, B, I = int(rng.integers(3, 9)), int(rng.integers(4, 9)), int(rng.integers(3, 8))
    dmg, cr = int(rng.integers(1, 4)) * 5, int(rng.integers(6, 12))
    total = P * B * I
    good = total - (total * dmg) // 100
    ans = good // cr
    q = (f"A warehouse has {P} pallets, each holding {B} boxes of {I} items. {dmg}% of "
         f"all items are damaged and removed. The rest are packed into crates of {cr} "
         f"(leftover items are not crated). How many full crates?")
    correct = (f"Total items = {P}*{B}*{I} = {total}. Damaged = {dmg}% of {total} = "
               f"{(total*dmg)//100}. Good = {good}. Crates = {good}//{cr} = {ans}.\nANSWER: {ans}")
    # WRONG METHOD: add instead of multiply for the total (coherent wrong chain)
    wt = P + B + I
    wg = wt - (wt * dmg) // 100
    wc = wg // cr
    wrong_method = (f"Total items = {P}+{B}+{I} = {wt}. Damaged = {dmg}% of {wt} = "
                    f"{(wt*dmg)//100}. Good = {wg}. Crates = {wg}//{cr} = {wc}.\nANSWER: {wc}")
    bad = ans + int(rng.choice([-2, -1, 1, 2, 3]))
    # anchored: corrupt ONLY the final ANSWER line (str.replace would also mangle
    # intermediate numbers containing the answer string)
    ans_corrupt = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {bad}", correct)
    return q, ans, correct, ans_corrupt, wrong_method


def pool(n, rng):
    out = []
    for _ in range(n):
        q, g, c, ac, wm = _arith_hard_item(rng)
        out.append({"problem": q, "gold": g, "correct": c, "ans_corrupt": ac, "wrong_method": wm})
    return out


def mem_for(items, cond):
    key = {"worked_correct": "correct", "worked_ans_corrupt": "ans_corrupt",
           "worked_wrong_method": "wrong_method"}[cond]
    return [{"problem": it["problem"], "sol": it[key]} for it in items]


def acc(be, mem, test, k, rng):
    corr = []
    for t in test:
        ex = [] if mem is None else [mem[i] for i in rng.permutation(len(mem))[:k]]
        a, _, _ = mem_solve(be, t["problem"], ex)
        corr.append(1.0 if a == t["gold"] else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--pooln", type=int, default=24)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/exp18_procedure.json")
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
            items = pool(args.pooln, np.random.default_rng(s + 11))
            test = pool(args.test, np.random.default_rng(s + 7777))
            row = {"model": key, "seed": s, "nomem": acc(be, None, test, 0, np.random.default_rng(s))}
            for c in CONDS[1:]:
                row[c] = acc(be, mem_for(items, c), test, args.k, np.random.default_rng(s + 1))
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 66)
    print(" exp18 — method-corruption vs answer-corruption (mean over seeds)")
    print("=" * 66)
    print(f"{'model':<16}" + "".join(f"{c.replace('worked_',''):>16}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>16.2f}" for c in CONDS))
    print("=" * 66)
    print("METHOD matters if wrong_method < correct ≈ ans_corrupt")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
