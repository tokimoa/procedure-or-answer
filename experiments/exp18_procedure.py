"""Method corruption versus answer corruption on the multi-step arithmetic task.

Exemplars are templated, so style is constant across conditions: nomem, worked_correct,
worked_ans_corrupt (correct method, wrong answer field) and worked_wrong_method (item
counts added instead of multiplied, ending in the answer that method gives).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_correct", "worked_ans_corrupt", "worked_wrong_method"]


def _arith_hard_item(rng):
    """Return (question, gold, correct solution, answer-corrupted and wrong-method variants)."""
    pallets, boxes, items = (
        int(rng.integers(3, 9)),
        int(rng.integers(4, 9)),
        int(rng.integers(3, 8)),
    )
    dmg, cr = int(rng.integers(1, 4)) * 5, int(rng.integers(6, 12))
    total = pallets * boxes * items
    good = total - (total * dmg) // 100
    ans = good // cr
    q = (
        f"A warehouse has {pallets} pallets, each holding {boxes} boxes of {items} items. "
        f"{dmg}% of all items are damaged and removed. The rest are packed into crates of {cr} "
        f"(leftover items are not crated). How many full crates?"
    )
    correct = (
        f"Total items = {pallets}*{boxes}*{items} = {total}. Damaged = {dmg}% of {total} = "
        f"{(total * dmg) // 100}. Good = {good}. Crates = {good}//{cr} = {ans}.\nANSWER: {ans}"
    )
    # Wrong method: add instead of multiply; the later steps follow from that total.
    wt = pallets + boxes + items
    wg = wt - (wt * dmg) // 100
    wc = wg // cr
    wrong_method = (
        f"Total items = {pallets}+{boxes}+{items} = {wt}. Damaged = {dmg}% of {wt} = "
        f"{(wt * dmg) // 100}. Good = {wg}. Crates = {wg}//{cr} = {wc}.\nANSWER: {wc}"
    )
    bad = ans + int(rng.choice([-2, -1, 1, 2, 3]))
    # Substitute only the ANSWER line; str.replace on the value would also change
    # intermediate numbers that contain the same digits.
    ans_corrupt = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {bad}", correct)
    return q, ans, correct, ans_corrupt, wrong_method


def pool(n, rng):
    out = []
    for _ in range(n):
        q, g, c, ac, wm = _arith_hard_item(rng)
        out.append({"problem": q, "gold": g, "correct": c, "ans_corrupt": ac, "wrong_method": wm})
    return out


def mem_for(items, cond):
    key = {
        "worked_correct": "correct",
        "worked_ans_corrupt": "ans_corrupt",
        "worked_wrong_method": "wrong_method",
    }[cond]
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
            row = {
                "model": key,
                "seed": s,
                "nomem": acc(be, None, test, 0, np.random.default_rng(s)),
            }
            for c in CONDS[1:]:
                row[c] = acc(be, mem_for(items, c), test, args.k, np.random.default_rng(s + 1))
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)
    print("\n" + "=" * 66)
    print(" exp18: method-corruption vs answer-corruption (mean over seeds)")
    print("=" * 66)
    print(f"{'model':<16}" + "".join(f"{c.replace('worked_', ''):>16}" for c in CONDS))
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
