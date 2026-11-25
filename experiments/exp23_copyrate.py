"""Copy rate when the test item itself is the only exemplar and carries a wrong answer W.

Procedural (multi-step arithmetic): the item's templated solution with the answer field
set to W. Fact lookup: the queried entity's fact record rebuilt with W. Records the
fraction of outputs equal to W and the fraction equal to the correct answer.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp15_knowledge import make_kb
from experiments.exp18_procedure import _arith_hard_item
from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed


def procedural_trial(be, rng):
    q, gold, correct, _, _ = _arith_hard_item(rng)
    wrong = gold + int(rng.choice([-3, -2, 2, 3]))
    bad_sol = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {wrong}", correct)
    ans, _, _ = mem_solve(be, q, [{"problem": q, "sol": bad_sol}])
    return ans, gold, wrong


def knowledge_trial(be, kb, it, rng):
    others = [x["gold"] for x in kb if x["gold"] != it["gold"]]
    wrong = int(rng.choice(others)) if others else it["gold"] + 1
    # Rebuild the record: str.replace on the value could also change the entity id.
    sol = f"Product #{it['eid']} record: the warranty period is {wrong} months.\nANSWER: {wrong}"
    ans, _, _ = mem_solve(be, it["problem"], [{"problem": it["problem"], "sol": sol}])
    return ans, it["gold"], wrong


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
            pc = pr = kc = kk = 0  # procedural: copied W, correct; fact lookup: copied W, correct
            for _ in range(args.n):
                ans, gold, wrong = procedural_trial(be, rng)
                pc += ans == wrong
                pr += ans == gold
            kb = make_kb(args.n, s)
            for it in kb:
                ans, gold, wrong = knowledge_trial(be, kb, it, rng)
                kc += ans == wrong
                kk += ans == gold
            row = {
                "model": key,
                "seed": s,
                "proc_copy": pc / args.n,
                "proc_recompute": pr / args.n,
                "know_copy": kc / len(kb),
                "know_correct": kk / len(kb),
            }
            records.append(row)
            print(
                f"  seed{s}: proc copy={row['proc_copy']:.2f} recompute={row['proc_recompute']:.2f}"
                f" | know copy={row['know_copy']:.2f} correct={row['know_correct']:.2f}",
                flush=True,
            )
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)
    print("\n" + "=" * 64)
    print(" exp23: copy-rate: P(output == planted wrong answer W)")
    print("=" * 64)
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(
            f"  {key:<14} procedural copy={np.mean([r['proc_copy'] for r in rs]):.2f}"
            f"  knowledge copy={np.mean([r['know_copy'] for r in rs]):.2f}"
        )
    print("=" * 64)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
