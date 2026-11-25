"""Copy rate with a planted wrong answer W shown among k exemplars from a full store.

Procedural (multi-step arithmetic): the test item's templated solution with its answer
field set to W, followed by k-1 random exemplars from the model's own correct solutions.
Fact lookup: the queried entity's fact record rebuilt with W inside the full KB, retrieved
top-k by cosine similarity.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp15_knowledge import make_kb
from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from experiments.pools import POOLS, build_clean_pool
from rsiguard.backends import MLXBackend, mx_seed


def proc_trials(be, pool, test, k, rng, n_save):
    """Return (copy rate, gold rate, other rate, per-trial answers, first n_save outputs)."""
    copy = gold = other = 0
    trials, saved = [], []
    for it in test:
        wrong = it["gold"] + int(rng.choice([-3, -2, 2, 3]))
        planted_sol, nsub = re.subn(r"ANSWER:\s*-?\d+", f"ANSWER: {wrong}", it["sol"])
        if nsub == 0:
            planted_sol = it["sol"] + f"\nANSWER: {wrong}"
        planted = {"problem": it["problem"], "sol": planted_sol}
        others = [pool[i] for i in rng.permutation(len(pool))[: max(0, k - 1)]]
        ans, _, txt = mem_solve(be, it["problem"], [planted] + others)
        copy += ans == wrong
        gold += ans == it["gold"]
        other += ans != wrong and ans != it["gold"]
        trials.append({"gold": it["gold"], "W": wrong, "ans": ans})
        if len(saved) < n_save:
            saved.append(txt)
    n = len(test)
    return copy / n, gold / n, other / n, trials, saved


def know_trials(be, kb, k, rng, n_save):
    """Return (copy rate, gold rate, other rate, per-trial answers, first n_save outputs)."""
    embs = be.embed([it["problem"] for it in kb])
    copy = gold = other = 0
    trials, saved = [], []
    for idx, it in enumerate(kb):
        others_vals = [x["gold"] for x in kb if x["gold"] != it["gold"]]
        wrong = int(rng.choice(others_vals)) if others_vals else it["gold"] + 1
        mem = []
        for j, rec in enumerate(kb):
            if j == idx:
                sol = (
                    f"Product #{rec['eid']} record: the warranty period is {wrong} months."
                    f"\nANSWER: {wrong}"
                )
            else:
                sol = rec["worked"]
            mem.append({"problem": rec["problem"], "sol": sol})
        qe = embs[idx]  # same question text as the planted record, which therefore ranks first
        sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
        ex = [mem[i] for i in np.argsort(-sims)[:k]]
        ans, _, txt = mem_solve(be, it["problem"], ex)
        copy += ans == wrong
        gold += ans == it["gold"]
        other += ans != wrong and ans != it["gold"]
        trials.append({"gold": it["gold"], "W": wrong, "ans": ans})
        if len(saved) < n_save:
            saved.append(txt)
    n = len(kb)
    return copy / n, gold / n, other / n, trials, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--n-save", type=int, default=2)
    ap.add_argument("--out", default="results/exp25_copyrate_k3.json")
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
            pool = build_clean_pool(be, POOLS["arith_hard"], args.n, s, "model")
            test = POOLS["arith_hard"](args.n, np.random.default_rng(s + 7777))
            pc, pg, po, ptr, psav = proc_trials(be, pool, test, args.k, rng, args.n_save)
            kb = make_kb(args.n, s)
            kc, kg, ko, ktr, ksav = know_trials(be, kb, args.k, rng, args.n_save)
            row = {
                "model": key,
                "seed": s,
                "pool": len(pool),
                "k": args.k,
                "proc_copy": pc,
                "proc_gold": pg,
                "proc_other": po,
                "know_copy": kc,
                "know_gold": kg,
                "know_other": ko,
                "proc_trials": ptr,
                "know_trials": ktr,
                "proc_samples": psav,
                "know_samples": ksav,
            }
            records.append(row)
            print(
                f"  seed{s}: proc copy={pc:.2f} gold={pg:.2f} other={po:.2f}"
                f" | know copy={kc:.2f} gold={kg:.2f} other={ko:.2f}"
                f" (pool={len(pool)})",
                flush=True,
            )
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)

    print("\n" + "=" * 70)
    print(f" exp25: copy-rate at k={args.k} with full store (mean over seeds)")
    print("=" * 70)
    print(
        f"{'model':<16}{'proc copy':>10}{'proc gold':>10}"
        f"{'know copy':>11}{'know gold':>10}{'know other':>11}"
    )
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(
            f"{key:<16}"
            f"{np.mean([r['proc_copy'] for r in rs]):>10.2f}"
            f"{np.mean([r['proc_gold'] for r in rs]):>10.2f}"
            f"{np.mean([r['know_copy'] for r in rs]):>11.2f}"
            f"{np.mean([r['know_gold'] for r in rs]):>10.2f}"
            f"{np.mean([r['know_other'] for r in rs]):>11.2f}"
        )
    print("=" * 70)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
