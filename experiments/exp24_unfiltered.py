"""Filtered versus unfiltered self-generated exemplar pools on synthetic procedural tasks.

Conditions: nomem, worked_filtered (the model's own correct solutions) and
worked_unfiltered (its first --pool solutions, correct or not). Each record also stores
the unfiltered pool's purity and its wrong exemplars, which exp24b_wrongonly.py reads.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from experiments.pools import POOLS, acc_with
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_filtered", "worked_unfiltered"]


def build_pools(be, pool_fn, target, seed):
    """Return (unfiltered, filtered, attempts, corrects) from one zero-shot pass.

    Candidates are the same problems, from the same rng stream, as in
    pools.build_clean_pool, so the filtered pool matches that function's output. The
    unfiltered pool keeps the first `target` attempts verbatim.
    """
    rng = np.random.default_rng(seed + 333)
    raw = pool_fn(target * 3, rng)
    unfiltered, filtered = [], []
    attempts = corrects = 0
    for it in raw:
        ans, _, txt = mem_solve(be, it["problem"], [])
        ok = ans == it["gold"]
        attempts += 1
        corrects += int(ok)
        if len(unfiltered) < target:
            unfiltered.append(
                {"problem": it["problem"], "gold": it["gold"], "sol": txt, "correct": bool(ok)}
            )
        if ok and len(filtered) < target:
            filtered.append({"problem": it["problem"], "gold": it["gold"], "sol": txt})
        if len(unfiltered) >= target and len(filtered) >= target:
            break
    return unfiltered, filtered, attempts, corrects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp24_unfiltered.json")
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
                unf, fil, attempts, corrects = build_pools(be, pf, args.pool, s)
                purity = float(np.mean([e["correct"] for e in unf])) if unf else 0.0

                # Same exemplar draws for every condition.
                def fresh_rng(seed=s):
                    return np.random.default_rng(seed + 1)

                row = {
                    "model": key,
                    "task": task,
                    "seed": s,
                    "nomem": acc_with(be, None, test, 0, fresh_rng()),
                    "worked_filtered": acc_with(be, fil, test, args.k, fresh_rng()),
                    "worked_unfiltered": acc_with(be, unf, test, args.k, fresh_rng()),
                    "pool_filtered": len(fil),
                    "pool_unfiltered": len(unf),
                    "purity_unfiltered": purity,
                    "gen_attempts": attempts,
                    "gen_corrects": corrects,
                    "wrong_exemplars": [
                        {"problem": e["problem"], "gold": e["gold"], "sol": e["sol"]}
                        for e in unf
                        if not e["correct"]
                    ],
                }
                records.append(row)
                print(
                    f"  {task:<11} seed{s}: "
                    + " ".join(f"{c}={row[c]:.2f}" for c in CONDS)
                    + f"  purity={purity:.2f} pools=({len(fil)},{len(unf)})",
                    flush=True,
                )
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)

    from scipy import stats as st

    def paired(rows, a, b):
        """Return (n, mean of a - b, 95% CI half-width, paired t-test p-value)."""
        d = np.array([r[a] - r[b] for r in rows], float)
        if len(d) < 2:
            return len(d), float(d.mean()) if len(d) else 0.0, 0.0, 1.0
        t, p = st.ttest_rel([r[a] for r in rows], [r[b] for r in rows])
        ci = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return len(d), float(d.mean()), float(ci), float(p)

    print("\n" + "=" * 74)
    print(" exp24: unfiltered vs filtered self-generated memory (paired)")
    print("=" * 74)
    print(f"{'model/task':<26}" + "".join(f"{c:>18}" for c in CONDS) + f"{'purity':>9}")
    for task in tasks:
        for key in [m.strip() for m in args.models.split(",")]:
            rs = [r for r in records if r["model"] == key and r["task"] == task]
            if not rs:
                continue
            line = f"{key + '/' + task:<26}" + "".join(
                f"{np.mean([r[c] for r in rs]):>18.2f}" for c in CONDS
            )
            print(line + f"{np.mean([r['purity_unfiltered'] for r in rs]):>9.2f}")
    for a, b, label in [
        ("worked_filtered", "worked_unfiltered", "filtered - unfiltered"),
        ("worked_unfiltered", "nomem", "unfiltered - nomem"),
        ("worked_filtered", "nomem", "filtered  - nomem"),
    ]:
        n, m, ci, p = paired(records, a, b)
        print(f"POOLED {label:<24} {m:+.3f} +/- {ci:.3f} (95% CI, n={n})  p={p:.4g}")
    print("per-model (filtered - unfiltered):")
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        n, m, ci, p = paired(rs, "worked_filtered", "worked_unfiltered")
        print(f"  {key:<16} {m:+.3f} +/- {ci:.3f} (n={n})  p={p:.4g}")
    print("=" * 74)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
