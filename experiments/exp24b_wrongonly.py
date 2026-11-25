"""Pools made only of each model's own wrong solutions, read from exp24_unfiltered.py output.

Regenerates the same test sets and compares wrong-only accuracy with the no-memory
accuracy stored in the input file, so --test must match the value used for that run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.models import MODELS
from experiments.pools import POOLS, acc_with
from rsiguard.backends import MLXBackend, mx_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/exp24_unfiltered.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp24b_wrongonly.json")
    args = ap.parse_args()

    src = json.loads(Path(args.inp).read_text())
    by_model = {}
    for r in src:
        by_model.setdefault(r["model"], []).append(r)

    records = []
    for key, rows in by_model.items():
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for r in sorted(rows, key=lambda x: (x["task"], x["seed"])):
            wrong = r["wrong_exemplars"]
            if not wrong:
                continue
            s = r["seed"]
            mx_seed(s)
            test = POOLS[r["task"]](args.test, np.random.default_rng(s + 7777))
            acc = acc_with(be, wrong, test, args.k, np.random.default_rng(s + 1))
            row = {
                "model": key,
                "task": r["task"],
                "seed": s,
                "wrongonly": acc,
                "pool_wrong": len(wrong),
                "nomem": r["nomem"],
                "worked_filtered": r["worked_filtered"],
                "worked_unfiltered": r["worked_unfiltered"],
            }
            records.append(row)
            print(
                f"  {r['task']:<11} seed{s}: wrongonly={acc:.2f} "
                f"(nomem={r['nomem']:.2f} unf={r['worked_unfiltered']:.2f} "
                f"pool={len(wrong)})",
                flush=True,
            )
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)

    from scipy import stats as st

    print("\n" + "=" * 70)
    print(" exp24b: wrong-only natural-error pools vs nomem (paired)")
    print("=" * 70)
    print(f"{'model':<16}{'nomem':>8}{'wrongonly':>11}{'unfiltered':>12}")
    for key in by_model:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(
            f"{key:<16}{np.mean([r['nomem'] for r in rs]):>8.2f}"
            f"{np.mean([r['wrongonly'] for r in rs]):>11.2f}"
            f"{np.mean([r['worked_unfiltered'] for r in rs]):>12.2f}"
        )
    d = [r["wrongonly"] - r["nomem"] for r in records]
    t, p = st.ttest_rel([r["wrongonly"] for r in records], [r["nomem"] for r in records])
    ci = st.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d))
    print(f"POOLED wrongonly - nomem: {np.mean(d):+.3f} +/- {ci:.3f} (n={len(d)}) p={p:.4g}")
    print("per-model:")
    for key in by_model:
        rs = [r for r in records if r["model"] == key]
        if len(rs) < 2:
            continue
        dd = [r["wrongonly"] - r["nomem"] for r in rs]
        t, p = st.ttest_rel([r["wrongonly"] for r in rs], [r["nomem"] for r in rs])
        print(f"  {key:<16} {np.mean(dd):+.3f}  p={p:.4g} (n={len(rs)})")
    print("=" * 70)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
