"""exp17 — retrieval-k sensitivity (robustness check of the dichotomy).

Does the procedure-benefit (procedural) and the fact-benefit (knowledge) hold
across retrieval sizes k? Sweeps k for worked-clean memory on a procedural task
(invoice) and the knowledge task.

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp17_ksens.py \
      --models phi4-mini,granite4.1-8b --ks 1,3,5,8 --seeds 2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend
from experiments.models import MODELS
from experiments.pools import POOLS, build_clean_pool, acc_with
from experiments.exp15_knowledge import make_kb
from experiments.exp16_knowledge_rho import mem_worked, acc as kn_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="phi4-mini,granite4.1-8b")
    ap.add_argument("--ks", default="1,3,5,8")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default="results/exp17_ksens.json")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for s in range(args.seeds):
            # procedural (invoice), worked-clean memory
            test = POOLS["invoice"](args.test, np.random.default_rng(s + 7777))
            clean = build_clean_pool(be, POOLS["invoice"], args.pool, s, "model")
            proc = {str(k): acc_with(be, clean, test, k, np.random.default_rng(s + 1)) for k in ks}
            # knowledge (clean facts)
            kb = make_kb(args.n, s)
            know = {str(k): kn_acc(be, kb, mem_worked(kb, 0.0, np.random.default_rng(s)), k) for k in ks}
            records.append({"model": key, "seed": s, "proc": proc, "know": know})
            print(f"  seed{s} proc(invoice): " + " ".join(f"k{k}={proc[str(k)]:.2f}" for k in ks) +
                  " | know: " + " ".join(f"k{k}={know[str(k)]:.2f}" for k in ks), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 60)
    print(" exp17 — accuracy vs retrieval k (mean over seeds)")
    print("=" * 60)
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        pc = " ".join(f"k{k}={np.mean([r['proc'][str(k)] for r in rs]):.2f}" for k in ks)
        kc = " ".join(f"k{k}={np.mean([r['know'][str(k)] for r in rs]):.2f}" for k in ks)
        print(f"  {key:<14} proc: {pc} | know: {kc}")
    print("=" * 60)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
