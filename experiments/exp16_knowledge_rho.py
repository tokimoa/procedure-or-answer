"""exp16 — soundness: accuracy vs memory pollution rho on a KNOWLEDGE task.

Pairs with the procedural side, where accuracy is flat in rho. Prediction: knowledge accuracy FALLS
sharply with rho (a cliff) — because the fact is the content and cannot be
recomputed. Together they show the actionable claim: *verifying memory pays off
only where rho matters (knowledge), and is useless where it doesn't (procedural).*

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/exp16_knowledge_rho.py \
      --models phi4-mini,ministral3-3b,granite4.1-8b --n 24 --k 3 --seeds 3
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
from experiments.exp15_knowledge import make_kb


def mem_worked(kb, rho, rng):
    idx = rng.permutation(len(kb))
    bad = set(idx[:int(round(rho * len(kb)))].tolist())
    out = []
    for i, it in enumerate(kb):
        if i in bad:
            wrong = it["gold"] + int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
            # rebuild the record with the wrong fact (a str.replace on the value
            # string could also mangle the entity id if it contains that substring)
            sol = (f"Product #{it['eid']} record: the warranty period is {wrong} "
                   f"months.\nANSWER: {wrong}")
        else:
            sol = it["worked"]
        out.append({"problem": it["problem"], "sol": sol})
    return out


def acc(be, kb, mem, k):
    embs = None if mem is None else be.embed([m["problem"] for m in mem])
    corr = []
    for it in kb:
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
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--rhos", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default="results/exp16_knowledge_rho.json")
    args = ap.parse_args()
    rhos = [float(x) for x in args.rhos.split(",")]

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
            row = {"model": key, "seed": s, "nomem": acc(be, kb, None, 0), "rho_acc": {}}
            for rho in rhos:
                row["rho_acc"][str(rho)] = acc(be, kb, mem_worked(kb, rho, np.random.default_rng(s + int(rho * 100))), args.k)
            records.append(row)
            accs = " ".join(f"{r}:{row['rho_acc'][str(r)]:.2f}" for r in rhos)
            print(f"  seed{s}: nomem={row['nomem']:.2f} | rho-> {accs}", flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 60)
    print(" exp16 — KNOWLEDGE accuracy vs pollution rho (mean over seeds)")
    print("=" * 60)
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        nm = np.mean([r["nomem"] for r in rs])
        curve = " ".join(f"{r}={np.mean([x['rho_acc'][str(r)] for x in rs]):.2f}" for r in rhos)
        print(f"  {key:<14} nomem={nm:.2f} | {curve}")
    print("=" * 60)
    print("CLIFF expected (vs flat procedural) -> verify memory only for lookup.")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
