"""exp22 — REAL procedural benchmark (GSM8K): the dichotomy's procedural side on real data.

Mirrors exp13 on a real, diverse math benchmark (not templated). Memory = the model's
OWN correct worked solutions to other GSM8K problems (matched style); retrieval = top-k
by embedding similarity. Prediction (procedural side, on real data):
worked_clean ≈ worked_ans_bad  >  answer_only ≈ nomem
(the demonstrated reasoning helps; corrupting its final answer is harmless — the model
recomputes).

Usage:
  HF_HUB_OFFLINE=0 .venv/bin/python experiments/exp22_gsm8k.py \
      --models llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b --n-pool 40 --n-test 24 --seeds 3
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

CONDS = ["nomem", "worked_clean", "worked_ans_bad", "answer_only"]
BBH_TASK = "multistep_arithmetic_two"  # default; override with --bbh-task


def load_data(dataset, n_pool, n_test, seed, bbh_task=BBH_TASK):
    from datasets import load_dataset
    rng = np.random.default_rng(seed + 909)
    items = []
    if dataset == "bbh":  # BIG-Bench Hard subtask with integer targets, recurring procedure
        ds = load_dataset("lukaemon/bbh", bbh_task)["test"]
        idx = rng.choice(len(ds), size=min(n_pool + n_test, len(ds)), replace=False)
        for i in idx:
            ex = ds[int(i)]
            items.append({"problem": ex["input"], "gold": int(str(ex["target"]).strip())})
    elif dataset == "svamp":
        ds = load_dataset("ChilleD/SVAMP")["test"]  # simple, structured, recurring by Type
        idx = rng.choice(len(ds), size=min(n_pool + n_test, len(ds)), replace=False)
        for i in idx:
            ex = ds[int(i)]
            items.append({"problem": ex["Body"] + " " + ex["Question"],
                          "gold": int(round(float(ex["Answer"])))})
    else:
        ds = load_dataset("openai/gsm8k", "main")["test"]
        idx = rng.choice(len(ds), size=n_pool + n_test, replace=False)
        for i in idx:
            ex = ds[int(i)]
            gold = int(ex["answer"].split("####")[-1].strip().replace(",", ""))
            items.append({"problem": ex["question"], "gold": gold})
    return items[:n_pool], items[n_pool:]


def build_clean_pool(be, pool_items):
    """Pool of the model's OWN correct worked solutions (matched style)."""
    clean = []
    for it in pool_items:
        ans, _, txt = mem_solve(be, it["problem"], [])
        if ans == it["gold"]:
            sol = txt if "ANSWER:" in txt else f"{txt}\nANSWER: {it['gold']}"
            clean.append({"problem": it["problem"], "gold": it["gold"], "sol": sol})
    return clean


def mem_variant(pool, cond, rng):
    out = []
    for p in pool:
        if cond == "answer_only":
            sol = f"ANSWER: {p['gold']}"
        elif cond == "worked_ans_bad":
            bad = p["gold"] + int(rng.choice([-3, -2, -1, 1, 2, 3]))
            sol = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {bad}", p["sol"])
            if sol == p["sol"]:
                sol = p["sol"] + f"\nANSWER: {bad}"
        else:
            sol = p["sol"]
        out.append({"problem": p["problem"], "sol": sol})
    return out


def acc(be, pool, test, cond, k, seed):
    rng = np.random.default_rng(seed)
    if cond == "nomem" or not pool:
        mem, embs = None, None
    else:
        mem = mem_variant(pool, cond, rng)
        embs = be.embed([m["problem"] for m in mem])
    corr = []
    for t in test:
        if mem is None:
            ex = []
        else:
            qe = be.embed([t["problem"]])[0]
            sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
            ex = [mem[i] for i in np.argsort(-sims)[:k]]
        a, _, _ = mem_solve(be, t["problem"], ex)
        corr.append(1.0 if a == t["gold"] else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--n-pool", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k","svamp","bbh"])
    ap.add_argument("--bbh-task", default=BBH_TASK)
    ap.add_argument("--out", default="results/exp22_gsm8k.json")
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
            pool_items, test = load_data(args.dataset, args.n_pool, args.n_test, s,
                                         bbh_task=args.bbh_task)
            pool = build_clean_pool(be, pool_items)
            row = {"model": key, "seed": s, "pool_size": len(pool)}
            for c in CONDS:
                row[c] = acc(be, pool, test, c, args.k, s)
            records.append(row)
            print(f"  seed{s} (pool={len(pool)}): " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\n" + "=" * 72)
    print(" exp22 — GSM8K (REAL procedural): procedure helps, answer-corruption harmless")
    print("=" * 72)
    print(f"{'model':<16}" + "".join(f"{c:>15}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>15.2f}" for c in CONDS))
    print("=" * 72)
    print("dichotomy confirmed on REAL data if worked_clean ≈ worked_ans_bad > answer_only ≈ nomem")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
