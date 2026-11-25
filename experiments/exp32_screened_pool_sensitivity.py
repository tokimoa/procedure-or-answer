"""Sensitivity of downstream accuracy to flawed exemplars that pass the answer filter.

Uses the archived answer-correct pools from exp31 with the diagnostics added by
audit_filtered_pool.py. The answer_filtered condition uses the full pool; screened removes
exemplars with a rounding conflict or a confirmed false equation.
Test items and k-shot sampling match exp24_unfiltered.py. Predictions are checkpointed per item.
"""

import argparse
import contextlib
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from experiments.pools import POOLS
from rsiguard.backends import MLXBackend, mx_seed

# (model, task, seed, exemplar_index) of correct chained calculations that the equation
# regex flags because it stops at the first intermediate equality.
EQUATION_PARSER_FALSE_POSITIVES = {
    ("llama3.2-3b", "invoice", 1, 16),
    ("ministral3-3b", "invoice", 1, 8),
}


def should_remove(x):
    d = x["diagnostic"]
    key = (x["model"], x["task"], x["seed"], x["exemplar_index"])
    confirmed_equation_error = bool(d["equation_issues"]) and (
        key not in EQUATION_PARSER_FALSE_POSITIVES
    )
    return bool(d["rounding_conflict"] or confirmed_equation_error)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit",
        default="results/exp31_filtered_audit_diagnostics_v3.json",
    )
    ap.add_argument(
        "--models",
        default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b",
    )
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument(
        "--conditions",
        default="answer_filtered,screened",
        help="comma-separated subset; screened-only can reuse exp24's baseline",
    )
    ap.add_argument("--out", default="results/exp32_screened_only.json")
    args = ap.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not set(conditions) <= {"answer_filtered", "screened"}:
        raise ValueError(f"Unknown conditions: {conditions}")
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    pools = defaultdict(list)
    for x in audit["exemplars"]:
        pools[(x["model"], x["task"], x["seed"])].append(x)
    for xs in pools.values():
        xs.sort(key=lambda x: x["exemplar_index"])

    out_path = Path(args.out)
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        data = {"meta": vars(args), "records": []}
    done = {
        (r["model"], r["task"], r["seed"], r["item_index"], r["condition"]) for r in data["records"]
    }

    for model in models:
        print(f"\n### model={model}", flush=True)
        be = MLXBackend(MODELS[model], max_tokens=args.max_tokens)
        for task in tasks:
            for seed in range(args.seeds):
                full = pools[(model, task, seed)]
                screened = [x for x in full if not should_remove(x)]
                variants = {
                    name: {"answer_filtered": full, "screened": screened}[name]
                    for name in conditions
                }
                test = POOLS[task](args.test, np.random.default_rng(seed + 7777))
                print(
                    f"  {task} seed{seed}: pool {len(full)} -> {len(screened)}",
                    flush=True,
                )
                for condition, pool in variants.items():
                    rng = np.random.default_rng(seed + 1)
                    sampled = [
                        rng.permutation(len(pool))[: args.k].tolist() if pool else [] for _ in test
                    ]
                    for i, item in enumerate(test):
                        key = (model, task, seed, i, condition)
                        if key in done:
                            continue
                        # The decoding seed depends only on the item, so the two
                        # conditions are paired and a resumed run is reproducible.
                        mx_seed(seed * 1000 + i)
                        ex = [
                            {
                                "problem": pool[j]["problem"],
                                "gold": pool[j]["gold"],
                                "sol": pool[j]["solution"],
                            }
                            for j in sampled[i]
                        ]
                        ans, _, _ = mem_solve(be, item["problem"], ex)
                        data["records"].append(
                            {
                                "model": model,
                                "task": task,
                                "seed": seed,
                                "item_index": i,
                                "condition": condition,
                                "prediction": ans,
                                "gold": item["gold"],
                                "correct": bool(ans == item["gold"]),
                                "pool_size": len(pool),
                            }
                        )
                        done.add(key)
                        save(out_path, data)
                    acc = np.mean(
                        [
                            r["correct"]
                            for r in data["records"]
                            if r["model"] == model
                            and r["task"] == task
                            and r["seed"] == seed
                            and r["condition"] == condition
                        ]
                    )
                    print(f"    {condition}: {acc:.3f}", flush=True)
        with contextlib.suppress(Exception):
            be.mx.clear_cache()
        del be
        gc.collect()

    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
