"""Mask the answer value inside the working while keeping the answer field.

Standalone occurrences of the gold answer on non-declaration lines are replaced by ``___``;
ANSWER/CONFIDENCE lines are left intact. Exemplars come from the archived pools of exp31.
Predictions are checkpointed per item.
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

from experiments.exp28_procedure_only import _is_decl, blank_value
from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from experiments.pools import POOLS
from rsiguard.backends import MLXBackend, mx_seed


def mask_working(solution, gold):
    """Blank the gold value on every line except the ANSWER/CONFIDENCE lines."""
    lines = []
    for line in solution.splitlines():
        lines.append(line if _is_decl(line) else blank_value(line, gold))
    return "\n".join(lines)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/exp31_filtered_audit_final.json")
    ap.add_argument(
        "--models",
        default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b",
    )
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp33_working_mask.json")
    args = ap.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    source = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    pools = defaultdict(list)
    for x in source["exemplars"]:
        pools[(x["model"], x["task"], x["seed"])].append(x)
    for values in pools.values():
        values.sort(key=lambda x: x["exemplar_index"])

    out_path = Path(args.out)
    data = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {"meta": vars(args), "records": []}
    )
    done = {(r["model"], r["task"], r["seed"], r["item_index"]) for r in data["records"]}
    for model in models:
        print(f"\n### model={model}", flush=True)
        be = MLXBackend(MODELS[model], max_tokens=args.max_tokens)
        for task in tasks:
            for seed in range(args.seeds):
                pool = pools[(model, task, seed)]
                test = POOLS[task](args.test, np.random.default_rng(seed + 7777))
                rng = np.random.default_rng(seed + 1)
                sampled = [rng.permutation(len(pool))[: args.k].tolist() for _ in test]
                for i, item in enumerate(test):
                    key = (model, task, seed, i)
                    if key in done:
                        continue
                    mx_seed(seed * 1000 + i)
                    ex = [
                        {
                            "problem": pool[j]["problem"],
                            "sol": mask_working(pool[j]["solution"], pool[j]["gold"]),
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
                            "condition": "working_value_masked_answer_preserved",
                            "prediction": ans,
                            "gold": item["gold"],
                            "correct": bool(ans == item["gold"]),
                            "pool_size": len(pool),
                        }
                    )
                    done.add(key)
                    save(out_path, data)
                rows = [
                    r
                    for r in data["records"]
                    if r["model"] == model and r["task"] == task and r["seed"] == seed
                ]
                print(
                    f"  {task} seed{seed}: {np.mean([r['correct'] for r in rows]):.3f}", flush=True
                )
        with contextlib.suppress(Exception):
            be.mx.clear_cache()
        del be
        gc.collect()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
