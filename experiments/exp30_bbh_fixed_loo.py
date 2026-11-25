"""BBH on the full fixed test set with leave-one-out memory.

Each model first solves every item with no memory; its correct generations form the pool.
With memory, an item's own record is excluded before cosine top-k retrieval.
Conditions: nomem, worked_clean, worked_ans_bad. Pool generations and predictions are
checkpointed per item, so an interrupted run resumes without regenerating them.
"""

import argparse
import contextlib
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_clean", "worked_ans_bad"]


def load_bbh(task):
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", task)["test"]
    return [
        {
            "item_id": str(i),
            "problem": str(ex["input"]),
            "gold": int(str(ex["target"]).strip()),
        }
        for i, ex in enumerate(ds)
    ]


def corrupt_solution(solution, gold, source_index):
    """Replace the answer field with gold plus a nonzero offset chosen by source_index."""
    offsets = (-3, -2, -1, 1, 2, 3)
    wrong = gold + offsets[source_index % len(offsets)]
    out, count = re.subn(r"ANSWER:\s*-?\d+", f"ANSWER: {wrong}", solution)
    return out if count else solution + f"\nANSWER: {wrong}"


def load_checkpoint(path):
    if not path.exists():
        return {"meta": {}, "pools": {}, "records": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not all(k in data for k in ("meta", "pools", "records")):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return data


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def retrieve(be, problem, pool, embeddings, k, excluded_item_id):
    """Indices of up to k pool exemplars most similar to the problem, excluding its own."""
    qe = be.embed([problem])[0]
    sims = embeddings @ qe / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qe) + 1e-9)
    for j, exemplar in enumerate(pool):
        if exemplar["item_id"] == excluded_item_id:
            sims[j] = -np.inf
    usable = min(k, sum(np.isfinite(sims)))
    return np.argsort(-sims)[:usable].tolist()


def summarize(records, model_keys):
    print("\n" + "=" * 78)
    print(" exp30: BBH full fixed-set leave-one-out evaluation")
    print("=" * 78)
    for model in model_keys:
        print(f"-- {model} --")
        for cond in CONDS:
            rows = [r for r in records if r["model"] == model and r["condition"] == cond]
            if rows:
                print(
                    f"  {cond:<16} n={len(rows):>3} acc={np.mean([r['correct'] for r in rows]):.3f}"
                )
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b",
    )
    ap.add_argument("--task", default="multistep_arithmetic_two")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--out", default="results/exp30_bbh_fixed.json")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    items = load_bbh(args.task)
    if args.max_items > 0:
        items = items[: args.max_items]
    out_path = Path(args.out)
    data = load_checkpoint(out_path)
    expected_meta = {
        "task": args.task,
        "k": args.k,
        "seed": args.seed,
        "max_items": args.max_items,
        "n_items": len(items),
        "item_ids": [x["item_id"] for x in items],
    }
    if data["meta"] and data["meta"] != expected_meta:
        raise ValueError("Checkpoint metadata does not match the requested experiment")
    data["meta"] = expected_meta
    done = {(r["model"], r["condition"], r["item_id"]) for r in data["records"]}
    print(f"fixed BBH set: task={args.task} n={len(items)} k={args.k}", flush=True)

    for model_key in model_keys:
        print(f"\n### model={model_key}", flush=True)
        be = MLXBackend(MODELS[model_key], max_tokens=args.max_tokens)
        model_pool_rows = data["pools"].setdefault(model_key, [])
        generated = {r["item_id"] for r in model_pool_rows}

        print("  building/resuming zero-shot pool", flush=True)
        for item_index, item in enumerate(items):
            if item["item_id"] in generated:
                continue
            mx_seed(args.seed * 100000 + item_index)
            pred, _, text = mem_solve(be, item["problem"], [])
            pool_row = {
                **item,
                "pred": pred,
                "correct": int(pred == item["gold"]),
                "sol": text,
            }
            model_pool_rows.append(pool_row)
            generated.add(item["item_id"])
            record_key = (model_key, "nomem", item["item_id"])
            if record_key not in done:
                data["records"].append(
                    {
                        "model": model_key,
                        "condition": "nomem",
                        "item_id": item["item_id"],
                        "gold": item["gold"],
                        "pred": pred,
                        "correct": int(pred == item["gold"]),
                        "retrieved_ids": [],
                    }
                )
                done.add(record_key)
            save_checkpoint(out_path, data)
            if (item_index + 1) % 10 == 0 or item_index + 1 == len(items):
                print(f"    pool {item_index + 1}/{len(items)} checkpointed", flush=True)

        # Restore dataset order after a resumed run.
        by_id = {r["item_id"]: r for r in model_pool_rows}
        model_pool_rows[:] = [by_id[x["item_id"]] for x in items]
        clean_pool = [r for r in model_pool_rows if r["correct"]]
        print(f"  answer-filtered pool size={len(clean_pool)}", flush=True)
        if not clean_pool:
            raise RuntimeError(f"No correct exemplars for {model_key}")
        embeddings = be.embed([r["problem"] for r in clean_pool])
        memories = {
            "worked_clean": [
                {"problem": r["problem"], "sol": r["sol"], "item_id": r["item_id"]}
                for r in clean_pool
            ],
            "worked_ans_bad": [
                {
                    "problem": r["problem"],
                    "sol": corrupt_solution(r["sol"], r["gold"], int(r["item_id"])),
                    "item_id": r["item_id"],
                }
                for r in clean_pool
            ],
        }

        for cond in CONDS[1:]:
            print(f"  condition={cond}", flush=True)
            memory = memories[cond]
            for item_index, item in enumerate(items):
                record_key = (model_key, cond, item["item_id"])
                if record_key in done:
                    continue
                idx = retrieve(be, item["problem"], clean_pool, embeddings, args.k, item["item_id"])
                exemplars = [memory[j] for j in idx]
                mx_seed(args.seed * 100000 + item_index)
                pred, _, _ = mem_solve(be, item["problem"], exemplars)
                data["records"].append(
                    {
                        "model": model_key,
                        "condition": cond,
                        "item_id": item["item_id"],
                        "gold": item["gold"],
                        "pred": pred,
                        "correct": int(pred == item["gold"]),
                        "retrieved_ids": [x["item_id"] for x in exemplars],
                    }
                )
                done.add(record_key)
                save_checkpoint(out_path, data)
                if (item_index + 1) % 10 == 0 or item_index + 1 == len(items):
                    print(f"    {item_index + 1}/{len(items)} checkpointed", flush=True)

        with contextlib.suppress(Exception):
            be.mx.clear_cache()
        del be
        gc.collect()
        summarize(data["records"], [model_key])

    summarize(data["records"], model_keys)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
