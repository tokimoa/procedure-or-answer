"""Archive answer-filtered procedural pools for auditing.

Builds each model's own pool with exp24_unfiltered.build_pools and saves every answer-correct
exemplar together with the expected intermediate values and rule-based hints. The output is
the input of audit_filtered_pool.py and finalize_filtered_pool_audit.py.
Checkpointed per (model, task, seed) cell.
"""

import argparse
import contextlib
import gc
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp24_unfiltered import build_pools
from experiments.models import MODELS
from experiments.pools import POOLS
from rsiguard.backends import MLXBackend, mx_seed


def expected_trace(task, problem, gold):
    """Expected intermediate values for a problem, recomputed from its text."""
    if task == "arith_hard":
        m = re.search(
            r"(\d+) pallets, each holding (\d+) boxes of (\d+) items\. "
            r"(\d+)%.*crates of (\d+)",
            problem,
        )
        if not m:
            raise ValueError(f"Could not parse arith_hard problem: {problem}")
        pallets, boxes, items, damage_pct, crate = map(int, m.groups())
        total = pallets * boxes * items
        damaged = (total * damage_pct) // 100
        good = total - damaged
        return {
            "inputs": [pallets, boxes, items, damage_pct, crate],
            "total": total,
            "damaged_floor": damaged,
            "remaining": good,
            "gold": gold,
        }
    if task == "invoice":
        m = re.search(
            r"(\d+) units x JPY (\d+); (\d+) units x JPY (\d+).*Add (\d+)%",
            problem,
        )
        if not m:
            raise ValueError(f"Could not parse invoice problem: {problem}")
        q1, p1, q2, p2, tax = map(int, m.groups())
        subtotal = q1 * p1 + q2 * p2
        return {
            "inputs": [q1, p1, q2, p2, tax],
            "line_1": q1 * p1,
            "line_2": q2 * p2,
            "subtotal": subtotal,
            "gold": gold,
        }
    raise ValueError(f"Unsupported audit task: {task}")


def audit_hints(task, solution, trace):
    """Rule-based hints for manual review; these are not final labels."""
    lower = solution.lower()

    def number_forms(value):
        return (str(value), f"{value:,}")

    def mentions(value):
        return any(
            re.search(rf"(?<!\d){re.escape(form)}(?!\d)", solution) for form in number_forms(value)
        )

    if task == "arith_hard":
        expected_mentions = {
            key: mentions(trace[key]) for key in ("total", "damaged_floor", "remaining", "gold")
        }
        structure = (
            ("*" in solution or "multiply" in lower or "×" in solution)
            and ("%" in solution or "percent" in lower)
            and ("/" in solution or "divid" in lower or "crate" in lower)
        )
    else:
        expected_mentions = {
            key: mentions(trace[key]) for key in ("line_1", "line_2", "subtotal", "gold")
        }
        structure = (
            ("*" in solution or "multiply" in lower or "×" in solution)
            and ("+" in solution or "subtotal" in lower)
            and ("%" in solution or "tax" in lower or ("110" in solution and "100" in solution))
        )
    return {
        "expected_mentions": expected_mentions,
        "has_expected_structure": bool(structure),
        "needs_manual_review": not structure or sum(expected_mentions.values()) < 2,
    }


def load_checkpoint(path):
    if not path.exists():
        return {"meta": {}, "cells": [], "exemplars": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not all(k in data for k in ("meta", "cells", "exemplars")):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return data


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b",
    )
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp31_filtered_audit.json")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown_models = [m for m in model_keys if m not in MODELS]
    unknown_tasks = [t for t in tasks if t not in POOLS]
    if unknown_models or unknown_tasks:
        raise ValueError(f"Unknown models={unknown_models} or unsupported tasks={unknown_tasks}")

    out_path = Path(args.out)
    data = load_checkpoint(out_path)
    expected_meta = {
        "models": model_keys,
        "tasks": tasks,
        "pool": args.pool,
        "seeds": args.seeds,
        "max_tokens": args.max_tokens,
    }
    if data["meta"] and data["meta"] != expected_meta:
        raise ValueError("Checkpoint metadata does not match the requested audit")
    data["meta"] = expected_meta
    done = {(c["model"], c["task"], c["seed"]) for c in data["cells"]}

    for model_key in model_keys:
        print(f"\n### model={model_key}", flush=True)
        be = MLXBackend(MODELS[model_key], max_tokens=args.max_tokens)
        for task in tasks:
            for seed in range(args.seeds):
                cell_key = (model_key, task, seed)
                if cell_key in done:
                    continue
                mx_seed(seed)
                _, filtered, attempts, corrects = build_pools(be, POOLS[task], args.pool, seed)
                for exemplar_index, exemplar in enumerate(filtered):
                    trace = expected_trace(task, exemplar["problem"], exemplar["gold"])
                    data["exemplars"].append(
                        {
                            "model": model_key,
                            "task": task,
                            "seed": seed,
                            "exemplar_index": exemplar_index,
                            "problem": exemplar["problem"],
                            "gold": exemplar["gold"],
                            "solution": exemplar["sol"],
                            "expected_trace": trace,
                            "audit_hints": audit_hints(task, exemplar["sol"], trace),
                            "human_label": None,
                            "human_notes": "",
                        }
                    )
                data["cells"].append(
                    {
                        "model": model_key,
                        "task": task,
                        "seed": seed,
                        "pool_size": len(filtered),
                        "attempts": attempts,
                        "corrects_seen": corrects,
                    }
                )
                done.add(cell_key)
                save_checkpoint(out_path, data)
                flagged = sum(
                    x["audit_hints"]["needs_manual_review"]
                    for x in data["exemplars"]
                    if x["model"] == model_key and x["task"] == task and x["seed"] == seed
                )
                print(
                    f"  {task:<11} seed{seed}: pool={len(filtered)} "
                    f"attempts={attempts} hints_flagged={flagged}",
                    flush=True,
                )

        with contextlib.suppress(Exception):
            be.mx.clear_cache()
        del be
        gc.collect()

    print(
        f"saved {len(data['exemplars'])} retained exemplars from "
        f"{len(data['cells'])} cells -> {out_path}"
    )


if __name__ == "__main__":
    main()
