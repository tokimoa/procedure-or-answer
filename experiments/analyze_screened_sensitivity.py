"""Summarize exp32 with seed-cell paired bootstrap intervals."""
import argparse
import json
from collections import defaultdict

import numpy as np


def ci_mean(values, rng, n_boot=20000):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    data = json.load(open(args.path, encoding="utf-8"))["records"]
    item = defaultdict(dict)
    for r in data:
        key = (r["model"], r["task"], r["seed"], r["item_index"])
        item[key][r["condition"]] = float(r["correct"])
    cells = defaultdict(lambda: defaultdict(list))
    for (model, task, seed, _), conditions in item.items():
        for condition, value in conditions.items():
            cells[(model, task, seed)][condition].append(value)
    rows = []
    for (model, task, seed), conditions in sorted(cells.items()):
        if not {"answer_filtered", "screened"} <= conditions.keys():
            continue
        rows.append({
            "model": model,
            "task": task,
            "seed": seed,
            "answer_filtered": np.mean(conditions["answer_filtered"]),
            "screened": np.mean(conditions["screened"]),
        })
    rng = np.random.default_rng(20260918)
    groups = [("POOLED", rows)]
    for model, task in sorted(set((r["model"], r["task"]) for r in rows)):
        groups.append((f"{model}/{task}", [
            r for r in rows if r["model"] == model and r["task"] == task
        ]))
    for name, group in groups:
        a = np.array([r["answer_filtered"] for r in group])
        b = np.array([r["screened"] for r in group])
        delta = b - a
        lo, hi = ci_mean(delta, rng)
        print(
            f"{name:<30} n={len(group):2d} filtered={a.mean():.3f} "
            f"screened={b.mean():.3f} delta={delta.mean():+.3f} "
            f"95%CI=[{lo:+.3f},{hi:+.3f}]"
        )


if __name__ == "__main__":
    main()
