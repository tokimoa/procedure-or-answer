"""PopQA on a fixed test set with direct and non-oracle retrieval.

Direct retrieval indexes each fact record by its verbatim question. Non-oracle retrieval indexes
a structured (subject, relation) key, and the store also holds other relations of the same
entity as distractors.
Conditions: nomem, direct_clean, direct_corrupt, nonoracle_clean, nonoracle_corrupt.
Every item is checkpointed, so a run can be resumed with the same arguments.
"""

import argparse
import contextlib
import gc
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp19_popqa import correct, norm, solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

CONDS = [
    "nomem",
    "direct_clean",
    "direct_corrupt",
    "nonoracle_clean",
    "nonoracle_corrupt",
]


def _qnorm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _aliases(row):
    """Normalized gold answers: the object and its aliases."""
    try:
        aliases = json.loads(row["o_aliases"]) if row.get("o_aliases") else []
    except (TypeError, json.JSONDecodeError):
        aliases = []
    return [norm(x) for x in [row["obj"], *aliases] if x]


def load_fixed_items(tail_n=5000, max_subjects=0):
    """Return (store, targets) for long-tail entities with at least two unambiguous relations.

    Questions that occur more than once are dropped before grouping. One deterministic relation
    per entity becomes the query; the entity's other relations stay in the store as distractors.
    """
    from datasets import load_dataset

    rows = list(load_dataset("akariasai/PopQA")["test"])
    rows.sort(key=lambda x: (float(x["s_pop"] or 0), int(x["id"])))
    rows = rows[:tail_n]
    qcounts = Counter(_qnorm(r["question"]) for r in rows)

    by_subject = defaultdict(list)
    for row in rows:
        if qcounts[_qnorm(row["question"])] == 1:
            by_subject[int(row["subj_id"])].append(row)

    eligible = []
    for subject_id, subject_rows in by_subject.items():
        by_prop = defaultdict(list)
        for row in subject_rows:
            by_prop[int(row["prop_id"])].append(row)
        unique = [group[0] for group in by_prop.values() if len(group) == 1]
        unique.sort(key=lambda x: (int(x["prop_id"]), int(x["id"])))
        if len(unique) >= 2:
            eligible.append((subject_id, unique))
    eligible.sort(key=lambda x: (float(x[1][0]["s_pop"] or 0), x[0]))
    if max_subjects > 0:
        eligible = eligible[:max_subjects]

    store, targets = [], []
    for subject_id, subject_rows in eligible:
        target_row = subject_rows[0]
        for row in subject_rows:
            rid = str(row["id"])
            item = {
                "rid": rid,
                "subject_id": subject_id,
                "subj": str(row["subj"]),
                "prop": str(row["prop"]),
                "obj": str(row["obj"]),
                "q": str(row["question"]),
                "golds": _aliases(row),
                "direct_key": str(row["question"]),
                "nonoracle_key": (f"Subject: {row['subj']}. Relation: {row['prop']}."),
            }
            store.append(item)
            if row is target_row:
                targets.append(item)
    return store, targets


def corrupted_values(store):
    """Map each record id to a deterministic wrong object taken from another record."""
    values = [item["obj"] for item in store]
    out = {}
    for i, item in enumerate(store):
        golds = set(item["golds"])
        wrong = None
        for offset in range(1, len(store)):
            candidate = values[(i + offset) % len(store)]
            if norm(candidate) not in golds:
                wrong = candidate
                break
        if wrong is None:
            raise RuntimeError(f"Could not find a wrong value for record {item['rid']}")
        out[item["rid"]] = wrong
    return out


def make_memory(store, corrupt_map, key_field, corrupt=False):
    memory = []
    for item in store:
        value = corrupt_map[item["rid"]] if corrupt else item["obj"]
        memory.append(
            {
                "q": item[key_field],
                "rid": item["rid"],
                "sol": (
                    f"Stored fact: for {item['subj']}, {item['prop']} is {value}.\nANSWER: {value}"
                ),
            }
        )
    return memory


def retrieve(be, query, keys, embeddings, k):
    """Indices of the k stored keys most similar to the query (cosine)."""
    qe = be.embed([query])[0]
    sims = embeddings @ qe / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qe) + 1e-9)
    return np.argsort(-sims)[:k].tolist()


def load_checkpoint(path):
    if not path.exists():
        return {"meta": {}, "records": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return data


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def summarize(records, models):
    print("\n" + "=" * 82)
    print(" exp29: fixed PopQA with direct vs non-oracle retrieval")
    print("=" * 82)
    for model in models:
        print(f"-- {model} --")
        for cond in CONDS:
            rows = [r for r in records if r["model"] == model and r["condition"] == cond]
            if not rows:
                continue
            acc = np.mean([r["correct"] for r in rows])
            hit = [r["hit_at_k"] for r in rows if r["hit_at_k"] is not None]
            suffix = f" hit@k={np.mean(hit):.3f}" if hit else ""
            print(f"  {cond:<20} n={len(rows):>3} acc={acc:.3f}{suffix}")
    print("=" * 82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b",
    )
    ap.add_argument("--tail-n", type=int, default=5000)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--out", default="results/exp29_popqa_fixed.json")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    store, targets = load_fixed_items(args.tail_n, args.max_subjects)
    if len(targets) < 2:
        raise ValueError("Fixed-set selection produced fewer than two target queries")
    corrupt_map = corrupted_values(store)
    out_path = Path(args.out)
    data = load_checkpoint(out_path)
    expected_meta = {
        "tail_n": args.tail_n,
        "max_subjects": args.max_subjects,
        "k": args.k,
        "seed": args.seed,
        "n_store": len(store),
        "n_targets": len(targets),
        "target_ids": [t["rid"] for t in targets],
    }
    if data["meta"] and data["meta"] != expected_meta:
        raise ValueError("Checkpoint metadata does not match the requested experiment")
    data["meta"] = expected_meta
    done = {(r["model"], r["condition"], r["target_rid"]) for r in data["records"]}
    print(
        f"fixed set: targets={len(targets)} store={len(store)} tail={args.tail_n} k={args.k}",
        flush=True,
    )

    direct_keys = [item["direct_key"] for item in store]
    nonoracle_keys = [item["nonoracle_key"] for item in store]
    memories = {
        ("direct", False): make_memory(store, corrupt_map, "direct_key", corrupt=False),
        ("direct", True): make_memory(store, corrupt_map, "direct_key", corrupt=True),
        ("nonoracle", False): make_memory(store, corrupt_map, "nonoracle_key", corrupt=False),
        ("nonoracle", True): make_memory(store, corrupt_map, "nonoracle_key", corrupt=True),
    }

    for model_key in model_keys:
        print(f"\n### model={model_key}", flush=True)
        be = MLXBackend(MODELS[model_key], max_tokens=args.max_tokens)
        direct_embs = be.embed(direct_keys)
        nonoracle_embs = be.embed(nonoracle_keys)

        for cond in CONDS:
            mode = None if cond == "nomem" else cond.split("_", 1)[0]
            is_corrupt = cond.endswith("corrupt")
            memory = [] if mode is None else memories[(mode, is_corrupt)]
            keys = direct_keys if mode == "direct" else nonoracle_keys
            embeddings = direct_embs if mode == "direct" else nonoracle_embs
            print(f"  condition={cond}", flush=True)
            for item_index, target in enumerate(targets):
                key = (model_key, cond, target["rid"])
                if key in done:
                    continue
                if cond == "nomem":
                    chosen = []
                    hit_at_1 = hit_at_k = None
                else:
                    idx = retrieve(be, target["q"], keys, embeddings, args.k)
                    chosen = [memory[i] for i in idx]
                    retrieved_ids = [store[i]["rid"] for i in idx]
                    hit_at_1 = int(bool(retrieved_ids) and retrieved_ids[0] == target["rid"])
                    hit_at_k = int(target["rid"] in retrieved_ids)
                # The decoding seed depends only on the item, so conditions are paired and
                # a resumed run reproduces the same outputs.
                mx_seed(args.seed * 100000 + item_index)
                pred = solve(be, target["q"], chosen)
                row = {
                    "model": model_key,
                    "condition": cond,
                    "target_rid": target["rid"],
                    "subject_id": target["subject_id"],
                    "q": target["q"],
                    "gold": target["obj"],
                    "pred": pred,
                    "correct": int(correct(pred, target["golds"])),
                    "hit_at_1": hit_at_1,
                    "hit_at_k": hit_at_k,
                    "retrieved_ids": [x["rid"] for x in chosen],
                }
                data["records"].append(row)
                done.add(key)
                save_checkpoint(out_path, data)
                if (item_index + 1) % 10 == 0 or item_index + 1 == len(targets):
                    print(
                        f"    {item_index + 1}/{len(targets)} checkpointed",
                        flush=True,
                    )

        with contextlib.suppress(Exception):
            be.mx.clear_cache()
        del be
        gc.collect()
        summarize(data["records"], [model_key])

    summarize(data["records"], model_keys)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
