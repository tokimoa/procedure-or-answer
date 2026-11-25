"""Two-fact queries: the answer requires combining two fact records.

Variants: single (direct lookup of one record, as a reference), aggregate (sum of the values
in two records), bridge (a product->supplier record chained to a supplier->lead-time record).
Each query is shown three records in shuffled order: the needed record(s) plus distractor(s).
Conditions: nomem, clean, and corruptions of one or both needed records. For corrupted
conditions, W is the answer obtained by composing the records as shown and
`follow` = (prediction == W). `hits` records, for each needed record, whether embedding
retrieval from the query alone would return it in the top k.
The result file is checkpointed after each condition; a run resumes with the same arguments.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

OFFSETS = [-5, -3, -2, -1, 1, 2, 3, 5]
CONDS = {
    "single": ["nomem", "clean", "corrupt"],
    "aggregate": ["nomem", "clean", "one_corrupt", "both_corrupt"],
    "bridge": ["nomem", "clean", "hop1_corrupt", "hop2_corrupt", "both_corrupt"],
}
TAIL = " End with EXACTLY: ANSWER: <integer>"


def warranty_rec(pid, val):
    return {
        "problem": f"What is the warranty period, in months, of Product #{pid}?",
        "sol": (
            f"Stored fact: the warranty period of Product #{pid} is {val} months.\nANSWER: {val}"
        ),
    }


def maker_rec(pid, sid):
    return {
        "problem": f"What is the ID number of the supplier that makes Product #{pid}?",
        "sol": (f"Stored fact: Product #{pid} is made by Supplier #{sid}.\nANSWER: {sid}"),
    }


def lead_rec(sid, days):
    return {
        "problem": f"What is the lead time, in days, of Supplier #{sid}?",
        "sol": (f"Stored fact: the lead time of Supplier #{sid} is {days} days.\nANSWER: {days}"),
    }


def make_items(n, seed):
    """Deterministic items for one seed, keyed by variant.

    Product ids are distinct 4-digit numbers and supplier ids are distinct 3-digit numbers.
    """
    rng = np.random.default_rng(seed + 3434)
    pids = [int(x) for x in rng.choice(np.arange(1000, 9999), size=5 * n, replace=False)]
    sids = [int(x) for x in rng.choice(np.arange(100, 999), size=2 * n, replace=False)]
    off = lambda: int(rng.choice(OFFSETS))  # noqa: E731

    items = {"single": [], "aggregate": [], "bridge": []}
    for i in range(n):
        # single: one needed record, two distractors
        p, d1, d2 = pids[5 * i], pids[5 * i + 1], pids[5 * i + 2]
        v, vd1, vd2 = (int(x) for x in rng.integers(6, 60, size=3))
        items["single"].append(
            {
                "q": f"What is the warranty period, in months, of Product #{p}?" + TAIL,
                "gold": v,
                "recs": {
                    "clean": [warranty_rec(p, v), warranty_rec(d1, vd1), warranty_rec(d2, vd2)],
                    "corrupt": [
                        warranty_rec(p, v + (o := off())),
                        warranty_rec(d1, vd1),
                        warranty_rec(d2, vd2),
                    ],
                },
                "W": {"corrupt": v + o},
                "need": [warranty_rec(p, v)["problem"]],
            }
        )
        # aggregate: two needed records, one distractor
        a, b, d = pids[5 * i], pids[5 * i + 3], pids[5 * i + 4]
        va, vb, vd = (int(x) for x in rng.integers(6, 60, size=3))
        oa, ob = off(), off()
        # Keep the both_corrupt sum different from gold.
        while oa + ob == 0:
            ob = off()
        items["aggregate"].append(
            {
                "q": (
                    f"What is the total warranty period, in months, of Product #{a} "
                    f"and Product #{b} combined?" + TAIL
                ),
                "gold": va + vb,
                "recs": {
                    "clean": [warranty_rec(a, va), warranty_rec(b, vb), warranty_rec(d, vd)],
                    "one_corrupt": [
                        warranty_rec(a, va + oa),
                        warranty_rec(b, vb),
                        warranty_rec(d, vd),
                    ],
                    "both_corrupt": [
                        warranty_rec(a, va + oa),
                        warranty_rec(b, vb + ob),
                        warranty_rec(d, vd),
                    ],
                },
                "W": {"one_corrupt": va + vb + oa, "both_corrupt": va + vb + oa + ob},
                "need": [warranty_rec(a, va)["problem"], warranty_rec(b, vb)["problem"]],
            }
        )
        # bridge: product -> supplier -> lead time, plus a distractor supplier record
        p = pids[5 * i]
        s, sd = sids[2 * i], sids[2 * i + 1]
        lt, ltd = (int(x) for x in rng.choice(np.arange(6, 60), size=2, replace=False))
        o2 = off()
        # Keep the corrupted lead time different from the distractor's.
        while lt + o2 == ltd:
            o2 = off()
        od = off()
        # Keep the corrupted distractor lead time different from gold.
        while ltd + od == lt:
            od = off()
        items["bridge"].append(
            {
                "q": (
                    f"What is the lead time, in days, of the supplier that makes "
                    f"Product #{p}?" + TAIL
                ),
                "gold": lt,
                "recs": {
                    "clean": [maker_rec(p, s), lead_rec(s, lt), lead_rec(sd, ltd)],
                    # hop 1 points to the distractor supplier, whose record is shown
                    "hop1_corrupt": [maker_rec(p, sd), lead_rec(s, lt), lead_rec(sd, ltd)],
                    "hop2_corrupt": [maker_rec(p, s), lead_rec(s, lt + o2), lead_rec(sd, ltd)],
                    "both_corrupt": [
                        maker_rec(p, sd),
                        lead_rec(s, lt + o2),
                        lead_rec(sd, ltd + od),
                    ],
                },
                "W": {"hop1_corrupt": ltd, "hop2_corrupt": lt + o2, "both_corrupt": ltd + od},
                "need": [maker_rec(p, s)["problem"], lead_rec(s, lt)["problem"]],
            }
        )
    return items


def retrieval_hits(be, items, k):
    """For each item, whether each needed record is in the top k by similarity to the query.

    The store is the clean records of every item of the variant. No text is generated.
    """
    keys = []
    for it in items:
        for rec in it["recs"]["clean"]:
            if rec["problem"] not in keys:
                keys.append(rec["problem"])
    embs = be.embed(keys)
    norms = np.linalg.norm(embs, axis=1)
    out = []
    for it in items:
        qe = be.embed([it["q"]])[0]
        sims = embs @ qe / (norms * np.linalg.norm(qe) + 1e-9)
        top = [keys[j] for j in np.argsort(-sims)[:k]]
        out.append([int(need in top) for need in it["need"]])
    return out


def load_checkpoint(path):
    if not path.exists():
        return {"meta": {}, "records": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("records"), list):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return data


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def summarize(records, models):
    print("\n" + "=" * 78)
    for model in models:
        print(f"-- {model} --")
        for variant, conds in CONDS.items():
            for cond in conds:
                rows = [
                    r
                    for r in records
                    if r["model"] == model and r["variant"] == variant and r["condition"] == cond
                ]
                if not rows:
                    continue
                acc = np.mean([r["correct"] for r in rows])
                fol = [r["follow"] for r in rows if r["follow"] is not None]
                pf = np.mean([r["pred"] is None for r in rows])
                extra = f" follow={np.mean(fol):.3f}" if fol else ""
                print(
                    f"  {variant:<9} {cond:<13} n={len(rows):>3} acc={acc:.3f}"
                    f"{extra} parsefail={pf:.3f}"
                )
    print("=" * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--variants", default="single,aggregate,bridge")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument(
        "--mem-limit-gb",
        type=float,
        default=14.0,
        help="hard Metal memory limit; generation fails rather than swap",
    )
    ap.add_argument("--out", default="results/exp34_two_fact.json")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_path = Path(args.out)
    data = load_checkpoint(out_path)
    meta = {"n": args.n, "k": args.k, "seeds": args.seeds, "max_tokens": args.max_tokens}
    if data["meta"] and data["meta"] != meta:
        raise ValueError("Checkpoint metadata does not match the requested experiment")
    data["meta"] = meta
    done = {
        (r["model"], r["variant"], r["seed"], r["item_index"], r["condition"])
        for r in data["records"]
    }

    import mlx.core as mx

    mx.set_memory_limit(int(args.mem_limit_gb * 1024**3))
    mx.set_cache_limit(2 * 1024**3)

    for model_key in model_keys:
        print(f"\n### model={model_key}", flush=True)
        be = MLXBackend(MODELS[model_key], max_tokens=args.max_tokens)
        for seed in range(args.seeds):
            items = make_items(args.n, seed)
            for variant in variants:
                hits = retrieval_hits(be, items[variant], args.k)
                order_rng = np.random.default_rng(seed * 1000 + 7)
                orders = [order_rng.permutation(3).tolist() for _ in range(args.n)]
                for cond in CONDS[variant]:
                    for i, it in enumerate(items[variant]):
                        key = (model_key, variant, seed, i, cond)
                        if key in done:
                            continue
                        if cond == "nomem":
                            shown = []
                        else:
                            recs = it["recs"][cond]
                            shown = [recs[j] for j in orders[i]]  # same order in every condition
                        # Same decoding seed in every condition, so contrasts are paired.
                        mx_seed(seed * 100000 + i)
                        pred, _, _ = mem_solve(be, it["q"], shown)
                        w = it["W"].get(cond)
                        data["records"].append(
                            {
                                "model": model_key,
                                "variant": variant,
                                "seed": seed,
                                "item_index": i,
                                "condition": cond,
                                "gold": it["gold"],
                                "W": w,
                                "pred": pred,
                                "correct": int(pred == it["gold"]),
                                "follow": None if w is None else int(pred == w),
                                "hits": hits[i],
                            }
                        )
                        done.add(key)
                    save_checkpoint(out_path, data)
            print(
                f"  seed {seed} done; active={mx.get_active_memory() / 1e9:.1f}GB "
                f"peak={mx.get_peak_memory() / 1e9:.1f}GB",
                flush=True,
            )
        summarize(data["records"], [model_key])
        del be
        gc.collect()
        mx.clear_cache()

    summarize(data["records"], model_keys)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
