"""Method corruption versus answer corruption with exemplars written by the model itself.

The model solves pool problems after k templated demonstrations of the target procedure
(correct: multiply; wrong: add). A generation is kept as an exemplar only if its answer and
working match that procedure; the templated demonstrations never enter the pool.
Conditions: nomem, worked_correct, worked_ans_corrupt (answer field of worked_correct replaced),
worked_wrong_method. One result row per model and seed, including pool retention counts.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import _ANS, build_prompt
from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_correct", "worked_ans_corrupt", "worked_wrong_method"]


def make_item(rng):
    """One warehouse problem with its gold answer and the values the additive method yields."""
    pallets, boxes, items = (
        int(rng.integers(3, 9)),
        int(rng.integers(4, 9)),
        int(rng.integers(3, 8)),
    )
    dmg, cr = int(rng.integers(1, 4)) * 5, int(rng.integers(6, 12))
    total = pallets * boxes * items
    good = total - (total * dmg) // 100
    ans = good // cr
    wt = pallets + boxes + items
    wg = wt - (wt * dmg) // 100
    q = (
        f"A warehouse has {pallets} pallets, each holding {boxes} boxes of {items} items. "
        f"{dmg}% of all items are damaged and removed. The rest are packed into crates of {cr} "
        f"(leftover items are not crated). How many full crates?"
    )
    return {
        "problem": q,
        "gold": ans,
        "total": total,
        "wrong_total": wt,
        "wrong_gold_int": wg // cr,
        "wrong_gold_float": math.floor(wt * (1 - dmg / 100) / cr),
    }


def template_sol(it, method):
    """Templated solution using the correct or the additive method; only seeds elicitation."""
    # The item dict does not carry the parameters, so parse them from the problem text.
    m = re.search(r"(\d+)% of all items.*crates of (\d+)", it["problem"])
    dmg, cr = int(m.group(1)), int(m.group(2))
    pbi = re.search(r"(\d+) pallets, each holding (\d+) boxes of (\d+) items", it["problem"])
    pallets, boxes, items = int(pbi.group(1)), int(pbi.group(2)), int(pbi.group(3))
    if method == "correct":
        total = pallets * boxes * items
        good = total - (total * dmg) // 100
        return (
            f"Total items = {pallets}*{boxes}*{items} = {total}. Damaged = {dmg}% of {total} = "
            f"{(total * dmg) // 100}. Good = {good}. Crates = {good}//{cr} = {good // cr}.\n"
            f"ANSWER: {good // cr}"
        )
    wt = pallets + boxes + items
    wg = wt - (wt * dmg) // 100
    return (
        f"Total items = {pallets}+{boxes}+{items} = {wt}. Damaged = {dmg}% of {wt} = "
        f"{(wt * dmg) // 100}. Good = {wg}. Crates = {wg}//{cr} = {wg // cr}.\n"
        f"ANSWER: {wg // cr}"
    )


def _shows(body, value):
    """True if value appears in body as a standalone number."""
    return re.search(rf"(?<![\d.]){value}(?![\d.])", body) is not None


def elicit(be, item, method, seed_demos):
    """Return the model's solution if it coherently executes the target method, else None."""
    txt = be.gen(build_prompt(item["problem"], seed_demos))
    m = _ANS.search(txt)
    if not m:
        return None
    a, body = int(m.group(1)), txt[: m.start()]
    if method == "correct":
        if a != item["gold"]:
            return None
        if not _shows(body, item["total"]) or _shows(body, item["wrong_total"]):
            return None
    else:
        if a not in (item["wrong_gold_int"], item["wrong_gold_float"]):
            return None
        if not _shows(body, item["wrong_total"]) or _shows(body, item["total"]):
            return None
    return txt


def build_matched_pools(be, target, cand_cap, rng, k_seed=3, verbose="", decouple=False):
    """Return (correct_pool, wrong_pool, stats).

    By default an item enters both pools only when both elicitations verify, so every memory
    condition demonstrates on the same problems. With decouple=True each pool fills
    independently from its own verified items.
    """
    correct_pool, wrong_pool = [], []
    tried = ok_c = ok_w = 0
    while tried < cand_cap:
        if decouple:
            if len(correct_pool) >= target and len(wrong_pool) >= target:
                break
        elif len(correct_pool) >= target:
            break
        it = make_item(rng)
        seeds = [make_item(rng) for _ in range(k_seed)]
        demos_c = [{"problem": s["problem"], "sol": template_sol(s, "correct")} for s in seeds]
        demos_w = [{"problem": s["problem"], "sol": template_sol(s, "wrong")} for s in seeds]
        tried += 1
        c = elicit(be, it, "correct", demos_c)
        if c is not None:
            ok_c += 1
        w = elicit(be, it, "wrong", demos_w)
        if w is not None:
            ok_w += 1
        if decouple:
            if c is not None and len(correct_pool) < target:
                correct_pool.append({"problem": it["problem"], "gold": it["gold"], "sol": c})
            if w is not None and len(wrong_pool) < target:
                wrong_pool.append({"problem": it["problem"], "gold": it["gold"], "sol": w})
        elif c is not None and w is not None:
            correct_pool.append({"problem": it["problem"], "gold": it["gold"], "sol": c})
            wrong_pool.append({"problem": it["problem"], "gold": it["gold"], "sol": w})
    stats = {
        "tried": tried,
        "verified_correct": ok_c,
        "verified_wrong": ok_w,
        "pool": len(correct_pool),
        "pool_wrong": len(wrong_pool),
        "decoupled": decouple,
    }
    print(f"    {verbose} pools: {stats}", flush=True)
    return correct_pool, wrong_pool, stats


def ans_corrupt(pool, rng):
    """Replace the answer field of each exemplar with a nearby wrong integer."""
    out = []
    for it in pool:
        bad = it["gold"] + int(rng.choice([-2, -1, 1, 2, 3]))
        sol, nsub = re.subn(r"ANSWER:\s*-?\d+", f"ANSWER: {bad}", it["sol"])
        if nsub == 0:
            sol = it["sol"] + f"\nANSWER: {bad}"
        out.append({"problem": it["problem"], "gold": it["gold"], "sol": sol})
    return out


def acc(be, mem, test, k, rng):
    corr = []
    for t in test:
        ex = [] if mem is None else [mem[i] for i in rng.permutation(len(mem))[:k]]
        a, _, _ = mem_solve(be, t["problem"], ex)
        corr.append(1.0 if a == t["gold"] else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--pooln", type=int, default=24)
    ap.add_argument("--cand-cap", type=int, default=90)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--decouple",
        action="store_true",
        help="fill each pool independently (see build_matched_pools)",
    )
    ap.add_argument("--out", default="results/exp26_ownstyle.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
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
            correct_pool, wrong_pool, pstats = build_matched_pools(
                be,
                args.pooln,
                args.cand_cap,
                np.random.default_rng(s + 11),
                k_seed=args.k,
                verbose=f"seed{s}",
                decouple=args.decouple,
            )
            trng = np.random.default_rng(s + 7777)
            test = [make_item(trng) for _ in range(args.test)]
            mems = {
                "worked_correct": correct_pool,
                "worked_ans_corrupt": ans_corrupt(correct_pool, np.random.default_rng(s)),
                "worked_wrong_method": wrong_pool,
            }
            row = {
                "model": key,
                "seed": s,
                **pstats,
                "nomem": acc(be, None, test, 0, np.random.default_rng(s)),
            }
            for c in CONDS[1:]:
                row[c] = acc(
                    be,
                    mems[c] if mems[c] else None,
                    test,
                    args.k if mems[c] else 0,
                    np.random.default_rng(s + 1),
                )
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
            with open(args.out, "w") as fp:
                json.dump(records, fp, indent=2)
        del be

    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)
    print("\n" + "=" * 70)
    print(" exp26: own-style method corruption (mean over seeds)")
    print("=" * 70)
    print(f"{'model':<16}" + "".join(f"{c.replace('worked_', ''):>16}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>16.2f}" for c in CONDS))
    print("=" * 70)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
