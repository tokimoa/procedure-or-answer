"""Procedure-only memory: exemplars with the answer field removed.

Conditions: nomem, worked_clean, proc_only (ANSWER/CONFIDENCE lines deleted), proc_blank
(additionally blanks the answer value wherever it appears in the working), answer_only.
Pools are the model's own correct solutions, on synthetic tasks (--regime proc) or BBH
(--regime bbh). Each row records accuracy and the parse-failure rate per condition.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from experiments.pools import POOLS, build_clean_pool
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_clean", "proc_only", "proc_blank", "answer_only"]

# A declaration line is an ANSWER:/CONFIDENCE: field, possibly behind markdown or bullet
# decoration. Such lines are deleted wherever they occur instead of truncating at the first
# one, because models sometimes state the answer before the reasoning.
_DECO = re.compile(r"^[\s*#>\-•]+")
_DECL = re.compile(r"^(?:ANSWER|CONFIDENCE)\s*\**\s*:", re.I)
BLANK = "___"


def _is_decl(line):
    return bool(_DECL.match(_DECO.sub("", line)))


def strip_answer(sol):
    """Remove the ANSWER/CONFIDENCE lines from a solution.

    The derived value usually still appears in the working; strip_and_blank removes that too.
    """
    out = "\n".join(ln for ln in sol.splitlines() if not _is_decl(ln)).strip()
    return out if out else sol  # never emit an empty exemplar


# Matches a standalone number, not part of a longer numeral. A separator blocks the match
# only when it joins digits ("1,535", "3.35"); a sentence period after the value does not.
def _num_re(form):
    return rf"(?<!\d)(?<!\d[.,]){re.escape(form)}(?!\d)(?![.,]\d)"


def blank_value(text, gold):
    """Replace standalone occurrences of the answer value (plain and comma-grouped) with BLANK.

    Digits inside larger numbers are left alone. An intermediate value equal to the answer is
    blanked as well.
    """
    out = text
    for form in {str(gold), f"{gold:,}"}:
        out = re.sub(_num_re(form), BLANK, out)
    return out


def strip_and_blank(sol, gold):
    """Remove the ANSWER/CONFIDENCE lines and blank the answer value in the working."""
    return blank_value(strip_answer(sol), gold)


def mem_variant(pool, cond, rng):
    """Pool entries are {'problem', 'gold', 'sol'}; sol is the model's own solution."""
    out = []
    for p in pool:
        if cond == "answer_only":
            sol = f"ANSWER: {p['gold']}"
        elif cond == "proc_only":
            sol = strip_answer(p["sol"])
        elif cond == "proc_blank":
            sol = strip_and_blank(p["sol"], p["gold"])
        else:  # worked_clean
            sol = p["sol"]
        out.append({"problem": p["problem"], "sol": sol})
    return out


def run_cond(be, pool, test, cond, k, seed, retrieval):
    """Return (accuracy, parse_failure_rate).

    retrieval is 'uniform' (k random exemplars) or 'cosine' (top-k by embedding similarity).
    """
    rng = np.random.default_rng(seed + 1)
    if cond == "nomem" or not pool:
        mem, embs = None, None
    else:
        mem = mem_variant(pool, cond, rng)
        embs = be.embed([m["problem"] for m in mem]) if retrieval == "cosine" else None
    corr, fails = [], []
    for t in test:
        if mem is None:
            ex = []
        elif retrieval == "cosine":
            qe = be.embed([t["problem"]])[0]
            sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
            ex = [mem[i] for i in np.argsort(-sims)[:k]]
        else:
            ex = [mem[i] for i in rng.permutation(len(mem))[:k]]
        a, _, _ = mem_solve(be, t["problem"], ex)
        fails.append(1.0 if a is None else 0.0)
        corr.append(1.0 if a == t["gold"] else 0.0)
    return float(np.mean(corr)), float(np.mean(fails))


def cells_proc(be, args, seed, task):
    """Synthetic task: return (pool, test items, retrieval mode), built as in exp13."""
    pf = POOLS[task]
    test = pf(args.test, np.random.default_rng(seed + 7777))
    pool = build_clean_pool(be, pf, args.pool, seed, "model")
    return pool, test, "uniform"


def cells_bbh(be, args, seed, task):
    """BBH task: return (pool, test items, retrieval mode), built as in exp22."""
    from experiments.exp22_gsm8k import build_clean_pool as bbh_pool
    from experiments.exp22_gsm8k import load_data

    pool_items, test = load_data("bbh", args.pool, args.test, seed, bbh_task=task)
    return bbh_pool(be, pool_items), test[: args.test], "cosine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="proc", choices=["proc", "bbh"])
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--tasks", default="arith_hard,invoice")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp28_proc_only.json")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    build = cells_proc if args.regime == "proc" else cells_bbh

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for task in tasks:
            for s in range(args.seeds):
                mx_seed(s)
                pool, test, retrieval = build(be, args, s, task)
                row = {
                    "model": key,
                    "task": task,
                    "seed": s,
                    "pool": len(pool),
                    "regime": args.regime,
                }
                for c in CONDS:
                    a, f = run_cond(be, pool, test, c, args.k, s, retrieval)
                    row[c], row[c + "_parsefail"] = a, f
                records.append(row)
                print(
                    f"  {task:<12} seed{s} (pool={len(pool)}): "
                    + " ".join(f"{c}={row[c]:.2f}" for c in CONDS)
                    + " | parsefail "
                    + " ".join(f"{c}={row[c + '_parsefail']:.2f}" for c in CONDS[1:]),
                    flush=True,
                )
                with open(args.out, "w") as fp:
                    json.dump(records, fp, indent=2)
        del be

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)

    print("\n" + "=" * 78)
    print(" exp28: procedure-only (mean over seeds)")
    print("=" * 78)
    print(f"{'model/task':<26}" + "".join(f"{c:>14}" for c in CONDS))
    for task in tasks:
        for key in [m.strip() for m in args.models.split(",") if m.strip()]:
            rs = [r for r in records if r["model"] == key and r["task"] == task]
            if rs:
                print(
                    f"{key + '/' + task:<26}"
                    + "".join(f"{np.mean([r[c] for r in rs]):>14.2f}" for c in CONDS)
                )
    d_po = [r["worked_clean"] - r["proc_only"] for r in records]
    d_pm = [r["worked_clean"] - r["proc_blank"] for r in records]
    d_ao = [r["worked_clean"] - r["answer_only"] for r in records]
    ben = [r["proc_only"] - r["nomem"] for r in records]
    for lbl, d in [
        ("worked - proc_only", d_po),
        ("worked - proc_blank", d_pm),
        ("worked - answer_only", d_ao),
        ("proc_only - nomem", ben),
    ]:
        print(f"  {lbl:<22} n={len(d)} mean={np.mean(d):+.4f}")


if __name__ == "__main__":
    main()
