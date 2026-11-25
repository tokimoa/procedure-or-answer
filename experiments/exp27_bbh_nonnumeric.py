"""Answer corruption on BBH tasks with non-numeric answers (word lists, option letters, labels).

The pool holds the model's own correct solutions to held-out items; retrieval is cosine top-k;
pool and test items are disjoint.
Conditions: nomem, worked_clean, worked_ans_bad (answer field replaced by a wrong value of the
same type), answer_only. One result row per model and seed.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

CONDS = ["nomem", "worked_clean", "worked_ans_bad", "answer_only"]
_ANS_LINE = re.compile(r"ANSWER:\s*([^\n]*)")
_OPT = re.compile(r"\(([a-z])\)")


def norm(s):
    s = s.strip().strip(".").lower().replace(",", " ")
    return re.sub(r"\s+", " ", s)


FLIP = {
    "true": "False",
    "false": "True",
    "yes": "No",
    "no": "Yes",
    "valid": "invalid",
    "invalid": "valid",
}


def parse_options(problem):
    """Map option letter to normalized option text for a BBH multiple-choice input."""
    opts = {}
    for m in re.finditer(r"\(([A-Z])\)\s*([^\n(]*)", problem.split("Options:")[-1]):
        opts[m.group(1).lower()] = norm(m.group(2))
    return opts


def build_prompt(problem, exemplars):
    shots = ""
    for ex in exemplars:
        shots += f"Solved example:\nProblem: {ex['problem']}\nSolution: {ex['sol']}\n\n"
    return (
        shots + "Solve the new problem, using the solved examples as guidance.\n"
        f"Problem: {problem}\nEnd with EXACTLY two lines:\n"
        "ANSWER: <your final answer>\nCONFIDENCE: <0-100>"
    )


def solve_str(be, problem, exemplars):
    """Return (text of the answer field or None, generated text)."""
    txt = be.gen(build_prompt(problem, exemplars))
    m = _ANS_LINE.search(txt)
    return (m.group(1).strip() if m else None), txt


def is_correct(pred, item):
    if pred is None:
        return False
    p, g = norm(pred), norm(item["gold"])
    if p == g:
        return True
    gl = _OPT.search(g)
    if gl:  # multiple choice: match by letter, or by the gold option's text
        pl = _OPT.search(p)
        if pl:
            return pl.group(1) == gl.group(1)
        opts = parse_options(item["problem"])
        return bool(opts.get(gl.group(1))) and p == opts[gl.group(1)]
    return False


def wrong_answer(item, rng):
    """A wrong answer of the same type as the gold answer."""
    g = item["gold"].strip()
    gl = _OPT.search(norm(g))
    if gl:  # multiple choice: a different option present in the problem
        letters = sorted(parse_options(item["problem"]).keys()) or ["a", "b", "c"]
        others = [letter for letter in letters if letter != gl.group(1)]
        return f"({rng.choice(others).upper()})"
    if norm(g) in FLIP:  # binary label (True/False etc.): the other label
        return FLIP[norm(g)]
    words = g.split()
    if len(words) > 1:  # word list: a different permutation of the same words
        for _ in range(20):
            perm = list(rng.permutation(words))
            if perm != words:
                return " ".join(perm)
    return g[::-1]


def load_data(bbh_task, n_pool, n_test, seed):
    from datasets import load_dataset

    rng = np.random.default_rng(seed + 909)
    ds = load_dataset("lukaemon/bbh", bbh_task)["test"]
    idx = rng.choice(len(ds), size=min(n_pool + n_test, len(ds)), replace=False)
    items = [
        {"problem": ds[int(i)]["input"], "gold": str(ds[int(i)]["target"]).strip()} for i in idx
    ]
    return items[:n_pool], items[n_pool:]


def build_clean_pool(be, pool_items):
    clean = []
    for it in pool_items:
        pred, txt = solve_str(be, it["problem"], [])
        if is_correct(pred, it):
            sol = txt if "ANSWER:" in txt else f"{txt}\nANSWER: {it['gold']}"
            clean.append({"problem": it["problem"], "gold": it["gold"], "sol": sol})
    return clean


def mem_variant(pool, cond, rng):
    out = []
    for p in pool:
        if cond == "answer_only":
            sol = f"ANSWER: {p['gold']}"
        elif cond == "worked_ans_bad":
            bad = wrong_answer(p, rng)
            sol = re.sub(r"ANSWER:[^\n]*", f"ANSWER: {bad}", p["sol"])
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
        pred, _ = solve_str(be, t["problem"], ex)
        corr.append(1.0 if is_correct(pred, t) else 0.0)
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--bbh-task", default="word_sorting")
    ap.add_argument("--n-pool", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=25)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/exp27_{args.bbh_task}.json"

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
            pool_items, test = load_data(args.bbh_task, args.n_pool, args.n_test, s)
            pool = build_clean_pool(be, pool_items)
            row = {"model": key, "seed": s, "pool_size": len(pool)}
            for c in CONDS:
                row[c] = acc(be, pool, test, c, args.k, s)
            records.append(row)
            print(
                f"  seed{s} (pool={len(pool)}): " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS),
                flush=True,
            )
            with open(out, "w") as fp:
                json.dump(records, fp, indent=2)
        del be

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump(records, fp, indent=2)
    print("\n" + "=" * 72)
    print(f" exp27: BBH {args.bbh_task} (non-numeric procedural)")
    print("=" * 72)
    print(f"{'model':<16}" + "".join(f"{c:>16}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>16.2f}" for c in CONDS))
    print("=" * 72)
    print("dichotomy holds beyond arithmetic if worked_clean ≈ worked_ans_bad")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
