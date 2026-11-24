"""exp28 — PROCEDURE-ONLY: does memory still help when the stored ANSWER is REMOVED?

The paper *corrupts* the stored answer and finds it nearly irrelevant on procedural
tasks, then recommends "store worked procedures, filter by answer correctness". But
the recommended design itself — a memory holding procedures with NO answers — is
never tested. Removal is also a stronger intervention than corruption: a corrupted
answer is still an answer (the model can notice it disagrees with its own
computation), whereas a removed one cannot be deferred to at all.

exp28 closes the gap with a 2x2 factorial over what the exemplar carries:

                        answer present      answer absent
   procedure present    worked_clean        proc_only
   procedure absent     answer_only         nomem

PROC_ONLY drops the declared answer (the deployable design); PROC_BLANK
additionally blanks the derived value wherever it appears, so the exemplar shows
the complete procedure and no answer at all. We log the parse-failure rate per
condition, so a format artifact (the model imitating an answer-less exemplar
instead of emitting ANSWER: <int>) is measured, not assumed.

Prediction (procedural): proc_only ~= worked_clean > answer_only ~ nomem.

Protocol is identical to exp13 (procedural pools) / exp22 (BBH): pools are the
MODEL'S OWN correct solutions, k=3, 24 test items (25 for BBH), 5 seeds, seeded
decoding via mx_seed.

Usage:
  HF_HUB_OFFLINE=0 .venv/bin/python experiments/exp28_procedure_only.py \
      --regime proc --models llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b \
      --tasks arith_hard,invoice --pool 24 --test 24 --k 3 --seeds 5 \
      --max-tokens 384 --out results/exp28_proc_only.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsiguard.backends import MLXBackend, mx_seed
from experiments.models import MODELS
from experiments.memory import solve as mem_solve
from experiments.pools import POOLS, build_clean_pool

CONDS = ["nomem", "worked_clean", "proc_only", "proc_blank", "answer_only"]

# A declaration line is one whose content (after markdown/bullet decoration) is the
# ANSWER:/CONFIDENCE: field the prompt asks for. We delete such lines WHEREVER they
# occur rather than truncating from the first one: models sometimes emit the answer
# BEFORE the reasoning (76/310 of the archived generations we checked), and
# truncation would then delete the whole procedure.
_DECO = re.compile(r"^[\s*#>\-•]+")
_DECL = re.compile(r"^(?:ANSWER|CONFIDENCE)\s*\**\s*:", re.I)
BLANK = "___"


def _is_decl(line):
    return bool(_DECL.match(_DECO.sub("", line)))


def strip_answer(sol):
    """PROC_ONLY — the deployable design: the worked procedure with the ANSWER
    field removed. NOTE: a worked procedure necessarily *derives* its result, so
    the final computed value normally remains visible in the working; what is
    removed is the declared answer. PROC_BLANK below removes the value too."""
    out = "\n".join(ln for ln in sol.splitlines() if not _is_decl(ln)).strip()
    return out if out else sol  # never emit an empty exemplar


# A standalone number: not part of a longer numeral. The separator classes must
# only bite when they really join digits ("1,535", "3.35", "35.5") — a sentence
# period after the value ("= 35.") must NOT block the match.
def _num_re(form):
    return rf"(?<!\d)(?<!\d[.,]){re.escape(form)}(?!\d)(?![.,]\d)"


def blank_value(text, gold):
    """Replace standalone occurrences of the answer value (plain and comma-grouped)
    with a blank, without touching digits inside larger numbers. If an intermediate
    step happens to equal the answer it is blanked too — conservative for our
    hypothesis, since it can only weaken PROC_BLANK."""
    out = text
    for form in {str(gold), f"{gold:,}"}:
        out = re.sub(_num_re(form), BLANK, out)
    return out


def strip_and_blank(sol, gold):
    """PROC_BLANK — the strict ablation: the full procedure with no answer anywhere
    (declaration removed AND the derived value blanked). The model still sees which
    operations, in what order and format, but never the number."""
    return blank_value(strip_answer(sol), gold)


def mem_variant(pool, cond, rng):
    """pool entries: {'problem', 'gold', 'sol'} where sol is the model's own solution."""
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
    """Returns (accuracy, parse_failure_rate). retrieval='uniform' matches the
    single-task controlled pools (every exemplar shows the same procedure);
    'cosine' matches the heterogeneous stores (BBH)."""
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
    """Controlled synthetic procedural pools — same construction as exp13."""
    pf = POOLS[task]
    test = pf(args.test, np.random.default_rng(seed + 7777))
    pool = build_clean_pool(be, pf, args.pool, seed, "model")
    return pool, test, "uniform"


def cells_bbh(be, args, seed, task):
    """Real benchmark — same construction as exp22 (pool and test items disjoint)."""
    from experiments.exp22_gsm8k import load_data, build_clean_pool as bbh_pool
    pool_items, test = load_data("bbh", args.pool, args.test, seed, bbh_task=task)
    return bbh_pool(be, pool_items), test[:args.test], "cosine"


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
                row = {"model": key, "task": task, "seed": s, "pool": len(pool),
                       "regime": args.regime}
                for c in CONDS:
                    a, f = run_cond(be, pool, test, c, args.k, s, retrieval)
                    row[c], row[c + "_parsefail"] = a, f
                records.append(row)
                print(f"  {task:<12} seed{s} (pool={len(pool)}): "
                      + " ".join(f"{c}={row[c]:.2f}" for c in CONDS)
                      + " | parsefail "
                      + " ".join(f"{c}={row[c + '_parsefail']:.2f}" for c in CONDS[1:]),
                      flush=True)
                json.dump(records, open(args.out, "w"), indent=2)  # crash-safe
        del be

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print(" exp28 — procedure-only (mean over seeds)")
    print("=" * 78)
    print(f"{'model/task':<26}" + "".join(f"{c:>14}" for c in CONDS))
    for task in tasks:
        for key in [m.strip() for m in args.models.split(",") if m.strip()]:
            rs = [r for r in records if r["model"] == key and r["task"] == task]
            if rs:
                print(f"{key + '/' + task:<26}"
                      + "".join(f"{np.mean([r[c] for r in rs]):>14.2f}" for c in CONDS))
    d_po = [r["worked_clean"] - r["proc_only"] for r in records]
    d_pm = [r["worked_clean"] - r["proc_blank"] for r in records]
    d_ao = [r["worked_clean"] - r["answer_only"] for r in records]
    ben = [r["proc_only"] - r["nomem"] for r in records]
    for lbl, d in [("worked - proc_only", d_po), ("worked - proc_blank", d_pm),
                   ("worked - answer_only", d_ao), ("proc_only - nomem", ben)]:
        print(f"  {lbl:<22} n={len(d)} mean={np.mean(d):+.4f}")


if __name__ == "__main__":
    main()
