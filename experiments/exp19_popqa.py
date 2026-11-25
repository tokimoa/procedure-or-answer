"""Compare exemplar formats on PopQA (Mallen et al., 2023), using long-tail entities.

Memory holds one record per test question, retrieved top-k by embedding similarity.
Conditions: nomem, answer_only, worked_clean and fact_corrupt (each record states the
answer to a randomly drawn question from the same set).
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.models import MODELS
from rsiguard.backends import MLXBackend

CONDS = ["nomem", "answer_only", "worked_clean", "fact_corrupt"]
_ANS = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).strip().lower()).strip()


def load_popqa(n, seed):
    from datasets import load_dataset

    d = load_dataset("akariasai/PopQA")["test"]
    # Sample from the 3000 least popular subjects.
    order = np.argsort([float(d[i]["s_pop"] or 0) for i in range(len(d))])
    pool = order[:3000]
    rng = np.random.default_rng(seed + 2222)
    sel = rng.choice(pool, size=n, replace=False)
    items = []
    for i in sel:
        ex = d[int(i)]
        try:
            aliases = json.loads(ex["o_aliases"]) if ex["o_aliases"] else []
        except Exception:  # unparseable alias list: score against the object alone
            aliases = []
        items.append(
            {
                "q": ex["question"],
                "obj": ex["obj"],
                "golds": [norm(x) for x in [ex["obj"], *aliases] if x],
            }
        )
    return items


def solve(be, q, exemplars):
    """Return the normalised answer: the text after ANSWER:, else the last output line."""
    shots = "".join(f"Q: {e['q']}\n{e['sol']}\n\n" for e in exemplars)
    u = shots + f"Answer the question. End with EXACTLY: ANSWER: <answer>\nQ: {q}"
    txt = be.gen(u)
    m = _ANS.search(txt)
    return norm(m.group(1)) if m else norm(txt.splitlines()[-1] if txt else "")


def correct(pred, golds):
    """1.0 if the prediction equals, contains or is contained in any gold alias."""
    return 1.0 if any(g and (g == pred or g in pred or pred in g) for g in golds) else 0.0


def mem_items(kb, cond, rng):
    out = []
    for it in kb:
        if cond == "answer_only":
            sol = f"ANSWER: {it['obj']}"
        elif cond == "fact_corrupt":
            wrong = kb[int(rng.integers(len(kb)))]["obj"]  # may draw the item itself
            sol = f"The answer is {wrong}.\nANSWER: {wrong}"
        else:  # worked_clean
            sol = f"The answer is {it['obj']}.\nANSWER: {it['obj']}"
        out.append({"q": it["q"], "sol": sol})
    return out


def acc(be, kb, cond, k, seed):
    rng = np.random.default_rng(seed)
    mem = None if cond == "nomem" else mem_items(kb, cond, rng)
    embs = None if mem is None else be.embed([m["q"] for m in mem])
    corr = []
    for it in kb:
        if mem is None:
            ex = []
        else:
            qe = be.embed([it["q"]])[0]
            sims = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-9)
            ex = [mem[i] for i in np.argsort(-sims)[:k]]
        corr.append(correct(solve(be, it["q"], ex), it["golds"]))
    return float(np.mean(corr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,phi4-mini,ministral3-3b,granite4.1-8b")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--out", default="results/exp19_popqa.json")
    args = ap.parse_args()

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        try:
            be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            continue
        for s in range(args.seeds):
            kb = load_popqa(args.n, s)
            row = {"model": key, "seed": s}
            for c in CONDS:
                row[c] = acc(be, kb, c, args.k, s)
            records.append(row)
            print(f"  seed{s}: " + " ".join(f"{c}={row[c]:.2f}" for c in CONDS), flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)
    print("\n" + "=" * 70)
    print(" exp19: PopQA accuracy by condition")
    print("=" * 70)
    print(f"{'model':<16}" + "".join(f"{c:>14}" for c in CONDS))
    for key in [m.strip() for m in args.models.split(",")]:
        rs = [r for r in records if r["model"] == key]
        if not rs:
            continue
        print(f"{key:<16}" + "".join(f"{np.mean([r[c] for r in rs]):>14.2f}" for c in CONDS))
    print("=" * 70)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
