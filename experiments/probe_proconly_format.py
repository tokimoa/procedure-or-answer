"""Output-format probe for the proc_only condition of exp28.

For worked_clean and proc_only exemplars, records per (task, seed): the parse-failure rate, the
mean number of generated tokens, and the fraction of generations that reach the token cap.
The generated texts are saved in the same file.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp28_procedure_only import mem_variant
from experiments.memory import _ANS, build_prompt
from experiments.models import MODELS
from experiments.pools import POOLS, build_clean_pool
from rsiguard.backends import MLXBackend, mx_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi4-mini")
    ap.add_argument("--tasks", default="invoice,arith_hard")
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--test", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/exp28_format_probe.json")
    args = ap.parse_args()

    be = MLXBackend(MODELS[args.model], max_tokens=args.max_tokens)
    rows, samples = [], {}
    for task in [t.strip() for t in args.tasks.split(",")]:
        pf = POOLS[task]
        for s in range(args.seeds):
            mx_seed(s)
            test = pf(args.test, np.random.default_rng(s + 7777))
            pool = build_clean_pool(be, pf, args.pool, s, "model")
            for cond in ["worked_clean", "proc_only"]:
                mem = mem_variant(pool, cond, np.random.default_rng(s + 1))
                rng = np.random.default_rng(s + 1)
                toks, fails, texts = [], [], []
                for t in test:
                    ex = [mem[i] for i in rng.permutation(len(mem))[: args.k]]
                    tx = be.gen(build_prompt(t["problem"], ex))
                    n = len(be.tok.encode(tx))
                    texts.append(tx)
                    toks.append(n)
                    fails.append(1.0 if _ANS.search(tx) is None else 0.0)
                row = {
                    "model": args.model,
                    "task": task,
                    "seed": s,
                    "cond": cond,
                    "parse_fail": float(np.mean(fails)),
                    "mean_tokens": float(np.mean(toks)),
                    "hit_cap": float(np.mean([t >= args.max_tokens - 6 for t in toks])),
                }
                rows.append(row)
                print(
                    f"  {task:<11} seed{s} {cond:<13} parse-fail={row['parse_fail']:.2f} "
                    f"mean_tokens={row['mean_tokens']:5.0f} hit_cap={row['hit_cap']:.2f}",
                    flush=True,
                )
                # Keep the first failing generation (if any), the first generation, and all texts.
                fail_ix = [i for i, f in enumerate(fails) if f]
                samples[f"{task}|s{s}|{cond}"] = {
                    "failed_example": texts[fail_ix[0]][-400:] if fail_ix else None,
                    "ok_example": texts[0][-300:],
                    "all_texts": texts,
                }
                with open(args.out, "w") as fp:
                    json.dump({"rows": rows, "samples": samples}, fp, indent=2)

    with open(args.out, "w") as fp:
        json.dump({"rows": rows, "samples": samples}, fp, indent=2)
    print("\n=== pooled by condition ===")
    for cond in ["worked_clean", "proc_only"]:
        rs = [r for r in rows if r["cond"] == cond]
        print(
            f"  {cond:<13} parse-fail={np.mean([r['parse_fail'] for r in rs]):.3f} "
            f"mean_tokens={np.mean([r['mean_tokens'] for r in rs]):5.0f} "
            f"hit_cap={np.mean([r['hit_cap'] for r in rs]):.3f}"
        )


if __name__ == "__main__":
    main()
