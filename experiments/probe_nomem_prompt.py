"""Compare two zero-exemplar prompts on BBH multistep_arithmetic_two.

The shared prompt (memory.build_prompt) says "using the solved examples as guidance" even when
no exemplar is shown. This script measures no-memory accuracy with that prompt and with a plain
zero-shot prompt on the same test items.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp22_gsm8k import load_data
from experiments.memory import solve as mem_solve
from experiments.models import MODELS
from rsiguard.backends import MLXBackend, mx_seed

_ANS = re.compile(r"ANSWER:\s*(-?\d+)")

CLEAN = (
    "Solve the problem.\n"
    "Problem: {q}\nEnd with EXACTLY two lines:\n"
    "ANSWER: <integer>\nCONFIDENCE: <0-100>"
)


def clean_solve(be, problem):
    txt = be.gen(CLEAN.format(q=problem))
    a = _ANS.search(txt)
    return int(a.group(1)) if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama3.2-3b,ministral3-3b")
    ap.add_argument("--n-test", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--out", default="results/probe_nomem_prompt.json")
    args = ap.parse_args()

    records = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n### model={key}", flush=True)
        be = MLXBackend(MODELS[key], max_tokens=args.max_tokens)
        for s in range(args.seeds):
            mx_seed(s)
            _, test = load_data("bbh", 60, args.n_test, s)  # same split as exp22 with --n-pool 60
            memstyle = float(
                np.mean(
                    [1.0 if mem_solve(be, t["problem"], [])[0] == t["gold"] else 0.0 for t in test]
                )
            )
            clean = float(
                np.mean([1.0 if clean_solve(be, t["problem"]) == t["gold"] else 0.0 for t in test])
            )
            records.append(
                {
                    "model": key,
                    "seed": s,
                    "nomem_memstyle_prompt": memstyle,
                    "nomem_clean_prompt": clean,
                }
            )
            print(f"  seed{s}: memstyle={memstyle:.2f} clean={clean:.2f}", flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(records, fp, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
