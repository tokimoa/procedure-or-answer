"""probe — is the nomem baseline depressed by the memory-style prompt?

The shared solve prompt (memory.build_prompt) says "using the solved examples as
guidance" even when zero exemplars are shown (the nomem condition). This probe
compares that nomem prompt against a clean zero-shot prompt on BBH
multistep_arithmetic_two, to check the baseline is not unfairly depressed
(which would inflate the reported memory benefit).

Usage:
  .venv/bin/python experiments/probe_nomem_prompt.py \
      --models llama3.2-3b,ministral3-3b --n-test 25 --seeds 5
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
from experiments.exp22_gsm8k import load_data

_ANS = re.compile(r"ANSWER:\s*(-?\d+)")

CLEAN = ("Solve the problem.\n"
         "Problem: {q}\nEnd with EXACTLY two lines:\n"
         "ANSWER: <integer>\nCONFIDENCE: <0-100>")


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
            _, test = load_data("bbh", 60, args.n_test, s)  # same split as exp22
            memstyle = float(np.mean([1.0 if mem_solve(be, t["problem"], [])[0] == t["gold"]
                                      else 0.0 for t in test]))
            clean = float(np.mean([1.0 if clean_solve(be, t["problem"]) == t["gold"]
                                   else 0.0 for t in test]))
            records.append({"model": key, "seed": s,
                            "nomem_memstyle_prompt": memstyle, "nomem_clean_prompt": clean})
            print(f"  seed{s}: memstyle={memstyle:.2f} clean={clean:.2f}", flush=True)
        del be

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print("\nbaseline is FAIR if memstyle ≈ clean (no depression from the prompt)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
