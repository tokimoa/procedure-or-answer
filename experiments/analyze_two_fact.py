"""Statistics for exp34 (multi-fact queries): per-model accuracies, follow-rates,
and paired clean - corrupt contrasts over model x seed cells (95% t-interval,
paired t-test), matching the paper's convention for the controlled tasks."""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats

CONDS = {
    "single": ["nomem", "clean", "corrupt"],
    "aggregate": ["nomem", "clean", "one_corrupt", "both_corrupt"],
    "bridge": ["nomem", "clean", "hop1_corrupt", "hop2_corrupt", "both_corrupt"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/exp34_two_fact.json")
    args = ap.parse_args()
    data = json.load(open(args.path, encoding="utf-8"))
    recs = data["records"]
    models = list(dict.fromkeys(r["model"] for r in recs))
    print("meta", data["meta"], "records", len(recs))

    cell = defaultdict(list)   # (model, variant, cond, seed) -> correct
    fol = defaultdict(list)    # same -> follow
    wrong_fol = defaultdict(list)  # follow among incorrect answers
    pfail = defaultdict(list)
    for r in recs:
        key = (r["model"], r["variant"], r["condition"], r["seed"])
        cell[key].append(r["correct"])
        pfail[key[:3]].append(r["pred"] is None)
        if r["follow"] is not None:
            fol[key].append(r["follow"])
    acc = {k: float(np.mean(v)) for k, v in cell.items()}

    for variant, conds in CONDS.items():
        print(f"\n=== {variant} ===")
        print(f"{'model':<16}" + "".join(f"{c:>14}" for c in conds))
        for m in models:
            row = [np.mean([acc[k] for k in acc if k[:3] == (m, variant, c)]) for c in conds]
            print(f"{m:<16}" + "".join(f"{x:>14.3f}" for x in row))
        print("follow-rate P(pred == W)")
        for m in models:
            row = []
            for c in conds:
                v = [x for k, xs in fol.items() if k[:3] == (m, variant, c) for x in xs]
                row.append(f"{np.mean(v):>14.3f}" if v else f"{'-':>14}")
            print(f"{m:<16}" + "".join(row))
        print("parse-fail")
        for m in models:
            print(f"{m:<16}" + "".join(
                f"{np.mean(pfail[(m, variant, c)]):>14.3f}" for c in conds))
        print("paired clean - corrupt over model x seed cells")
        for c in conds:
            if c in ("nomem", "clean"):
                continue
            keys = sorted({(k[0], k[3]) for k in acc if k[1] == variant and k[2] == c})
            d = np.array([acc[(m, variant, "clean", s)] - acc[(m, variant, c, s)]
                          for m, s in keys])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            p = stats.ttest_1samp(d, 0).pvalue
            allfol = [x for k, xs in fol.items() if k[1] == variant and k[2] == c for x in xs]
            print(f"  {c:<13} n={len(d)} drop={d.mean():+.3f} ±{half:.3f} p={p:.2g} "
                  f"pooled follow={np.mean(allfol):.3f}")
            for m in models:
                dm = np.array([acc[(m, variant, 'clean', s)] - acc[(m, variant, c, s)]
                               for mm, s in keys if mm == m])
                pm = stats.ttest_1samp(dm, 0).pvalue if dm.std() > 0 else float("nan")
                print(f"      {m:<16} drop={dm.mean():+.3f} p={pm:.2g}")
        # memory benefit
        keys = sorted({(k[0], k[3]) for k in acc if k[1] == variant})
        d = np.array([acc[(m, variant, "clean", s)] - acc[(m, variant, "nomem", s)]
                      for m, s in keys])
        print(f"  clean - nomem  n={len(d)} {d.mean():+.3f}")

    print("\n=== retrieval from the query alone (model's own embeddings, top-k) ===")
    for variant in CONDS:
        for m in models:
            rows = [r["hits"] for r in recs if r["model"] == m and r["variant"] == variant
                    and r["condition"] == "clean"]
            h = np.array(rows, dtype=float)
            per = " ".join(f"{x:.3f}" for x in h.mean(axis=0))
            print(f"  {variant:<9} {m:<16} need-hit@k: {per}  all-needed: "
                  f"{np.mean(h.min(axis=1)):.3f}")


if __name__ == "__main__":
    main()
