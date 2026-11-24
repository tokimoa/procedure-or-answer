"""Statistical summaries for the fixed-set evaluations added after review (exp29/exp30)."""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def bootstrap_mean(values, rng, draws=20000):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means[start : start + n] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_rows(records, model, clean, corrupt, id_key):
    a = {
        r[id_key]: r
        for r in records
        if r["model"] == model and r["condition"] == clean
    }
    b = {
        r[id_key]: r
        for r in records
        if r["model"] == model and r["condition"] == corrupt
    }
    ids = sorted(set(a) & set(b))
    return [(a[i], b[i]) for i in ids]


def exact_mcnemar(pairs):
    clean_only = sum(a["correct"] == 1 and b["correct"] == 0 for a, b in pairs)
    corrupt_only = sum(a["correct"] == 0 and b["correct"] == 1 for a, b in pairs)
    discordant = clean_only + corrupt_only
    p = (
        float(stats.binomtest(clean_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return clean_only, corrupt_only, p


def analyze_file(path, rng):
    data = json.load(open(path, encoding="utf-8"))
    records = data["records"]
    models = sorted({r["model"] for r in records})
    conditions = sorted({r["condition"] for r in records})
    id_key = "target_rid" if "target_rid" in records[0] else "item_id"
    print(f"\nFILE {path}")
    print(f"meta={data['meta']}")
    print("condition accuracies (item bootstrap 95% CI)")
    for model in models:
        print(f"-- {model} --")
        for cond in conditions:
            rows = [r for r in records if r["model"] == model and r["condition"] == cond]
            if not rows:
                continue
            mean, lo, hi = bootstrap_mean([r["correct"] for r in rows], rng)
            hit_rows = [r for r in rows if r.get("hit_at_k") is not None]
            hit = np.mean([r["hit_at_k"] for r in hit_rows]) if hit_rows else None
            hit1 = np.mean([r["hit_at_1"] for r in hit_rows]) if hit_rows else None
            extra = f" hit@1={hit1:.3f} hit@k={hit:.3f}" if hit is not None else ""
            print(f"  {cond:<20} n={len(rows):>3} {mean:.3f} [{lo:.3f},{hi:.3f}]{extra}")

    if "direct_clean" in conditions:
        contrasts = [
            ("direct_clean", "direct_corrupt"),
            ("nonoracle_clean", "nonoracle_corrupt"),
        ]
    else:
        contrasts = [("worked_clean", "worked_ans_bad")]
    print("paired clean-corrupt effects")
    macro = defaultdict(list)
    for model in models:
        for clean, corrupt in contrasts:
            pairs = paired_rows(records, model, clean, corrupt, id_key)
            if not pairs:
                continue
            diffs = [a["correct"] - b["correct"] for a, b in pairs]
            mean, lo, hi = bootstrap_mean(diffs, rng)
            n10, n01, p = exact_mcnemar(pairs)
            macro[(clean, corrupt)].append(mean)
            print(
                f"  {model:<18} {clean}-{corrupt}: {mean:+.3f} "
                f"[{lo:+.3f},{hi:+.3f}] discordant={n10}/{n01} p={p:.4g}"
            )
    for (clean, corrupt), effects in macro.items():
        print(
            f"  MACRO {clean}-{corrupt}: "
            f"{np.mean(effects):+.3f} over {len(effects)} models"
        )

    if "nonoracle_clean" in conditions:
        print("non-oracle clean accuracy conditional on retrieval")
        for model in models:
            rows = [
                r
                for r in records
                if r["model"] == model and r["condition"] == "nonoracle_clean"
            ]
            for hit in (1, 0):
                subset = [r["correct"] for r in rows if r["hit_at_k"] == hit]
                value = np.mean(subset) if subset else float("nan")
                print(f"  {model:<18} hit@k={hit} n={len(subset):>3} acc={value:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for path in args.paths:
        analyze_file(path, rng)


if __name__ == "__main__":
    main()
