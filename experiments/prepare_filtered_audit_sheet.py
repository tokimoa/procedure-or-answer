"""Create a deterministic author-review sheet for the filtered-pool audit."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--clean-per-stratum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default="results/exp31_audit_sheet.json")
    args = ap.parse_args()

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    strata = defaultdict(list)
    selected = []
    for exemplar in data["exemplars"]:
        key = (exemplar["model"], exemplar["task"])
        strata[key].append(exemplar)
        if exemplar["diagnostic"]["requires_author_review"]:
            selected.append(exemplar)

    rng = np.random.default_rng(args.seed)
    for key in sorted(strata):
        clean = [
            x for x in strata[key]
            if not x["diagnostic"]["requires_author_review"]
        ]
        n = min(args.clean_per_stratum, len(clean))
        for index in rng.choice(len(clean), size=n, replace=False):
            selected.append(clean[int(index)])

    rows = []
    seen = set()
    for exemplar in selected:
        audit_id = (
            f"{exemplar['model']}|{exemplar['task']}|"
            f"s{exemplar['seed']}|e{exemplar['exemplar_index']}"
        )
        if audit_id in seen:
            continue
        seen.add(audit_id)
        rows.append({
            "audit_id": audit_id,
            "model": exemplar["model"],
            "task": exemplar["task"],
            "seed": exemplar["seed"],
            "exemplar_index": exemplar["exemplar_index"],
            "sample_source": (
                "diagnostic" if exemplar["diagnostic"]["requires_author_review"]
                else "random_clean"
            ),
            "diagnostic_reasons": exemplar["diagnostic"]["reasons"],
            "problem": exemplar["problem"],
            "gold": exemplar["gold"],
            "expected_trace": exemplar["expected_trace"],
            "solution": exemplar["solution"],
            "label": None,
            "error_type": None,
            "notes": "",
        })

    rows.sort(key=lambda x: (
        x["model"], x["task"], x["sample_source"], x["seed"], x["exemplar_index"]
    ))
    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"saved {len(rows)} rows: "
        f"diagnostic={sum(r['sample_source'] == 'diagnostic' for r in rows)} "
        f"random_clean={sum(r['sample_source'] == 'random_clean' for r in rows)} "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
