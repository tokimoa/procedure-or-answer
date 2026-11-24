"""Apply the predeclared audit rubric and summarize the full exp31 population."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp32_screened_pool_sensitivity import (
    EQUATION_PARSER_FALSE_POSITIVES,
)


def classify(x):
    d = x["diagnostic"]
    key = (x["model"], x["task"], x["seed"], x["exemplar_index"])
    labels = []
    if d["rounding_conflict"]:
        labels.append("protocol_rule_mismatch")
    if d["equation_issues"] and key not in EQUATION_PARSER_FALSE_POSITIVES:
        labels.append("explicit_arithmetic_error")
    return labels or ["no_issue_identified"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument(
        "--out", default="results/exp31_filtered_audit_final.json"
    )
    args = ap.parse_args()
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    counts = defaultdict(Counter)
    for x in data["exemplars"]:
        x["audit_labels"] = classify(x)
        key = f"{x['model']}/{x['task']}"
        counts[key]["total"] += 1
        for label in x["audit_labels"]:
            counts[key][label] += 1
    summary = {key: dict(value) for key, value in sorted(counts.items())}
    total = Counter()
    for value in counts.values():
        total.update(value)
    data["audit_summary"] = summary
    data["audit_total"] = dict(total)
    Path(args.out).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for key, value in summary.items():
        print(key, value)
    print("TOTAL", dict(total))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
