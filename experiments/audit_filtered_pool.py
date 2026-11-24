"""Deterministic diagnostics for the retained-exemplar audit population.

This is deliberately conservative: it identifies records requiring author
inspection but never turns an automatic flag into a scientific annotation.
It checks explicit numeric equations, missing procedural structure, and a
specific compensating-error pattern in which the text says to round to nearest
but outputs the protocol's floor value.
"""
import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


_EQ = re.compile(
    r"(?<![\d.])"
    r"(\d+(?:\.\d+)?(?:\s*[+*/-]\s*\d+(?:\.\d+)?){1,4})"
    r"\s*=\s*(\d+(?:\.\d+)?)"
)


def normalize_math(text):
    out = text.replace("\\times", "*").replace("\\div", "/")
    out = out.replace("×", "*").replace("÷", "/").replace("−", "-")
    out = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", out)
    out = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", out)
    out = re.sub(r"(?<=\d)\s+[xX]\s+(?=\d)", "*", out)
    return out


def safe_eval(expr):
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.USub,
        ast.UnaryOp,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(expr)
    return float(eval(compile(tree, "<audit>", "eval"), {"__builtins__": {}}))


def equation_issues(text):
    issues = []
    for match in _EQ.finditer(normalize_math(text)):
        expr, stated = match.groups()
        try:
            actual = safe_eval(expr)
            given = float(stated)
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        # Decimal working is commonly shown to two places (e.g. 143/6=23.83).
        acceptable = math.isclose(actual, given, rel_tol=0, abs_tol=0.011)
        # Procedural tasks frequently state integer quotients directly.  Accept
        # an explicit floor result when division is present.
        if "/" in expr and math.isclose(math.floor(actual), given, abs_tol=1e-6):
            acceptable = True
        if not acceptable:
            issues.append({
                "equation": match.group(0),
                "computed": actual,
                "stated": given,
            })
    return issues


def has_structure(task, text):
    normalized = normalize_math(text).lower()
    has_mul = "*" in normalized or "multiply" in normalized
    if task == "arith_hard":
        return has_mul and (
            "/" in normalized or "divid" in normalized or "crate" in normalized
        ) and (
            "damag" in normalized or "%" in normalized
        )
    return has_mul and ("+" in normalized or "subtotal" in normalized) and (
        "%" in normalized
        or "tax" in normalized
        or ("110" in normalized and "100" in normalized)
    )


def rounding_conflict(exemplar):
    """Detect nearest-vs-floor wording that can hide a compensating error."""
    text = exemplar["solution"].lower()
    contexts = []
    # Inspect every use of ``round`` as well as ``nearest``.  The earlier,
    # narrower expression only caught phrases such as "rounding off" and
    # missed ordinary wording like "round 4.5 to 5".
    for match in re.finditer(r"nearest|round", text):
        contexts.append(text[max(0, match.start() - 80) : match.end() + 80])
    # "round down to the nearest whole number" states the intended floor rule,
    # despite containing the word nearest.
    suspect_contexts = [
        c for c in contexts if "down" not in c and "floor" not in c
    ]
    if not suspect_contexts:
        return False
    trace = exemplar["expected_trace"]
    if exemplar["task"] == "invoice":
        exact = trace["subtotal"] * 1.10
    else:
        total = trace["total"]
        damage_pct = trace["inputs"][3]
        exact_damage = total * damage_pct / 100
        # The protocol floors damaged items before subtracting them.
        exact = exact_damage
    frac = exact - math.floor(exact)
    # Any non-integral quantity handled with an unspecified/nearest rounding
    # rule is non-transferable under the benchmark's explicit floor protocol,
    # even when it happens to agree with floor for this particular item.
    return frac > 1e-9


def diagnose(exemplar):
    equations = equation_issues(exemplar["solution"])
    structure = has_structure(exemplar["task"], exemplar["solution"])
    rounding = rounding_conflict(exemplar)
    reasons = []
    if equations:
        reasons.append("explicit_equation_mismatch")
    if not structure:
        reasons.append("missing_or_unrecognized_structure")
    if rounding:
        reasons.append("nearest_vs_floor_conflict")
    return {
        "reasons": reasons,
        "equation_issues": equations,
        "has_expected_structure": structure,
        "rounding_conflict": rounding,
        "requires_author_review": bool(reasons),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = defaultdict(Counter)
    for exemplar in data["exemplars"]:
        diag = diagnose(exemplar)
        exemplar["diagnostic"] = diag
        key = f"{exemplar['model']}/{exemplar['task']}"
        counts[key]["total"] += 1
        counts[key]["review"] += int(diag["requires_author_review"])
        for reason in diag["reasons"]:
            counts[key][reason] += 1

    for key in sorted(counts):
        print(key, dict(counts[key]))
    total = Counter()
    for count in counts.values():
        total.update(count)
    print("TOTAL", dict(total))

    out = Path(args.out) if args.out else path
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(f"saved diagnostics -> {out}")


if __name__ == "__main__":
    main()
