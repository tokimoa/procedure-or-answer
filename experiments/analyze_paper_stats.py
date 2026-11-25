"""Single source of truth for every statistic quoted in the paper.

Reads the *_v2 (anchored-corruption) results when present, falling back to the
originals, and prints (a) per-model condition means for the tables, (b) pooled
paired contrasts (mean +/- 95% CI, t, p over model x seed [x task] cells), and
(c) per-model paired contrasts over seeds — the hierarchy-aware check that the
pooled result is not driven by one model.

Usage:
  .venv/bin/python experiments/analyze_paper_stats.py
"""
import json
import os

import numpy as np
from scipy import stats

R = "results"
MODELS4 = ["llama3.2-3b", "phi4-mini", "ministral3-3b", "granite4.1-8b"]
MODELS18 = ["llama3.2-3b", "phi4-mini", "granite4.1-8b", "phi4", "ministral3-14b"]


def load(*names):
    for n in names:
        p = os.path.join(R, n)
        if os.path.exists(p):
            print(f"[{n}]")
            return json.load(open(p))
    raise FileNotFoundError(names)


def paired(diffs, label):
    d = np.asarray(diffs, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), n - 1)
    ci = stats.t.ppf(0.975, n - 1) * se
    print(f"  {label}: n={n} mean={m:+.4f} +/-{ci:.4f} t={t:.2f} p={p:.3g}")
    return m, ci, p


def tost(diffs, label, margins=(0.05, 0.10)):
    """Equivalence test (TOST): is the paired effect bounded within +/-margin?
    Reported for every 'harmless / bounded by a few points' claim in the paper."""
    d = np.asarray(diffs, float)
    n, m = len(d), d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    outs = []
    for delta in margins:
        p_lo = stats.t.sf((m + delta) / se, n - 1)
        p_hi = stats.t.cdf((m - delta) / se, n - 1)
        outs.append(f"+/-{delta}: p={max(p_lo, p_hi):.3g}"
                    f"{' EQUIV' if max(p_lo, p_hi) < 0.05 else ''}")
    print(f"  TOST ({label}): " + "  ".join(outs))


def mixedlm_check(rows, fn, label):
    """Random-intercept (model) mixed model — the hierarchy-aware alternative to
    pooling model x seed cells as if independent."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"d": [fn(r) for r in rows], "model": [r["model"] for r in rows]})
        # the default BFGS start does not converge on the headline contrast
        res = smf.mixedlm("d ~ 1", df, groups=df["model"]).fit(
            reml=True, method=["lbfgs", "powell", "nm"])
        if not res.converged:
            raise RuntimeError("mixed model did not converge")
        b, se = res.params["Intercept"], res.bse["Intercept"]
        z = b / se
        p = 2 * stats.norm.sf(abs(z))
        print(f"  mixed-effects ({label}): b={b:+.4f} se={se:.4f} z={z:.2f} p={p:.3g}"
              f"  (random intercept per model)")
    except Exception as e:
        print(f"  mixed-effects ({label}): unavailable ({e})")


def holm(named_ps, label):
    """Holm step-down adjustment over a family of secondary residual-effect tests."""
    items = sorted(named_ps.items(), key=lambda kv: kv[1])
    k = len(items)
    print(f"  Holm-adjusted ({label}):")
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (k - i) * p))
        prev = adj
        print(f"    {name:38s} raw p={p:.3g} -> adj p={adj:.3g}"
              f"{'  *' if adj < 0.05 else ''}")


def r2(x):
    """Half-up 2dp rounding — the paper's table convention (0.145 -> 0.15)."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def permodel_paired(rows, models, fn, label):
    print(f"  per-model ({label}):")
    for mo in models:
        d = np.asarray([fn(r) for r in rows if r["model"] == mo], float)
        if len(d) == 0:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        ci = stats.t.ppf(0.975, len(d) - 1) * se if se > 0 else 0.0
        t = d.mean() / se if se > 0 else float("inf")
        p = 2 * stats.t.sf(abs(t), len(d) - 1) if se > 0 else 0.0
        print(f"    {mo:15s} n={len(d)} mean={d.mean():+.3f} +/-{ci:.3f} p={p:.3g}")


def table(rows, models, conds, label):
    print(f"  table ({label}):")
    for mo in models:
        rs = [r for r in rows if r["model"] == mo]
        if not rs:
            continue
        cells = " ".join(f"{c}={np.mean([r[c] for r in rs]):.2f}" for c in conds)
        print(f"    {mo:15s} {cells}")


print("=== exp13 procedural (Table 1) ===")
d = load("exp13_ci_v2.json", "exp13_ci.json")
table(d, MODELS4, ["nomem", "worked_clean", "worked_ans_bad", "answer_only"], "avg over 2 tasks")
paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
mixedlm_check(d, lambda r: r["worked_clean"] - r["worked_ans_bad"], "answer-corruption drop")
paired([r["worked_clean"] - r["answer_only"] for r in d], "worked - answer_only")
permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"], "answer-corruption drop")
_p_exp13_pooled = stats.ttest_rel([r["worked_clean"] for r in d],
                                  [r["worked_ans_bad"] for r in d]).pvalue
_p_exp13_ministral = stats.ttest_rel(
    [r["worked_clean"] for r in d if r["model"] == "ministral3-3b"],
    [r["worked_ans_bad"] for r in d if r["model"] == "ministral3-3b"]).pvalue

print("\n=== exp15 knowledge (Table 3) ===")
d = load("exp15_ci_v2.json", "exp15_ci.json")
table(d, MODELS4, ["nomem", "answer_only", "worked_clean", "worked_ans_bad"], "")
paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "fact-corruption drop")
permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"], "fact-corruption drop")

print("\n=== exp18 method vs answer (Table 2) ===")
d = load("exp18_ci_v2.json", "exp18_ci.json")
table(d, MODELS18, ["nomem", "worked_correct", "worked_ans_corrupt", "worked_wrong_method"], "")
paired([r["worked_correct"] - r["worked_wrong_method"] for r in d], "method-corruption damage")
paired([r["worked_correct"] - r["worked_ans_corrupt"] for r in d], "answer-corruption damage")
tost([r["worked_correct"] - r["worked_ans_corrupt"] for r in d], "answer-corruption damage")

print("\n=== exp26 OWN-STYLE method vs answer (repairs exp18's templated exception) ===")
try:
    d = load("exp26_ownstyle.json")
    try:
        dec = load("exp26_decoupled.json")
        dec_models = {r["model"] for r in dec}
        d = [r for r in d if r["model"] not in dec_models] + dec
        print(f"  (merged decoupled rows for: {sorted(dec_models)})")
    except FileNotFoundError:
        pass
    table(d, MODELS4, ["nomem", "worked_correct", "worked_ans_corrupt", "worked_wrong_method"], "")
    for mo in MODELS4:
        rs = [r for r in d if r["model"] == mo]
        if rs:
            pw = np.mean([r.get("pool_wrong", r["pool"]) for r in rs])
            print(f"    {mo:15s} pool={np.mean([r['pool'] for r in rs]):.1f} "
                  f"pool_wrong={pw:.1f} "
                  f"retention: correct={np.mean([r['verified_correct']/r['tried'] for r in rs]):.2f} "
                  f"wrong={np.mean([r['verified_wrong']/r['tried'] for r in rs]):.2f}")
    # wrong-method pools are constructible only for the deferring models; quote method
    # damage over models whose wrong pool reached the target
    dm = [r for r in d if np.mean([x.get("pool_wrong", x["pool"]) for x in d
                                   if x["model"] == r["model"]]) >= 8]
    dm_models = sorted({r["model"] for r in dm})
    paired([r["worked_correct"] - r["worked_wrong_method"] for r in dm],
           f"method-corruption damage (constructible pools: {dm_models})")
    paired([r["worked_correct"] - r["worked_ans_corrupt"] for r in d], "answer-corruption damage")
    tost([r["worked_correct"] - r["worked_ans_corrupt"] for r in d], "answer-corruption damage")
    permodel_paired(d, MODELS4, lambda r: r["worked_correct"] - r["worked_ans_corrupt"],
                    "answer-corruption damage")
    permodel_paired(dm, dm_models, lambda r: r["worked_correct"] - r["worked_wrong_method"],
                    "method-corruption damage")
except FileNotFoundError:
    print("  (not run yet)")

print("\n=== exp21 compositional ===")
d = load("exp21_mixed.json")
paired([r["worked_clean"] - r["fact_corrupt"] for r in d], "fact-corruption drop")
paired([r["worked_clean"] - r["method_corrupt"] for r in d], "method-corruption drop")
paired([r["worked_clean"] - r["answer_corrupt"] for r in d], "answer-corruption drop")
tost([r["worked_clean"] - r["answer_corrupt"] for r in d], "answer-corruption drop")
_p_exp21_answer = stats.ttest_rel([r["worked_clean"] for r in d],
                                  [r["answer_corrupt"] for r in d]).pvalue

print("\n=== exp22 BBH multistep (Table 4 top) ===")
d = load("exp22_bbh_ci.json")
table(d, MODELS4, ["nomem", "answer_only", "worked_clean", "worked_ans_bad"], "")
paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"], "answer-corruption drop")

print("\n=== exp22 BBH object_counting (2nd procedural real bench) ===")
try:
    d = load("exp22_objcount_ci.json")
    table(d, MODELS4, ["nomem", "answer_only", "worked_clean", "worked_ans_bad"], "")
    paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
    tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
    permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"], "answer-corruption drop")
    _p_objcount_pooled = stats.ttest_rel([r["worked_clean"] for r in d],
                                         [r["worked_ans_bad"] for r in d]).pvalue
except FileNotFoundError:
    _p_objcount_pooled = None
    print("  (not run yet)")

print("\n=== exp27 non-numeric BBH procedural (tracking_three / boolean_expressions) ===")
_p_boolean_granite = _p_boolean_ministral = None
for name, fn in [("tracking_shuffled_objects_three", "exp27_tracking3.json"),
                 ("boolean_expressions", "exp27_boolean.json")]:
    try:
        d = load(fn)
        print(f"  -- {name} --")
        table(d, MODELS4, ["nomem", "answer_only", "worked_clean", "worked_ans_bad"], "")
        paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
        tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
        paired([r["worked_clean"] - r["nomem"] for r in d], "memory benefit (worked - nomem)")
        permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"],
                        "answer-corruption drop")
        if name == "boolean_expressions":
            _p_boolean_granite = stats.ttest_rel(
                [r["worked_clean"] for r in d if r["model"] == "granite4.1-8b"],
                [r["worked_ans_bad"] for r in d if r["model"] == "granite4.1-8b"]).pvalue
            _p_boolean_ministral = stats.ttest_rel(
                [r["worked_clean"] for r in d if r["model"] == "ministral3-3b"],
                [r["worked_ans_bad"] for r in d if r["model"] == "ministral3-3b"]).pvalue
    except FileNotFoundError:
        print(f"  ({name}: not run yet)")

print("\n=== exp19 PopQA (Table 4 bottom) ===")
d = load("exp19_ci.json")
table(d, MODELS4, ["nomem", "answer_only", "worked_clean", "fact_corrupt"], "")
paired([r["worked_clean"] - r["fact_corrupt"] for r in d], "fact-corruption drop")
permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["fact_corrupt"], "fact-corruption drop")

print("\n=== exp25 copy-rate @ k=3 / full store (paper, Sec 4.3) ===")
try:
    d = load("exp25_copyrate_k3.json")
    for mo in [m for m in MODELS4 if any(r["model"] == m for r in d)]:
        rs = [r for r in d if r["model"] == mo]
        print(f"  {mo:15s} proc_copy={np.mean([r['proc_copy'] for r in rs]):.3f} "
              f"proc_gold={np.mean([r['proc_gold'] for r in rs]):.3f} "
              f"know_copy={np.mean([r['know_copy'] for r in rs]):.3f} seeds={len(rs)}")
except FileNotFoundError:
    print("  (not run yet)")

print("\n=== exp23 copy-rate @ k=1 single planted exemplar (appendix robustness) ===")
d = load("exp23_copyrate_v2.json", "exp23_copyrate.json")
for mo in [m for m in MODELS4 if any(r["model"] == m for r in d)]:
    rs = [r for r in d if r["model"] == mo]
    print(f"  {mo:15s} proc_copy={np.mean([r['proc_copy'] for r in rs]):.3f} "
          f"know_copy={np.mean([r['know_copy'] for r in rs]):.3f} seeds={len(rs)}")

print("\n=== exp24 unfiltered self-generated pools (paper, Sec 4.4) ===")
try:
    d = load("exp24_unfiltered.json")
    table(d, MODELS4, ["nomem", "worked_filtered", "worked_unfiltered"], "avg over 2 tasks")
    for mo in MODELS4:
        rs = [r for r in d if r["model"] == mo]
        if rs:
            print(f"    {mo:15s} purity={np.mean([r['purity_unfiltered'] for r in rs]):.2f}")
    paired([r["worked_filtered"] - r["worked_unfiltered"] for r in d], "filtered - unfiltered")
    paired([r["worked_unfiltered"] - r["nomem"] for r in d], "unfiltered - nomem")
    paired([r["worked_filtered"] - r["nomem"] for r in d], "filtered - nomem")
    permodel_paired(d, MODELS4, lambda r: r["worked_filtered"] - r["worked_unfiltered"],
                    "filtered - unfiltered")
    permodel_paired(d, MODELS4, lambda r: r["worked_unfiltered"] - r["nomem"],
                    "unfiltered - nomem")
    pur = np.array([1 - r["purity_unfiltered"] for r in d])
    dmg = np.array([r["worked_filtered"] - r["worked_unfiltered"] for r in d])
    rr, pp = stats.pearsonr(pur, dmg)
    print(f"  corr(impurity, damage) over {len(d)} cells: r={rr:.2f} p={pp:.3g}")
except FileNotFoundError:
    print("  (not run yet)")

print("\n=== exp24b wrong-only control (paper, Sec 4.4) ===")
try:
    d = load("exp24b_wrongonly.json")
    table(d, MODELS4, ["nomem", "wrongonly", "worked_unfiltered"], "avg over 2 tasks")
    paired([r["wrongonly"] - r["nomem"] for r in d], "wrongonly - nomem")
    permodel_paired(d, MODELS4, lambda r: r["wrongonly"] - r["nomem"], "wrongonly - nomem")
except FileNotFoundError:
    print("  (not run yet)")

print("\n=== exp16 rho cliff ===")
d = load("exp16_ci_v2.json", "exp16_roster.json")
for mo in [m for m in MODELS4 if any(r["model"] == m for r in d)]:
    rs = [r for r in d if r["model"] == mo]
    curve = " ".join(f"{rho}:{np.mean([r['rho_acc'][rho] for r in rs]):.2f}"
                     for rho in ["0.0", "0.25", "0.5", "0.75", "1.0"])
    print(f"  {mo:15s} {curve} (seeds={len(rs)})")

print("\n=== exp13 roster v2 (Appendix B, Table 6; half-up rounding) ===")
ROSTER = ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b", "phi4-mini", "phi4",
          "ministral3-3b", "ministral3-8b", "ministral3-14b",
          "granite4.1-8b", "granite4.1-30b"]
d = load("exp13_roster_v2.json", "exp13_roster.json")
for mo in ROSTER:
    rs = [r for r in d if r["model"] == mo]
    if not rs:
        continue
    dd = [r["worked_clean"] - r["worked_ans_bad"] for r in rs]
    print(f"  {mo:15s} nomem={r2(np.mean([r['nomem'] for r in rs])):.2f} "
          f"worked={r2(np.mean([r['worked_clean'] for r in rs])):.2f} "
          f"delta={r2(np.mean(dd)):+.2f} cell-range=[{min(dd):+.2f},{max(dd):+.2f}] "
          f"n={len(rs)}")
paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "roster answer-corruption drop (pooled)")
tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "roster answer-corruption drop")

print("\n=== exp22 GSM8K (GSM8K appendix; diverse-task boundary; half-up rounding) ===")
d = load("exp22_gsm8k.json")
for mo in MODELS4:
    rs = [r for r in d if r["model"] == mo]
    if not rs:
        continue
    print(f"  {mo:15s} " + " ".join(
        f"{c}={r2(np.mean([r[c] for r in rs])):.2f}"
        for c in ["nomem", "worked_clean", "worked_ans_bad", "answer_only"]) + f" n={len(rs)}")
paired([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
tost([r["worked_clean"] - r["worked_ans_bad"] for r in d], "answer-corruption drop")
paired([r["worked_clean"] - r["nomem"] for r in d], "memory benefit (worked - nomem)")
permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["worked_ans_bad"],
                "answer-corruption drop")

print("\n=== exp28 PROCEDURE-ONLY: memory with the stored answer REMOVED (not corrupted) ===")
for _f, _lbl in [("exp28_proc_only.json", "controlled procedural (arith_hard, invoice)"),
                 ("exp28_bbh_proc_only.json", "BBH multistep_arithmetic_two")]:
    try:
        d = load(_f)
    except FileNotFoundError:
        print(f"  ({_lbl}: not run yet)")
        continue
    print(f"  -- {_lbl}, n={len(d)} cells --")
    C28 = ["nomem", "worked_clean", "proc_only", "proc_blank", "answer_only"]
    table(d, MODELS4, C28, "")
    for mo in MODELS4:
        rs = [r for r in d if r["model"] == mo]
        if rs:
            print(f"    {mo:15s} parse-fail " + " ".join(
                f"{c}={np.mean([r[c + '_parsefail'] for r in rs]):.2f}" for c in C28[1:]))
    # Does dropping the answer cost anything? (positive = removal hurts)
    for _a, _b, _lab in [("worked_clean", "proc_only", "worked - proc_only (answer field dropped)"),
                         ("worked_clean", "proc_blank", "worked - proc_blank (no answer at all)"),
                         ("worked_clean", "answer_only", "worked - answer_only")]:
        paired([r[_a] - r[_b] for r in d], _lab)
        if _b != "answer_only":
            tost([r[_a] - r[_b] for r in d], _lab)
    # Does answer-free memory still beat no memory? (positive = it helps)
    for _c in ["worked_clean", "proc_only", "proc_blank", "answer_only"]:
        paired([r[_c] - r["nomem"] for r in d], f"{_c} - nomem (memory benefit)")
    permodel_paired(d, MODELS4, lambda r: r["worked_clean"] - r["proc_only"],
                    "worked - proc_only")
    permodel_paired(d, MODELS4, lambda r: r["proc_blank"] - r["nomem"],
                    "proc_blank - nomem")
    mixedlm_check(d, lambda r: r["worked_clean"] - r["proc_blank"], "worked - proc_blank")

print("\n=== exp28 POOLED over both regimes (the numbers quoted in Sec 4.1 and the answer-removal appendix) ===")
try:
    _p28 = (json.load(open(os.path.join(R, "exp28_proc_only.json")))
            + json.load(open(os.path.join(R, "exp28_bbh_proc_only.json"))))
    print(f"  pooled cells n={len(_p28)}")
    for _c in ["proc_only", "proc_blank"]:
        _da = np.array([r["worked_clean"] - r[_c] for r in _p28])
        _dp = np.array([r[_c + "_parsefail"] - r["worked_clean_parsefail"] for r in _p28])
        paired(_da, f"worked - {_c} (pooled)")
        _r, _rp = stats.pearsonr(_dp, _da)
        _sl, _ic, _rv, _pv, _se = stats.linregress(_dp, _da)
        print(f"    mediation: corr(parse-fail rise, drop) r={_r:+.3f} p={_rp:.3g}; "
              f"slope={_sl:.2f} intercept={_ic:+.3f} R2={_rv ** 2:.2f}")
        # the null that matters: cells where output formatting is essentially unaffected
        _sub = _da[_dp <= 0.02]
        paired(_sub, f"worked - {_c} | parse-fail rise <= 0.02")
        tost(_sub, f"worked - {_c} | parse-fail rise <= 0.02")
except FileNotFoundError:
    print("  (not run yet)")

print("\n=== Ceiling control (paper, Sec 4.1): the answer-corruption null on cells "
      "where memory DEMONSTRABLY helps ===")
# A model already at ceiling cannot be damaged by corrupting a memory it does not need.
# Restrict to procedural cells with a real memory benefit and re-test the null.
_ceil = []
for _f in ["exp13_ci_v2.json", "exp22_bbh_ci.json", "exp22_objcount_ci.json",
           "exp27_tracking3.json", "exp27_boolean.json"]:
    for _r in json.load(open(os.path.join(R, _f))):
        _ceil.append((_r["worked_clean"] - _r["nomem"],
                      _r["worked_clean"] - _r["worked_ans_bad"]))
print(f"  all procedural cells: n={len(_ceil)} "
      f"mean benefit={np.mean([b for b, _ in _ceil]):+.4f} "
      f"mean drop={np.mean([d for _, d in _ceil]):+.4f}")
for _thr in (0.05, 0.10, 0.15):
    _sub = [d for b, d in _ceil if b >= _thr]
    print(f"  -- benefit >= {_thr:.2f} (mean benefit "
          f"{np.mean([b for b, _ in _ceil if b >= _thr]):+.3f}) --")
    paired(_sub, f"answer-corruption drop | benefit>={_thr:.2f}")
    tost(_sub, f"answer-corruption drop | benefit>={_thr:.2f}")
_rb, _rd = [b for b, _ in _ceil], [d for _, d in _ceil]
_r, _rp = stats.pearsonr(_rb, _rd)
print(f"  corr(memory benefit, answer-corruption drop) over {len(_ceil)} cells: "
      f"r={_r:+.3f} p={_rp:.3g}")

print("\n=== Holm step-down over the secondary residual-effect family ===")
fam = {"exp13 pooled answer-corruption": _p_exp13_pooled,
       "exp13 Ministral answer-corruption": _p_exp13_ministral,
       "exp21 compositional answer-corruption": _p_exp21_answer}
if _p_objcount_pooled is not None:
    fam["objcount pooled answer-corruption"] = _p_objcount_pooled
if _p_boolean_granite is not None:
    fam["boolean Granite answer-corruption"] = _p_boolean_granite
    fam["boolean Ministral answer-corruption (neg.)"] = _p_boolean_ministral
holm(fam, "procedural residual effects (both signs)")
