"""Regenerate the paper figures from the result files in results/ and write them to figures/.

Bars and points are means with 95% t-intervals over model x seed cells.
Run from the repository root.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

R = "results"
OUT = "figures"
BLUE, VERM, ORANGE, GREEN = "#0072B2", "#D55E00", "#E69F00", "#009E73"
MODELS4 = ["llama3.2-3b", "phi4-mini", "ministral3-3b", "granite4.1-8b"]
NAME = {
    "llama3.2-3b": "Llama-3.2-3B",
    "phi4-mini": "Phi-4-mini",
    "ministral3-3b": "Ministral-3-3B",
    "granite4.1-8b": "Granite-4.1-8B",
    "phi4": "Phi-4 (14B)",
    "ministral3-14b": "Ministral-3-14B",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.grid.axis": "y",
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load(*names):
    """Load the first existing result file among names."""
    for n in names:
        p = os.path.join(R, n)
        if os.path.exists(p):
            print(f"  using {p}")
            return json.loads(Path(p).read_text())
    raise FileNotFoundError(names)


def ci95(xs):
    """Half-width of the 95% t-interval for the mean; 0 for fewer than two or constant values."""
    xs = np.asarray(xs, float)
    if len(xs) < 2 or np.std(xs) == 0:
        return 0.0
    return stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))


def permodel(rows, model, fn):
    return [fn(r) for r in rows if r["model"] == model]


def grouped_drops(ax, drops_a, drops_b, label_a, label_b, ylabel):
    x = np.arange(len(MODELS4))
    w = 0.38
    ma = [np.mean(drops_a[m]) for m in MODELS4]
    mb = [np.mean(drops_b[m]) for m in MODELS4]
    ea = [ci95(drops_a[m]) for m in MODELS4]
    eb = [ci95(drops_b[m]) for m in MODELS4]
    ax.bar(x - w / 2, ma, w, yerr=ea, capsize=3, color=BLUE, label=label_a)
    ax.bar(x + w / 2, mb, w, yerr=eb, capsize=3, color=VERM, label=label_b)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in MODELS4], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)


def fig_dichotomy():
    proc = load("exp13_ci_v2.json", "exp13_ci.json")
    know = load("exp15_ci_v2.json", "exp15_ci.json")
    dp = {m: permodel(proc, m, lambda r: r["worked_clean"] - r["worked_ans_bad"]) for m in MODELS4}
    dk = {m: permodel(know, m, lambda r: r["worked_clean"] - r["worked_ans_bad"]) for m in MODELS4}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    grouped_drops(
        ax,
        dp,
        dk,
        "procedural (synthetic)",
        "knowledge (synthetic)",
        "drop from answer/fact corruption",
    )
    fig.savefig(f"{OUT}/fig_dichotomy_synthetic.png")
    plt.close(fig)


def fig_real():
    bbh = load("exp22_bbh_ci.json")
    pop = load("exp19_ci.json")
    dp = {m: permodel(bbh, m, lambda r: r["worked_clean"] - r["worked_ans_bad"]) for m in MODELS4}
    dk = {m: permodel(pop, m, lambda r: r["worked_clean"] - r["fact_corrupt"]) for m in MODELS4}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    grouped_drops(
        ax, dp, dk, "procedural (BBH)", "knowledge (PopQA)", "drop from answer/fact corruption"
    )
    fig.savefig(f"{OUT}/real_dichotomy.png")
    plt.close(fig)


def fig_a3():
    d = load("exp18_ci_v2.json", "exp18_ci.json")
    models = ["llama3.2-3b", "phi4-mini", "granite4.1-8b", "phi4", "ministral3-14b"]
    dm = {
        m: permodel(d, m, lambda r: r["worked_correct"] - r["worked_wrong_method"]) for m in models
    }
    da = {
        m: permodel(d, m, lambda r: r["worked_correct"] - r["worked_ans_corrupt"]) for m in models
    }
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(
        x - w / 2,
        [np.mean(dm[m]) for m in models],
        w,
        yerr=[ci95(dm[m]) for m in models],
        capsize=3,
        color=VERM,
        label="corrupt the method",
    )
    ax.bar(
        x + w / 2,
        [np.mean(da[m]) for m in models],
        w,
        yerr=[ci95(da[m]) for m in models],
        capsize=3,
        color=BLUE,
        label="corrupt the answer",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in models], fontsize=8)
    ax.set_ylabel("drop vs correct memory")
    ax.legend(frameon=False)
    fig.savefig(f"{OUT}/fig_a3_method_answer.png")
    plt.close(fig)


def fig_copyrate():
    # Prefer the k=3 copy-rate results; the k=1 files are fallbacks.
    d = load("exp25_copyrate_k3.json", "exp23_copyrate_v2.json", "exp23_copyrate.json")
    models = [m for m in MODELS4 if any(r["model"] == m for r in d)]
    cp = {m: permodel(d, m, lambda r: r["proc_copy"]) for m in models}
    ck = {m: permodel(d, m, lambda r: r["know_copy"]) for m in models}
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(
        x - w / 2,
        [np.mean(cp[m]) for m in models],
        w,
        yerr=[ci95(cp[m]) for m in models],
        capsize=3,
        color=BLUE,
        label="procedural",
    )
    ax.bar(
        x + w / 2,
        [np.mean(ck[m]) for m in models],
        w,
        yerr=[ci95(ck[m]) for m in models],
        capsize=3,
        color=VERM,
        label="knowledge",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in models], fontsize=9)
    ax.set_ylabel("copy-rate  P(output = planted W)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.savefig(f"{OUT}/fig_copyrate.png")
    plt.close(fig)


def fig_mixed():
    d = load("exp21_mixed.json")
    conds = [
        ("worked_clean", "worked", BLUE),
        ("fact_corrupt", "fact corrupt", VERM),
        ("method_corrupt", "method corrupt", ORANGE),
        ("answer_corrupt", "answer corrupt", GREEN),
    ]
    x = np.arange(len(MODELS4))
    w = 0.2
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for j, (c, lab, col) in enumerate(conds):
        vals = {m: permodel(d, m, lambda r, c=c: r[c]) for m in MODELS4}
        ax.bar(
            x + (j - 1.5) * w,
            [np.mean(vals[m]) for m in MODELS4],
            w,
            yerr=[ci95(vals[m]) for m in MODELS4],
            capsize=2,
            color=col,
            label=lab,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in MODELS4], fontsize=9)
    ax.set_ylabel("accuracy")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.savefig(f"{OUT}/fig_mixed.png")
    plt.close(fig)


def fig_rho():
    d = load("exp16_ci_v2.json", "exp16_roster.json")
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    # Thick and low z-order so the reference stays visible under a model line that coincides.
    ax.plot(
        rhos,
        [1 - r for r in rhos],
        "--",
        color="gray",
        lw=3.0,
        alpha=0.55,
        zorder=1,
        label=r"$(1-\rho)$ reference",
    )
    for m, col in zip(MODELS4, [GREEN, ORANGE, "#CC79A7", BLUE], strict=True):
        rows = [r for r in d if r["model"] == m]
        if not rows:
            continue
        mean = [np.mean([r["rho_acc"][str(rho)] for r in rows]) for rho in rhos]
        err = [ci95([r["rho_acc"][str(rho)] for r in rows]) for rho in rhos]
        ax.errorbar(
            rhos, mean, yerr=err, fmt="-o", ms=4, lw=1.6, capsize=2, color=col, label=NAME[m]
        )
    ax.set_xlabel(r"fraction of corrupted facts  $\rho$")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/fig_rho_cliff.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_dichotomy, fig_real, fig_a3, fig_copyrate, fig_mixed, fig_rho):
        print(f.__name__)
        f()
    print("done ->", OUT)
