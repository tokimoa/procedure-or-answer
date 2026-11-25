"""Synthetic procedural tasks, exemplar pools and answer-field corruption.

Each task generator in POOLS returns items with a problem, its gold integer answer and a
templated worked solution ending in an ANSWER line.
"""

import re

import numpy as np

from experiments.memory import solve as mem_solve


def _pool_arith_hard(n, rng):
    pool = []
    for _ in range(n):
        pa, bo, it = int(rng.integers(3, 9)), int(rng.integers(4, 9)), int(rng.integers(3, 8))
        dmg, cr = int(rng.integers(1, 4)) * 5, int(rng.integers(6, 12))
        total = pa * bo * it
        good = total - (total * dmg) // 100
        ans = good // cr
        q = (
            f"A warehouse has {pa} pallets, each holding {bo} boxes of {it} items. "
            f"{dmg}% of all items are damaged and removed. The rest are packed into "
            f"crates of {cr} (leftover items are not crated). How many full crates?"
        )
        sol = (
            f"Total items = {pa}*{bo}*{it} = {total}. Damaged = {dmg}% of {total} = "
            f"{(total * dmg) // 100}. Good = {good}. Crates = {good}//{cr} = {ans}.\n"
            f"ANSWER: {ans}"
        )
        pool.append({"problem": q, "gold": ans, "sol": sol})
    return pool


def _pool_invoice(n, rng):
    pool = []
    for _ in range(n):
        q1, p1 = int(rng.integers(2, 10)), int(rng.integers(100, 900))
        q2, p2 = int(rng.integers(2, 10)), int(rng.integers(100, 900))
        sub = q1 * p1 + q2 * p2
        ans = (sub * 110) // 100
        q = (
            f"Invoice #{int(rng.integers(10000, 99999))}. Line items: {q1} units x JPY "
            f"{p1}; {q2} units x JPY {p2}. Add 10% consumption tax to the subtotal. "
            f"What is the total payable in JPY (integer)?"
        )
        sol = (
            f"Subtotal = {q1}*{p1} + {q2}*{p2} = {sub}. Total = {sub}*110//100 = {ans}.\n"
            f"ANSWER: {ans}"
        )
        pool.append({"problem": q, "gold": ans, "sol": sol})
    return pool


def _pool_date_add(n, rng):
    pool = []
    for _ in range(n):
        d = int(rng.integers(1, 28))
        m = int(rng.integers(1, 13))
        y = int(rng.integers(2020, 2027))
        nn = int(rng.integers(3, 31))
        tot = (m - 1) + nn
        ey, em = y + tot // 12, tot % 12 + 1
        ans = ey * 10000 + em * 100 + d
        q = (
            f"A contract starts on {d:02d}/{m:02d}/{y} (day/month/year) and lasts "
            f"{nn} months. Give the END date (same day-of-month) as an integer in "
            f"YYYYMMDD format."
        )
        sol = (
            f"Add {nn} months to month {m}: total = ({m}-1)+{nn} = {tot}. End year = "
            f"{y} + {tot}//12 = {ey}. End month = {tot}%12 + 1 = {em}. Day stays "
            f"{d:02d}.\nANSWER: {ans}"
        )
        pool.append({"problem": q, "gold": ans, "sol": sol})
    return pool


POOLS = {"arith_hard": _pool_arith_hard, "invoice": _pool_invoice, "date_add": _pool_date_add}


def corrupt(pool, rho, rng):
    """Return a copy of the pool with the answer field wrong in a fraction rho of exemplars.

    Only the ANSWER line is substituted; str.replace on the value would also change
    intermediate numbers that contain the same digits.
    """
    out, idx = [], rng.permutation(len(pool))
    n_bad = int(round(rho * len(pool)))
    bad = set(idx[:n_bad].tolist())
    for i, it in enumerate(pool):
        if i in bad:
            wrong = it["gold"] + int(rng.choice([-2, -1, 1, 2, 3]))
            sol, nsub = re.subn(r"ANSWER:\s*-?\d+", f"ANSWER: {wrong}", it["sol"])
            if nsub == 0:
                sol = it["sol"] + f"\nANSWER: {wrong}"
            out.append({"problem": it["problem"], "gold": it["gold"], "sol": sol})
        else:
            out.append(it)
    return out


def build_clean_pool(be, pool_fn, target, seed, source):
    """Build a pool of up to `target` exemplars with correct answers.

    source='templated' keeps the templated solutions. source='model' replaces them with
    the model's own correct zero-shot solutions, so exemplar style matches the model's
    output; it tries 3 * target problems and may return fewer than target.
    """
    rng = np.random.default_rng(seed + 333)
    raw = pool_fn(target if source == "templated" else target * 3, rng)
    if source == "templated":
        return raw[:target]
    clean = []
    for it in raw:
        ans, _, txt = mem_solve(be, it["problem"], [])
        if ans == it["gold"]:
            clean.append({"problem": it["problem"], "gold": it["gold"], "sol": txt})
        if len(clean) >= target:
            break
    return clean


def acc_with(be, pool, test, k, rng):
    """Accuracy on `test` with k exemplars drawn at random per item (pool=None: no-memory)."""
    corr = []
    for t in test:
        ex = [] if pool is None else [pool[i] for i in rng.permutation(len(pool))[:k]]
        ans, _, _ = mem_solve(be, t["problem"], ex)
        corr.append(1.0 if ans == t["gold"] else 0.0)
    return float(np.mean(corr))
