"""Paired comparisons over results/plateaus.csv (per-seed plateau = mean of last 20 rounds).

    python scripts/paired_stats.py                      # built-in comparison families
    python scripts/paired_stats.py femnist "0.5%" pure matched

For each A - B (paired by seed): mean difference, 95% t-interval, number of seeds with A > B, paired t-test p,
exact two-sided sign-flip permutation p (the smallest possible is 2/2^n: 0.25 at n=3, 0.031 at n=6), and
Holm-adjusted p within the family (both tests). All inference here is exploratory: the comparisons were chosen
after seeing the data. Rows whose source is not JSON are flagged.
"""

import csv
import itertools
import os
import sys
import numpy as np
from scipy import stats

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "plateaus.csv",
)


def load():
    d = {}
    with open(CSV, encoding="utf-8") as file:
        for r in csv.DictReader(file):
            key = (r["dataset"], r["setting"], r["variant"], int(r["seed"]))
            if key in d:
                raise ValueError(f"Duplicate experimental unit: {key}")
            d[key] = (float(r["plateau_last20"]), r["source"])
    return d


def perm_p(diff):
    n = len(diff)
    obs = abs(diff.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    return float(np.mean(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-12))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def family(data, ds, setting, pairs, label=None):
    res = []
    for a, b in pairs:
        seeds = sorted(
            set(s for (d_, st, v, s) in data if (d_, st, v) == (ds, setting, a))
            & set(s for (d_, st, v, s) in data if (d_, st, v) == (ds, setting, b))
        )
        if len(seeds) < 3:
            print(f"  skip {a}-{b}: only {len(seeds)} shared seeds")
            continue
        x = np.array([data[(ds, setting, a, s)][0] for s in seeds])
        y = np.array([data[(ds, setting, b, s)][0] for s in seeds])
        d = x - y
        tp = stats.ttest_rel(x, y).pvalue
        pp = perm_p(d)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        nj = sum(
            1
            for s in seeds
            for v in (a, b)
            if not data[(ds, setting, v, s)][1].startswith("JSON")
        )
        res.append((a, b, len(d), d.mean(), ci, int((d > 0).sum()), tp, pp, nj))
    if not res:
        return
    ht = holm([r[6] for r in res])
    hp = holm([r[7] for r in res])
    print(
        f"\n{label or ds + ' ' + setting}  (family of {len(res)}; Holm within family)"
    )
    print(
        f"  {'A - B':26s} {'n':>2} {'diff':>7} {'95% CI':>17} {'A>B':>5} {'t p':>7} {'t Holm':>7} {'perm p':>7} {'perm Holm':>9}  non-JSON values"
    )
    for (a, b, n, m, ci, w, tp, pp, nj), th, ph in zip(res, ht, hp):
        print(
            f"  {a + ' - ' + b:26s} {n:>2} {m:+7.2f} [{ci[0]:+6.2f},{ci[1]:+6.2f}] {w:>3}/{n} {tp:7.3f} {th:7.3f} {pp:7.3f} {ph:9.3f}  {nj}"
        )


FAMILIES = [
    (
        "femnist",
        "0.5%",
        [
            ("pure", "purerandom"),
            ("pure", "matched"),
            ("purerandom", "matched"),
            ("pureinvert", "pure"),
            ("pureinvert", "matched"),
            ("matched", "uniform"),
        ],
        "FEMNIST 0.5%, EF on (eight seeds)",
    ),
    (
        "femnist",
        "1%",
        [
            ("pure", "purerandom"),
            ("pure", "matched"),
            ("purerandom", "matched"),
            ("pureinvert", "pure"),
            ("pureinvert", "matched"),
            ("matched", "uniform"),
        ],
        "FEMNIST 1%",
    ),
    (
        "femnist",
        "0.5%-noEF",
        [
            ("pure", "purerandom"),
            ("pure", "matched"),
            ("purerandom", "matched"),
            ("matched", "uniform"),
        ],
        "FEMNIST 0.5%, EF off",
    ),
    (
        "femnist",
        "0.5%",
        [
            ("pure_sizeinvert", "pureinvert"),
            ("pure_size", "pure"),
            ("pure_sizeinvert", "pure_size"),
            ("pure_sizeinvert", "matched"),
        ],
        "FEMNIST 0.5%, size routing vs divergence routing (seeds 42-44)",
    ),
    (
        "cifar10",
        "0.5%",
        [
            ("pure", "purerandom"),
            ("pure", "matched"),
            ("purerandom", "matched"),
            ("pureinvert", "pure"),
            ("pureinvert", "matched"),
            ("matched", "uniform"),
        ],
        "CIFAR-10 0.5%",
    ),
]

for dataset in ("femnist", "cifar10"):
    FAMILIES.append((
        dataset, "0.5%",
        [("dagc", "matched"), ("dagc", "dagcrandom"), ("dagc", "dagcrev"),
         ("dagcrandom", "matched"), ("dagcrev", "matched")],
        f"{dataset.upper()} 0.5%, DAGC-inspired allocation (seeds 42-44)",
    ))

if __name__ == "__main__":
    data = load()
    if len(sys.argv) == 5:
        family(data, sys.argv[1], sys.argv[2], [(sys.argv[3], sys.argv[4])])
    else:
        for ds, st, pairs, lab in FAMILIES:
            family(data, ds, st, pairs, lab)
