"""
analyze_multiseed.py
====================
Publication-quality statistical analysis and plotting for DivRoute-FL.

Reads:   results/multiseed_summary.csv
Writes:  results/accuracy_comparison.png
         results/communication_comparison.png
         results/statistical_report.txt

Usage:
    python analyze_multiseed.py
"""

import csv
import math
import os
import statistics
import sys
import textwrap
from pathlib import Path

# Force UTF-8 output on Windows consoles (cp1252 cannot render box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend, safe on all platforms
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — plots will be skipped")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH      = Path("results/multiseed_summary.csv")
ACC_PLOT      = Path("results/accuracy_comparison.png")
COMM_PLOT     = Path("results/communication_comparison.png")
REPORT_PATH   = Path("results/statistical_report.txt")

RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Colour palette (publication-friendly, colour-blind safe) ───────────────────
C_FEDAVG   = "#4878CF"   # steel blue
C_DIVROUTE = "#E87722"   # burnt orange
C_GRID     = "#E8E8E8"
C_BG       = "#FAFAFA"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_csv(path: Path) -> dict[str, list]:
    """Returns {'FedAvg': [...], 'DivRoute': [...]} keyed on method."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = {"FedAvg": [], "DivRoute": []}
    for row in rows:
        method = row["method"]
        data[method].append({
            "seed":        int(row["seed"]),
            "accuracy":    float(row["accuracy"]),
            "download_mb": float(row["download_mb"]),
            "upload_mb":   float(row["upload_mb"]),
            "bidir_mb":    float(row["bidir_mb"]),
            "saving_pct":  float(row["saving_pct"]),
        })
    return data


# ── Statistics helpers ─────────────────────────────────────────────────────────
def _mean(xs):  return statistics.mean(xs)
def _std(xs):   return statistics.stdev(xs) if len(xs) > 1 else 0.0

def _ci95(xs):
    """95 % confidence interval half-width using t-distribution."""
    n = len(xs)
    if n < 2:
        return 0.0
    se = _std(xs) / math.sqrt(n)
    # t critical value for 95 % CI (two-tailed) — hardcoded for n ≤ 10
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
               5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
               9: 2.262, 10: 2.228}
    t_crit = t_table.get(n - 1, 2.0)   # fallback to z≈2 for large n
    return t_crit * se


def compute_stats(method_rows: list) -> dict:
    accs  = [r["accuracy"]   for r in method_rows]
    saves = [r["saving_pct"] for r in method_rows]
    bidirs = [r["bidir_mb"]  for r in method_rows]
    return {
        "n":            len(accs),
        "accs":         accs,
        "saves":        saves,
        "bidirs":       bidirs,
        "mean_acc":     _mean(accs),
        "std_acc":      _std(accs),
        "ci95_acc":     _ci95(accs),
        "mean_save":    _mean(saves),
        "std_save":     _std(saves),
        "mean_bidir":   _mean(bidirs),
        "std_bidir":    _std(bidirs),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────
def _apply_style(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(C_BG)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(labelsize=10)


def plot_accuracy(data: dict, fa_stats: dict, dr_stats: dict) -> None:
    seeds = sorted({r["seed"] for r in data["FedAvg"]})
    fa_map = {r["seed"]: r["accuracy"] * 100 for r in data["FedAvg"]}
    dr_map = {r["seed"]: r["accuracy"] * 100 for r in data["DivRoute"]}

    x     = list(range(len(seeds)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    bars_fa = ax.bar([i - width/2 for i in x],
                     [fa_map[s] for s in seeds], width,
                     color=C_FEDAVG, alpha=0.88, label="FedAvg", zorder=3,
                     edgecolor="white", linewidth=0.5)
    bars_dr = ax.bar([i + width/2 for i in x],
                     [dr_map[s] for s in seeds], width,
                     color=C_DIVROUTE, alpha=0.88, label="DivRoute", zorder=3,
                     edgecolor="white", linewidth=0.5)

    # mean lines with CI shading
    ax.axhline(fa_stats["mean_acc"] * 100, color=C_FEDAVG,
               linestyle="--", linewidth=1.4, alpha=0.7, zorder=4)
    ax.axhline(dr_stats["mean_acc"] * 100, color=C_DIVROUTE,
               linestyle="--", linewidth=1.4, alpha=0.7, zorder=4)

    # value labels on bars
    for bar in bars_fa:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5, color=C_FEDAVG)
    for bar in bars_dr:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7.5, color=C_DIVROUTE)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds], fontsize=9.5)
    ax.set_ylim(max(0, min(list(fa_map.values()) + list(dr_map.values())) - 5), 100)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    _apply_style(ax, "Test Accuracy (%)",
                 "Test Accuracy by Seed — FedAvg vs DivRoute")

    # annotation box
    gap = (fa_stats["mean_acc"] - dr_stats["mean_acc"]) * 100
    textstr = (f"FedAvg:   {fa_stats['mean_acc']*100:.2f}% ± {fa_stats['std_acc']*100:.2f}%\n"
               f"DivRoute: {dr_stats['mean_acc']*100:.2f}% ± {dr_stats['std_acc']*100:.2f}%\n"
               f"Gap:      {gap:.2f} pp")
    ax.text(0.98, 0.04, textstr, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    fig.tight_layout()
    fig.savefig(ACC_PLOT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {ACC_PLOT}")


def plot_communication(data: dict, dr_stats: dict) -> None:
    seeds  = sorted({r["seed"] for r in data["DivRoute"]})
    dr_map = {r["seed"]: r for r in data["DivRoute"]}
    fa_map = {r["seed"]: r for r in data["FedAvg"]}

    dr_bidirs = [dr_map[s]["bidir_mb"] for s in seeds]
    fa_bidirs = [fa_map[s]["bidir_mb"] for s in seeds]
    savings   = [dr_map[s]["saving_pct"] for s in seeds]

    x     = list(range(len(seeds)))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # ── Left: bidirectional MB comparison ────────────────────────────────────
    ax1.bar([i - width/2 for i in x], fa_bidirs, width,
            color=C_FEDAVG, alpha=0.88, label="FedAvg", zorder=3,
            edgecolor="white")
    ax1.bar([i + width/2 for i in x], dr_bidirs, width,
            color=C_DIVROUTE, alpha=0.88, label="DivRoute", zorder=3,
            edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"seed={s}" for s in seeds], fontsize=9.5)
    ax1.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    _apply_style(ax1, "Bidirectional Data (MB)",
                 "Communication Volume by Seed")

    # ── Right: saving % per seed ───────────────────────────────────────────
    bars = ax2.bar(x, savings, color=C_DIVROUTE, alpha=0.88, zorder=3,
                   edgecolor="white")
    ax2.axhline(_mean(savings), color=C_DIVROUTE, linestyle="--",
                linewidth=1.4, alpha=0.7, zorder=4,
                label=f"mean = {_mean(savings):.1f}%")
    for bar, pct in zip(bars, savings):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, color=C_DIVROUTE)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seed={s}" for s in seeds], fontsize=9.5)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10, framealpha=0.9, edgecolor="#DDDDDD")
    _apply_style(ax2, "Bidirectional Communication Saving (%)",
                 "DivRoute Communication Saving by Seed")

    fig.tight_layout(pad=2.5)
    fig.savefig(COMM_PLOT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {COMM_PLOT}")


# ── Statistical report ─────────────────────────────────────────────────────────
def build_report(data: dict, fa: dict, dr: dict) -> str:
    """
    Formulas used
    -------------
    mean(x)   = sum(x) / n
    std(x)    = sqrt(sum((xi - mean)^2) / (n-1))    [Bessel's correction]
    CI95      = t_{alpha/2, n-1} * std / sqrt(n)
    gap_pp    = mean_acc_FedAvg - mean_acc_DivRoute   [percentage points]
    comm_red  = 1 - mean_bidir_DivRoute / mean_bidir_FedAvg
    """
    gap_pp   = (fa["mean_acc"] - dr["mean_acc"]) * 100
    comm_red = 1.0 - dr["mean_bidir"] / fa["mean_bidir"] if fa["mean_bidir"] else 0.0

    # paired t-test on accuracy (same seeds → paired)
    if HAS_SCIPY:
        fa_accs = [r["accuracy"] for r in data["FedAvg"]]
        dr_accs = [r["accuracy"] for r in data["DivRoute"]]
        # sort by seed to ensure pairing is correct
        fa_accs = [r["accuracy"] for r in sorted(data["FedAvg"],  key=lambda r: r["seed"])]
        dr_accs = [r["accuracy"] for r in sorted(data["DivRoute"], key=lambda r: r["seed"])]
        t_stat, p_val = sp_stats.ttest_rel(fa_accs, dr_accs)
        sig_note = ("statistically significant (p < 0.05)"
                    if p_val < 0.05 else "not statistically significant (p ≥ 0.05)")
        ttest_block = textwrap.dedent(f"""
            Paired t-test (FedAvg vs DivRoute accuracy)
            ─────────────────────────────────────────────
            t-statistic : {t_stat:+.4f}
            p-value     : {p_val:.4f}
            Result      : {sig_note}
        """)
    else:
        ttest_block = "\nPaired t-test: scipy not available — skipped.\n"

    seeds_run = sorted({r["seed"] for r in data["FedAvg"]})

    report = textwrap.dedent(f"""
        DivRoute-FL — Multi-Seed Statistical Report
        ============================================
        Generated from : {CSV_PATH}
        Seeds evaluated: {seeds_run}
        Rounds per run : 20
        Dataset        : CIFAR-10 (alpha=0.9 Dirichlet)

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ACCURACY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        FedAvg
          n              : {fa['n']}
          mean accuracy  : {fa['mean_acc']*100:.2f}%
          std accuracy   : {fa['std_acc']*100:.2f}%
          95% CI         : ± {fa['ci95_acc']*100:.2f}%
          per-seed       : {[f"{r['accuracy']*100:.2f}%" for r in sorted(data['FedAvg'],  key=lambda r: r['seed'])]}

        DivRoute (recommended config)
          n              : {dr['n']}
          mean accuracy  : {dr['mean_acc']*100:.2f}%
          std accuracy   : {dr['std_acc']*100:.2f}%
          95% CI         : ± {dr['ci95_acc']*100:.2f}%
          per-seed       : {[f"{r['accuracy']*100:.2f}%" for r in sorted(data['DivRoute'], key=lambda r: r['seed'])]}

        Absolute accuracy gap (FedAvg − DivRoute)
          mean gap       : {gap_pp:.2f} pp
          interpretation : DivRoute trades {gap_pp:.2f} pp accuracy for communication savings

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        COMMUNICATION
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        FedAvg (bidirectional MB over 20 rounds)
          mean           : {fa['mean_bidir']:.2f} MB
          std            : {fa['std_bidir']:.2f} MB

        DivRoute (bidirectional MB over 20 rounds)
          mean           : {dr['mean_bidir']:.2f} MB
          std            : {dr['std_bidir']:.2f} MB

        Communication saving
          formula        : 1 − (DivRoute_bidir / FedAvg_bidir)
          mean saving    : {dr['mean_save']:.1f}% ± {dr['std_save']:.1f}%
          relative reduc : {comm_red*100:.1f}% reduction vs FedAvg
        {ttest_block}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        FORMULAS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        mean(x)    = Σxᵢ / n
        std(x)     = sqrt(Σ(xᵢ − mean)² / (n−1))   [Bessel's correction]
        CI95       = t_{{α/2, n−1}} × std / sqrt(n)
        gap_pp     = mean_acc_FedAvg − mean_acc_DivRoute
        comm_red   = 1 − mean_bidir_DivRoute / mean_bidir_FedAvg
        saving_pct = comm_red × 100
    """).lstrip()

    return report


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if not CSV_PATH.exists():
        print(f"[error] CSV not found: {CSV_PATH}")
        print("        Run 'python run_multiseed.py' first to generate results.")
        return

    print(f"[info] loading {CSV_PATH}")
    data = load_csv(CSV_PATH)

    if not data["FedAvg"] or not data["DivRoute"]:
        print("[error] CSV is empty or missing method rows.")
        return

    fa = compute_stats(data["FedAvg"])
    dr = compute_stats(data["DivRoute"])

    # ── Plots ──────────────────────────────────────────────────────────────────
    if HAS_MPL:
        plot_accuracy(data, fa, dr)
        plot_communication(data, dr)
    else:
        print("[warn] Skipping plots (matplotlib not available).")
        print("       Install with: pip install matplotlib")

    # ── Report ─────────────────────────────────────────────────────────────────
    report = build_report(data, fa, dr)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[report] saved {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
