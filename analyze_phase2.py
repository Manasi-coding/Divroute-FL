"""
analyze_phase2.py
=================
Publication-quality analysis for DivRoute-FL Phase 2.

Reads:
    results/phase2_main_results.csv
    results/alpha_sweep.csv

Generates:
    results/phase2_accuracy.png        — grouped bar: accuracy mean ± std
    results/phase2_communication.png   — grouped bar: bidir communication
    results/phase2_efficiency.png      — scatter: accuracy vs communication
    results/alpha_sweep_accuracy.png   — line: accuracy vs alpha
    results/alpha_sweep_communication.png  — line: bidir MB vs alpha
    results/phase2_report.txt          — full statistical report

Usage:
    python analyze_phase2.py
"""

import csv
import math
import statistics
import sys
import textwrap
from pathlib import Path

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not available — plots will be skipped")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR    = Path("results")
PHASE2_CSV     = RESULTS_DIR / "phase2_main_results.csv"
ALPHA_CSV      = RESULTS_DIR / "alpha_sweep.csv"

P2_ACC_PLOT    = RESULTS_DIR / "phase2_accuracy.png"
P2_COMM_PLOT   = RESULTS_DIR / "phase2_communication.png"
P2_EFF_PLOT    = RESULTS_DIR / "phase2_efficiency.png"
AS_ACC_PLOT    = RESULTS_DIR / "alpha_sweep_accuracy.png"
AS_COMM_PLOT   = RESULTS_DIR / "alpha_sweep_communication.png"
REPORT_PATH    = RESULTS_DIR / "phase2_report.txt"

# ── Design tokens ──────────────────────────────────────────────────────────────
PALETTE = {
    "FedAvg":      "#4878CF",   # steel blue
    "UniformTop5": "#6ACC65",   # sage green
    "DivRoute":    "#E87722",   # burnt orange
}
METHODS   = ["FedAvg", "UniformTop5", "DivRoute"]
ALPHAS    = [0.1, 0.3, 0.5, 0.9]
C_GRID    = "#EBEBEB"
C_BG      = "#FAFAFA"
DPI       = 180
FONT_SM   = 9
FONT_MD   = 11
FONT_LG   = 13


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _group(rows: list[dict], **filters) -> list[dict]:
    """Return rows matching all filter key=value pairs."""
    result = rows
    for k, v in filters.items():
        result = [r for r in result if r.get(k) == v]
    return result


# ── Statistics helpers ─────────────────────────────────────────────────────────
def _mean(xs):  return statistics.mean(xs)  if xs else float("nan")
def _std(xs):   return statistics.stdev(xs) if len(xs) > 1 else 0.0

def _ci95(xs: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    se = _std(xs) / math.sqrt(n)
    t_table = {1:12.706, 2:4.303, 3:3.182, 4:2.776,
               5:2.571,  6:2.447, 7:2.365, 8:2.306,
               9:2.262, 10:2.228}
    t_crit = t_table.get(n - 1, 1.96)
    return t_crit * se


def _paired_ttest(xs: list, ys: list) -> tuple[float, float] | None:
    if not HAS_SCIPY or len(xs) < 2:
        return None
    t, p = sp_stats.ttest_rel(xs, ys)
    return float(t), float(p)


def _method_stats(rows: list[dict], method: str, field: str = "accuracy") -> dict:
    vals = [r[field] for r in rows if r.get("method") == method
            and r.get(field) is not None]
    return {
        "vals":   vals,
        "n":      len(vals),
        "mean":   _mean(vals),
        "std":    _std(vals),
        "ci95":   _ci95(vals),
    }


# ── Shared plot style ──────────────────────────────────────────────────────────
def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(C_BG)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_title(title,   fontsize=FONT_LG, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=FONT_MD, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FONT_MD, labelpad=6)
    ax.tick_params(labelsize=FONT_SM)


# ── Plot 1: phase2_accuracy.png ────────────────────────────────────────────────
def plot_phase2_accuracy(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    x      = range(len(METHODS))
    means  = []
    ci95s  = []
    colors = []

    for m in METHODS:
        s = _method_stats(rows, m, "accuracy")
        means.append(s["mean"] * 100)
        ci95s.append(s["ci95"] * 100)
        colors.append(PALETTE[m])

    bars = ax.bar(x, means, yerr=ci95s, capsize=5, width=0.55,
                  color=colors, alpha=0.88, zorder=3,
                  edgecolor="white", linewidth=0.5,
                  error_kw=dict(elinewidth=1.2, ecolor="#555555", capthick=1.2))

    for bar, mean, ci in zip(bars, means, ci95s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + ci + 0.3,
                f"{mean:.2f}%",
                ha="center", va="bottom", fontsize=FONT_SM, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(METHODS, fontsize=FONT_MD)
    ymin = max(0, min(means) - max(ci95s) - 5)
    ax.set_ylim(ymin, min(100, max(means) + max(ci95s) + 4))
    ax.set_ylabel("Test Accuracy (%)", fontsize=FONT_MD, labelpad=6)
    _style_ax(ax,
              title="Phase 2: Mean Accuracy ± 95% CI\n(3 seeds, 100 rounds, CIFAR-10)",
              xlabel="Method", ylabel="Test Accuracy (%)")

    fig.tight_layout()
    fig.savefig(P2_ACC_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {P2_ACC_PLOT}")


# ── Plot 2: phase2_communication.png ──────────────────────────────────────────
def plot_phase2_communication(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    x      = range(len(METHODS))
    means  = []
    ci95s  = []
    colors = []

    for m in METHODS:
        s = _method_stats(rows, m, "bidir_mb")
        means.append(s["mean"])
        ci95s.append(s["ci95"])
        colors.append(PALETTE[m])

    bars = ax.bar(x, means, yerr=ci95s, capsize=5, width=0.55,
                  color=colors, alpha=0.88, zorder=3,
                  edgecolor="white", linewidth=0.5,
                  error_kw=dict(elinewidth=1.2, ecolor="#555555", capthick=1.2))

    for bar, mean, ci in zip(bars, means, ci95s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + ci + max(means) * 0.01,
                f"{mean:.0f} MB",
                ha="center", va="bottom", fontsize=FONT_SM, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(METHODS, fontsize=FONT_MD)
    _style_ax(ax,
              title="Phase 2: Mean Bidirectional Communication\n(3 seeds, 100 rounds)",
              xlabel="Method", ylabel="Bidirectional Data (MB)")

    fig.tight_layout()
    fig.savefig(P2_COMM_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {P2_COMM_PLOT}")


# ── Plot 3: phase2_efficiency.png (accuracy vs communication scatter) ──────────
def plot_phase2_efficiency(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    for method in METHODS:
        subset = [r for r in rows if r.get("method") == method]
        xs = [r["bidir_mb"]   for r in subset if r.get("bidir_mb")   is not None]
        ys = [r["accuracy"]*100 for r in subset if r.get("accuracy") is not None]
        if not xs:
            continue

        ax.scatter(xs, ys, color=PALETTE[method], alpha=0.85, s=80,
                   zorder=4, label=method, edgecolors="white", linewidths=0.5)

        # mean marker
        mx, my = _mean(xs), _mean(ys)
        ax.scatter([mx], [my], color=PALETTE[method], s=220, marker="D",
                   zorder=5, edgecolors="white", linewidths=1.0)
        ax.annotate(f"{method}\n{my:.1f}%",
                    xy=(mx, my), xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=FONT_SM, color=PALETTE[method], fontweight="bold")

    _style_ax(ax,
              title="Phase 2: Accuracy vs Communication Trade-off\n(each point = one seed)",
              xlabel="Bidirectional Communication (MB)",
              ylabel="Test Accuracy (%)")
    ax.legend(fontsize=FONT_SM, framealpha=0.9, edgecolor="#DDDDDD")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    fig.tight_layout()
    fig.savefig(P2_EFF_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {P2_EFF_PLOT}")


# ── Plot 4: alpha_sweep_accuracy.png ──────────────────────────────────────────
def plot_alpha_accuracy(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    alpha_vals = sorted({r["alpha"] for r in rows if r.get("alpha") is not None})

    for method in METHODS:
        means, ci95s = [], []
        for alpha in alpha_vals:
            s = _method_stats(
                [r for r in rows if r.get("alpha") == alpha], method, "accuracy")
            means.append(s["mean"] * 100)
            ci95s.append(s["ci95"] * 100)

        ax.plot(alpha_vals, means, marker="o", linewidth=2.2,
                color=PALETTE[method], label=method, zorder=4, markersize=7)
        ax.fill_between(alpha_vals,
                        [m - c for m, c in zip(means, ci95s)],
                        [m + c for m, c in zip(means, ci95s)],
                        color=PALETTE[method], alpha=0.13, zorder=3)

    ax.set_xticks(alpha_vals)
    ax.set_xticklabels([str(a) for a in alpha_vals], fontsize=FONT_MD)
    ax.legend(fontsize=FONT_MD, framealpha=0.9, edgecolor="#DDDDDD")
    _style_ax(ax,
              title="Alpha Sweep: Accuracy vs Data Heterogeneity\n"
                    "(lower alpha = more heterogeneous; shading = 95% CI)",
              xlabel="Dirichlet Alpha", ylabel="Test Accuracy (%)")

    fig.tight_layout()
    fig.savefig(AS_ACC_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {AS_ACC_PLOT}")


# ── Plot 5: alpha_sweep_communication.png ─────────────────────────────────────
def plot_alpha_communication(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    alpha_vals = sorted({r["alpha"] for r in rows if r.get("alpha") is not None})

    for method in METHODS:
        means, ci95s = [], []
        for alpha in alpha_vals:
            s = _method_stats(
                [r for r in rows if r.get("alpha") == alpha], method, "bidir_mb")
            means.append(s["mean"])
            ci95s.append(s["ci95"])

        ax.plot(alpha_vals, means, marker="s", linewidth=2.2,
                color=PALETTE[method], label=method, zorder=4, markersize=7)
        ax.fill_between(alpha_vals,
                        [m - c for m, c in zip(means, ci95s)],
                        [m + c for m, c in zip(means, ci95s)],
                        color=PALETTE[method], alpha=0.13, zorder=3)

    ax.set_xticks(alpha_vals)
    ax.set_xticklabels([str(a) for a in alpha_vals], fontsize=FONT_MD)
    ax.legend(fontsize=FONT_MD, framealpha=0.9, edgecolor="#DDDDDD")
    _style_ax(ax,
              title="Alpha Sweep: Communication vs Data Heterogeneity\n"
                    "(lower alpha = more heterogeneous; shading = 95% CI)",
              xlabel="Dirichlet Alpha", ylabel="Bidirectional Communication (MB)")

    fig.tight_layout()
    fig.savefig(AS_COMM_PLOT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {AS_COMM_PLOT}")


# ── Report builder ─────────────────────────────────────────────────────────────
def _ttest_block(label: str, xs: list, ys: list) -> str:
    result = _paired_ttest(xs, ys)
    if result is None:
        if not HAS_SCIPY:
            return f"  {label}: scipy not available — skipped\n"
        return f"  {label}: insufficient data (n={len(xs)})\n"
    t, p = result
    sig = "SIGNIFICANT (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"
    return (f"  {label}\n"
            f"    t = {t:+.4f}   p = {p:.4f}   [{sig}]\n")


def build_report(p2_rows: list[dict], alpha_rows: list[dict]) -> str:
    lines = []

    def h(text):
        lines.append("")
        lines.append(text)
        lines.append("-" * len(text))

    lines.append("DivRoute-FL Phase 2 — Statistical Report")
    lines.append("=" * 50)
    lines.append(f"CI formula: CI95 = t_(n-1, alpha/2) * std / sqrt(n)")
    lines.append(f"t-test     : paired, matched on seed")
    lines.append(f"scipy      : {'available' if HAS_SCIPY else 'NOT available — t-tests skipped'}")

    # ── Phase 2 main results ──────────────────────────────────────────────────
    if p2_rows:
        h("Phase 2 Main Results  (100 rounds, seeds=[42,123,456])")
        for method in METHODS:
            s_acc  = _method_stats(p2_rows, method, "accuracy")
            s_comm = _method_stats(p2_rows, method, "bidir_mb")
            s_save = _method_stats(p2_rows, method, "saving_pct")
            lines.append(f"\n  [{method}]  n={s_acc['n']}")
            lines.append(f"    accuracy : {s_acc['mean']*100:.2f}% "
                         f"+/- {s_acc['std']*100:.2f}%   CI95 +/- {s_acc['ci95']*100:.2f}%")
            lines.append(f"    bidir MB : {s_comm['mean']:.1f} "
                         f"+/- {s_comm['std']:.1f}   CI95 +/- {s_comm['ci95']:.1f}")
            lines.append(f"    saving % : {s_save['mean']:.1f}% "
                         f"+/- {s_save['std']:.1f}%")

        h("Accuracy gap (FedAvg - method, in pp)")
        fa_mean = _method_stats(p2_rows, "FedAvg", "accuracy")["mean"]
        for method in ("UniformTop5", "DivRoute"):
            m_mean = _method_stats(p2_rows, method, "accuracy")["mean"]
            lines.append(f"  FedAvg - {method:<14}: {(fa_mean - m_mean)*100:+.2f} pp")

        h("Paired t-tests on accuracy (FedAvg seed order = DivRoute seed order)")
        seeds = sorted({int(r["seed"]) for r in p2_rows if r.get("seed") is not None})
        fa_accs = [r["accuracy"] for s in seeds
                   for r in p2_rows if int(r["seed"])==s and r["method"]=="FedAvg"]
        u5_accs = [r["accuracy"] for s in seeds
                   for r in p2_rows if int(r["seed"])==s and r["method"]=="UniformTop5"]
        dr_accs = [r["accuracy"] for s in seeds
                   for r in p2_rows if int(r["seed"])==s and r["method"]=="DivRoute"]
        lines.append(_ttest_block("FedAvg vs DivRoute",    fa_accs, dr_accs))
        lines.append(_ttest_block("UniformTop5 vs DivRoute", u5_accs, dr_accs))
    else:
        lines.append("\n[warn] phase2_main_results.csv not found or empty.")

    # ── Alpha sweep results ───────────────────────────────────────────────────
    if alpha_rows:
        h("Alpha Sweep Results  (100 rounds, seeds=[42,123,456])")
        alpha_vals = sorted({r["alpha"] for r in alpha_rows
                             if r.get("alpha") is not None})
        header = f"  {'Alpha':>6}  {'Method':<14}  {'Acc mean':>9}  {'Acc std':>8}  {'CI95':>7}  {'Save':>7}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for alpha in alpha_vals:
            for method in METHODS:
                s = _method_stats(
                    [r for r in alpha_rows if r.get("alpha") == alpha],
                    method, "accuracy")
                lines.append(
                    f"  {alpha:>6.1f}  {method:<14}  "
                    f"{s['mean']*100:>8.2f}%  "
                    f"{s['std']*100:>7.2f}%  "
                    f"+/-{s['ci95']*100:>5.2f}%  "
                )
            lines.append("")

        h("Alpha Sweep: DivRoute vs FedAvg gap across heterogeneity levels")
        for alpha in alpha_vals:
            fa = _method_stats([r for r in alpha_rows if r.get("alpha")==alpha],
                               "FedAvg", "accuracy")
            dr = _method_stats([r for r in alpha_rows if r.get("alpha")==alpha],
                               "DivRoute", "accuracy")
            gap = (fa["mean"] - dr["mean"]) * 100
            lines.append(f"  alpha={alpha}: FedAvg={fa['mean']*100:.2f}%  "
                         f"DivRoute={dr['mean']*100:.2f}%  "
                         f"gap={gap:+.2f}pp")
    else:
        lines.append("\n[warn] alpha_sweep.csv not found or empty.")

    lines.append("")
    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    p2_rows    = _load_csv(PHASE2_CSV)
    alpha_rows = _load_csv(ALPHA_CSV)

    if not p2_rows:
        print(f"[warn] {PHASE2_CSV} not found or empty — phase2 plots will be skipped")
    if not alpha_rows:
        print(f"[warn] {ALPHA_CSV} not found or empty — alpha plots will be skipped")

    if HAS_MPL:
        if p2_rows:
            plot_phase2_accuracy(p2_rows)
            plot_phase2_communication(p2_rows)
            plot_phase2_efficiency(p2_rows)
        if alpha_rows:
            plot_alpha_accuracy(alpha_rows)
            plot_alpha_communication(alpha_rows)
    else:
        print("[warn] matplotlib not available — install with: pip install matplotlib")

    report = build_report(p2_rows, alpha_rows)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[report] {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
