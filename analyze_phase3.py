"""
analyze_phase3.py
=================
Reads all Phase 3 logs (logs/phase3/) and produces:

  1. results/phase3_summary.csv  — per-seed rows for both methods
  2. A publication-ready table printed to stdout showing:
8:        Method | Final Accuracy (mean±std) | Best Accuracy (mean±std) |
9:        Download MB | Upload MB | Bidirectional MB | Communication Saving %

Usage:
    python analyze_phase3.py
"""

import csv
import json
import statistics
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/phase3")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "phase3_summary.csv"

SEEDS   = [1, 42, 84]
METHODS = ["fedavg", "divroute"]

# Phase 3 experiment parameters (needed for baseline byte accounting)
CLIENTS_PER_ROUND = 10

CSV_COLUMNS = [
    "seed", "method",
    "final_accuracy", "best_accuracy",
    "download_mb", "upload_mb", "bidir_mb",
    "fedavg_bidir_mb", "saving_pct",
]


# ── Log parsing ───────────────────────────────────────────────────────────────
def _parse_log(log_path: Path) -> dict:
    """Read a single run log and return the metrics needed for the CSV row."""
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc   = history[-1]["test_accuracy"]
    best_acc    = max(e["test_accuracy"] for e in history)
    total_dl    = sum(e["total_download_bytes"] for e in history)
    total_ul    = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul

    # FedAvg bidirectional reference: full float32 delta both ways, every round
    delta_numel        = history[0]["delta_numel"]
    fedavg_per_round   = CLIENTS_PER_ROUND * delta_numel * 4 * 2   # bytes
    actual_rounds      = len(history)
    fedavg_total_bidir = fedavg_per_round * actual_rounds

    saving_pct = (
        100.0 * (1.0 - total_bidir / fedavg_total_bidir)
        if fedavg_total_bidir else 0.0
    )

    return {
        "final_accuracy": final_acc,
        "best_accuracy":  best_acc,
        "download_mb":    total_dl    / 1e6,
        "upload_mb":      total_ul    / 1e6,
        "bidir_mb":       total_bidir / 1e6,
        "fedavg_bidir_mb":fedavg_total_bidir / 1e6,
        "saving_pct":     saving_pct,
    }


# ── Summary statistics ─────────────────────────────────────────────────────────
def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[str] = []

    for seed in SEEDS:
        for method in METHODS:
            log_path = LOGS_DIR / f"{method}_seed{seed}.json"
            if not log_path.exists():
                missing.append(str(log_path))
                continue
            metrics = _parse_log(log_path)
            rows.append({"seed": seed, "method": method, **metrics})

    if missing:
        print(f"\n[WARNING] {len(missing)} log file(s) not found — "
              f"they will be absent from the summary:")
        for p in missing:
            print(f"    {p}")
        print()

    if not rows:
        print("[ERROR] No log files found in logs/phase3/. "
              "Run  python run_phase3_scalability.py  first.")
        return

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed":           row["seed"],
                "method":         row["method"],
                "final_accuracy": f"{row['final_accuracy']:.4f}",
                "best_accuracy":  f"{row['best_accuracy']:.4f}",
                "download_mb":    f"{row['download_mb']:.2f}",
                "upload_mb":      f"{row['upload_mb']:.2f}",
                "bidir_mb":       f"{row['bidir_mb']:.2f}",
                "fedavg_bidir_mb":f"{row['fedavg_bidir_mb']:.2f}",
                "saving_pct":     f"{row['saving_pct']:.1f}",
            })
    print(f"[analyze_phase3] CSV saved -> {CSV_PATH}")

    # ── Publication-ready table ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  PHASE 3 — CIFAR-100 / ResNet-18 / 75 clients / 100 rounds  "
          f"({len(SEEDS)} seeds)")
    print(f"{'='*72}\n")

    display_methods = {
        "fedavg":   "FedAvg",
        "divroute": "DivRoute",
    }

    summary_rows: list[dict] = []
    for method_key, method_label in display_methods.items():
        subset = [r for r in rows if r["method"] == method_key]
        if not subset:
            continue

        accs       = [r["final_accuracy"] for r in subset]
        best_accs  = [r["best_accuracy"]  for r in subset]
        dls        = [r["download_mb"]    for r in subset]
        uls        = [r["upload_mb"]      for r in subset]
        bidirs     = [r["bidir_mb"]       for r in subset]
        saves      = [r["saving_pct"]     for r in subset]

        m_acc,      s_acc      = _mean_std(accs)
        m_best_acc, s_best_acc = _mean_std(best_accs)
        m_dl,       s_dl       = _mean_std(dls)
        m_ul,       s_ul       = _mean_std(uls)
        m_bidir,    s_bidir    = _mean_std(bidirs)
        m_save,     s_save     = _mean_std(saves)

        summary_rows.append({
            "label":          method_label,
            "acc_mean":       m_acc,
            "acc_std":        s_acc,
            "best_acc_mean":  m_best_acc,
            "best_acc_std":   s_best_acc,
            "dl_mean":        m_dl,
            "ul_mean":        m_ul,
            "bidir_mean":     m_bidir,
            "save_mean":      m_save,
            "save_std":       s_save,
        })

    # Header
    col_w = [12, 28, 28, 18, 18, 22, 22]
    headers = [
        "Method",
        "Final Accuracy (mean±std)",
        "Best Accuracy (mean±std)",
        "Download MB",
        "Upload MB",
        "Bidirectional MB",
        "Comm Saving %",
    ]
    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    sep_line    = "  " + "  ".join("-" * w for w in col_w)
    print(header_line)
    print(sep_line)

    for sr in summary_rows:
        acc_str        = f"{sr['acc_mean']*100:.2f}% ± {sr['acc_std']*100:.2f}%"
        best_acc_str   = f"{sr['best_acc_mean']*100:.2f}% ± {sr['best_acc_std']*100:.2f}%"
        dl_str         = f"{sr['dl_mean']:.1f}"
        ul_str         = f"{sr['ul_mean']:.1f}"
        bidir_str      = f"{sr['bidir_mean']:.1f}"
        save_str       = f"{sr['save_mean']:.1f}% ± {sr['save_std']:.1f}%"
        row_str = "  " + "  ".join([
            sr["label"].ljust(col_w[0]),
            acc_str.ljust(col_w[1]),
            best_acc_str.ljust(col_w[2]),
            dl_str.ljust(col_w[3]),
            ul_str.ljust(col_w[4]),
            bidir_str.ljust(col_w[5]),
            save_str.ljust(col_w[6]),
        ])
        print(row_str)

    print()

    # Per-seed breakdown
    print("  Per-seed accuracy:")
    for method_key, method_label in display_methods.items():
        subset = [r for r in rows if r["method"] == method_key]
        if not subset:
            continue
        detail = ", ".join(
            f"seed={r['seed']} -> {r['final_accuracy']*100:.2f}%"
            for r in sorted(subset, key=lambda x: x["seed"])
        )
        print(f"    [{method_label}] {detail}")

    print()


if __name__ == "__main__":
    main()
