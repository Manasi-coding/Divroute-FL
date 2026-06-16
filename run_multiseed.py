"""
run_multiseed.py
================
Multi-seed evaluation framework for DivRoute-FL Phase 2.

For each seed in SEEDS, runs:
  1. FedAvg baseline
  2. Recommended DivRoute configuration

Saves per-run logs to logs/multiseed/ and a summary CSV to
results/multiseed_summary.csv.

Usage:
    python run_multiseed.py
"""

import csv
import json
import os
import statistics
from pathlib import Path

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

# ── Configuration ──────────────────────────────────────────────────────────────
SEEDS = [1, 7, 21, 42, 84]

LOGS_DIR    = Path("logs/multiseed")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "multiseed_summary.csv"

CSV_COLUMNS = [
    "seed", "method", "accuracy",
    "download_mb", "upload_mb", "bidir_mb", "saving_pct",
]



def _parse_log(log_path: str, num_rounds: int, clients_per_round: int) -> dict:
    """Read a run log and return the metrics needed for the CSV row."""
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc  = history[-1]["test_accuracy"]
    total_dl   = sum(e["total_download_bytes"] for e in history)
    total_ul   = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul

    # FedAvg bidirectional reference: full float32 delta both ways, every round
    delta_numel          = history[0]["delta_numel"]
    fedavg_per_round     = clients_per_round * delta_numel * 4 * 2   # bytes
    fedavg_total_bidir   = fedavg_per_round * num_rounds

    saving_pct = 100.0 * (1.0 - total_bidir / fedavg_total_bidir) if fedavg_total_bidir else 0.0

    return {
        "accuracy":    final_acc,
        "download_mb": total_dl    / 1e6,
        "upload_mb":   total_ul    / 1e6,
        "bidir_mb":    total_bidir / 1e6,
        "saving_pct":  saving_pct,
    }


# ── Experiment runners ─────────────────────────────────────────────────────────
def _run_fedavg(seed: int) -> str:
    log_path = str(LOGS_DIR / f"fedavg_seed{seed}.json")
    run(Config(
        fedavg_baseline_mode = True,
        seed                 = seed,
        skip_plot_prompt     = True,
        log_path             = log_path,
    ))
    return log_path


def _run_divroute(seed: int) -> str:
    log_path = str(LOGS_DIR / f"divroute_seed{seed}.json")
    run(get_recommended_divroute_config(
        seed             = seed,
        skip_plot_prompt = True,
        log_path         = log_path,
    ))
    return log_path


# ── Summary statistics ─────────────────────────────────────────────────────────
def _print_summary(rows: list[dict]) -> None:
    for method in ("FedAvg", "DivRoute"):
        subset = [r for r in rows if r["method"] == method]
        accs   = [r["accuracy"]   for r in subset]
        saves  = [r["saving_pct"] for r in subset]

        mean_acc  = statistics.mean(accs)
        std_acc   = statistics.stdev(accs) if len(accs) > 1 else 0.0
        mean_save = statistics.mean(saves)
        std_save  = statistics.stdev(saves) if len(saves) > 1 else 0.0

        print(f"\n  [{method}]")
        print(f"    accuracy  : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"    comm save : {mean_save:.1f}% ± {std_save:.1f}%")
        print(f"    per seed  : " +
              ", ".join(f"seed={r['seed']} → {r['accuracy']*100:.2f}%" for r in subset))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}  —  1/2: FedAvg baseline")
        print(f"{'='*70}\n")
        fedavg_log = _run_fedavg(seed)
        fedavg_metrics = _parse_log(fedavg_log, num_rounds=20, clients_per_round=15)
        rows.append({"seed": seed, "method": "FedAvg", **fedavg_metrics})

        print(f"\n{'='*70}")
        print(f"  SEED {seed}  —  2/2: DivRoute (recommended config)")
        print(f"{'='*70}\n")
        divroute_log = _run_divroute(seed)
        divroute_metrics = _parse_log(divroute_log, num_rounds=20, clients_per_round=15)
        rows.append({"seed": seed, "method": "DivRoute", **divroute_metrics})

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed":        row["seed"],
                "method":      row["method"],
                "accuracy":    f"{row['accuracy']:.4f}",
                "download_mb": f"{row['download_mb']:.2f}",
                "upload_mb":   f"{row['upload_mb']:.2f}",
                "bidir_mb":    f"{row['bidir_mb']:.2f}",
                "saving_pct":  f"{row['saving_pct']:.1f}",
            })

    print(f"\n\n{'='*70}")
    print("  MULTI-SEED EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  CSV saved to: {CSV_PATH}")
    print(f"  Per-run logs: {LOGS_DIR}/")

    _print_summary(rows)

    # ── Formatted results table ────────────────────────────────────────────────
    print(f"\n  {'Seed':>6}  {'Method':<10}  {'Accuracy':>9}  {'Bidir MB':>10}  {'Saving':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*8}")
    for row in rows:
        print(f"  {row['seed']:>6}  {row['method']:<10}  "
              f"{row['accuracy']*100:>8.2f}%  "
              f"{row['bidir_mb']:>10.2f}  "
              f"{row['saving_pct']:>7.1f}%")

    print()


if __name__ == "__main__":
    main()
