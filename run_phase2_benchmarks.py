"""
run_phase2_benchmarks.py
========================
Phase 2 main benchmark: FedAvg vs Uniform Top-5% vs DivRoute.

For each seed in SEEDS, runs all three methods using identical dataset
partitions and client sampling sequences (guaranteed by shared seed).

Features:
  - Resumable: skips any run whose log file already exists.
  - Deterministic: each method receives the same seed → same partition.
  - Produces results/phase2_main_results.csv and a terminal summary.

Usage:
    python run_phase2_benchmarks.py
"""

import csv
import json
import statistics
import sys
import time
from pathlib import Path

from divroute_fl.config import (
    Config,
    get_recommended_divroute_config,
    get_uniform_top5_config,
)
from divroute_fl.main import run

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Experiment parameters ──────────────────────────────────────────────────────
SEEDS      = [42, 123, 456]
NUM_ROUNDS = 100

NUM_CLIENTS       = 25
CLIENTS_PER_ROUND = 15

METHODS = ["FedAvg", "UniformTop5", "DivRoute"]

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/phase2")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "phase2_main_results.csv"

CSV_COLUMNS = [
    "seed", "method", "accuracy",
    "download_mb", "upload_mb", "bidir_mb",
    "saving_pct", "runtime_sec",
]


# ── Config factories ───────────────────────────────────────────────────────────
def _make_config(method: str, seed: int, log_path: str) -> Config:
    shared = dict(
        num_clients       = NUM_CLIENTS,
        clients_per_round = CLIENTS_PER_ROUND,
        num_rounds        = NUM_ROUNDS,
        seed              = seed,
        skip_plot_prompt  = True,
        log_path          = log_path,
    )
    if method == "FedAvg":
        return Config(fedavg_baseline_mode=True, **shared)
    elif method == "UniformTop5":
        return get_uniform_top5_config(**shared)
    elif method == "DivRoute":
        return get_recommended_divroute_config(**shared)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _log_path(method: str, seed: int) -> Path:
    tag = method.lower().replace("uniformtop5", "uniform")
    return LOGS_DIR / f"{tag}_seed{seed}.json"


# ── Log parsing ────────────────────────────────────────────────────────────────
def _parse_log(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc   = history[-1]["test_accuracy"]
    total_dl    = sum(e["total_download_bytes"] for e in history)
    total_ul    = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul
    num_rounds  = len(history)

    # FedAvg reference: full float32 delta, both directions, every round
    delta_numel        = history[0]["delta_numel"]
    fedavg_bidir_total = CLIENTS_PER_ROUND * delta_numel * 4 * 2 * num_rounds
    saving_pct = (
        100.0 * (1.0 - total_bidir / fedavg_bidir_total)
        if fedavg_bidir_total else 0.0
    )

    return {
        "accuracy":    final_acc,
        "download_mb": total_dl    / 1e6,
        "upload_mb":   total_ul    / 1e6,
        "bidir_mb":    total_bidir / 1e6,
        "saving_pct":  saving_pct,
    }


# ── Summary printer ────────────────────────────────────────────────────────────
def _mean(xs):  return statistics.mean(xs)  if xs else float("nan")
def _std(xs):   return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*68}")
    print("  PHASE 2 BENCHMARK SUMMARY")
    print(f"{'='*68}")
    print(f"\n  Seeds   : {SEEDS}")
    print(f"  Rounds  : {NUM_ROUNDS}")
    print(f"  Clients : {NUM_CLIENTS} total, {CLIENTS_PER_ROUND} selected/round")

    print(f"\n  {'Method':<14} {'Acc mean':>9}  {'Acc std':>8}  "
          f"{'Save mean':>10}  {'Save std':>9}  {'Bidir MB':>10}")
    print(f"  {'-'*14} {'-'*9}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*10}")

    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        accs   = [r["accuracy"]   for r in subset]
        saves  = [r["saving_pct"] for r in subset]
        bidirs = [r["bidir_mb"]   for r in subset]
        print(
            f"  {method:<14} "
            f"{_mean(accs)*100:>8.2f}%  "
            f"{_std(accs)*100:>7.2f}%  "
            f"{_mean(saves):>9.1f}%  "
            f"{_std(saves):>8.1f}%  "
            f"{_mean(bidirs):>10.1f}"
        )

    # Per-seed breakdown
    print(f"\n  Per-seed accuracy:")
    print(f"  {'Seed':>6}  {'FedAvg':>8}  {'UniformTop5':>12}  {'DivRoute':>9}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*9}")
    for seed in SEEDS:
        accs_by_method = {}
        for r in rows:
            if r["seed"] == seed:
                accs_by_method[r["method"]] = r["accuracy"] * 100
        fa  = accs_by_method.get("FedAvg",       float("nan"))
        u5  = accs_by_method.get("UniformTop5",  float("nan"))
        dr  = accs_by_method.get("DivRoute",     float("nan"))
        print(f"  {seed:>6}  {fa:>7.2f}%  {u5:>11.2f}%  {dr:>8.2f}%")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    total_runs  = len(SEEDS) * len(METHODS)
    skipped     = 0
    completed   = 0

    rows: list[dict] = []

    # Check for already-completed runs so we can resume
    pending = []
    for seed in SEEDS:
        for method in METHODS:
            lp = _log_path(method, seed)
            if lp.exists():
                print(f"[skip] {method} seed={seed} — log exists: {lp}")
                skipped += 1
            else:
                pending.append((seed, method))

    print(f"\nTotal runs: {total_runs}  |  "
          f"Skipped (already done): {skipped}  |  "
          f"To run: {len(pending)}")
    if not pending:
        print("[info] All runs already complete. Loading results from logs...\n")
    else:
        est_min_per_run = NUM_ROUNDS * CLIENTS_PER_ROUND / (15 * 20) * 3
        est_total = len(pending) * est_min_per_run
        h, m = divmod(int(est_total), 60)
        est_str = f"{h}h {m}m" if h else f"{int(est_total)}m"
        print(f"Estimated remaining time: ~{est_str}\n")

    run_num = skipped
    for seed, method in pending:
        run_num += 1
        lp = _log_path(method, seed)
        cfg = _make_config(method, seed, str(lp))

        print(f"\n{'='*68}")
        print(f"  Run {run_num}/{total_runs}: {method} | seed={seed} | {NUM_ROUNDS} rounds")
        print(f"{'='*68}\n")

        t0 = time.perf_counter()
        run(cfg)
        elapsed = time.perf_counter() - t0

        completed += 1
        print(f"\n  → done in {elapsed:.0f}s")

    # ── Collect all results (completed + previously skipped) ──────────────────
    for seed in SEEDS:
        for method in METHODS:
            lp = _log_path(method, seed)
            if not lp.exists():
                print(f"[warn] Missing log: {lp} — skipping row")
                continue
            metrics = _parse_log(lp)
            # runtime is only available for runs executed in this session
            runtime = None
            rows.append({
                "seed":        seed,
                "method":      method,
                "runtime_sec": runtime,
                **metrics,
            })

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
                "runtime_sec": f"{row['runtime_sec']:.1f}" if row["runtime_sec"] else "",
            })
    print(f"\n[done] CSV written to {CSV_PATH}")

    _print_summary(rows)


if __name__ == "__main__":
    main()
