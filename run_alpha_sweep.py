"""
run_alpha_sweep.py
==================
Heterogeneity sweep for DivRoute-FL Phase 2.

Sweeps Dirichlet alpha across [0.1, 0.3, 0.5, 0.9] to characterise how
data heterogeneity affects FedAvg, Uniform Top-5%, and DivRoute.

Lower alpha  = more heterogeneous (each client holds fewer classes)
Higher alpha = more homogeneous   (IID-like distribution)

For every (alpha, method, seed) combination:
  - Runs 100 rounds of federated training
  - Saves per-round log to logs/alpha_sweep/
  - Records accuracy + communication metrics

Features:
  - Resumable: any run whose log already exists is skipped
  - Deterministic: seed controls partition + client sampling
  - No changes to training, aggregation, compression, or routing

Usage:
    python run_alpha_sweep.py
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
ALPHAS     = [0.1, 0.9]
SEEDS      = [42]
NUM_ROUNDS = 100
METHODS    = ["FedAvg", "UniformTop5", "DivRoute"]

NUM_CLIENTS       = 25
CLIENTS_PER_ROUND = 15

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/alpha_sweep")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "alpha_sweep.csv"

CSV_COLUMNS = [
    "alpha", "seed", "method",
    "accuracy", "download_mb", "upload_mb", "bidir_mb",
    "saving_pct", "runtime_sec",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _alpha_tag(alpha: float) -> str:
    """Stable filename fragment for an alpha value: 0.1 → 'a01', 0.9 → 'a09'."""
    return f"a{str(alpha).replace('.', '')}"


def _log_path(alpha: float, method: str, seed: int) -> Path:
    tag = method.lower().replace("uniformtop5", "uniform")
    return LOGS_DIR / f"{_alpha_tag(alpha)}_{tag}_seed{seed}.json"


def _make_config(alpha: float, method: str, seed: int, log_path: str) -> Config:
    shared = dict(
        alpha             = alpha,
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


def _parse_log(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc   = history[-1]["test_accuracy"]
    total_dl    = sum(e["total_download_bytes"] for e in history)
    total_ul    = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul
    n_rounds    = len(history)

    delta_numel        = history[0]["delta_numel"]
    fedavg_bidir_total = CLIENTS_PER_ROUND * delta_numel * 4 * 2 * n_rounds
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


def _mean(xs): return statistics.mean(xs)  if xs else float("nan")
def _std(xs):  return statistics.stdev(xs) if len(xs) > 1 else 0.0


# ── Summary printer ────────────────────────────────────────────────────────────
def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  ALPHA SWEEP SUMMARY")
    print(f"{'='*72}")
    print(f"  Seeds  : {SEEDS}")
    print(f"  Rounds : {NUM_ROUNDS}")
    print(f"  Alphas : {ALPHAS}")
    print(f"  (Lower alpha = more heterogeneous data distribution)\n")

    header = (f"  {'Alpha':>6}  {'Method':<14}  "
              f"{'Acc mean':>9}  {'Acc std':>8}  "
              f"{'Save mean':>10}  {'Save std':>9}  "
              f"{'Bidir MB':>10}")
    separator = f"  {'-'*6}  {'-'*14}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*10}"

    prev_alpha = None
    for alpha in ALPHAS:
        print(header)
        print(separator)
        for method in METHODS:
            subset = [r for r in rows
                      if r["alpha"] == alpha and r["method"] == method]
            accs   = [r["accuracy"]   for r in subset]
            saves  = [r["saving_pct"] for r in subset]
            bidirs = [r["bidir_mb"]   for r in subset]
            print(
                f"  {alpha:>6.1f}  {method:<14}  "
                f"{_mean(accs)*100:>8.2f}%  "
                f"{_std(accs)*100:>7.2f}%  "
                f"{_mean(saves):>9.1f}%  "
                f"{_std(saves):>8.1f}%  "
                f"{_mean(bidirs):>10.1f}"
            )
        print()

    # DivRoute vs FedAvg gap across alphas
    print(f"  DivRoute accuracy gap vs FedAvg (mean over seeds):")
    print(f"  {'Alpha':>6}  {'FedAvg':>8}  {'DivRoute':>9}  {'Gap (pp)':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*10}")
    for alpha in ALPHAS:
        fa_accs = [r["accuracy"] for r in rows
                   if r["alpha"] == alpha and r["method"] == "FedAvg"]
        dr_accs = [r["accuracy"] for r in rows
                   if r["alpha"] == alpha and r["method"] == "DivRoute"]
        if fa_accs and dr_accs:
            gap = (_mean(fa_accs) - _mean(dr_accs)) * 100
            print(f"  {alpha:>6.1f}  {_mean(fa_accs)*100:>7.2f}%  "
                  f"{_mean(dr_accs)*100:>8.2f}%  {gap:>+10.2f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Build full job list and identify which are already done
    all_jobs = [
        (alpha, method, seed)
        for alpha  in ALPHAS
        for method in METHODS
        for seed   in SEEDS
    ]
    total_runs = len(all_jobs)

    pending  = []
    skipped  = 0
    for alpha, method, seed in all_jobs:
        lp = _log_path(alpha, method, seed)
        if lp.exists():
            skipped += 1
        else:
            pending.append((alpha, method, seed))

    print(f"Alpha sweep: {len(ALPHAS)} alphas × {len(METHODS)} methods × "
          f"{len(SEEDS)} seeds = {total_runs} runs")
    print(f"Skipped (log exists): {skipped}  |  To run: {len(pending)}")

    if pending:
        # ~3 min per run on RTX 3050 at 20 rounds; scale for 100 rounds
        est_min = len(pending) * 3 * (NUM_ROUNDS / 20)
        h, m = divmod(int(est_min), 60)
        est_str = f"{h}h {m}m" if h else f"{int(est_min)}m"
        print(f"Estimated remaining time: ~{est_str}\n")
    else:
        print("[info] All runs already complete. Loading from logs...\n")

    # Execute pending runs
    run_num = skipped
    for alpha, method, seed in pending:
        run_num += 1
        lp  = _log_path(alpha, method, seed)
        cfg = _make_config(alpha, method, seed, str(lp))

        print(f"\n{'='*68}")
        print(f"  Run {run_num}/{total_runs}: {method} | "
              f"alpha={alpha} | seed={seed} | {NUM_ROUNDS} rounds")
        print(f"{'='*68}\n")

        t0 = time.perf_counter()
        run(cfg)
        elapsed = time.perf_counter() - t0
        print(f"\n  → done in {elapsed:.0f}s")

    # Collect results from all log files (completed this session + resumed)
    rows: list[dict] = []
    for alpha, method, seed in all_jobs:
        lp = _log_path(alpha, method, seed)
        if not lp.exists():
            print(f"[warn] Missing log: {lp} — skipping row")
            continue
        metrics = _parse_log(lp)
        rows.append({
            "alpha":       alpha,
            "seed":        seed,
            "method":      method,
            "runtime_sec": None,   # only available for runs executed this session
            **metrics,
        })

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "alpha":       row["alpha"],
                "seed":        row["seed"],
                "method":      row["method"],
                "accuracy":    f"{row['accuracy']:.4f}",
                "download_mb": f"{row['download_mb']:.2f}",
                "upload_mb":   f"{row['upload_mb']:.2f}",
                "bidir_mb":    f"{row['bidir_mb']:.2f}",
                "saving_pct":  f"{row['saving_pct']:.1f}",
                "runtime_sec": (f"{row['runtime_sec']:.1f}"
                                if row["runtime_sec"] is not None else ""),
            })
    print(f"\n[done] CSV written to {CSV_PATH}")

    _print_summary(rows)


if __name__ == "__main__":
    main()
