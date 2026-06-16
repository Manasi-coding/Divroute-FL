"""
run_scalability.py
==================
Scalability experiment harness for DivRoute-FL.

Sweeps:
    num_clients        : [25, 50, 100]
    clients_per_round  : [15, 20, 30]

For every valid (num_clients, clients_per_round) pair where
clients_per_round < num_clients, runs:
    1. FedAvg baseline
    2. Recommended DivRoute configuration

Collects per run:
    accuracy, download_mb, upload_mb, bidir_mb, runtime_sec

Writes:
    results/scalability.csv
    results/scalability_report.txt

Usage:
    python run_scalability.py
"""

import csv
import json
import sys
import textwrap
import time
from pathlib import Path

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Sweep parameters ───────────────────────────────────────────────────────────
NUM_CLIENTS_SWEEP       = [25, 50, 100]
CLIENTS_PER_ROUND_SWEEP = [15, 20, 30]
NUM_ROUNDS              = 20
SEED                    = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/scalability")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "scalability.csv"
REPORT_PATH = RESULTS_DIR / "scalability_report.txt"

CSV_COLUMNS = [
    "num_clients", "clients_per_round", "method",
    "accuracy", "download_mb", "upload_mb", "bidir_mb", "runtime_sec",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_log(log_path: str, num_rounds: int, clients_per_round: int) -> dict:
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc   = history[-1]["test_accuracy"]
    total_dl    = sum(e["total_download_bytes"] for e in history)
    total_ul    = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul

    delta_numel        = history[0]["delta_numel"]
    fedavg_bidir_total = clients_per_round * delta_numel * 4 * 2 * num_rounds

    return {
        "accuracy":    final_acc,
        "download_mb": total_dl    / 1e6,
        "upload_mb":   total_ul    / 1e6,
        "bidir_mb":    total_bidir / 1e6,
    }


def _run_one(cfg: Config, label: str) -> tuple[dict, float]:
    """Run training, return (metrics dict, wall_time_sec)."""
    t0 = time.perf_counter()
    run(cfg)
    elapsed = time.perf_counter() - t0
    metrics = _parse_log(cfg.log_path, cfg.num_rounds, cfg.clients_per_round)
    return metrics, elapsed


def _valid_pairs() -> list[tuple[int, int]]:
    """Return (num_clients, clients_per_round) pairs where cpr < nc."""
    return [
        (nc, cpr)
        for nc  in NUM_CLIENTS_SWEEP
        for cpr in CLIENTS_PER_ROUND_SWEEP
        if cpr < nc
    ]


def _estimate_runtime(pairs: list[tuple[int, int]]) -> str:
    """
    Rough estimate: single-seed, 20-round baseline on 25c/15s ≈ 3 min.
    Scale linearly with clients_per_round (dominant cost).
    """
    base_min = 3.0   # measured on RTX 3050 for 25 clients / 15 selected
    total = 0.0
    for nc, cpr in pairs:
        scale = cpr / 15
        total += 2 * base_min * scale    # ×2 for FedAvg + DivRoute
    h, m = divmod(int(total), 60)
    return f"~{h}h {m}m" if h else f"~{int(total)}m"


# ── Report builder ─────────────────────────────────────────────────────────────
def _build_report(rows: list[dict]) -> str:
    pairs = sorted({(r["num_clients"], r["clients_per_round"]) for r in rows})

    lines = []
    lines.append("DivRoute-FL — Scalability Experiment Report")
    lines.append("=" * 52)
    lines.append(f"Rounds  : {NUM_ROUNDS}")
    lines.append(f"Seed    : {SEED}")
    lines.append(f"Dataset : CIFAR-10 (alpha=0.9 Dirichlet)")
    lines.append("")

    def _row(nc, cpr, method):
        for r in rows:
            if (r["num_clients"] == nc and
                    r["clients_per_round"] == cpr and
                    r["method"] == method):
                return r
        return None

    for nc, cpr in pairs:
        fa = _row(nc, cpr, "FedAvg")
        dr = _row(nc, cpr, "DivRoute")
        if fa is None or dr is None:
            continue

        gap   = (fa["accuracy"] - dr["accuracy"]) * 100
        saving = (1 - dr["bidir_mb"] / fa["bidir_mb"]) * 100 if fa["bidir_mb"] else 0.0

        lines.append(f"  Clients={nc:>3}, Selected={cpr:>2}")
        lines.append(f"  {'':─<48}")
        lines.append(f"  {'Method':<12} {'Accuracy':>9}  {'Bidir MB':>10}  {'Runtime':>9}")
        lines.append(f"  {'':─<12} {'':─>9}  {'':─>10}  {'':─>9}")
        lines.append(f"  {'FedAvg':<12} {fa['accuracy']*100:>8.2f}%"
                     f"  {fa['bidir_mb']:>10.1f}  {fa['runtime_sec']:>8.1f}s")
        lines.append(f"  {'DivRoute':<12} {dr['accuracy']*100:>8.2f}%"
                     f"  {dr['bidir_mb']:>10.1f}  {dr['runtime_sec']:>8.1f}s")
        lines.append(f"  Gap: {gap:.2f} pp  |  Comm saving: {saving:.1f}%")
        lines.append("")

    # summary: saving vs clients_per_round
    lines.append("  Communication saving summary")
    lines.append(f"  {'Clients':<10} {'Selected':<10} {'Saving%':>8}")
    lines.append(f"  {'':─<10} {'':─<10} {'':─>8}")
    for nc, cpr in pairs:
        fa = _row(nc, cpr, "FedAvg")
        dr = _row(nc, cpr, "DivRoute")
        if fa and dr and fa["bidir_mb"]:
            saving = (1 - dr["bidir_mb"] / fa["bidir_mb"]) * 100
            lines.append(f"  {nc:<10} {cpr:<10} {saving:>7.1f}%")

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    pairs = _valid_pairs()
    n_runs = len(pairs) * 2   # FedAvg + DivRoute

    print(f"Scalability sweep: {len(pairs)} (nc, cpr) pairs × 2 methods = {n_runs} runs")
    print(f"Parameter grid:")
    for nc, cpr in pairs:
        print(f"  num_clients={nc:>3}, clients_per_round={cpr:>2}")
    print(f"Estimated runtime: {_estimate_runtime(pairs)}")
    print()

    rows: list[dict] = []
    run_idx = 0

    for nc, cpr in pairs:
        for method in ("FedAvg", "DivRoute"):
            run_idx += 1
            tag = f"nc{nc}_cpr{cpr}_{method.lower()}"
            log_path = str(LOGS_DIR / f"{tag}.json")

            print(f"\n{'='*65}")
            print(f"  Run {run_idx}/{n_runs}: {method} | "
                  f"num_clients={nc}, clients_per_round={cpr}")
            print(f"{'='*65}\n")

            if method == "FedAvg":
                cfg = Config(
                    num_clients        = nc,
                    clients_per_round  = cpr,
                    num_rounds         = NUM_ROUNDS,
                    seed               = SEED,
                    fedavg_baseline_mode = True,
                    skip_plot_prompt   = True,
                    log_path           = log_path,
                )
            else:
                cfg = get_recommended_divroute_config(
                    num_clients        = nc,
                    clients_per_round  = cpr,
                    num_rounds         = NUM_ROUNDS,
                    seed               = SEED,
                    skip_plot_prompt   = True,
                    log_path           = log_path,
                )

            metrics, elapsed = _run_one(cfg, method)
            row = {
                "num_clients":       nc,
                "clients_per_round": cpr,
                "method":            method,
                **metrics,
                "runtime_sec":       round(elapsed, 1),
            }
            rows.append(row)

            print(f"\n  → accuracy={row['accuracy']*100:.2f}%  "
                  f"bidir={row['bidir_mb']:.1f}MB  "
                  f"time={row['runtime_sec']:.0f}s")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "num_clients":       row["num_clients"],
                "clients_per_round": row["clients_per_round"],
                "method":            row["method"],
                "accuracy":          f"{row['accuracy']:.4f}",
                "download_mb":       f"{row['download_mb']:.2f}",
                "upload_mb":         f"{row['upload_mb']:.2f}",
                "bidir_mb":          f"{row['bidir_mb']:.2f}",
                "runtime_sec":       f"{row['runtime_sec']:.1f}",
            })
    print(f"\n[done] CSV written to {CSV_PATH}")

    # ── Write report ───────────────────────────────────────────────────────────
    report = _build_report(rows)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report written to {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
