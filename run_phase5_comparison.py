"""
run_phase5_comparison.py
=========================
Phase 5 — Comparison Against Published Baselines

Runs all five methods on ResNet-18 / CIFAR-100 / 100 clients at 10% participation
across 3 seeds to produce the paper's main comparison table (Table 1 / Table 3).

Methods:
  1. FedAvg          — uncompressed full-delta baseline (upper-bound accuracy)
  2. FedZip          — Top-1% sparsification + k-means quantisation (byte-cost
                       computed post-hoc using the FedZip formula; training uses
                       full deltas to preserve accuracy fidelity)
  3. FedSparse-0.01  — L1 proximity regularisation, lambda=0.01
  4. Uniform Top-5%  — flat top-5% compression, no routing intelligence
  5. DivRoute-FL     — three-tier adaptive divergence-routed compression

Features:
  - Resumable: skips any run whose log file already exists.
  - Produces results/phase5_comparison_results.csv and a terminal summary.
  - Acc@XMB comparison table at budgets 500 MB, 1 GB, 2 GB, 5 GB.

Usage:
    python run_phase5_comparison.py                     # all methods, all seeds
    python run_phase5_comparison.py --methods fedavg divroute   # subset of methods
    python run_phase5_comparison.py --seeds 42          # single seed
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from divroute_fl.config import Config, get_recommended_divroute_config, get_uniform_top5_config
from divroute_fl.main import run
from baselines.fedzip_baseline import get_fedzip_config, fedzip_bytes_for_delta
from baselines.fedsparse_baseline import get_fedsparse_config

# ── Experiment parameters ──────────────────────────────────────────────────────
SEEDS             = [42, 123, 456]
NUM_ROUNDS        = 100
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 10        # 10% participation — realistic FL
DATASET           = "cifar100"
MODEL             = "resnet18"

ALL_METHODS = ["fedavg", "fedzip", "fedsparse", "uniform", "divroute"]

# FedZip byte-accounting parameters (from the FedZip paper defaults)
FEDZIP_Z_RATIO    = 0.01
FEDZIP_K_CLUSTERS = 3

# FedSparse lambda — must be scaled to model size.
# SimpleCNN (200K params): 0.01 works.  ResNet-18 (11M params): 0.0001.
# At 0.01 on ResNet-18 the L1 penalty overwhelms cross-entropy → accuracy ≈ random.
FEDSPARSE_LAMBDA  = 0.0001

# Acc@MB budget checkpoints (in bytes) — tuned for ResNet-18 scale
ACC_AT_BUDGETS_MB = [500, 1000, 2000, 5000]

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/phase5")
RESULTS_DIR = Path("results")
CSV_PATH    = RESULTS_DIR / "phase5_comparison_results.csv"

CSV_COLUMNS = [
    "method", "seed",
    "final_acc", "download_mb", "upload_mb", "bidir_mb",
    "bidir_saving_pct", "runtime_sec",
] + [f"acc_at_{b}mb" for b in ACC_AT_BUDGETS_MB]


# ── Config factories ───────────────────────────────────────────────────────────
def _shared(seed: int, log_path: str) -> dict:
    return dict(
        num_clients       = NUM_CLIENTS,
        clients_per_round = CLIENTS_PER_ROUND,
        num_rounds        = NUM_ROUNDS,
        seed              = seed,
        dataset_name      = DATASET,
        model_name        = MODEL,
        skip_plot_prompt  = True,
        log_path          = log_path,
    )


def _make_config(method: str, seed: int, log_path: str) -> Config:
    s = _shared(seed, log_path)
    if method == "fedavg":
        return Config(fedavg_baseline_mode=True, **s)
    elif method == "fedzip":
        # FedZip runs full-delta training; bytes are recalculated post-hoc.
        return Config(fedavg_baseline_mode=True, **s)
    elif method == "fedsparse":
        return get_fedsparse_config(fedsparse_lambda=FEDSPARSE_LAMBDA, **s)
    elif method == "uniform":
        return get_uniform_top5_config(**s)
    elif method == "divroute":
        return get_recommended_divroute_config(**s)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _log_path(method: str, seed: int) -> Path:
    return LOGS_DIR / f"{method}_seed{seed}.json"


# ── Log parsing ────────────────────────────────────────────────────────────────
def _parse_log(path: Path, method: str) -> dict:
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc = history[-1]["test_accuracy"]
    num_rounds = len(history)

    # Download bytes ── prefer the explicit field, fall back to legacy field
    total_dl = sum(
        e.get("total_download_bytes", e.get("total_bytes_transmitted", 0))
        for e in history
    )

    # Upload bytes ── prefer explicit field, fall back to per-client sum
    total_ul = 0
    for e in history:
        if "total_upload_bytes" in e:
            total_ul += e["total_upload_bytes"]
        elif "clients" in e:
            total_ul += sum(c.get("upload_bytes", 0) for c in e["clients"])

    # ── FedZip post-hoc byte recalculation ───────────────────────────────────
    # FedZip's download bytes are *much* lower than the raw full-delta that was
    # actually transmitted during training.  We recalculate them using the
    # FedZip formula applied to the delta sizes recorded in the log.
    if method == "fedzip":
        fedzip_dl = 0
        for e in history:
            delta_numel = e.get("delta_numel", None)
            if delta_numel is not None:
                # Build a dummy tensor of the right size to run the formula
                dummy = torch.zeros(delta_numel)
                per_client = fedzip_bytes_for_delta(
                    dummy, z_ratio=FEDZIP_Z_RATIO, k_clusters=FEDZIP_K_CLUSTERS
                )
                n_clients = e.get("num_selected_clients", CLIENTS_PER_ROUND)
                fedzip_dl += per_client * n_clients
            else:
                # Fallback: apply z_ratio to raw download bytes
                fedzip_dl += int(total_dl / num_rounds * FEDZIP_Z_RATIO)
        total_dl = fedzip_dl
    # ─────────────────────────────────────────────────────────────────────────

    total_bidir = total_dl + total_ul

    # FedAvg reference (full float32 delta, both directions, every round)
    delta_numel = history[0].get("delta_numel", 0)
    fedavg_ref  = CLIENTS_PER_ROUND * delta_numel * 4 * 2 * num_rounds
    saving_pct  = (
        100.0 * (1.0 - total_bidir / fedavg_ref)
        if fedavg_ref else 0.0
    )

    # ── Acc @ fixed budget ────────────────────────────────────────────────────
    acc_at = {}
    for budget_mb in ACC_AT_BUDGETS_MB:
        budget_bytes = budget_mb * 1e6
        cum_dl = 0
        acc_found = None
        for e in history:
            dl = e.get("total_download_bytes", e.get("total_bytes_transmitted", 0))
            if method == "fedzip" and "delta_numel" in e:
                dummy = torch.zeros(e["delta_numel"])
                n = e.get("num_selected_clients", CLIENTS_PER_ROUND)
                dl = fedzip_bytes_for_delta(dummy, FEDZIP_Z_RATIO, FEDZIP_K_CLUSTERS) * n
            cum_dl += dl
            if cum_dl >= budget_bytes:
                acc_found = e["test_accuracy"]
                break
        acc_at[budget_mb] = acc_found if acc_found is not None else final_acc

    return {
        "final_acc":       final_acc,
        "download_mb":     total_dl    / 1e6,
        "upload_mb":       total_ul    / 1e6,
        "bidir_mb":        total_bidir / 1e6,
        "bidir_saving_pct": saving_pct,
        "acc_at":          acc_at,
    }


# ── Summary helpers ────────────────────────────────────────────────────────────
def _mean(xs): return statistics.mean(xs)  if xs else float("nan")
def _std(xs):  return statistics.stdev(xs) if len(xs) > 1 else 0.0

METHOD_LABELS = {
    "fedavg":    "Full FedAvg",
    "fedzip":    "FedZip (z=1%)",
    "fedsparse": f"FedSparse (λ={FEDSPARSE_LAMBDA})",
    "uniform":   "Uniform Top-5%",
    "divroute":  "DivRoute-FL",
}


def _print_summary(rows: list[dict], methods: list[str]) -> None:
    print(f"\n{'='*76}")
    print("  PHASE 5 COMPARISON SUMMARY")
    print(f"{'='*76}")
    print(f"\n  Dataset : {DATASET.upper()}   Model : {MODEL.upper()}")
    print(f"  Clients : {NUM_CLIENTS} total, {CLIENTS_PER_ROUND}/round  "
          f"Rounds : {NUM_ROUNDS}   Seeds : {SEEDS}")

    # Main table
    print(f"\n  {'Method':<22} {'Acc%':>8} {'±':>4} {'Bidir MB':>10} {'Saving%':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*4} {'-'*10} {'-'*9}")
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        accs   = [r["final_acc"]        for r in subset]
        bidirs = [r["bidir_mb"]         for r in subset]
        saves  = [r["bidir_saving_pct"] for r in subset]
        label  = METHOD_LABELS.get(m, m)
        print(f"  {label:<22} {_mean(accs)*100:>7.2f}% {_std(accs)*100:>3.2f}%"
              f" {_mean(bidirs):>10.1f} {_mean(saves):>8.1f}%")

    # Acc@MB table
    print(f"\n  Acc @ fixed download budget (mean over seeds):")
    header = f"  {'Method':<22}" + "".join(f" {b}MB:>8" for b in ACC_AT_BUDGETS_MB)
    # Build header row
    hdr = f"  {'Method':<22}"
    for b in ACC_AT_BUDGETS_MB:
        hdr += f"  {str(b)+'MB':>8}"
    print(hdr)
    print(f"  {'-'*22}" + f"  {'-'*8}" * len(ACC_AT_BUDGETS_MB))
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        label  = METHOD_LABELS.get(m, m)
        row = f"  {label:<22}"
        for b in ACC_AT_BUDGETS_MB:
            vals = [r["acc_at"][b] for r in subset if b in r["acc_at"]]
            row += f"  {_mean(vals)*100:>7.2f}%" if vals else f"  {'N/A':>8}"
        print(row)

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 baseline comparisons")
    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS,
        help="Which methods to run (default: all five)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to use (default: 42 123 456)"
    )
    args = parser.parse_args()

    methods = args.methods
    seeds   = args.seeds

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    total_runs = len(methods) * len(seeds)
    pending    = []
    skipped    = 0

    for method in methods:
        for seed in seeds:
            lp = _log_path(method, seed)
            if lp.exists():
                print(f"[skip] {METHOD_LABELS.get(method, method)} seed={seed} — log exists")
                skipped += 1
            else:
                pending.append((method, seed))

    print(f"\nTotal runs : {total_runs}  |  "
          f"Skipped : {skipped}  |  To run : {len(pending)}")
    if pending:
        # Rough time estimate: ResNet-18 on CIFAR-100, ~90s per round on a mid GPU
        est_s = len(pending) * NUM_ROUNDS * 90
        h, rem = divmod(int(est_s), 3600)
        m = rem // 60
        print(f"Estimated remaining time: ~{h}h {m}m  (assumes ~90s/round on mid-range GPU)\n")

    run_num = skipped
    for method, seed in pending:
        run_num += 1
        lp  = _log_path(method, seed)
        cfg = _make_config(method, seed, str(lp))
        label = METHOD_LABELS.get(method, method)

        print(f"\n{'='*76}")
        print(f"  Run {run_num}/{total_runs}: {label} | seed={seed} | "
              f"{NUM_ROUNDS} rounds | {DATASET.upper()} / {MODEL.upper()}")
        print(f"{'='*76}\n")

        t0 = time.perf_counter()
        run(cfg)
        elapsed = time.perf_counter() - t0
        print(f"\n  done in {elapsed:.0f}s")

    # ── Collect all results ────────────────────────────────────────────────────
    rows: list[dict] = []
    for method in methods:
        for seed in seeds:
            lp = _log_path(method, seed)
            if not lp.exists():
                print(f"[warn] Missing log: {lp} — skipping")
                continue
            metrics = _parse_log(lp, method)
            rows.append({
                "method": method,
                "seed":   seed,
                **metrics,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "method":            r["method"],
                "seed":              r["seed"],
                "final_acc":         f"{r['final_acc']:.4f}",
                "download_mb":       f"{r['download_mb']:.2f}",
                "upload_mb":         f"{r['upload_mb']:.2f}",
                "bidir_mb":          f"{r['bidir_mb']:.2f}",
                "bidir_saving_pct":  f"{r['bidir_saving_pct']:.1f}",
                "runtime_sec":       "",
                **{f"acc_at_{b}mb": f"{r['acc_at'].get(b, float('nan')):.4f}"
                   for b in ACC_AT_BUDGETS_MB},
            })

    print(f"\n[done] CSV written to {CSV_PATH}")
    _print_summary(rows, methods)


if __name__ == "__main__":
    main()
