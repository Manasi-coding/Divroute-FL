"""
run_phase4_ablations.py
========================
Phase 4 Ablation Studies: thresholds, k-ratios, gamma decay, local epochs.

Runs parameter sweeps over different configurations.
Each run uses identical dataset partitions and client sampling sequences (guaranteed by shared seed).

Features:
  - Selectively run specific ablations via command line arguments.
  - Resumable: skips any run whose log file already exists.
  - Produces CSVs in `results/` and a terminal summary with means and standard deviations.

Usage:
    python run_phase4_ablations.py --ablation thresholds
    python run_phase4_ablations.py --ablation kratios
    python run_phase4_ablations.py --ablation gamma
    python run_phase4_ablations.py --ablation epochs
    python run_phase4_ablations.py --ablation all
"""

import argparse
import csv
import json
import statistics
import sys
import time
from itertools import product
from pathlib import Path

from divroute_fl.config import get_recommended_divroute_config
from divroute_fl.main import run

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Experiment parameters ──────────────────────────────────────────────────────
SEEDS      = [42]
NUM_ROUNDS = 50
NUM_CLIENTS       = 25
CLIENTS_PER_ROUND = 15

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR    = Path("logs/phase4")
RESULTS_DIR = Path("results")

def get_base_config(seed, log_path):
    return get_recommended_divroute_config(
        num_clients=NUM_CLIENTS,
        clients_per_round=CLIENTS_PER_ROUND,
        num_rounds=NUM_ROUNDS,
        seed=seed,
        skip_plot_prompt=True,
        log_path=str(log_path)
    )

# Define the ablations to run
ABLATIONS = {
    "thresholds": {
        "params": ["tau_high", "tau_low"],
        "values": [
            (0.01, 0.005),
            (0.015, 0.0075),
            (0.02, 0.01),
            (0.03, 0.015),
            (0.05, 0.025)
        ],
        "config_kwargs": lambda v: {"use_adaptive_tau": False, "tau_high": v[0], "tau_low": v[1]},
        "name_func": lambda v: f"tau_high_{v[0]}"
    },
    # the "tau low & high values" used in the kratios run are the best ones obtained from run of thresholds ablation
    "kratios": {
        "params": ["k_ratio_tier1", "k_ratio_tier2"],
        "values": list(product([0.10, 0.20, 0.30], [0.02, 0.05, 0.10])), 
        "config_kwargs": lambda v: {"k_ratio_tier1": v[0], "k_ratio_tier2": v[1], "use_adaptive_tau": False, "tau_high": 0.03, "tau_low": 0.015},
        "name_func": lambda v: f"k1_{v[0]:.2f}_k2_{v[1]:.2f}"
    },
    "gamma": {
        "params": ["gamma"],
        "values": [(g,) for g in [0.70, 0.80, 0.85, 0.90, 1.00]],
        "config_kwargs": lambda v: {"gamma": v[0], "use_adaptive_tau": False, "tau_high": 0.03, "tau_low": 0.015},
        "name_func": lambda v: f"gamma_{v[0]:.2f}"
    },
    "epochs": {
        "params": ["local_epochs"],
        "values": [(e,) for e in [1, 2, 5, 10]],
        "config_kwargs": lambda v: {"local_epochs": v[0], "use_epoch_warmup": False, "use_adaptive_tau": False, "tau_high": 0.03, "tau_low": 0.015},
        "name_func": lambda v: f"epochs_{v[0]}"
    }
}

def _parse_log(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc = history[-1]["test_accuracy"]
    
    # Safely extract bytes regardless of the logger version
    total_dl = sum(e.get("total_download_bytes", e.get("total_bytes_transmitted", 0)) for e in history)
    
    # Extract upload bytes
    total_ul = 0
    for e in history:
        if "total_upload_bytes" in e:
            total_ul += e["total_upload_bytes"]
        elif "clients" in e:
            total_ul += sum(c.get("upload_bytes", 0) for c in e["clients"])
            
    total_bidir = total_dl + total_ul

    return {
        "accuracy":    final_acc,
        "download_mb": total_dl    / 1e6,
        "upload_mb":   total_ul    / 1e6,
        "bidir_mb":    total_bidir / 1e6,
    }

def _mean(xs):  return statistics.mean(xs)  if xs else float("nan")
def _std(xs):   return statistics.stdev(xs) if len(xs) > 1 else 0.0

def print_summary(ablation_name, values, rows):
    print(f"\n{'='*70}")
    print(f"  PHASE 4 SUMMARY: {ablation_name.upper()}")
    print(f"{'='*70}")
    print(f"\n  Seeds   : {SEEDS}")
    print(f"  Rounds  : {NUM_ROUNDS}")
    
    print(f"\n  {'Config':<20} {'Acc mean':>9}  {'Acc std':>8}  {'Bidir MB':>10}")
    print(f"  {'-'*20} {'-'*9}  {'-'*8}  {'-'*10}")
    
    for val in values:
        config_name = ABLATIONS[ablation_name]["name_func"](val)
        subset = [r for r in rows if r["config"] == config_name]
        accs = [r["accuracy"] for r in subset]
        bidirs = [r["bidir_mb"] for r in subset]
        
        if accs:
            print(f"  {config_name:<20} {_mean(accs)*100:>8.2f}%  {_std(accs)*100:>7.2f}%  {_mean(bidirs):>10.1f}")
        else:
            print(f"  {config_name:<20} {'N/A':>9}  {'N/A':>8}  {'N/A':>10}")

def run_ablation(ablation_name):
    abl = ABLATIONS[ablation_name]
    logs_dir = LOGS_DIR / ablation_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    csv_path = RESULTS_DIR / f"phase4_{ablation_name}_results.csv"
    
    rows = []
    total_runs = len(abl["values"]) * len(SEEDS)
    
    print(f"\nChecking pending runs for ablation: {ablation_name} ({total_runs} total runs)")
    
    for val in abl["values"]:
        config_name = abl["name_func"](val)
        kwargs = abl["config_kwargs"](val)
        
        for seed in SEEDS:
            log_path = logs_dir / f"{config_name}_seed{seed}.json"
            
            if not log_path.exists():
                cfg = get_base_config(seed, log_path)
                for k, v in kwargs.items():
                    setattr(cfg, k, v)
                
                print(f"\n--- Running {ablation_name} | {config_name} | seed={seed} ---")
                t0 = time.perf_counter()
                run(cfg)
                elapsed = time.perf_counter() - t0
                print(f"  → done in {elapsed:.0f}s")
            else:
                print(f"[skip] {ablation_name} | {config_name} | seed={seed} — log exists")
                
            metrics = _parse_log(log_path)
            rows.append({
                "config": config_name,
                "seed": seed,
                **metrics
            })
            
    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "seed", "accuracy", "download_mb", "upload_mb", "bidir_mb"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "config": r["config"],
                "seed": r["seed"],
                "accuracy": f"{r['accuracy']:.4f}",
                "download_mb": f"{r['download_mb']:.2f}",
                "upload_mb": f"{r['upload_mb']:.2f}",
                "bidir_mb": f"{r['bidir_mb']:.2f}",
            })
            
    print(f"\n[done] CSV written to {csv_path}")
    print_summary(ablation_name, abl["values"], rows)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 Ablations")
    parser.add_argument("--ablation", choices=["thresholds", "kratios", "gamma", "epochs", "all"], 
                        required=True, help="Which ablation to run (or 'all' for all of them)")
    args = parser.parse_args()
    
    if args.ablation == "all":
        for a in ABLATIONS.keys():
            run_ablation(a)
    else:
        run_ablation(args.ablation)

if __name__ == "__main__":
    main()
