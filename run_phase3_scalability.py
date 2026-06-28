"""
run_phase3_scalability.py
=========================
Phase 3 publication-quality scalability experiment for DivRoute-FL.

Evaluates DivRoute and FedAvg on:
  - Dataset   : CIFAR-100
  - Model     : ResNet-18
  - Clients   : 75 total, 10 selected per round
  - Rounds    : 100
  - Local SGD : 2 epochs per round
  - Alpha     : 0.1  (extreme non-IID)
  - Seeds     : [1, 42, 84]

Logs are written to logs/phase3/:
  logs/phase3/fedavg_seed{seed}.json
  logs/phase3/divroute_seed{seed}.json

Usage:
    python run_phase3_scalability.py
"""

import os
import time
from pathlib import Path

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

# ── Experiment configuration ──────────────────────────────────────────────────
SEEDS   = [1, 42, 84]
LOGS_DIR = Path("logs/phase3")

# Shared Phase 3 hyperparameters
PHASE3_BASE = dict(
    dataset_name       = "cifar100",
    model_name         = "resnet18",
    num_clients        = 75,
    clients_per_round  = 10,
    num_rounds         = 100,
    local_epochs       = 2,
    alpha              = 0.1,
    skip_plot_prompt   = True,
)


# ── Runners ───────────────────────────────────────────────────────────────────
def _run_fedavg(seed: int) -> str:
    log_path = str(LOGS_DIR / f"fedavg_seed{seed}.json")
    run(Config(
        **PHASE3_BASE,
        fedavg_baseline_mode = True,
        use_epoch_warmup     = False,
        seed                 = seed,
        log_path             = log_path,
    ))
    return log_path


def _run_divroute(seed: int) -> str:
    log_path = str(LOGS_DIR / f"divroute_seed{seed}.json")
    run(get_recommended_divroute_config(
        **PHASE3_BASE,
        seed     = seed,
        log_path = log_path,
    ))
    return log_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    start_time = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(SEEDS) * 2
    run_no = 0

    for seed in SEEDS:
        run_no += 1
        print(f"\n{'='*72}")
        print(f"  [{run_no}/{total_runs}]  SEED {seed}  -  FedAvg baseline  "
              f"(CIFAR-100 / ResNet-18 / 75 clients / 100 rounds)")
        print(f"{'='*72}\n")
        fedavg_log = _run_fedavg(seed)
        print(f"\n  [phase3] FedAvg log -> {fedavg_log}")

        run_no += 1
        print(f"\n{'='*72}")
        print(f"  [{run_no}/{total_runs}]  SEED {seed}  -  DivRoute  "
              f"(CIFAR-100 / ResNet-18 / 75 clients / 100 rounds)")
        print(f"{'='*72}\n")
        divroute_log = _run_divroute(seed)
        print(f"\n  [phase3] DivRoute log -> {divroute_log}")

    print(f"\n\n{'='*72}")
    print("  PHASE 3 SCALABILITY EXPERIMENT COMPLETE")
    print(f"{'='*72}")
    print(f"\n  Logs saved to: {LOGS_DIR}/")
    print(f"  Run  python analyze_phase3.py  to generate the summary CSV.\n")

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nTotal runtime: {elapsed_hours:.2f} hours")


if __name__ == "__main__":
    main()
