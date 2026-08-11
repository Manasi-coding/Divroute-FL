"""
run_ablation_a_fedavg.py
========================
Experiment A: FedAvg-equivalent sanity baseline.

Reuses fedavg_baseline_mode=True (the project's validated FedAvg path).
Purpose: verify the underlying FL pipeline achieves healthy CIFAR-100 accuracy.

Expected outcomes
-----------------
  Accuracy >> 35%  ->  failure is definitively inside DivRoute routing/compression.
  Accuracy ~= 35%  ->  investigate local training pipeline or aggregation.

Checkpoint : checkpoints/ablation_a_fedavg_cifar100_seed42/
Log        : logs/ablation_a_fedavg.json
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config
from divroute_fl.main import run

cfg = Config(
    # Checkpoint isolation via run_label (no algorithmic effect).
    run_label              = "ablation_a_fedavg",
    # Existing validated FedAvg path: disables all DivRoute features internally.
    fedavg_baseline_mode   = True,
    use_epoch_warmup       = False,   # match the project's existing FedAvg runner
    dataset_name           = "cifar100",
    model_name             = "resnet18",
    num_clients            = 100,
    clients_per_round      = 20,
    num_rounds             = 100,
    local_epochs           = 5,
    local_lr               = 0.1,
    batch_size             = 32,
    alpha                  = 0.9,
    seed                   = 42,
    # Diagnostics (read-only output; no algorithmic effect)
    enable_routing_diagnostics      = True,
    enable_routing_quality_analysis = True,
    enable_tau_diagnostics          = True,
    enable_compression_analysis     = False,
    enable_gradient_diagnostics     = False,
    diag_print_interval             = 10,
    log_path               = "logs/ablation_a_fedavg.json",
    skip_plot_prompt       = True,
    fresh                  = True,
    resume                 = False,
)

print("=" * 72)
print("  ABLATION A -- FedAvg-equivalent sanity baseline")
print("=" * 72)
print(f"  dataset      : {cfg.dataset_name.upper()} / {cfg.model_name}")
print(f"  clients      : {cfg.num_clients}  |  per round: {cfg.clients_per_round}")
print(f"  rounds       : {cfg.num_rounds}")
print(f"  local_epochs : {cfg.local_epochs}  |  lr: {cfg.local_lr}  |  bs: {cfg.batch_size}")
print(f"  alpha        : {cfg.alpha}  |  seed: {cfg.seed}")
print(f"  fedavg_mode  : {cfg.fedavg_baseline_mode}  (ALL DivRoute features OFF)")
print(f"  checkpoint   : checkpoints/{cfg.run_label}_{cfg.dataset_name}_seed{cfg.seed}/")
print(f"  log          : {cfg.log_path}")
print()
run(cfg)
