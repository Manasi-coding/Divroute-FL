"""
run_ablation_b_routing_no_compression.py
========================================
Experiment B: DivRoute routing + divergence weighting ON; compression OFF.

All tier assignments, EMA updates, adaptive-tau thresholds, and divergence
weights run exactly as in a normal DivRoute run.  The only change: k_ratio
is forced to 1.0 for every tier so no parameter coordinates are discarded.

Purpose: isolate whether routing/weighting ALONE causes the accuracy drop,
independently of top-k compression.

Expected outcomes
-----------------
  Accuracy recovers toward Exp A  ->  compression is the primary culprit.
  Accuracy stays near 35%          ->  tier weighting/aggregation is suspect.

Checkpoint : checkpoints/ablation_b_no_compression_cifar100_seed42/
Log        : logs/ablation_b_no_compression.json
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

cfg = get_recommended_divroute_config(
    # Checkpoint isolation via run_label (no algorithmic effect).
    run_label              = "ablation_b_no_compression",
    # Experiment B flag: forces k=1.0 for all tiers at startup and every round.
    ablation_routing_no_compression = True,
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
    # Diagnostics
    enable_routing_diagnostics      = True,
    enable_routing_quality_analysis = True,
    enable_tau_diagnostics          = True,
    enable_compression_analysis     = True,
    enable_gradient_diagnostics     = False,
    diag_print_interval             = 10,
    log_path               = "logs/ablation_b_no_compression.json",
    skip_plot_prompt       = True,
    fresh                  = True,
    resume                 = False,
)

print("=" * 72)
print("  ABLATION B -- DivRoute routing ON / compression OFF (k=1.0)")
print("=" * 72)
print(f"  dataset      : {cfg.dataset_name.upper()} / {cfg.model_name}")
print(f"  clients      : {cfg.num_clients}  |  per round: {cfg.clients_per_round}")
print(f"  rounds       : {cfg.num_rounds}")
print(f"  local_epochs : {cfg.local_epochs}  |  lr: {cfg.local_lr}  |  bs: {cfg.batch_size}")
print(f"  alpha        : {cfg.alpha}  |  seed: {cfg.seed}")
print(f"  div_weight   : {cfg.use_divergence_weighting}  ({cfg.divergence_weight_mode})")
print(f"  adaptive_tau : {cfg.use_adaptive_tau}")
print(f"  k_ratio_t1/t2: {cfg.k_ratio_tier1} / {cfg.k_ratio_tier2}  (will be forced to 1.0)")
print(f"  ablation_B   : {cfg.ablation_routing_no_compression}")
print(f"  checkpoint   : checkpoints/{cfg.run_label}_{cfg.dataset_name}_seed{cfg.seed}/")
print(f"  log          : {cfg.log_path}")
print()
run(cfg)
