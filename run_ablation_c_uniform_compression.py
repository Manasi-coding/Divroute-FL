"""
run_ablation_c_uniform_compression.py
======================================
Experiment C: uniform compression (k=0.05) for every client; routing OFF.

Divergence scores and EMA are still computed each round (for diagnostic
logging) but have NO effect on tier assignment or aggregation weighting.
Every selected client is routed through Tier-1 with k=0.05, giving exactly
the same communication budget as DivRoute's Tier-2 clients.

Purpose: determine whether it is the adaptive routing DECISION (not
compression itself) that hurts accuracy.

Expected outcomes
-----------------
  Accuracy >> DivRoute's 35%  ->  adaptive routing misroutes clients.
  Accuracy ~= DivRoute's 35%  ->  compression at k=0.05 is independently harmful.

Checkpoint : checkpoints/ablation_c_uniform_compression_cifar100_seed42/
Log        : logs/ablation_c_uniform_compression.json
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config
from divroute_fl.main import run

cfg = Config(
    # Checkpoint isolation via run_label (no algorithmic effect).
    run_label              = "ablation_c_uniform_compression",
    # Experiment C flags: routing OFF, uniform k=0.05 for all clients.
    ablation_uniform_compression = True,
    ablation_uniform_k_ratio     = 0.05,
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
    # Diagnostic flags
    enable_routing_diagnostics      = True,
    enable_routing_quality_analysis = True,
    enable_tau_diagnostics          = True,
    enable_compression_analysis     = True,
    enable_gradient_diagnostics     = False,
    diag_print_interval             = 10,
    # Keep adaptive tau active so the diagnostic output is comparable with
    # the cosine/L2 runs, even though the tau values don't affect routing.
    use_adaptive_tau       = True,
    tau_alpha              = 0.5,
    tau_beta               = 1.0,
    log_path               = "logs/ablation_c_uniform_compression.json",
    skip_plot_prompt       = True,
    fresh                  = True,
    resume                 = False,
)

print("=" * 72)
print("  ABLATION C -- Uniform compression k=0.05 / routing OFF")
print("=" * 72)
print(f"  dataset      : {cfg.dataset_name.upper()} / {cfg.model_name}")
print(f"  clients      : {cfg.num_clients}  |  per round: {cfg.clients_per_round}")
print(f"  rounds       : {cfg.num_rounds}")
print(f"  local_epochs : {cfg.local_epochs}  |  lr: {cfg.local_lr}  |  bs: {cfg.batch_size}")
print(f"  alpha        : {cfg.alpha}  |  seed: {cfg.seed}")
print(f"  uniform_k    : {cfg.ablation_uniform_k_ratio}")
print(f"  div_weight   : OFF (forced by ablation_uniform_compression)")
print(f"  ablation_C   : {cfg.ablation_uniform_compression}")
print(f"  checkpoint   : checkpoints/{cfg.run_label}_{cfg.dataset_name}_seed{cfg.seed}/")
print(f"  log          : {cfg.log_path}")
print()
run(cfg)
