"""
run_ablation_d_per_client_logging.py
=====================================
Experiment D: full DivRoute (unchanged) + dense per-client forensic logging.

NO algorithmic change.  All routing, compression, divergence weighting, EMA,
and adaptive-tau logic run exactly as in the divergence-metric experiments.
The only addition: ablation_per_client_logging=True prints a detailed table
every round with client_id, d_raw, d_ema, tier, k_ratio, selection weight,
aggregation weight, local_loss, and update norm (where available).

Purpose: reveal which specific clients are being aggressively compressed
and whether high-loss / useful clients are getting Tier-2 / Tier-3 routing.

Checkpoint : checkpoints/ablation_d_full_divroute_cifar100_seed42/
Log        : logs/ablation_d_per_client_logging.json
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

cfg = get_recommended_divroute_config(
    # Checkpoint isolation via run_label (no algorithmic effect).
    run_label              = "ablation_d_full_divroute",
    # Experiment D: only adds the forensic logging table; no algorithm change.
    ablation_per_client_logging = True,
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
    # Enable all diagnostics for maximum forensic data
    enable_routing_diagnostics      = True,
    enable_routing_quality_analysis = True,
    enable_tau_diagnostics          = True,
    enable_compression_analysis     = True,
    enable_gradient_diagnostics     = True,   # enables update_norm in table
    diag_print_interval             = 10,
    log_path               = "logs/ablation_d_per_client_logging.json",
    skip_plot_prompt       = True,
    fresh                  = True,
    resume                 = False,
)

print("=" * 72)
print("  ABLATION D -- Full DivRoute + per-client forensic logging")
print("=" * 72)
print(f"  dataset      : {cfg.dataset_name.upper()} / {cfg.model_name}")
print(f"  clients      : {cfg.num_clients}  |  per round: {cfg.clients_per_round}")
print(f"  rounds       : {cfg.num_rounds}")
print(f"  local_epochs : {cfg.local_epochs}  |  lr: {cfg.local_lr}  |  bs: {cfg.batch_size}")
print(f"  alpha        : {cfg.alpha}  |  seed: {cfg.seed}")
print(f"  div_weight   : {cfg.use_divergence_weighting}  ({cfg.divergence_weight_mode})")
print(f"  k_ratio_t1/t2: {cfg.k_ratio_tier1} / {cfg.k_ratio_tier2}")
print(f"  adaptive_tau : {cfg.use_adaptive_tau}")
print(f"  per_client_log: {cfg.ablation_per_client_logging}  (diagnostic only)")
print(f"  checkpoint   : checkpoints/{cfg.run_label}_{cfg.dataset_name}_seed{cfg.seed}/")
print(f"  log          : {cfg.log_path}")
print()
run(cfg)
