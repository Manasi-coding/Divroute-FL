"""
write_ablation_runners.py
=========================
One-shot helper script: writes the four ablation runner files and then
deletes itself.  Run with:

    python write_ablation_runners.py
"""

import os

# ── Shared CIFAR-100 / ResNet-18 base settings ───────────────────────────────
SHARED = """
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
    skip_plot_prompt       = True,
    fresh                  = True,
    resume                 = False,
"""

# ── Runner A: FedAvg-equivalent sanity baseline ───────────────────────────────
runner_a = '''\
"""
run_ablation_a_fedavg.py
========================
Experiment A: FedAvg-equivalent sanity baseline.

Reuses fedavg_baseline_mode=True (the project\'s validated FedAvg path).
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
    use_epoch_warmup       = False,   # match the project\'s existing FedAvg runner
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
'''

# ── Runner B: DivRoute routing ON, compression OFF ───────────────────────────
runner_b = '''\
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
'''

# ── Runner C: Uniform compression, routing OFF ───────────────────────────────
runner_c = '''\
"""
run_ablation_c_uniform_compression.py
======================================
Experiment C: uniform compression (k=0.05) for every client; routing OFF.

Divergence scores and EMA are still computed each round (for diagnostic
logging) but have NO effect on tier assignment or aggregation weighting.
Every selected client is routed through Tier-1 with k=0.05, giving exactly
the same communication budget as DivRoute\'s Tier-2 clients.

Purpose: determine whether it is the adaptive routing DECISION (not
compression itself) that hurts accuracy.

Expected outcomes
-----------------
  Accuracy >> DivRoute\'s 35%  ->  adaptive routing misroutes clients.
  Accuracy ~= DivRoute\'s 35%  ->  compression at k=0.05 is independently harmful.

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
    # the cosine/L2 runs, even though the tau values don\'t affect routing.
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
'''

# ── Runner D: Full DivRoute + per-client forensic logging ────────────────────
runner_d = '''\
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
'''

files = {
    "run_ablation_a_fedavg.py":                runner_a,
    "run_ablation_b_routing_no_compression.py": runner_b,
    "run_ablation_c_uniform_compression.py":    runner_c,
    "run_ablation_d_per_client_logging.py":     runner_d,
}

for fname, content in files.items():
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"Written: {fname}")

print("All four runners written successfully.")
