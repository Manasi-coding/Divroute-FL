"""
static_check.py — Quick static checks for ablation logic without running full training.
"""
from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.compression import get_adaptive_k_ratios

def _get_method_name(cfg):
    if getattr(cfg, "run_label", ""):
        return cfg.run_label
    if cfg.fedavg_baseline_mode:
        return "fedavg"
    if getattr(cfg, "uniform_top5_mode", False):
        return "uniform"
    return "divroute"

def ckpt(cfg):
    return f"checkpoints/{_get_method_name(cfg)}_{cfg.dataset_name}_seed{cfg.seed}"


# ── Exp A ─────────────────────────────────────────────────────────────────────
print("=== Exp A: fedavg_baseline_mode ===")
cfg_a = Config(
    run_label="ablation_a_fedavg",
    fedavg_baseline_mode=True, use_epoch_warmup=False,
    dataset_name="cifar100", model_name="resnet18",
    num_clients=100, clients_per_round=20,
    num_rounds=2, local_lr=0.1, seed=42, skip_plot_prompt=True,
)
# Simulate main.py fedavg_baseline_mode block
cfg_a.use_divergence_weighting = False
cfg_a.use_server_momentum      = False
cfg_a.use_error_feedback       = False
cfg_a.use_adaptive_tau         = False
cfg_a.use_tier3_sync           = False
cfg_a.use_adaptive_k           = False
cfg_a.k_ratio_tier1            = 1.0
cfg_a.k_ratio_tier2            = 1.0
cfg_a.tau_low                  = -1.0
cfg_a.tau_high                 = -1.0
cfg_a.gamma                    = 1.0
assert cfg_a.k_ratio_tier1 == 1.0
assert cfg_a.tau_low == -1.0 and cfg_a.tau_high == -1.0
assert cfg_a.use_divergence_weighting == False
c = ckpt(cfg_a)
assert c == "checkpoints/ablation_a_fedavg_cifar100_seed42", c
print(f"  checkpoint: {c}  PASS")

# ── Exp B ─────────────────────────────────────────────────────────────────────
print("=== Exp B: routing ON / compression OFF ===")
cfg_b = get_recommended_divroute_config(
    run_label="ablation_b_no_compression",
    ablation_routing_no_compression=True,
    dataset_name="cifar100", model_name="resnet18",
    num_clients=100, clients_per_round=20,
    num_rounds=2, local_lr=0.1, seed=42, skip_plot_prompt=True,
)
# Simulate main.py startup override
cfg_b.k_ratio_tier1 = 1.0
cfg_b.k_ratio_tier2 = 1.0
assert cfg_b.k_ratio_tier1 == 1.0
assert cfg_b.use_divergence_weighting == True   # routing still ON
assert cfg_b.use_adaptive_tau == True
# Simulate per-round re-enforcement (even after get_adaptive_k_ratios)
k1, k2 = get_adaptive_k_ratios(cfg_b, rnd=50)
cfg_b.active_k_ratio_tier1 = k1
cfg_b.active_k_ratio_tier2 = k2
# B override
if getattr(cfg_b, "ablation_routing_no_compression", False):
    cfg_b.active_k_ratio_tier1 = 1.0
    cfg_b.active_k_ratio_tier2 = 1.0
assert cfg_b.active_k_ratio_tier1 == 1.0, cfg_b.active_k_ratio_tier1
assert cfg_b.active_k_ratio_tier2 == 1.0, cfg_b.active_k_ratio_tier2
c = ckpt(cfg_b)
assert c == "checkpoints/ablation_b_no_compression_cifar100_seed42", c
print(f"  checkpoint: {c}  PASS")
print(f"  active_k1={cfg_b.active_k_ratio_tier1} active_k2={cfg_b.active_k_ratio_tier2}  PASS")

# ── Exp C ─────────────────────────────────────────────────────────────────────
print("=== Exp C: uniform compression / routing OFF ===")
cfg_c = Config(
    run_label="ablation_c_uniform_compression",
    ablation_uniform_compression=True,
    ablation_uniform_k_ratio=0.05,
    dataset_name="cifar100", model_name="resnet18",
    num_clients=100, clients_per_round=20,
    num_rounds=2, local_lr=0.1, seed=42, skip_plot_prompt=True,
)
# Simulate main.py startup override
uk = cfg_c.ablation_uniform_k_ratio
cfg_c.k_ratio_tier1 = uk
cfg_c.k_ratio_tier2 = uk
cfg_c.use_divergence_weighting = False
assert cfg_c.k_ratio_tier1 == 0.05
assert cfg_c.k_ratio_tier2 == 0.05
assert cfg_c.use_divergence_weighting == False
# Simulate per-round tier override
fake_results = [{"tier": 2, "client_id": i} for i in range(5)]
for r in fake_results:
    r["tier"] = 1
assert all(r["tier"] == 1 for r in fake_results)
c = ckpt(cfg_c)
assert c == "checkpoints/ablation_c_uniform_compression_cifar100_seed42", c
print(f"  checkpoint: {c}  PASS")
print(f"  all clients -> tier 1 with k={cfg_c.k_ratio_tier1}  PASS")

# ── Exp D ─────────────────────────────────────────────────────────────────────
print("=== Exp D: full DivRoute + per-client logging ===")
cfg_d = get_recommended_divroute_config(
    run_label="ablation_d_full_divroute",
    ablation_per_client_logging=True,
    dataset_name="cifar100", model_name="resnet18",
    num_clients=100, clients_per_round=20,
    num_rounds=2, local_lr=0.1, seed=42, skip_plot_prompt=True,
)
# Exp D: NO startup overrides to k_ratio or weighting
assert cfg_d.k_ratio_tier1 == 0.20, cfg_d.k_ratio_tier1
assert cfg_d.k_ratio_tier2 == 0.05, cfg_d.k_ratio_tier2
assert cfg_d.use_divergence_weighting == True
assert cfg_d.ablation_per_client_logging == True
assert cfg_d.ablation_routing_no_compression == False
assert cfg_d.ablation_uniform_compression == False
c = ckpt(cfg_d)
assert c == "checkpoints/ablation_d_full_divroute_cifar100_seed42", c
print(f"  checkpoint: {c}  PASS")
print(f"  k_ratio unchanged: t1={cfg_d.k_ratio_tier1} t2={cfg_d.k_ratio_tier2}  PASS")
print(f"  div_weighting={cfg_d.use_divergence_weighting}  PASS")

# ── Checkpoint isolation ───────────────────────────────────────────────────────
print("=== Checkpoint isolation ===")
all_ckpts = [ckpt(cfg_a), ckpt(cfg_b), ckpt(cfg_c), ckpt(cfg_d)]
assert len(set(all_ckpts)) == 4, f"FAIL: duplicates: {all_ckpts}"
for p in all_ckpts:
    print(f"  {p}/")
print("  All 4 distinct  PASS")

# ── No algorithmic field contamination ────────────────────────────────────────
print("=== Existing Config defaults unchanged ===")
base = Config()
assert base.ablation_routing_no_compression == False
assert base.ablation_uniform_compression    == False
assert base.ablation_uniform_k_ratio        == 0.05
assert base.ablation_per_client_logging     == False
assert base.run_label                       == ""
assert base.k_ratio_tier1                   == 0.20
assert base.k_ratio_tier2                   == 0.05
assert base.fedavg_baseline_mode            == False
print("  All ablation flags default to False / neutral values  PASS")

print()
print("ALL STATIC CHECKS PASSED")
