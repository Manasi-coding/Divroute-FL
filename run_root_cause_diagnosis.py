"""
run_root_cause_diagnosis.py
===========================
Root-cause diagnostic suite — 4 experiments to isolate why DivRoute-FL
loses 16-20 pp relative to FedAvg, testing different divergence metrics.
"""

import json
import shutil
import os
from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.main import run

# ── Shared overrides for CIFAR-100 Phase-5 and Diagnostics ──────────────
SHARED_OVERRIDES = dict(
    dataset_name      = "cifar100",
    model_name        = "resnet18",
    num_clients       = 100,
    clients_per_round = 20,
    num_rounds        = 100,
    local_epochs      = 5,
    local_lr          = 0.1,
    batch_size        = 32,
    alpha             = 0.9,
    seed              = 42,
    
    skip_plot_prompt  = True,
    
    # ── Diagnostics ───────────────────────────────────────────────────────
    enable_gradient_diagnostics     = True,
    enable_routing_quality_analysis = True,
    enable_compression_analysis     = True,
    enable_tau_diagnostics          = True,
    
    # ── Checkpoint / State Isolation ──────────────────────────────────────
    resume = False,
    fresh = True,
)

# ── Experiments ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "metric": "cosine",
        "overrides": {"divergence_metric": "cosine"},
        "log_path": "logs/diag_cosine.json",
    },
    {
        "metric": "l2",
        "overrides": {"divergence_metric": "l2"},
        "log_path": "logs/diag_l2.json",
    },
    {
        "metric": "relative_l2",
        "overrides": {"divergence_metric": "relative_l2"},
        "log_path": "logs/diag_relative_l2.json",
    },
    {
        "metric": "layerwise_cosine",
        "overrides": {"divergence_metric": "layerwise_cosine"},
        "log_path": "logs/diag_layerwise_cosine.json",
    },
]


def _read_summary(log_path: str, clients_per_round: int) -> dict:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
        
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    if len(history) != 100:
        raise ValueError(f"Experiment finished with {len(history)} rounds instead of 100 in log {log_path}.")

    final_acc   = history[-1]["test_accuracy"]
    total_dl    = sum(e["total_download_bytes"] for e in history)
    total_ul    = sum(e["total_upload_bytes"]   for e in history)
    total_bidir = total_dl + total_ul

    delta_numel            = history[0]["delta_numel"]
    fedavg_bidir_per_round = clients_per_round * delta_numel * 4 * 2
    fedavg_total_bidir     = fedavg_bidir_per_round * len(history)

    bidir_saving = 100.0 * (1.0 - total_bidir / fedavg_total_bidir) if fedavg_total_bidir else 0.0

    return {
        "final_acc":         final_acc,
        "upload_mb":         total_ul / 1e6,
        "download_mb":       total_dl / 1e6,
        "bidir_mb":          total_bidir / 1e6,
        "bidir_saving_pct":  bidir_saving,
    }


def main():
    os.makedirs("logs", exist_ok=True)
    results = []
    
    # Track cosine accuracy for difference calculation
    cosine_acc = None

    for i, exp in enumerate(EXPERIMENTS, 1):
        metric = exp["metric"]
        log_path = exp["log_path"]
        print(f"\n{'='*72}")
        print(f"  DIAGNOSTIC {i}/4: Metric = {metric}")
        print(f"{'='*72}\n")
        
        print("Starting from scratch: no previous checkpoint will be used.")

        # Clean old log
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"  [Setup] Removed old log file: {log_path}")

        # Checkpoint isolation
        checkpoint_dir = Config().checkpoint_dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            print(f"  [Setup] Deleted existing checkpoints directory: {checkpoint_dir}")
            
        # Construct Config using the validated baseline
        cfg_dict = {**SHARED_OVERRIDES, **exp["overrides"], "log_path": log_path}
        cfg = get_recommended_divroute_config(**cfg_dict)
        
        print(f"  [Config] dataset           = {cfg.dataset_name}")
        print(f"  [Config] model             = {cfg.model_name}")
        print(f"  [Config] clients           = {cfg.num_clients}")
        print(f"  [Config] clients_per_round = {cfg.clients_per_round}")
        print(f"  [Config] rounds            = {cfg.num_rounds}")
        print(f"  [Config] local_epochs      = {cfg.local_epochs}")
        print(f"  [Config] seed              = {cfg.seed}")
        print(f"  [Config] divergence_metric = {cfg.divergence_metric}")
        print(f"  [Config] diagnostics       = GRAD: {cfg.enable_gradient_diagnostics} | ROUTING: {cfg.enable_routing_quality_analysis} | COMPRESSION: {cfg.enable_compression_analysis} | TAU: {cfg.enable_tau_diagnostics}")
        print("\n")

        run(cfg)

        summary = _read_summary(log_path, cfg.clients_per_round)
        if metric == "cosine":
            cosine_acc = summary["final_acc"]
            acc_diff = 0.0
        else:
            acc_diff = summary["final_acc"] - cosine_acc if cosine_acc is not None else 0.0
            
        results.append({
            "metric": metric, 
            "acc_diff": acc_diff,
            **summary
        })

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*102}")
    print("  ROOT-CAUSE DIAGNOSTIC RESULTS")
    print(f"{'='*102}")
    print(f"\n  {'-'*100}")
    print(f"  {'Metric':<20}  {'Final Acc':>9}  {'Diff vs Cos':>11}  {'Upload MB':>11}  {'Download MB':>11}  {'Bidir MB':>10}  {'Saving %':>8}")
    print(f"  {'-'*100}")

    for r in results:
        diff_str = f"{r['acc_diff']*100:>+10.2f}%" if r['metric'] != "cosine" else f"{'-':>10} "
        print(
            f"  {r['metric']:<20}  "
            f"{r['final_acc']*100:>8.2f}%  "
            f"{diff_str}  "
            f"{r['upload_mb']:>11.2f}  "
            f"{r['download_mb']:>11.2f}  "
            f"{r['bidir_mb']:>10.2f}  "
            f"{r['bidir_saving_pct']:>7.1f}%"
        )
    print(f"  {'-'*100}\n")

if __name__ == "__main__":
    main()
