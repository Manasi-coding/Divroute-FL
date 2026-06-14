"""
run_phase1_ablations.py
=======================
Phase 1 ablation suite — measures the individual and joint contribution of:
  - Error Feedback (use_error_feedback)
  - Tier-3 Heartbeat Sync (use_tier3_sync)

Runs 4 experiments sequentially, saves separate logs, prints a summary table.
No DivRoute algorithm components are modified.
"""

import json
import os

from divroute_fl.config import Config
from divroute_fl.main import run

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name":            "Full DivRoute       (EF=ON,  HB=ON) ",
        "use_error_feedback": True,
        "use_tier3_sync":     True,
        "log_path":        "logs/phase1_full_divroute.json",
    },
    {
        "name":            "No Error Feedback   (EF=OFF, HB=ON) ",
        "use_error_feedback": False,
        "use_tier3_sync":     True,
        "log_path":        "logs/phase1_no_error_feedback.json",
    },
    {
        "name":            "No Heartbeat        (EF=ON,  HB=OFF)",
        "use_error_feedback": True,
        "use_tier3_sync":     False,
        "log_path":        "logs/phase1_no_heartbeat.json",
    },
    {
        "name":            "Neither             (EF=OFF, HB=OFF)",
        "use_error_feedback": False,
        "use_tier3_sync":     False,
        "log_path":        "logs/phase1_neither.json",
    },
]

# Shared config — identical across all four runs except the two ablation flags.
SHARED = dict(
    num_clients       = 25,
    clients_per_round = 15,
    num_rounds        = 20,
    local_epochs      = 5,
    local_lr          = 0.1,
    batch_size        = 32,
    alpha             = 0.9,
    seed              = 42,
    use_epoch_warmup  = False,   # fixed epochs for clean comparison
    skip_plot_prompt  = True,    # suppress interactive prompt in automated runs
)


# ---------------------------------------------------------------------------
# Log reader
# ---------------------------------------------------------------------------
def _read_summary(log_path: str, clients_per_round: int) -> dict:
    """Parse a completed run log and return scalar summary metrics."""
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc    = history[-1]["test_accuracy"]
    total_dl     = sum(e["total_download_bytes"] for e in history)
    total_ul     = sum(e["total_upload_bytes"]   for e in history)
    total_bidir  = total_dl + total_ul

    # FedAvg bidirectional baseline: full float32 delta both directions,
    # for every selected client, every round.
    delta_numel           = history[0]["delta_numel"]
    fedavg_per_round_bidir = clients_per_round * delta_numel * 4 * 2
    fedavg_total_bidir    = fedavg_per_round_bidir * len(history)

    bidir_saving = 100.0 * (1.0 - total_bidir / fedavg_total_bidir) if fedavg_total_bidir else 0.0

    return {
        "final_acc":       final_acc,
        "dl_mb":           total_dl   / 1e6,
        "ul_mb":           total_ul   / 1e6,
        "bidir_mb":        total_bidir / 1e6,
        "fedavg_bidir_mb": fedavg_total_bidir / 1e6,
        "bidir_saving_pct": bidir_saving,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT {i}/4: {exp['name'].strip()}")
        print(f"{'='*70}\n")

        cfg = Config(
            **SHARED,
            use_error_feedback = exp["use_error_feedback"],
            use_tier3_sync     = exp["use_tier3_sync"],
            log_path           = exp["log_path"],
        )
        run(cfg)

        summary = _read_summary(exp["log_path"], SHARED["clients_per_round"])
        results.append({"name": exp["name"], **summary})
        print(f"\n  [done] {exp['name'].strip()} — acc={summary['final_acc']:.4f}, "
              f"bidir={summary['bidir_mb']:.2f} MB (saving {summary['bidir_saving_pct']:.1f}%)")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  PHASE 1 ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Experiment':<42} {'Acc':>7}  {'DL MB':>8}  {'UL MB':>8}  {'Bidir MB':>10}  {'Bidir Save':>11}")
    print(f"  {'-'*42} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*11}")
    for r in results:
        print(
            f"  {r['name']:<42} "
            f"{r['final_acc']*100:>6.2f}%  "
            f"{r['dl_mb']:>8.2f}  "
            f"{r['ul_mb']:>8.2f}  "
            f"{r['bidir_mb']:>10.2f}  "
            f"{r['bidir_saving_pct']:>9.1f}%"
        )

    # Contribution of each component
    full   = results[0]
    no_ef  = results[1]
    no_hb  = results[2]
    neither = results[3]

    ef_contribution  = full["final_acc"] - no_ef["final_acc"]
    hb_contribution  = full["final_acc"] - no_hb["final_acc"]
    both_contribution = full["final_acc"] - neither["final_acc"]

    fedavg_bidir = full["fedavg_bidir_mb"]

    print(f"\n  FedAvg bidirectional baseline : {fedavg_bidir:.2f} MB")
    print(f"\n  Component contribution to accuracy (vs Full DivRoute):")
    print(f"    Error Feedback alone          : {ef_contribution*100:+.2f} pp")
    print(f"    Tier-3 Heartbeat alone         : {hb_contribution*100:+.2f} pp")
    print(f"    Both components together       : {both_contribution*100:+.2f} pp")
    print(f"\n  Log files written to:")
    for exp in EXPERIMENTS:
        print(f"    {exp['log_path']}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
