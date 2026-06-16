"""
run_root_cause_diagnosis.py
===========================
Root-cause diagnostic suite — 4 experiments to isolate why DivRoute-FL
loses 16-20 pp relative to FedAvg.

No algorithm changes. All experiments use existing config flags.
Baseline from Phase 1: "Neither" (EF=OFF, HB=OFF) at 49.57%.
"""

import json
from divroute_fl.config import Config
from divroute_fl.main import run

# ── Shared config (matches Phase 1 ablation settings) ──────────────────────
SHARED = dict(
    num_clients       = 25,
    clients_per_round = 15,
    num_rounds        = 20,
    local_epochs      = 5,
    local_lr          = 0.1,
    batch_size        = 32,
    alpha             = 0.9,
    seed              = 42,
    use_epoch_warmup  = False,
    skip_plot_prompt  = True,
    # Phase 1 result: EF and HB are non-contributing or harmful,
    # so the diagnostic baseline disables them.
    use_error_feedback = False,
    use_tier3_sync     = False,
)

# ── Experiments ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name":  "Exp 1: No compression      (k=1.0, div-w ON, mom ON)  ",
        "overrides": {
            "k_ratio_tier1":         1.0,
            "k_ratio_tier2":         1.0,
            "use_adaptive_k":        False,
        },
        "log_path": "logs/diag_no_compression.json",
        "tests": "Is compression the dominant accuracy cost?",
    },
    {
        "name":  "Exp 2: No div-weighting     (k default, div-w OFF, mom ON) ",
        "overrides": {
            "use_divergence_weighting": False,
        },
        "log_path": "logs/diag_no_divweight.json",
        "tests": "Is sqrt weighting upweighting damaged Tier-2 updates?",
    },
    {
        "name":  "Exp 3: No momentum          (k default, div-w ON, mom OFF)",
        "overrides": {
            "use_server_momentum": False,
        },
        "log_path": "logs/diag_no_momentum.json",
        "tests": "Is momentum compounding compression bias?",
    },
    {
        "name":  "Exp 4: No compression/dw/mom (k=1.0, div-w OFF, mom OFF)",
        "overrides": {
            "k_ratio_tier1":           1.0,
            "k_ratio_tier2":           1.0,
            "use_adaptive_k":          False,
            "use_divergence_weighting": False,
            "use_server_momentum":     False,
        },
        "log_path": "logs/diag_tier_only.json",
        "tests": "Upper bound: FedAvg minus Tier-3 exclusion only",
    },
]

# ── Reference baselines (from previous runs — not re-run) ──────────────────
REFERENCES = {
    "FedAvg":                0.6595,
    "DivRoute (neither)":    0.4957,
}


def _read_final_acc(log_path: str) -> float:
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)
    return history[-1]["test_accuracy"]


def _read_summary(log_path: str, clients_per_round: int) -> dict:
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

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
        "bidir_mb":          total_bidir / 1e6,
        "fedavg_bidir_mb":   fedavg_total_bidir / 1e6,
        "bidir_saving_pct":  bidir_saving,
    }


def main():
    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*72}")
        print(f"  DIAGNOSTIC {i}/4: {exp['name'].strip()}")
        print(f"  Tests: {exp['tests']}")
        print(f"{'='*72}\n")

        cfg_dict = {**SHARED, **exp["overrides"], "log_path": exp["log_path"]}
        cfg = Config(**cfg_dict)
        run(cfg)

        summary = _read_summary(exp["log_path"], SHARED["clients_per_round"])
        results.append({"name": exp["name"], **summary})

    # ── Summary table ───────────────────────────────────────────────────────
    fedavg_acc   = REFERENCES["FedAvg"]
    neither_acc  = REFERENCES["DivRoute (neither)"]

    print(f"\n\n{'='*72}")
    print("  ROOT CAUSE DIAGNOSTIC RESULTS")
    print(f"{'='*72}")

    print(f"\n  Reference baselines (from previous runs):")
    print(f"    FedAvg baseline            : {fedavg_acc*100:6.2f}%")
    print(f"    DivRoute (neither, EF/HB off): {neither_acc*100:6.2f}%")
    print(f"    Gap to explain             : {(fedavg_acc - neither_acc)*100:6.2f} pp")

    print(f"\n  {'Experiment':<56} {'Acc':>7}  {'Δ vs Neither':>13}  {'Δ vs FedAvg':>12}  {'Bidir Save':>11}")
    print(f"  {'-'*56} {'-'*7}  {'-'*13}  {'-'*12}  {'-'*11}")

    for r in results:
        delta_vs_neither = (r["final_acc"] - neither_acc) * 100
        delta_vs_fedavg  = (r["final_acc"] - fedavg_acc)  * 100
        print(
            f"  {r['name']:<56} "
            f"{r['final_acc']*100:>6.2f}%  "
            f"{delta_vs_neither:>+12.2f}pp  "
            f"{delta_vs_fedavg:>+11.2f}pp  "
            f"{r['bidir_saving_pct']:>9.1f}%"
        )

    # ── Attribution ─────────────────────────────────────────────────────────
    exp1_acc = results[0]["final_acc"]  # no compression
    exp4_acc = results[3]["final_acc"]  # tier-only (no comp, no dw, no mom)

    compression_cost   = (exp1_acc - neither_acc) * 100
    mechanism_cost     = (fedavg_acc - exp4_acc) * 100
    divweight_isolated = (results[1]["final_acc"] - neither_acc) * 100
    momentum_isolated  = (results[2]["final_acc"] - neither_acc) * 100

    print(f"\n  ── Attribution (approximate, pp recovered from 'Neither') ──")
    print(f"    Removing compression       : {compression_cost:>+7.2f} pp")
    print(f"    Removing div-weighting     : {divweight_isolated:>+7.2f} pp")
    print(f"    Removing momentum          : {momentum_isolated:>+7.2f} pp")
    print(f"  ── Residual gap ──")
    print(f"    Tier-3 exclusion cost      : {mechanism_cost:>+7.2f} pp  (FedAvg − Exp 4)")
    print(f"    Total gap                  : {(fedavg_acc - neither_acc)*100:>7.2f} pp")

    # ── Interpretation ──────────────────────────────────────────────────────
    print(f"\n  ── Interpretation ──")
    if compression_cost > 10:
        print(f"    ✗  COMPRESSION IS DOMINANT ({compression_cost:+.1f}pp).")
        print(f"       k_ratio_tier2=0.05 is too aggressive. Raise to 0.15-0.25.")
        print(f"       Adaptive k decay compounds this. Disable or reduce decay rate.")
    elif compression_cost > 5:
        print(f"    ⚠  Compression is a major contributor ({compression_cost:+.1f}pp) but not sole cause.")
    else:
        print(f"    ✓  Compression is not the primary issue ({compression_cost:+.1f}pp).")

    if divweight_isolated > 3:
        print(f"    ✗  Divergence weighting is harmful ({divweight_isolated:+.1f}pp).")
        print(f"       1/√d upweights Tier-2 (most compressed) clients. Disable or use uniform.")
    elif divweight_isolated > 1:
        print(f"    ⚠  Divergence weighting has a moderate effect ({divweight_isolated:+.1f}pp).")
    else:
        print(f"    ─  Divergence weighting is neutral ({divweight_isolated:+.1f}pp).")

    if momentum_isolated > 3:
        print(f"    ✗  Server momentum is harmful ({momentum_isolated:+.1f}pp).")
        print(f"       β=0.9 amplifies top-k selection bias across rounds.")
    elif momentum_isolated > 1:
        print(f"    ⚠  Server momentum has a moderate effect ({momentum_isolated:+.1f}pp).")
    else:
        print(f"    ─  Server momentum is neutral ({momentum_isolated:+.1f}pp).")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
