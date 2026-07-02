"""
run_phase5_comparison.py
=========================
Phase 5 — Comparison Against Published Baselines

Runs all five methods across a chosen dataset/model configuration to produce
the paper's main comparison table (Table 1 / Table 3).

Methods:
  1. FedAvg          — uncompressed full-delta baseline (upper-bound accuracy)
  2. FedZip          — Top-1% sparsification + k-means quantisation (byte-cost
                       computed post-hoc using the FedZip formula; training uses
                       full deltas to preserve accuracy fidelity)
  3. FedSparse       — L1 proximity regularisation (lambda auto-scaled to model)
  4. Uniform Top-5%  — flat top-5% compression, no routing intelligence
  5. DivRoute-FL     — three-tier compression with FIXED thresholds from Phase 4
                       ablation (tau_high=0.03, tau_low=0.015). Fixed thresholds
                       allow tier distribution to move naturally across rounds,
                       unlike adaptive percentiles which lock tiers at 4/8/3.

Features:
  - --dataset flag selects the full experimental preset (model, clients, seeds,
    lambda, Acc@MB budgets). Logs/CSVs are namespaced per dataset.
  - --methods and --seeds allow further subsetting within the chosen preset.
  - Resumable: skips any run whose log file already exists.

Usage:
    python run_phase5_comparison.py --dataset cifar100          # full ResNet-18 run
    python run_phase5_comparison.py --dataset cifar10           # SimpleCNN quick run
    python run_phase5_comparison.py --dataset cifar10 --seeds 42
    python run_phase5_comparison.py --dataset cifar100 --methods fedavg divroute
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from divroute_fl.config import Config, get_recommended_divroute_config, get_uniform_top5_config
from divroute_fl.main import run
from baselines.fedzip_actual import get_fedzip_actual_config
from baselines.fedsparse_baseline import get_fedsparse_config

# ── Dataset presets ────────────────────────────────────────────────────────────
# Each preset is a self-contained experimental configuration.
# Selecting a preset via --dataset updates ALL derived settings automatically.
#
# DivRoute uses FIXED thresholds (use_adaptive_tau=False) in BOTH presets.
# Reason: adaptive percentile thresholds lock the tier distribution to a fixed
# 4/8/3 split regardless of k-ratios or training dynamics, making the routing
# behaviour indistinguishable across ablation conditions. Fixed thresholds from
# the Phase 4 sweep (tau_high=0.03, tau_low=0.015) allow tiers to move naturally.

DATASET_PRESETS = {
    "cifar10": {
        "dataset_name":      "cifar10",
        "model_name":        "simplecnn",
        "num_clients":       25,
        "clients_per_round": 15,
        "num_rounds":        100,
        "seeds":             [42, 123, 456],
        # FedSparse: 0.01 is appropriate for SimpleCNN (200K params)
        "fedsparse_lambda":  0.01,
        # FedZip byte-accounting params (paper defaults)
        "fedzip_z_ratio":    0.01,
        "fedzip_k_clusters": 3,
        # Acc@MB budgets tuned for SimpleCNN scale
        "acc_at_budgets_mb": [50, 100, 200, 500],
        # DivRoute fixed thresholds from Phase 4.1 ablation sweep
        "tau_high":          0.03,
        "tau_low":           0.015,
        "log_subdir":        "logs/phase5/cifar10",
        "csv_name":          "phase5_cifar10_results.csv",
    },
    "cifar100": {
        "dataset_name":      "cifar100",
        "model_name":        "resnet18",
        "num_clients":       100,
        "clients_per_round": 20,    # raised from 10 → 20% participation for faster convergence
        "num_rounds":        150,   # raised from 100 → 150 for credible accuracy plateau
        # local_lr tuned down from 0.1 (SimpleCNN default) — ResNet-18 on CIFAR-100
        # needs smaller steps to reduce client drift across 5 local epochs.
        "local_lr":          0.01,
        "seeds":             [42, 123, 456],
        # FedSparse: scaled down 100x for ResNet-18 (11M params) vs SimpleCNN (200K).
        # At 0.01, the L1 penalty overwhelms cross-entropy → accuracy ≈ random chance.
        "fedsparse_lambda":  0.0001,
        # FedZip byte-accounting params (paper defaults)
        "fedzip_z_ratio":    0.01,
        "fedzip_k_clusters": 3,
        # Acc@MB budgets tuned for ResNet-18 / CIFAR-100 scale
        "acc_at_budgets_mb": [500, 1000, 2000, 5000],
        # DivRoute fixed thresholds tuned for true float64 parameter divergence:
        # Without BN buffer contamination, ResNet-18 divergence is ~0.0007 - 0.0012.
        # tau_high=0.0010 ensures top divergence gets Tier 1, tau_low=0.0007 ensures
        # converged clients fall to Tier 3 naturally as gradients shrink over time.
        "tau_high":          0.0008,
        "tau_low":           0.0006,
        # Adaptive tau settings (disabled by default to show communication savings trajectory,
        # but available here if you want to test fixed percentile assignments later).
        "use_adaptive_tau":  False,
        "tau_low_pct":       20.0,
        "tau_high_pct":      75.0,
        # DivRoute k-ratios scaled up for ResNet-18 / CIFAR-100:
        # Phase 4 values (0.20 / 0.05) were tuned on SimpleCNN/CIFAR-10.
        # At k=0.20, all early-round Tier-1 clients receive only 2.24M of 11.2M params
        # — the backbone barely updates. k=0.35 sends 3.92M params, covering all layers.
        # k_tier2=0.10 gives Tier-2 clients 1.12M params vs only 560K at 0.05.
        "k_ratio_tier1":     0.35,
        "k_ratio_tier2":     0.10,
        "log_subdir":        "logs/phase5/cifar100",
        "csv_name":          "phase5_cifar100_results.csv",
    },
}

# Active preset — overridden by --dataset at runtime
_PRESET: dict = DATASET_PRESETS["cifar100"]

ALL_METHODS = ["fedavg", "fedzip", "fedsparse", "uniform", "divroute"]


# ── Paths (resolved at runtime from preset) ────────────────────────────────────
RESULTS_DIR = Path("results")


def _logs_dir() -> Path:
    return Path(_PRESET["log_subdir"])


def _csv_path() -> Path:
    return RESULTS_DIR / _PRESET["csv_name"]


def _csv_columns() -> list:
    return [
        "method", "seed",
        "final_acc", "download_mb", "upload_mb", "bidir_mb",
        "bidir_saving_pct", "runtime_sec",
    ] + [f"acc_at_{b}mb" for b in _PRESET["acc_at_budgets_mb"]]


# ── Config factories ───────────────────────────────────────────────────────────
def _shared(seed: int, log_path: str) -> dict:
    p = _PRESET
    return dict(
        num_clients       = p["num_clients"],
        clients_per_round = p["clients_per_round"],
        num_rounds        = p["num_rounds"],
        local_lr          = p.get("local_lr", 0.1),  # preset-specific LR; falls back to Config default
        seed              = seed,
        dataset_name      = p["dataset_name"],
        model_name        = p["model_name"],
        skip_plot_prompt  = True,
        log_path          = log_path,
    )


def _make_config(method: str, seed: int, log_path: str) -> Config:
    p = _PRESET
    s = _shared(seed, log_path)
    if method == "fedavg":
        return Config(fedavg_baseline_mode=True, **s)
    elif method == "fedzip":
        return get_fedzip_actual_config(
            z_ratio=p["fedzip_z_ratio"],
            k_clusters=p["fedzip_k_clusters"],
            **s
        )
    elif method == "fedsparse":
        # Lambda is auto-scaled to the model size via the preset.
        return get_fedsparse_config(fedsparse_lambda=p["fedsparse_lambda"], **s)
    elif method == "uniform":
        return get_uniform_top5_config(**s)
    elif method == "divroute":
        # Fixed thresholds from Phase 4.1 ablation (tau_high=0.03, tau_low=0.015).
        # use_adaptive_tau=False so tier distribution moves naturally with training
        # dynamics instead of being locked at a fixed percentile split every round.
        # k-ratios read from preset so cifar100 can use scale-appropriate values
        # (0.35/0.10) while cifar10 keeps the Phase 4 ablation values (0.20/0.05).
        return get_recommended_divroute_config(
            use_adaptive_tau = p.get("use_adaptive_tau", False),
            tau_high         = p["tau_high"],
            tau_low          = p["tau_low"],
            tau_low_pct      = p.get("tau_low_pct", 20.0),
            tau_high_pct     = p.get("tau_high_pct", 75.0),
            k_ratio_tier1    = p.get("k_ratio_tier1", 0.20),
            k_ratio_tier2    = p.get("k_ratio_tier2", 0.05),
            **s
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _log_path(method: str, seed: int) -> Path:
    return _logs_dir() / f"{method}_seed{seed}.json"


# ── Log parsing ────────────────────────────────────────────────────────────────
def _parse_log(path: Path, method: str) -> dict:
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    final_acc = history[-1]["test_accuracy"]
    num_rounds = len(history)

    # Download bytes ── prefer the explicit field, fall back to legacy field
    total_dl = sum(
        e.get("total_download_bytes", e.get("total_bytes_transmitted", 0))
        for e in history
    )

    # Upload bytes ── prefer explicit field, fall back to per-client sum
    total_ul = 0
    for e in history:
        if "total_upload_bytes" in e:
            total_ul += e["total_upload_bytes"]
        elif "clients" in e:
            total_ul += sum(c.get("upload_bytes", 0) for c in e["clients"])



    total_bidir = total_dl + total_ul

    # FedAvg reference (full float32 delta, both directions, every round)
    delta_numel = history[0].get("delta_numel", 0)
    fedavg_ref  = _PRESET["clients_per_round"] * delta_numel * 4 * 2 * num_rounds
    saving_pct  = (
        100.0 * (1.0 - total_bidir / fedavg_ref)
        if fedavg_ref else 0.0
    )

    # ── Acc @ fixed budget ────────────────────────────────────────────────────
    acc_at = {}
    for budget_mb in _PRESET["acc_at_budgets_mb"]:
        budget_bytes = budget_mb * 1e6
        cum_dl = 0
        acc_found = None
        for e in history:
            dl = e.get("total_download_bytes", e.get("total_bytes_transmitted", 0))

            cum_dl += dl
            if cum_dl >= budget_bytes:
                acc_found = e["test_accuracy"]
                break
        acc_at[budget_mb] = acc_found if acc_found is not None else final_acc

    return {
        "final_acc":       final_acc,
        "download_mb":     total_dl    / 1e6,
        "upload_mb":       total_ul    / 1e6,
        "bidir_mb":        total_bidir / 1e6,
        "bidir_saving_pct": saving_pct,
        "acc_at":          acc_at,
    }


# ── Summary helpers ────────────────────────────────────────────────────────────
def _mean(xs): return statistics.mean(xs)  if xs else float("nan")
def _std(xs):  return statistics.stdev(xs) if len(xs) > 1 else 0.0

def _method_labels() -> dict:
    """Returns display labels — reads FedSparse lambda from the active preset."""
    return {
        "fedavg":    "Full FedAvg",
        "fedzip":    "FedZip (z=1%)",
        "fedsparse": f"FedSparse (lambda={_PRESET['fedsparse_lambda']})",
        "uniform":   "Uniform Top-5%",
        "divroute":  "DivRoute-FL (fixed-tau)",
    }


def _print_summary(rows: list[dict], methods: list[str]) -> None:
    print(f"\n{'='*76}")
    print("  PHASE 5 COMPARISON SUMMARY")
    print(f"{'='*76}")
    p = _PRESET
    print(f"\n  Dataset : {p['dataset_name'].upper()}   Model : {p['model_name'].upper()}")
    print(f"  Clients : {p['num_clients']} total, {p['clients_per_round']}/round  "
          f"Rounds : {p['num_rounds']}   Seeds : (see header above)")

    # Main table
    print(f"\n  {'Method':<22} {'Acc%':>8} {'±':>4} {'Bidir MB':>10} {'Saving%':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*4} {'-'*10} {'-'*9}")
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        accs   = [r["final_acc"]        for r in subset]
        bidirs = [r["bidir_mb"]         for r in subset]
        saves  = [r["bidir_saving_pct"] for r in subset]
        label  = _method_labels().get(m, m)
        print(f"  {label:<22} {_mean(accs)*100:>7.2f}% {_std(accs)*100:>3.2f}%"
              f" {_mean(bidirs):>10.1f} {_mean(saves):>8.1f}%")

    budgets = _PRESET["acc_at_budgets_mb"]
    print(f"\n  Acc @ fixed download budget (mean over seeds):")
    hdr = f"  {'Method':<22}"
    for b in budgets:
        hdr += f"  {str(b)+'MB':>8}"
    print(hdr)
    print(f"  {'-'*22}" + f"  {'-'*8}" * len(budgets))
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        label  = _method_labels().get(m, m)
        row = f"  {label:<22}"
        for b in budgets:
            vals = [r["acc_at"][b] for r in subset if b in r["acc_at"]]
            row += f"  {_mean(vals)*100:>7.2f}%" if vals else f"  {'N/A':>8}"
        print(row)

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    global _PRESET

    parser = argparse.ArgumentParser(description="Run Phase 5 baseline comparisons")
    parser.add_argument(
        "--dataset", choices=list(DATASET_PRESETS.keys()), default="cifar100",
        help=(
            "Dataset/model preset to use.\n"
            "  cifar10  → SimpleCNN, 25 clients, seeds {42,123,456}, lambda=0.01\n"
            "  cifar100 → ResNet-18, 100 clients, seeds {42,123,456}, lambda=0.0001\n"
            "All other settings (model, clients/round, Acc@MB budgets, FedSparse\n"
            "lambda, FedZip params, DivRoute fixed-tau values) update automatically."
        )
    )
    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS,
        help="Which methods to run (default: all five)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Override seeds (default: use preset's seed list)"
    )
    args = parser.parse_args()

    # ── Apply preset ──────────────────────────────────────────────────────────
    _PRESET = DATASET_PRESETS[args.dataset]
    p       = _PRESET
    methods = args.methods
    seeds   = args.seeds if args.seeds is not None else p["seeds"]

    print(f"\n{'='*76}")
    print(f"  PHASE 5 — {args.dataset.upper()} PRESET")
    print(f"  Model       : {p['model_name']}")
    print(f"  Dataset     : {p['dataset_name']}")
    print(f"  Clients     : {p['num_clients']} total, {p['clients_per_round']}/round")
    print(f"  Rounds      : {p['num_rounds']}")
    print(f"  Seeds       : {seeds}")
    print(f"  Methods     : {methods}")
    print(f"  FedSparse λ : {p['fedsparse_lambda']}")
    print(f"  DivRoute τ  : fixed  tau_high={p['tau_high']}  tau_low={p['tau_low']}")
    print(f"  Logs dir    : {p['log_subdir']}")
    print(f"{'='*76}\n")

    _logs_dir().mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    total_runs = len(methods) * len(seeds)
    pending    = []
    skipped    = 0

    for method in methods:
        for seed in seeds:
            lp = _log_path(method, seed)
            if lp.exists():
                print(f"[skip] {_method_labels().get(method, method)} seed={seed} — log exists")
                skipped += 1
            else:
                pending.append((method, seed))

    print(f"\nTotal runs : {total_runs}  |  "
          f"Skipped : {skipped}  |  To run : {len(pending)}")
    if pending:
        est_s = len(pending) * p["num_rounds"] * 90
        h, rem = divmod(int(est_s), 3600)
        m_min = rem // 60
        secs_per = 90 if p["model_name"] == "resnet18" else 20
        print(f"Estimated remaining time: ~{h}h {m_min}m  "
              f"(assumes ~{secs_per}s/round on mid-range GPU)\n")

    run_num = skipped
    for method, seed in pending:
        run_num += 1
        lp  = _log_path(method, seed)
        cfg = _make_config(method, seed, str(lp))
        label = _method_labels().get(method, method)

        print(f"\n{'='*76}")
        print(f"  Run {run_num}/{total_runs}: {label} | seed={seed} | "
              f"{p['num_rounds']} rounds | {p['dataset_name'].upper()} / {p['model_name'].upper()}")
        print(f"{'='*76}\n")

        t0 = time.perf_counter()
        run(cfg)
        elapsed = time.perf_counter() - t0
        print(f"\n  done in {elapsed:.0f}s")

    # ── Collect all results ────────────────────────────────────────────────────
    rows: list[dict] = []
    for method in methods:
        for seed in seeds:
            lp = _log_path(method, seed)
            if not lp.exists():
                print(f"[warn] Missing log: {lp} — skipping")
                continue
            metrics = _parse_log(lp, method)
            rows.append({
                "method": method,
                "seed":   seed,
                **metrics,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = _csv_path()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_columns())
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "method":            r["method"],
                "seed":              r["seed"],
                "final_acc":         f"{r['final_acc']:.4f}",
                "download_mb":       f"{r['download_mb']:.2f}",
                "upload_mb":         f"{r['upload_mb']:.2f}",
                "bidir_mb":          f"{r['bidir_mb']:.2f}",
                "bidir_saving_pct":  f"{r['bidir_saving_pct']:.1f}",
                "runtime_sec":       "",
                **{f"acc_at_{b}mb": f"{r['acc_at'].get(b, float('nan')):.4f}"
                   for b in _PRESET["acc_at_budgets_mb"]},
            })

    print(f"\n[done] CSV written to {csv_path}")
    _print_summary(rows, methods)


if __name__ == "__main__":
    main()
