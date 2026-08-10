"""
run_fedavg_diagnostic.py
========================
Matched FedAvg diagnostic for CIFAR-100 / ResNet-18 / seed 42.

PURPOSE
-------
Determine whether the ~70% global accuracy ceiling is the training-recipe
limit (FedAvg also reaches ~70%) or a DivRoute penalty (FedAvg > DivRoute
by a meaningful margin).

COMPARABILITY GUARANTEE
-----------------------
Every training hyper-parameter is identical to the current DivRoute seed-42
run.  Only the federated mechanism differs:

  DivRoute                        FedAvg (this script)
  ──────────────────────────────  ──────────────────────────────────────
  Tiered routing (T1/T2/T3)       All 20 clients contribute equally
  k1=0.70->0.35/k2=0.30->0.10    k=1.0 (full delta, no sparsification)
  sqrt(d_ema) aggregation weight  Sample-count only (n_i / sum n_j)
  gamma=0.85 selection decay      gamma=1.0 (uniform selection always)
  Tier-3 exclusion active         No exclusion -- all 20 clients aggregate
  Adaptive tau                    Not used
  Error feedback                  Not used (also off in DivRoute)
  Server momentum                 Not used (also off in DivRoute)

FEDAVG VERIFICATION (from divroute_fl/main.py L66-81)
-------------------------------------------------------
fedavg_baseline_mode forces:
  use_divergence_weighting = False   -> sample-weighted aggregation only
  k_ratio_tier1 = k_ratio_tier2 = 1.0  -> top-k keeps ALL params
  tau_low = tau_high = -1.0          -> all clients d >= 0 > -1 -> Tier 1
  gamma = 1.0                        -> no selection decay

Mathematical result:
  agg_delta = sum_i  (n_i / sum_j n_j) * full_delta_i
  new_w = w_global + agg_delta
        = sum_i (n_i / sum_j n_j) * w_i_local   <- canonical FedAvg

USAGE
-----
  python run_fedavg_diagnostic.py                    # run + compare
  python run_fedavg_diagnostic.py --no-compare       # run only
  python run_fedavg_diagnostic.py --compare-only     # compare only
  python run_fedavg_diagnostic.py --rounds 30        # shorter run
  python run_fedavg_diagnostic.py --fresh            # ignore checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import divroute_fl.client as _fl_client   # import for DEBUG_BN mutation

# Force UTF-8 on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config
from divroute_fl.main import run

# ── Paths ──────────────────────────────────────────────────────────────────────
FEDAVG_LOG   = Path("logs/debug/fedavg_cifar100_seed42_50r.json")
DIVROUTE_LOG = Path("logs/phase5/cifar100/divroute_seed42.json")

COMPARE_ROUNDS = [10, 20, 30, 40, 50]

# ── Experiment constants (must match DivRoute seed-42 preset exactly) ──────────
NUM_ROUNDS        = 50
SEED              = 42
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 20
LOCAL_EPOCHS      = 5
LOCAL_LR          = 0.05
ALPHA             = 0.9
BATCH_SIZE        = 32          # Config default, same in both runs
DATASET           = "cifar100"
MODEL             = "resnet18"


# ── Config factory ─────────────────────────────────────────────────────────────
def make_fedavg_config(num_rounds: int, log_path: str, fresh: bool) -> Config:
    """
    Build a Config for vanilla FedAvg that is mechanistically comparable
    to the DivRoute seed-42 run while keeping all training hyper-parameters
    identical.
    """
    return Config(
        # -- Identity (identical to DivRoute) ----------------------------------
        dataset_name         = DATASET,
        model_name           = MODEL,
        seed                 = SEED,
        num_clients          = NUM_CLIENTS,
        clients_per_round    = CLIENTS_PER_ROUND,
        num_rounds           = num_rounds,
        local_epochs         = LOCAL_EPOCHS,
        local_lr             = LOCAL_LR,
        alpha                = ALPHA,
        batch_size           = BATCH_SIZE,
        # -- Epoch warmup: DISABLED to match DivRoute --------------------------
        # get_recommended_divroute_config() sets use_epoch_warmup=False
        use_epoch_warmup     = False,
        # -- FedAvg mode (verified genuine -- see module docstring) ------------
        fedavg_baseline_mode = True,
        # -- Misc --------------------------------------------------------------
        skip_plot_prompt     = True,
        log_path             = log_path,
        fresh                = fresh,
        local_val_fraction   = 0.0,  # validation is a separate experiment
    )


# ── Comparability table ────────────────────────────────────────────────────────
def print_comparability_table() -> None:
    W = 72
    print()
    print("=" * W)
    print("  COMPARABILITY CHECK: FedAvg Diagnostic vs DivRoute seed-42")
    print("=" * W)

    identical = [
        ("dataset",              f"{DATASET}",               f"{DATASET}"),
        ("model",                f"{MODEL}",                 f"{MODEL}"),
        ("seed",                 f"{SEED}",                  f"{SEED}"),
        ("num_clients",          f"{NUM_CLIENTS}",           f"{NUM_CLIENTS}"),
        ("clients_per_round",    f"{CLIENTS_PER_ROUND}",     f"{CLIENTS_PER_ROUND}"),
        ("local_epochs",         f"{LOCAL_EPOCHS}",          f"{LOCAL_EPOCHS}"),
        ("local_lr",             f"{LOCAL_LR}",              f"{LOCAL_LR}"),
        ("alpha (Dirichlet)",    f"{ALPHA}",                 f"{ALPHA}"),
        ("batch_size",           f"{BATCH_SIZE}",            f"{BATCH_SIZE}"),
        ("use_epoch_warmup",     "False",                    "False"),
        ("LR schedule",          "cosine min=5e-4",          "cosine min=5e-4"),
        ("optimiser",            "SGD nesterov",             "SGD nesterov"),
        ("momentum",             "0.9",                      "0.9"),
        ("weight_decay",         "5e-4",                     "5e-4"),
        ("MixUp",                "Beta(0.2,0.2) lam>=0.5",  "Beta(0.2,0.2) lam>=0.5"),
        ("label smoothing",      "e=0.1",                   "e=0.1"),
        ("train augmentation",   "RandCrop+Flip+Erase",     "RandCrop+Flip+Erase"),
        ("normalisation",        "CIFAR-100 mean/std",      "CIFAR-100 mean/std"),
        ("test transform",       "Normalise only",           "Normalise only"),
        ("architecture",         "CIFAR-ResNet18 3x3 stem", "CIFAR-ResNet18 3x3 stem"),
        ("NTD",                  "off (ntd_beta=0.0)",       "off (ntd_beta=0.0)"),
        ("error feedback",       "off",                      "off"),
        ("server momentum",      "off",                      "off"),
        ("FP16 upload/download", "off",                      "off"),
        ("local_val_fraction",   "0.0 (disabled)",          "0.0 (disabled)"),
    ]

    differs = [
        ("k_ratio_tier1",       "1.0 (full delta)",         "0.70->0.35"),
        ("k_ratio_tier2",       "1.0 (full delta)",         "0.30->0.10"),
        ("tau_low / tau_high",  "-1.0 / -1.0 (all T1)",    "adaptive mu+/-sigma"),
        ("aggregation weight",  "sample n_i/sum(n_j)",      "sample x sqrt(d_ema)"),
        ("gamma (select decay)","1.0 (uniform)",            "0.85"),
        ("Tier-3 exclusion",    "none (all 20 contribute)", "Tier-3 -> weight 0"),
        ("compression",         "none (full state_dict)",   "top-k sparse"),
        ("adaptive tau",        "off",                      "on (alpha=0.5,beta=1.0)"),
    ]

    print("\n  IDENTICAL parameters (training conditions)")
    print(f"  {'Parameter':<28}  {'Value'}")
    print(f"  {'─'*28}  {'─'*30}")
    for name, fv, _ in identical:
        print(f"  {name:<28}  {fv}")

    print("\n  MECHANISM differences (intentional)")
    print(f"  {'Parameter':<28}  {'FedAvg':<25}  {'DivRoute'}")
    print(f"  {'─'*28}  {'─'*25}  {'─'*25}")
    for name, fv, dv in differs:
        print(f"  {name:<28}  {fv:<25}  {dv}")

    print()
    print("  NOTE: DivRoute k-ratio warmup (rounds 0-29: k1=0.70, k2=0.30) is")
    print("  active in the ongoing run.  This script does NOT modify that run.")
    print()


# ── Log utilities ──────────────────────────────────────────────────────────────
def _load_log(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _acc_at(history: list, target_round_idx: int):
    """Return test_accuracy for round==target_round_idx, or None if missing."""
    for e in history:
        if e["round"] == target_round_idx:
            return e["test_accuracy"]
    return None


def _comm_at(history: list, target_round_idx: int):
    """Return (upload_mb, download_mb) for round==target_round_idx."""
    for e in history:
        if e["round"] == target_round_idx:
            ul = e.get("total_upload_bytes", 0) / 1e6
            dl = e.get("total_download_bytes", 0) / 1e6
            return ul, dl
    return None, None


# ── Comparison printer ─────────────────────────────────────────────────────────
def print_comparison(fedavg_log: Path, divroute_log: Path) -> None:
    W = 72
    print("=" * W)
    print("  TRAJECTORY COMPARISON  (FedAvg vs DivRoute, seed 42)")
    print("=" * W)

    fa_hist = _load_log(fedavg_log)
    dr_hist = _load_log(divroute_log)

    print(f"\n  {'Round':<7}  {'FedAvg Acc':>11}  {'DivRoute Acc':>13}  {'Gap (F-D)':>10}")
    print(f"  {'─'*7}  {'─'*11}  {'─'*13}  {'─'*10}")

    gaps = []
    for rnd in COMPARE_ROUNDS:
        fa = _acc_at(fa_hist, rnd - 1)
        dr = _acc_at(dr_hist, rnd - 1)
        fa_s = f"{fa*100:6.2f}%" if fa is not None else "  N/A  "
        dr_s = f"{dr*100:6.2f}%" if dr is not None else "  N/A  "
        if fa is not None and dr is not None:
            gap = (fa - dr) * 100
            gaps.append(gap)
            print(f"  R{rnd:<6}  {fa_s:>11}  {dr_s:>13}  {gap:+10.2f}pp")
        else:
            print(f"  R{rnd:<6}  {fa_s:>11}  {dr_s:>13}  {'N/A':>10}")

    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        print(f"\n  Mean gap (FedAvg - DivRoute) over {len(gaps)} rounds: {mean_gap:+.2f}pp\n")
        if abs(mean_gap) <= 2.0:
            verdict = ("<=2pp  No strong evidence DivRoute causes an accuracy penalty.\n"
                       "         Training-recipe ceiling likely explains ~70% ceiling.\n"
                       "         Routing/compression ablations are LOW priority.")
        elif abs(mean_gap) <= 5.0:
            verdict = ("2-5pp  Moderate DivRoute penalty. Ablate in order:\n"
                       "         (a) compression k-ratios -> (b) Tier-3 inclusion\n"
                       "         -> (c) divergence aggregation weight.")
        else:
            verdict = (">5pp   Strong evidence DivRoute mechanics cost accuracy.\n"
                       "         Prioritise: no-compression ablation, then Tier-3 inclusion.")
        print(f"  INTERPRETATION: {verdict}")

    print(f"\n  COMMUNICATION USAGE (per round)")
    print(f"  {'Round':<7}  {'FA Upload':>10}  {'FA Download':>12}  "
          f"{'DR Upload':>10}  {'DR Download':>12}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*12}")
    for rnd in COMPARE_ROUNDS:
        fa_ul, fa_dl = _comm_at(fa_hist, rnd - 1)
        dr_ul, dr_dl = _comm_at(dr_hist, rnd - 1)
        fa_ul_s  = f"{fa_ul:.0f}MB"  if fa_ul  is not None else "N/A"
        fa_dl_s  = f"{fa_dl:.0f}MB"  if fa_dl  is not None else "N/A"
        dr_ul_s  = f"{dr_ul:.0f}MB"  if dr_ul  is not None else "N/A"
        dr_dl_s  = f"{dr_dl:.0f}MB"  if dr_dl  is not None else "N/A"
        print(f"  R{rnd:<6}  {fa_ul_s:>10}  {fa_dl_s:>12}  {dr_ul_s:>10}  {dr_dl_s:>12}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched FedAvg diagnostic (CIFAR-100, seed 42)."
    )
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS,
                        help=f"Rounds to run (default: {NUM_ROUNDS})")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison table")
    parser.add_argument("--compare-only", action="store_true",
                        help="Print comparison only; requires both logs to exist")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--log", type=str, default=str(FEDAVG_LOG),
                        help=f"Output log path (default: {FEDAVG_LOG})")
    parser.add_argument("--debug-bn", action="store_true",
                        help="Enable BatchNorm propagation diagnostics (1 round, 1 client)")
    args = parser.parse_args()

    # Activate BN diagnostics if requested
    if args.debug_bn:
        _fl_client.DEBUG_BN = True
        print("[DEBUG_BN] Diagnostic mode enabled.  "
              "Will print BN state for client",
              _fl_client._DEBUG_BN_CLIENT,
              "layer", _fl_client._DEBUG_BN_LAYER)

    fedavg_log_path = Path(args.log)

    # Pre-flight output
    print_comparability_table()

    print("=" * 72)
    print("  fedavg_baseline_mode VERIFICATION (code-level)")
    print("=" * 72)
    print("""
  Source: divroute_fl/main.py L66-81

  fedavg_baseline_mode sets:
    k_ratio_tier1 = k_ratio_tier2 = 1.0
      -> compress_delta(): k >= numel branch fires
      -> returns {"indices": None, "values": full_delta}
      -> reconstruct_delta(): indices is None -> returns values unchanged
      -> NO sparsification occurs

    tau_low = tau_high = -1.0
      -> assign_tier(d): d > tau_high always (d >= 0 > -1.0)
      -> ALL clients assigned Tier 1
      -> agg_clients = ALL 20 selected clients (no Tier-3 filtering)

    use_divergence_weighting = False
      -> server.aggregate(): weight = n_i / total_samples (pure sample weight)

    gamma = 1.0
      -> update_selection_weights(): Tier-3 clients get weight * 1.0 = no decay
      -> (moot since no client is Tier 3)

  Resulting aggregation:
    new_w = w_global + sum_i  (n_i / sum_j n_j) * (w_i_local - w_global)
          = sum_i (n_i / sum_j n_j) * w_i_local   <- canonical FedAvg OK
""")

    if args.compare_only:
        if not fedavg_log_path.exists():
            sys.exit(f"[error] FedAvg log not found: {fedavg_log_path}")
        if not DIVROUTE_LOG.exists():
            sys.exit(f"[error] DivRoute log not found: {DIVROUTE_LOG}")
        print_comparison(fedavg_log_path, DIVROUTE_LOG)
        return

    # Guard: skip if already complete
    if fedavg_log_path.exists() and not args.fresh:
        try:
            existing = _load_log(fedavg_log_path)
            if len(existing) >= args.rounds:
                print(f"[skip] Log already has {len(existing)} rounds"
                      f" (>= {args.rounds}).  Use --fresh to rerun.\n")
                if not args.no_compare and DIVROUTE_LOG.exists():
                    print_comparison(fedavg_log_path, DIVROUTE_LOG)
                return
        except Exception:
            pass  # corrupt or truncated log -- proceed

    fedavg_log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = make_fedavg_config(args.rounds, str(fedavg_log_path), args.fresh)

    print("=" * 72)
    print(f"  RUNNING: FedAvg  |  CIFAR-100  |  ResNet-18  |  seed={SEED}")
    print(f"           {args.rounds} rounds  |"
          f"  {CLIENTS_PER_ROUND}/{NUM_CLIENTS} clients/round  |"
          f"  E={LOCAL_EPOCHS}  LR={LOCAL_LR}")
    print(f"           log -> {fedavg_log_path}")
    print("=" * 72)
    print()

    run(cfg)

    # Disable BN diagnostics after the run (idempotent if never enabled)
    _fl_client.DEBUG_BN = False

    print()
    print(f"[done] FedAvg complete. Log: {fedavg_log_path}")

    if not args.no_compare:
        if DIVROUTE_LOG.exists():
            print()
            print_comparison(fedavg_log_path, DIVROUTE_LOG)
        else:
            print(f"\n[info] DivRoute log not found at {DIVROUTE_LOG}.")
            print("       Run with --compare-only once the DivRoute log exists.")


if __name__ == "__main__":
    main()
