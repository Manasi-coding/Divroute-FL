"""
run_bn_ablation_diagnostic.py
==============================
Corrected BN-aggregation ablation vs Phase-5 CIFAR-100 DivRoute (seed 42).

PREVIOUS MISMATCH (now fixed)
------------------------------
The original runner called get_recommended_divroute_config("cifar100") which
raises TypeError (function accepts only **kwargs).  On Kaggle this crashed and
was patched by changing the call to get_recommended_divroute_config() — i.e.
no arguments — which fell back to the generic preset with:
    k_ratio_tier1 = 0.20   (should be 0.35)
    k_ratio_tier2 = 0.05   (should be 0.10)
    use_adaptive_tau = True (correct)
    tau_low/tau_high = Config dataclass defaults (wrong values)
    local_lr = 0.1         (should be 0.05)
    num_clients = 25        (should be 100)
    clients_per_round = 15  (should be 20)
    dataset_name = 'cifar10' (should be 'cifar100')
    model_name = 'simplecnn' (should be 'resnet18')
The resulting run was NOT comparable to Phase-5.  68.42% at R50 is therefore
from an entirely different experimental configuration, not a BN-fix measurement.

CORRECTION
----------
This runner now calls get_recommended_divroute_config() with EXACTLY the same
keyword arguments that _make_config('divroute', 42, ...) passes in
run_phase5_comparison.py, derived from DATASET_PRESETS['cifar100'].
A startup comparability table verifies every field before training starts.
Any deviation (except bn_ablation_mode, num_rounds, log_path, checkpoint_dir)
aborts the run with an AssertionError.

ONLY experimental difference from Phase-5 DivRoute
----------------------------------------------------
server.py BN buffer aggregation now spans `clean` (all 20 valid selected
clients) instead of `agg_clients` (Tier-1+2 only, ~12 clients).
The include_tier3_in_aggregation server.py bug is also fixed (L75), but the
flag remains False, so Tier-3 clients are still excluded from parameter
aggregation.  BN stats from Tier-3 clients cost 38,480 bytes/client in a real
deployment; this overhead is accounted for in upload_bytes.

WHAT IS NOT CHANGED
-------------------
k-ratios, gamma, divergence weighting, thresholds, tau, EMA, grad clip,
data split, augmentation, LR schedule, local_epochs, selection, NTD, FP16,
error feedback, server momentum, Tier-3 staleness sync, routing score.

USAGE
-----
  python run_bn_ablation_diagnostic.py               # run then compare
  python run_bn_ablation_diagnostic.py --no-compare  # run only
  python run_bn_ablation_diagnostic.py --compare-only
  python run_bn_ablation_diagnostic.py --fresh       # discard checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from divroute_fl.config import Config, get_recommended_divroute_config
from divroute_fl.compression import get_adaptive_k_ratios
from divroute_fl.main import run

# ── Paths ──────────────────────────────────────────────────────────────────────
BN_LOG       = Path("logs/debug/bn_ablation_cifar100_seed42_50r_v2.json")
PHASE5_LOG   = Path("logs/phase5/cifar100/divroute_seed42.json")
FEDAVG_LOG   = Path("logs/debug/fedavg_cifar100_seed42_50r.json")
BN_CKPT_DIR  = "checkpoints_bn_ablation"

COMPARE_ROUNDS = [10, 20, 30, 40, 50]


# ── Config factory ─────────────────────────────────────────────────────────────
def make_phase5_divroute_config(num_rounds: int, log_path: str,
                                fresh: bool, checkpoint_dir: str) -> Config:
    """
    Reproduce _make_config('divroute', 42, ...) from run_phase5_comparison.py
    DATASET_PRESETS['cifar100'] exactly, then apply the BN ablation flag.
    Every keyword argument below comes from either DATASET_PRESETS['cifar100']
    or _shared().  Nothing is invented.
    """
    cfg = get_recommended_divroute_config(
        # ── From _shared() / DATASET_PRESETS['cifar100'] ────────────────────
        dataset_name      = "cifar100",
        model_name        = "resnet18",
        num_clients       = 100,
        clients_per_round = 20,
        num_rounds        = num_rounds,        # overridden to 50 for this ablation
        local_lr          = 0.05,
        local_epochs      = 5,
        seed              = 42,
        skip_plot_prompt  = True,
        log_path          = log_path,
        resume            = False,
        fresh             = fresh,
        checkpoint_dir    = checkpoint_dir,
        # ── DivRoute CIFAR-100 specific: tau and k-ratios ─────────────────
        use_adaptive_tau  = True,
        tau_high          = 0.00045,
        tau_low           = 0.00030,
        tau_alpha         = 0.5,
        tau_beta          = 1.0,
        k_ratio_tier1     = 0.35,
        k_ratio_tier2     = 0.10,
    )
    # bn_ablation_mode is not a declared Config dataclass field; set dynamically.
    # It controls two things:
    #   1. server.py L236: BN buffers averaged over `clean` (all 20 valid clients)
    #      instead of `agg_clients` (Tier-1+2 only) — patched globally in server.py.
    #   2. main.py: adds 38,480 bytes to upload_bytes per client to account for
    #      the real deployment cost of syncing ResNet-18 BN buffers from all clients.
    cfg.bn_ablation_mode = True
    return cfg



# ── Comparability check ────────────────────────────────────────────────────────
# Fields compared with Phase-5 control.  Tuple: (field_name, expected_value).
# All values from materialising _make_config('divroute', 42) above.
_PHASE5_EXPECTED = [
    ("dataset_name",              "cifar100"),
    ("model_name",                "resnet18"),
    ("num_clients",               100),
    ("clients_per_round",         20),
    ("local_lr",                  0.05),
    ("local_epochs",              5),
    ("batch_size",                32),
    ("alpha",                     0.9),
    ("seed",                      42),
    ("k_ratio_tier1",             0.35),
    ("k_ratio_tier2",             0.10),
    ("k_warmup_tier1",            0.70),
    ("k_warmup_tier2",            0.30),
    ("k_warmup_rounds",           30),
    ("use_k_warmup",              True),
    ("use_adaptive_k",            False),
    ("use_adaptive_tau",          True),
    ("tau_low",                   0.00030),
    ("tau_high",                  0.00045),
    ("tau_alpha",                 0.5),
    ("tau_beta",                  1.0),
    ("ema_beta",                  0.6),
    ("use_divergence_weighting",  True),
    ("divergence_weight_mode",    "sqrt"),
    ("gamma",                     0.85),
    ("grad_clip_norm",            10.0),
    ("use_epoch_warmup",          False),
    ("include_tier3_in_aggregation", False),
    ("use_server_momentum",       False),
    ("use_error_feedback",        False),
    ("use_tier3_sync",            False),
    ("fedavg_baseline_mode",      False),
    ("ntd_beta",                  0.0),
    ("ntd_tau",                   3.0),
    ("use_fp16_upload",           False),
    ("use_fp16_download",         False),
    ("routing_score",             "raw"),
    ("local_val_fraction",        0.0),
]

# Fields that are ALLOWED to differ between Phase-5 control and this ablation.
_ALLOWED_DIFFER = {"num_rounds", "log_path", "checkpoint_dir", "bn_ablation_mode"}


def verify_config(cfg: Config) -> None:
    """
    Print the comparability table and abort on any unexpected deviation.
    """
    W = 78
    print()
    print("=" * W)
    print("  COMPARABILITY TABLE: BN Ablation vs Phase-5 DivRoute Control")
    print("=" * W)
    print(f"  {'Field':<38}  {'Phase-5 (expected)':<18}  {'Ablation':<18}  Status")
    print(f"  {'-'*38}  {'-'*18}  {'-'*18}  ------")

    errors = []
    for field, expected in _PHASE5_EXPECTED:
        actual = getattr(cfg, field)
        match = (actual == expected) or (
            isinstance(expected, float) and abs(actual - expected) < 1e-12
        )
        status = "OK" if match else "MISMATCH ***"
        print(f"  {field:<38}  {str(expected):<18}  {str(actual):<18}  {status}")
        if not match:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")

    # BN ablation flag
    print(f"  {'bn_ablation_mode':<38}  {'False (control)':<18}  "
          f"  {str(cfg.bn_ablation_mode):<18}  EXPECTED DIFF")

    # k-ratios at milestone rounds
    print()
    print(f"  {'k-ratio at milestone rounds':}")
    print(f"  {'Round':<8}  {'k1':>6}  {'k2':>6}  {'Branch'}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*12}")
    expected_k = {1: (0.70, 0.30), 29: (0.70, 0.30), 30: (0.70, 0.30),
                  31: (0.35, 0.10), 50: (0.35, 0.10)}
    for disp_rnd in [1, 29, 30, 31, 50]:
        rnd = disp_rnd - 1
        k1, k2 = get_adaptive_k_ratios(cfg, rnd)
        in_wu = cfg.use_k_warmup and rnd < cfg.k_warmup_rounds
        branch = "WARMUP" if in_wu else "POST-WARMUP"
        ek1, ek2 = expected_k[disp_rnd]
        ok = (abs(k1 - ek1) < 1e-9 and abs(k2 - ek2) < 1e-9)
        flag = "" if ok else " *** MISMATCH"
        print(f"  R{disp_rnd:<6}  {k1:>6.2f}  {k2:>6.2f}  {branch}{flag}")
        if not ok:
            errors.append(f"R{disp_rnd} k-ratios: expected ({ek1},{ek2}), got ({k1},{k2})")

    print()
    if errors:
        print("  ABORT: The following mismatches were detected:")
        for e in errors:
            print(f"    - {e}")
        print()
        raise AssertionError(
            f"Config does not match Phase-5 DivRoute in {len(errors)} field(s). "
            "Fix the config factory before running."
        )

    print("  All fields verified. Ablation is config-comparable to Phase-5 DivRoute.")
    print("  Only BN aggregation scope differs (clean vs agg_clients in server.py).")
    print("=" * W)
    print()


# ── Log utilities ──────────────────────────────────────────────────────────────
def _load_log(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _acc_at(history: list, rnd_idx: int):
    for e in history:
        if e["round"] == rnd_idx:
            return e["test_accuracy"]
    return None


def _comm_at(history: list, rnd_idx: int):
    for e in history:
        if e["round"] == rnd_idx:
            ul = e.get("total_upload_bytes", 0) / 1e6
            dl = e.get("total_download_bytes", 0) / 1e6
            return ul, dl
    return None, None


def _tier_counts(history: list, rnd_idx: int):
    for e in history:
        if e["round"] == rnd_idx:
            t1 = sum(1 for c in e["clients"] if c.get("tier") == 1)
            t2 = sum(1 for c in e["clients"] if c.get("tier") == 2)
            t3 = sum(1 for c in e["clients"] if c.get("tier") == 3)
            return t1, t2, t3
    return None, None, None


# ── Comparison printer ─────────────────────────────────────────────────────────
def print_comparison(bn_log: Path) -> None:
    W = 90
    print()
    print("=" * W)
    print("  TRAJECTORY COMPARISON  (BN Ablation v2 vs Phase-5 DivRoute vs FedAvg, seed 42)")
    print("=" * W)

    bn_hist = _load_log(bn_log)

    p5_hist = _load_log(PHASE5_LOG) if PHASE5_LOG.exists() else []
    fa_hist = _load_log(FEDAVG_LOG) if FEDAVG_LOG.exists() else []

    print(f"\n  {'Round':<7}  {'T1/T2/T3':>9}  {'BN-Ablation':>13}  "
          f"{'Phase-5 DR':>12}  {'FedAvg':>8}  {'Gap AB-P5':>10}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*13}  {'-'*12}  {'-'*8}  {'-'*10}")

    for rnd in COMPARE_ROUNDS:
        ab  = _acc_at(bn_hist,  rnd - 1)
        p5  = _acc_at(p5_hist,  rnd - 1)
        fa  = _acc_at(fa_hist,  rnd - 1)
        t1, t2, t3 = _tier_counts(bn_hist, rnd - 1)

        tier_s = f"{t1}/{t2}/{t3}" if t1 is not None else "?"
        ab_s   = f"{ab*100:6.2f}%" if ab is not None else "  N/A  "
        p5_s   = f"{p5*100:6.2f}%" if p5 is not None else "  N/A  "
        fa_s   = f"{fa*100:6.2f}%" if fa is not None else "  N/A  "
        gap_s  = (f"{(ab-p5)*100:+.2f}pp" if (ab is not None and p5 is not None)
                  else "  N/A")

        print(f"  R{rnd:<6}  {tier_s:>9}  {ab_s:>13}  {p5_s:>12}  "
              f"{fa_s:>8}  {gap_s:>10}")

    # BN upload overhead
    if bn_hist:
        r1 = bn_hist[0]
        num_clients = len(r1.get("clients", []))
        t3_r1 = sum(1 for c in r1["clients"] if c.get("tier") == 3)
        print()
        print(f"  BN sync overhead:  {num_clients} clients × 38,480 bytes = "
              f"{num_clients * 38480 / 1e3:.1f} KB/round "
              f"(Tier-3 clients syncing BN only: {t3_r1})")

    # Communication
    print()
    print(f"  UPLOAD (MB/round)")
    print(f"  {'Round':<7}  {'BN Ablation':>12}  {'Phase-5 DR':>12}  "
          f"{'FedAvg':>8}  {'BN-overhead':>12}")
    print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*12}")
    for rnd in COMPARE_ROUNDS:
        ab_ul, _ = _comm_at(bn_hist, rnd - 1)
        p5_ul, _ = _comm_at(p5_hist, rnd - 1)
        fa_ul, _ = _comm_at(fa_hist, rnd - 1)
        bn_oh    = 20 * 38480 / 1e6  # 0.77 MB

        ab_s  = f"{ab_ul:.2f}" if ab_ul is not None else "N/A"
        p5_s  = f"{p5_ul:.2f}" if p5_ul is not None else "N/A"
        fa_s  = f"{fa_ul:.2f}" if fa_ul is not None else "N/A"
        print(f"  R{rnd:<6}  {ab_s:>12}  {p5_s:>12}  {fa_s:>8}  {bn_oh:>12.3f}")

    print()

    # Decision rule
    if bn_hist and p5_hist:
        bn_r50 = _acc_at(bn_hist, 49)
        p5_r50 = _acc_at(p5_hist, 49)
        if bn_r50 is not None and p5_r50 is not None:
            gain = (bn_r50 - p5_r50) * 100
            print(f"  R50 gain from BN-all: {gain:+.2f}pp")
            if gain >= 3.0:
                print("  >= 3pp: BN aggregation scope is a significant bottleneck.")
                print("  Next: compression ablation (k1=0.50, k2=0.20).")
            elif gain >= 1.0:
                print("  1-3pp: Modest BN effect. Also ablate compression (k1=0.50/k2=0.20).")
            else:
                print("  < 1pp: BN scope is not a primary bottleneck.")
                print("  Next: ablate compression or Tier-3 inclusion.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-5-matched BN ablation (CIFAR-100, seed 42, 50 rounds).")
    parser.add_argument("--rounds",      type=int, default=50)
    parser.add_argument("--no-compare",  action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--fresh",       action="store_true")
    parser.add_argument("--log",         type=str, default=str(BN_LOG))
    args = parser.parse_args()

    log_path = Path(args.log)

    if args.compare_only:
        if not log_path.exists():
            sys.exit(f"[error] BN ablation log not found: {log_path}")
        print_comparison(log_path)
        return

    # Skip if already complete
    if log_path.exists() and not args.fresh:
        try:
            existing = _load_log(log_path)
            if len(existing) >= args.rounds:
                print(f"[skip] Log already has {len(existing)} rounds "
                      f"(>= {args.rounds}).  Use --fresh to rerun.\n")
                if not args.no_compare:
                    print_comparison(log_path)
                return
        except Exception:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = make_phase5_divroute_config(
        num_rounds     = args.rounds,
        log_path       = str(log_path),
        fresh          = args.fresh,
        checkpoint_dir = BN_CKPT_DIR,
    )

    # Verify before training starts — aborts on any mismatch
    verify_config(cfg)

    print("=" * 72)
    print(f"  RUNNING: BN Ablation (Phase-5 matched)  |  CIFAR-100  |  ResNet-18")
    print(f"           seed=42  |  {args.rounds} rounds  |  20/100 clients/round")
    print(f"           k1=0.70->0.35  k2=0.30->0.10  (warmup R1-R30, post R31+)")
    print(f"           BN stats: all 20 valid clients  (vs Tier-1+2 only in control)")
    print(f"           log -> {log_path}")
    print(f"           checkpoint -> {BN_CKPT_DIR}/")
    print("=" * 72)
    print()

    run(cfg)

    print()
    print(f"[done] BN Ablation complete. Log: {log_path}")

    if not args.no_compare:
        print_comparison(log_path)


if __name__ == "__main__":
    main()
