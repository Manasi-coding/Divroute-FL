"""
DivRoute-FL Phase-5 (Revised) Runner
=====================================
Explicit configuration for the CIFAR-100 / ResNet-18 Phase-5 experiment.
All values are set explicitly here so this script is self-contained and
does NOT rely on get_recommended_divroute_config() or any implicit defaults.

Key changes vs. previous DivRoute-FL run:
  - Directional divergence (delta-space cosine) instead of weight-space cosine
  - EMA-smoothed hybrid routing score (dir_div 70% + loss_improvement 30%)
  - ema_beta = 0.95 (slow smoothing as specified)
  - Rolling-window tau (window=5) with EMA smoothing (alpha=0.80)
  - Server momentum ON (beta=0.9, lr=1.0)
  - use_epoch_warmup = False  (fixed 3 epochs every round)
  - use_adaptive_k  = False  (static k_ratio_tier1=0.20, tier2=0.05)
  - use_error_feedback = False
  - Distinct checkpoint dir: divroute_phase5_revised_cifar100_seed42
"""
import argparse
from divroute_fl.config import Config
from divroute_fl.main import run


def build_phase5_config(num_rounds: int, seed: int,
                        dataset_name: str = "cifar100",
                        model_name: str = "resnet18",
                        alpha: float = 0.9,
                        smoke_test: bool = False) -> Config:
    """Return the Phase-5 Config with every field set explicitly."""
    cfg = Config(
        # ── Checkpoint isolation ──────────────────────────────────────────────
        # run_label overrides the auto-derived method_name so this experiment
        # writes to checkpoints/divroute_phase5_revised_cifar100_seed<N>/
        # and never conflicts with legacy divroute or fedavg checkpoints.
        run_label="divroute_phase5_revised",

        # ── Experiment identity ───────────────────────────────────────────────
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,

        # ── Federation setup ──────────────────────────────────────────────────
        num_clients=100,
        clients_per_round=20,
        num_rounds=10 if smoke_test else num_rounds,

        # ── Non-IID partitioning ──────────────────────────────────────────────
        alpha=alpha,

        # ── Local training (instruction #8, #9) ──────────────────────────────
        local_epochs=3,          # STRICTLY 3 every round
        local_lr=0.1,            # SGD learning rate
        batch_size=32,
        use_epoch_warmup=False,  # instruction #8: no warmup ramp

        # ── Validation split (needed for hybrid loss-improvement score) ───────
        local_val_fraction=0.10,  # 10% of each client shard held out for val

        # ── Divergence / routing (instructions #1-#6) ────────────────────────
        use_directional_divergence=True,   # delta-space cosine, not weight-space
        use_divergence_ema=True,           # EMA is the actual routing signal
        ema_beta=0.95,                     # slow smoothing (instruction #4)

        # Hybrid weights (instruction #3)
        directional_div_weight=0.70,
        loss_improvement_weight=0.30,

        # Rolling-window + smoothed tau (instruction #5)
        use_adaptive_tau=True,
        tau_alpha=0.5,
        tau_beta=1.0,
        tau_window=5,
        tau_smoothing=0.80,

        # ── Aggregation / server (instruction #7) ────────────────────────────
        use_divergence_weighting=True,
        divergence_weight_mode="sqrt",
        use_server_momentum=True,
        server_momentum=0.9,
        server_lr=1.0,
        server_clip_updates=False,

        # ── Compression (instruction #10 — keep unchanged) ───────────────────
        k_ratio_tier1=0.20,
        k_ratio_tier2=0.05,
        use_adaptive_k=False,       # static ratios
        use_k_warmup=False,         # no warmup ramp on k either

        # ── Features to keep OFF (instruction #10) ───────────────────────────
        use_error_feedback=False,
        use_tier3_sync=False,
        fedsparse_lambda=0.0,
        fedzip_actual_mode=False,
        uniform_top5_mode=False,
        fedavg_baseline_mode=False,

        # ── FedNTD off ────────────────────────────────────────────────────────
        ntd_beta=0.0,

        # ── Selection weight decay ────────────────────────────────────────────
        gamma=0.85,

        # ── Tier-3 warm-up is handled inside main.py (first 15 rounds) ────────
        include_tier3_in_aggregation=False,

        # ── Diagnostics (instruction #13) ────────────────────────────────────
        enable_routing_diagnostics=True,
        enable_gradient_diagnostics=True,
        ablation_per_client_logging=True,
        diag_print_interval=25,      # full DIAG block every 25 rounds in full run
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DivRoute-FL Phase-5 Revised — CIFAR-100 / ResNet-18"
    )
    parser.add_argument("--num_rounds", type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--alpha",      type=float, default=0.9)
    parser.add_argument("--smoke_test", action="store_true",
                        help="10-round sanity check; does NOT launch full training")
    parser.add_argument("--diag_every", type=int, default=None,
                        help="Override diag_print_interval (default: 25 full / 1 smoke)")
    args = parser.parse_args()

    config = build_phase5_config(
        num_rounds=args.num_rounds,
        seed=args.seed,
        alpha=args.alpha,
        smoke_test=args.smoke_test,
    )

    # Override diag interval for smoke / override flag
    if args.smoke_test:
        config.diag_print_interval = 1   # print every round during smoke test
    if args.diag_every is not None:
        config.diag_print_interval = args.diag_every

    suffix = "smoke" if args.smoke_test else "full"
    config.log_path = (
        f"./logs/phase5_revised_seed{args.seed}"
        f"_{config.dataset_name}"
        f"_alpha{args.alpha}"
        f"_{suffix}.json"
    )

    # ── Startup banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DivRoute-FL Phase-5 (Revised) — Startup Configuration")
    print("=" * 60)
    print(f"  dataset        : {config.dataset_name}")
    print(f"  model          : {config.model_name}")
    print(f"  seed           : {config.seed}")
    print(f"  rounds         : {config.num_rounds}")
    print(f"  clients        : {config.num_clients}  ({config.clients_per_round}/round)")
    print(f"  local_epochs   : {config.local_epochs}  (warmup={config.use_epoch_warmup})")
    print(f"  local_lr       : {config.local_lr}")
    print(f"  batch_size     : {config.batch_size}")
    print(f"  alpha (Dirich) : {config.alpha}")
    print(f"  val_fraction   : {config.local_val_fraction}")
    print(f"  directional    : {config.use_directional_divergence}")
    print(f"  ema_routing    : {config.use_divergence_ema}  (beta={config.ema_beta})")
    print(f"  dir_weight     : {config.directional_div_weight}")
    print(f"  loss_weight    : {config.loss_improvement_weight}")
    print(f"  tau_window     : {config.tau_window}")
    print(f"  tau_smoothing  : {config.tau_smoothing}")
    print(f"  server_mom     : {config.use_server_momentum}  (beta={config.server_momentum}  lr={config.server_lr})")
    print(f"  k_ratio        : tier1={config.k_ratio_tier1}  tier2={config.k_ratio_tier2}  adaptive={config.use_adaptive_k}")
    print(f"  error_feedback : {config.use_error_feedback}")
    print(f"  log_path       : {config.log_path}")
    print("=" * 60 + "\n")

    run(config)


if __name__ == "__main__":
    main()
