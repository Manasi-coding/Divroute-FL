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
                        # Switched from BatchNorm to GroupNorm: BatchNorm running-stats get
                        # corrupted by non-IID cross-client averaging under this pipeline's
                        # 100-client, alpha=0.9 Dirichlet partition (see model.py's bn_mode
                        # design notes — "default" BatchNorm aggregates running_mean/running_var
                        # via sample-weighted averaging across clients with very different local
                        # distributions, a documented FL instability). GroupNorm computes
                        # normalization stats per-example, no cross-client BN buffer aggregation
                        # involved. Already implemented and available via bn_mode; not previously
                        # tested at scale on this specific CIFAR-100 pipeline.
                        bn_mode: str = "groupnorm",
                        threshold_mode: str = "percentile",
                        preset: str = "divroute",
                        server_momentum: float = None,
                        error_feedback: bool = None,
                        smoke_test: bool = False) -> Config:
    """Return the Phase-5 Config with every field set explicitly."""
    cfg = Config(
        # â”€â”€ Checkpoint isolation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # run_label overrides the auto-derived method_name so this experiment
        # writes to checkpoints/divroute_phase5_revised_cifar100_seed<N>/
        # and never conflicts with legacy divroute or fedavg checkpoints.
        run_label="divroute_phase5_revised",

        # â”€â”€ Experiment identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,

        # ──────────────────────────────────────────────────────────────────
        num_clients=100,
        clients_per_round=20,
        num_rounds=10 if smoke_test else num_rounds,

        # ──────────────────────────────────────────────────────────────────
        alpha=alpha,
        
        # ── Ablations ────────────────────────────────────────────────────────
        bn_mode=bn_mode,
        threshold_mode=threshold_mode,

        # ── Local training (instruction #8, #9) ──────────────────────────────
        local_epochs=3,          # STRICTLY 3 every round
        local_lr=0.1,            # SGD learning rate
        batch_size=32,
        use_epoch_warmup=False,  # instruction #8: no warmup ramp

        # ── Validation split (needed for hybrid loss-improvement score) ──────
        local_val_fraction=0.10,  # 10% of each client shard held out for val

        # â”€â”€ Divergence / routing (instructions #1-#6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Aggregation / server (instruction #7) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        use_divergence_weighting=True,
        divergence_weight_mode="sqrt",
        # Disabled: 20-round diagnostic (2026-08-24) confirmed this regresses CIFAR-100
        # accuracy from ~10.5% to ~1.1% at round 20, consistent with documented CIFAR-10
        # findings (-15.9pp / -3.8pp).
        use_server_momentum=False,
        server_momentum=0.9,
        server_lr=1.0,
        server_clip_updates=False,

        # â”€â”€ Compression (instruction #10 â€” keep unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        k_ratio_tier1=0.20,
        k_ratio_tier2=0.05,
        use_adaptive_k=False,       # static ratios
        use_k_warmup=False,         # no warmup ramp on k either

        # ── Compression Features ────────────────────────────────────────────────
        # Disabled: 20-round diagnostic (2026-08-24) confirmed this regresses CIFAR-100
        # accuracy from ~10.5% to ~1.1% at round 20, consistent with documented CIFAR-10
        # findings (-15.9pp / -3.8pp).
        use_error_feedback=False,
        error_feedback_momentum=0.9,
        use_layerwise_topk=True,
        use_tier3_sync=False,
        fedsparse_lambda=0.0,
        fedzip_actual_mode=False,
        uniform_top5_mode=False,
        fedavg_baseline_mode=False,

        # â”€â”€ FedNTD off â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ntd_beta=0.0,

        # â”€â”€ Selection weight decay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        gamma=0.85,

        # â”€â”€ Tier-3 warm-up is handled inside main.py (first 15 rounds) â”€â”€â”€â”€â”€â”€â”€â”€
        include_tier3_in_aggregation=False,

        # â”€â”€ Diagnostics (instruction #13) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        enable_routing_quality_analysis=True,
        enable_gradient_diagnostics=True,
        ablation_per_client_logging=True,
        diag_print_interval=25,      # full DIAG block every 25 rounds in full run
    )

    # Apply presets
    if preset.startswith("fedavg"):
        cfg.fedavg_baseline_mode = True
        cfg.ablation_routing_no_compression = False
    elif preset == "uniform_topk":
        cfg.ablation_uniform_compression = True
        cfg.ablation_uniform_k_ratio = 0.20
    elif preset == "divroute_nocomp":
        cfg.ablation_routing_no_compression = True
    elif preset.startswith("divroute"):
        pass

    if preset.endswith("localbn"):
        cfg.bn_mode = "local_bn"
    elif preset.endswith("gn"):
        cfg.bn_mode = "groupnorm"

    if preset.endswith("alpha03"):
        cfg.alpha = 0.3

    if preset == "divroute_ef":
        cfg.use_error_feedback = True

    # Explicit overrides if provided
    if server_momentum is not None:
        cfg.use_server_momentum = (server_momentum > 0.0)
        cfg.server_momentum = server_momentum
        
    if error_feedback is not None:
        cfg.use_error_feedback = error_feedback

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DivRoute-FL Phase-5 Revised â€” CIFAR-100 / ResNet-18"
    )
    parser.add_argument("--num_rounds", type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dataset",    type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--preset",     type=str, default="divroute", 
                        choices=["fedavg", "fedavg_localbn", "uniform_topk", "divroute_nocomp", "divroute", "divroute_ef", "fedavg_gn", "divroute_gn", "fedavg_alpha03", "divroute_alpha03"])
    parser.add_argument("--alpha",      type=float, default=0.9)
    # Switched from BatchNorm to GroupNorm: BatchNorm running-stats get corrupted by
    # non-IID cross-client averaging under this pipeline's 100-client, alpha=0.9
    # Dirichlet partition (see model.py's bn_mode design notes — "default" BatchNorm
    # aggregates running_mean/running_var via sample-weighted averaging across clients
    # with very different local distributions, a documented FL instability). GroupNorm
    # computes normalization stats per-example, no cross-client BN buffer aggregation
    # involved. Already implemented and available via bn_mode; not previously tested
    # at scale on this specific CIFAR-100 pipeline.
    parser.add_argument("--bn_mode",    type=str, default="groupnorm", choices=["default", "local_bn", "groupnorm", "ws_groupnorm"])
    parser.add_argument("--threshold_mode", type=str, default="percentile", choices=["fixed", "adaptive_tau", "percentile"])
    parser.add_argument("--momentum",   type=float, default=None, choices=[0.0, 0.5, 0.9])
    parser.add_argument("--error_feedback", action="store_true", default=None)
    parser.add_argument("--no_error_feedback", action="store_false", dest="error_feedback", default=None)
    parser.add_argument("--smoke_test", action="store_true",
                        help="10-round sanity check; does NOT launch full training")
    parser.add_argument("--diag_every", type=int, default=None,
                        help="Override diag_print_interval (default: 25 full / 1 smoke)")
    args = parser.parse_args()

    model_name = "simplecnn" if args.dataset == "cifar10" else "resnet18"
    config = build_phase5_config(
        num_rounds=args.num_rounds,
        seed=args.seed,
        dataset_name=args.dataset,
        model_name=model_name,
        alpha=args.alpha,
        bn_mode=args.bn_mode,
        threshold_mode=args.threshold_mode,
        preset=args.preset,
        server_momentum=args.momentum,
        error_feedback=args.error_feedback,
        smoke_test=args.smoke_test,
    )

    # Override diag interval for smoke / override flag
    if args.smoke_test:
        config.diag_print_interval = 1   # print every round during smoke test
    if args.diag_every is not None:
        config.diag_print_interval = args.diag_every

    suffix = "smoke" if args.smoke_test else "full"
    config.log_path = (
        f"./logs/phase5_{args.preset}_seed{args.seed}"
        f"_{config.dataset_name}"
        f"_alpha{config.alpha}"
        f"_bn{config.bn_mode}"
        f"_thresh{config.threshold_mode}"
        f"_{suffix}.json"
    )

    # â”€â”€ Startup banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("=" * 60)
    print("  DivRoute-FL Phase-5 (Revised) â€” Startup Configuration")
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
    print(f"  bn_mode        : {config.bn_mode}")
    print(f"  threshold_mode : {config.threshold_mode}")
    print(f"  error_feedback : {config.use_error_feedback}")
    print(f"  log_path       : {config.log_path}")
    print("=" * 60 + "\n")

    run(config)


if __name__ == "__main__":
    main()
