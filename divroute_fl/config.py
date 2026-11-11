from dataclasses import dataclass


@dataclass
class Config:
    # federation setup
    num_clients: int = 25
    clients_per_round: int = 15
    num_rounds: int = 20

    # local training
    local_epochs: int = 5
    local_lr: float = 0.1
    batch_size: int = 32

    # non-IID partitioning (Dirichlet alpha — lower = more heterogeneous)
    alpha: float = 0.9

    partition_mode: str = "dirichlet"
    shards_per_client: int = 2

    # divergence thresholds
    # if use_adaptive_tau=True, tau_low/tau_high are recomputed each round
    # using percentiles of the current round's divergence distribution
    tau_low: float = 0.01
    tau_high: float = 0.02
    use_adaptive_tau: bool = True
    tau_alpha: float = 0.5  # tau_low = mu - alpha * sigma
    tau_beta: float = 1.0  # tau_high = mu + beta * sigma

    tau_window: int = 5  # number of past rounds to keep in rolling window
    tau_smoothing: float = 0.80  # EMA coefficient applied to candidate tau each round
    # 0 = instant (no smoothing), 1 = frozen

    # EMA smoothing for divergence scores
    ema_beta: float = 0.85

    use_divergence_ema: bool = False

    use_directional_divergence: bool = False

    directional_div_weight: float = 0.50
    loss_improvement_weight: float = 0.50

    # top-k compression ratios per tier
    k_ratio_tier1: float = 0.20
    k_ratio_tier2: float = 0.05
    use_k_warmup: bool = True
    k_warmup_rounds: int = 30
    k_warmup_tier1: float = 0.70
    k_warmup_tier2: float = 0.30
    use_adaptive_k: bool = False

    # When True, Tier-3 clients participate in aggregation (compressed at k_ratio_tier2)
    # instead of being excluded. Used for ablation experiments.
    include_tier3_in_aggregation: bool = False

    # layer-wise dynamic top-k
    use_layerwise_topk: bool = False
    layerwise_ema_beta: float = 0.9

    use_error_feedback: bool = False
    error_feedback_momentum: float = 0.9  # momentum factor for residual accumulation

    error_feedback_tier_aware: bool = False

    use_tier3_sync: bool = False
    tier3_sync_interval: int = (
        5  # every Nth round, Tier-3 clients get a tier-2 heartbeat
    )

    # aggregation
    use_divergence_weighting: bool = True
    # weighting multiplier on sample share: "sqrt" = sqrt(d)+eps (higher divergence -> more weight), "inverse" = 1/(d+eps), "exp" = exp(-10d), "softmax" = softmax(-50d)
    divergence_weight_mode: str = "sqrt"  # sqrt is more stable than raw inverse
    use_server_momentum: bool = False
    server_momentum: float = 0.9
    server_lr: float = 1.0
    server_clip_updates: bool = False
    grad_clip_norm: float = 10.0

    # FedNTD: Not-True Distillation (NeurIPS 2022)
    # Preserves global knowledge about non-true classes during local training.
    # beta=0.0 disables NTD. Recommended: beta=1.0, tau=3.0 for CIFAR-100.
    ntd_beta: float = 0.0
    ntd_tau: float = 3.0

    use_fedprox: bool = False
    fedprox_mu: float = 0.01

    # selection weight decay
    gamma: float = 0.85

    # Local epoch warmup: ramps from E//2 to E over the first 15 rounds.
    # Set False for baselines/ablations to use fixed local_epochs every round.
    use_epoch_warmup: bool = True

    # Baseline mode: disables all DivRoute features to reproduce vanilla FedAvg.
    # k_ratio=1.0, tau thresholds below any real score, gamma=1, no momentum/EF/sync.
    fedavg_baseline_mode: bool = False

    uniform_top5_mode: bool = False

    fedsparse_lambda: float = 0.0

    fedsparse_sparsify_upload: bool = False
    fedsparse_threshold: float = 1e-4

    fedzip_actual_mode: bool = False
    fedzip_z_ratio: float = 0.01
    fedzip_k_clusters: int = 3

    ablation_routing_no_compression: bool = False

    ablation_uniform_compression: bool = False
    ablation_uniform_k_ratio: float = 0.05  # mirrors DivRoute's Tier-2 ratio

    ablation_per_client_logging: bool = False

    invert_tier_polarity: bool = False

    # Control: shuffle tier labels among the round's clients (same tier counts
    # and bytes). Isolates whether divergence carries routing signal.
    random_tier_assignment: bool = False

    dagc_mean_k: float = 0.0
    dagc_direction: int = 1
    dagc_shuffle: bool = False
    # Console-only routing diagnostics block in main.py (was read via getattr but not a field).
    enable_routing_diagnostics: bool = False

    run_label: str = ""

    # Set True to suppress the interactive "Generate plots?" prompt.
    # Required for automated / multi-run scripts.
    skip_plot_prompt: bool = False

    # Checkpoint and resume options
    resume: bool = False
    fresh: bool = False
    checkpoint_dir: str = "checkpoints"

    # FP16 Transmission Options
    use_fp16_upload: bool = False
    use_fp16_download: bool = False

    # parallelism (set to 0 to disable multiprocessing — sequential on GPU)
    num_workers: int = 0

    seed: int = 42
    log_path: str = "logs/run.json"

    local_val_fraction: float = 0.0

    # If True, triggers threshold_mode="adaptive_tau" logic (legacy compat).
    # Modes: "fixed", "adaptive_tau", "percentile", "rolling_percentile"
    threshold_mode: str = "adaptive_tau"
    rolling_window_size: int = 5
    convergence_floor: float = 1e-4

    bn_mode: str = "default"

    routing_score: str = "raw"  # "raw" | "ema"

    # Defaults preserve the original CIFAR-10 / SimpleCNN behaviour exactly.
    dataset_name: str = "cifar10"  # "cifar10" | "cifar100"
    model_name: str = (
        "simplecnn"  # "simplecnn" | "resnet18" | "efficientnet_b0_pretrained"
    )

    local_lr_min: float = 0.001

    # ALL flags default to False / "cosine" so that existing experiments are
    # completely unaffected when these fields are absent or at their defaults.

    enable_gradient_diagnostics: bool = False

    divergence_metric: str = "cosine"  # "cosine" | "l2" | "relative_l2" | "layerwise_cosine" | "num_samples" (routing control)

    enable_routing_quality_analysis: bool = False

    # PART 4 + 5: Tier Contribution and Compression Analysis
    # When True, compute total update norm separately for T1/T2/T3, and measure
    # per-tier compression error before and after compression.
    enable_compression_analysis: bool = False

    # PART 6: Adaptive Tau Diagnostics
    # When True, log mu, sigma, tau_low, tau_high, spread, coefficient of
    # variation, and per-tier client counts every round.
    enable_tau_diagnostics: bool = False

    # Interval (in rounds) at which diagnostic summaries are printed to stdout.
    # Has no effect when all diagnostic flags are False.
    diag_print_interval: int = 25


def get_recommended_divroute_config(**overrides) -> Config:
    """Returns a Config pre-set to the validated DivRoute operating point:"""
    base = dict(
        use_server_momentum=False,
        use_error_feedback=False,
        use_tier3_sync=False,
        use_adaptive_k=False,
        include_tier3_in_aggregation=False,
        use_divergence_weighting=True,
        use_adaptive_tau=True,
        use_epoch_warmup=True,
        divergence_weight_mode="sqrt",
        ntd_beta=0.1,
        k_ratio_tier1=0.20,
        k_ratio_tier2=0.05,
        use_layerwise_topk=True,
        ema_beta=0.85,
    )
    base.update(overrides)
    return Config(**base)


def get_uniform_top5_config(**overrides) -> Config:
    """
    Returns a Config for the Uniform Top-5% baseline.

    Every selected client transmits exactly top-5% of its gradient delta.
    No divergence scoring, no tier assignment, no adaptive routing.
    Communication volume matches DivRoute Tier-2 for all clients.

    This baseline answers: does DivRoute's routing intelligence
    outperform naive uniform compression at the same budget?

    Any keyword argument in `overrides` is forwarded to Config().
    """
    base = dict(
        uniform_top5_mode=True,
        use_server_momentum=False,
        use_error_feedback=False,
        use_tier3_sync=False,
        use_adaptive_k=False,
        use_divergence_weighting=False,  # no divergence scores to weight by
        use_adaptive_tau=False,  # no tier assignment to threshold
        use_epoch_warmup=False,
        k_ratio_tier1=0.05,  # unused — all clients routed as Tier-2
        k_ratio_tier2=0.05,  # the single uniform compression ratio
    )
    base.update(overrides)
    return Config(**base)


def get_pretrained_finetune_config(**overrides) -> Config:
    """Config for DivRoute's pretrained-backbone fine-tuning run — the "main"""
    base = get_recommended_divroute_config().__dict__.copy()
    base.update(
        dataset_name="cifar100",
        model_name="efficientnet_b0_pretrained",
        bn_mode="default",
        include_tier3_in_aggregation=True,
        local_lr=0.01,
        local_lr_min=0.001,
        threshold_mode="rolling_percentile",
        convergence_floor=1e-6,
        use_k_warmup=False,
    )
    base.update(overrides)
    return Config(**base)


def get_fedavg_pretrained_config(**overrides) -> Config:
    """Config for the "Full FedAvg" companion baseline (master plan §6) under the
    SAME pretrained-backbone fine-tuning condition as
    get_pretrained_finetune_config() — the iso-condition accuracy reference
    that the DivRoute pretrained run must be compared against for
    comm_vs_accuracy.png to be a valid claim."""
    base = dict(
        fedavg_baseline_mode=True,
        include_tier3_in_aggregation=True,
        threshold_mode="fixed",
        dataset_name="cifar100",
        model_name="efficientnet_b0_pretrained",
        bn_mode="default",
        local_lr=0.01,
        local_lr_min=0.001,
        use_k_warmup=False,
        ntd_beta=0.1,
    )
    base.update(overrides)
    return Config(**base)


def get_uniform_top5_pretrained_config(**overrides) -> Config:
    """Config for the "Uniform Top-5%" companion baseline (master plan §6) under"""
    base = get_uniform_top5_config().__dict__.copy()
    base.update(
        dataset_name="cifar100",
        model_name="efficientnet_b0_pretrained",
        bn_mode="default",
        local_lr=0.01,
        local_lr_min=0.001,
        use_k_warmup=False,
        ntd_beta=0.1,  # see get_fedavg_pretrained_config()'s matching note
        use_epoch_warmup=True,
    )
    base.update(overrides)
    return Config(**base)


def _femnist_common() -> dict:
    return dict(
        dataset_name="femnist",
        model_name="femnist_cnn",
        bn_mode="default",  # FemnistCNN has no BatchNorm; ignored
        partition_mode="natural",  # one client = one real writer
        num_clients=100,
        clients_per_round=15,
        num_rounds=150,
        # lr chosen on dense FedAvg only (60 rounds, seed 42, plateau = last
        # 10 rounds): 0.01 -> 77.77%, 0.03 -> 79.42%, 0.1 -> 79.10%. A coarse
        # choice: 0.03's margin over 0.1 is about one round-to-round std.
        local_lr=0.03,
        local_lr_min=0.003,
        use_k_warmup=False,
        use_epoch_warmup=True,
        ntd_beta=0.1,
    )


def get_femnist_fedavg_config(**overrides) -> Config:
    """Dense FedAvg on FEMNIST (from scratch). Same data/model/optimisation as
    the two compressed methods. Set run_label via overrides (num_rounds
    defaults to 150, see _femnist_common())."""
    base = _femnist_common()
    base.update(
        fedavg_baseline_mode=True,
        include_tier3_in_aggregation=True,  # required, see get_fedavg_pretrained_config()
        threshold_mode="fixed",
    )
    base.update(overrides)
    return Config(**base)


def get_femnist_uniform_config(**overrides) -> Config:
    """Uniform Top-5% + error feedback on FEMNIST (from scratch)."""
    base = get_uniform_top5_config().__dict__.copy()
    base.update(_femnist_common())
    base.update(use_error_feedback=True)
    base.update(overrides)
    return Config(**base)


def get_femnist_divroute_config(**overrides) -> Config:
    """DivRoute (routing + tiered top-k) + plain error feedback on FEMNIST
    (from scratch). Thresholds mirror get_pretrained_finetune_config()."""
    base = get_recommended_divroute_config().__dict__.copy()
    base.update(_femnist_common())
    base.update(
        use_error_feedback=True,
        include_tier3_in_aggregation=True,
        threshold_mode="rolling_percentile",
        convergence_floor=1e-6,
    )
    base.update(overrides)
    return Config(**base)


def _cifar10s_common() -> dict:
    return dict(
        dataset_name="cifar10",
        model_name="simplecnn",
        bn_mode="default",  # SimpleCNN has no BatchNorm; ignored
        partition_mode="dirichlet",
        alpha=0.1,
        num_clients=25,
        clients_per_round=15,
        num_rounds=100,  # provisional
        local_lr=0.03,  # provisional
        local_lr_min=0.003,
        use_k_warmup=False,
        use_epoch_warmup=True,
        ntd_beta=0.1,
    )


def get_cifar10s_fedavg_config(**overrides) -> Config:
    base = _cifar10s_common()
    base.update(
        fedavg_baseline_mode=True,
        include_tier3_in_aggregation=True,
        threshold_mode="fixed",
    )
    base.update(overrides)
    return Config(**base)


def get_cifar10s_uniform_config(**overrides) -> Config:
    base = get_uniform_top5_config().__dict__.copy()
    base.update(_cifar10s_common())
    base.update(use_error_feedback=True)
    base.update(overrides)
    return Config(**base)


def get_cifar10s_divroute_config(**overrides) -> Config:
    base = get_recommended_divroute_config().__dict__.copy()
    base.update(_cifar10s_common())
    base.update(
        use_error_feedback=True,
        include_tier3_in_aggregation=True,
        threshold_mode="rolling_percentile",
        convergence_floor=1e-6,
    )
    base.update(overrides)
    return Config(**base)
