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

    # divergence thresholds
    # if use_adaptive_tau=True, tau_low/tau_high are recomputed each round
    # using percentiles of the current round's divergence distribution
    tau_low: float = 0.01
    tau_high: float = 0.02
    use_adaptive_tau: bool = True
    tau_alpha: float = 0.5       # tau_low = mu - alpha * sigma
    tau_beta: float = 1.0        # tau_high = mu + beta * sigma

    # EMA smoothing for divergence scores
    ema_beta: float = 0.6

    # top-k compression ratios per tier
    k_ratio_tier1: float = 0.20
    k_ratio_tier2: float = 0.05
    use_k_warmup: bool = True
    k_warmup_rounds: int = 30
    k_warmup_tier1: float = 0.70
    k_warmup_tier2: float = 0.30
    use_adaptive_k: bool = False     # validated: static ratios outperform decay schedule

    # When True, Tier-3 clients participate in aggregation (compressed at k_ratio_tier2)
    # instead of being excluded. Used for ablation experiments.
    include_tier3_in_aggregation: bool = False

    # layer-wise dynamic top-k
    use_layerwise_topk: bool = False
    layerwise_ema_beta: float = 0.9

    # error feedback (Phase 1.1)
    use_error_feedback: bool = False   # validated: EF decreases accuracy at k_ratio_tier2=0.05

    # Tier-3 staleness sync (Phase 1.2, Option A — periodic heartbeat)
    use_tier3_sync: bool = False       # validated: zero contribution at 20-round horizon
    tier3_sync_interval: int = 5     # every Nth round, Tier-3 clients get a tier-2 heartbeat

    # aggregation
    use_divergence_weighting: bool = True
    # weighting formula: "inverse" (1/d), "sqrt" (1/sqrt(d)), "exp" (exp(-d)), "softmax"
    divergence_weight_mode: str = "sqrt"   # sqrt is more stable than raw inverse
    use_server_momentum: bool = False  # validated: β=0.9 with η=1.0 → 10× effective step
    server_momentum: float = 0.9
    server_lr: float = 1.0
    server_clip_updates: bool = False
    grad_clip_norm: float = 10.0

    # FedNTD: Not-True Distillation (NeurIPS 2022)
    # Preserves global knowledge about non-true classes during local training.
    # beta=0.0 disables NTD. Recommended: beta=1.0, tau=3.0 for CIFAR-100.
    ntd_beta: float = 0.0
    ntd_tau: float = 3.0

    # selection weight decay
    gamma: float = 0.85

    # Local epoch warmup: ramps from E//2 to E over the first 15 rounds.
    # Set False for baselines/ablations to use fixed local_epochs every round.
    use_epoch_warmup: bool = True

    # Baseline mode: disables all DivRoute features to reproduce vanilla FedAvg.
    # k_ratio=1.0, tau thresholds below any real score, gamma=1, no momentum/EF/sync.
    fedavg_baseline_mode: bool = False

    # Uniform Top-5% baseline: every selected client receives a top-k(0.05)
    # compressed update, with no divergence scoring, no tier assignment,
    # and no adaptive routing. Used as a communication-matched baseline
    # to isolate the contribution of DivRoute's routing intelligence.
    uniform_top5_mode: bool = False

    # FedSparse baseline (Phase 5): L1 proximity regularisation added to each
    # client's loss to encourage sparse gradient updates.
    # lambda=0.0 (default) → standard training, no regularisation.
    # lambda>0  → FedSparse mode; recommended values: 0.01, 0.04.
    fedsparse_lambda: float = 0.0
    # FedSparse upload sparsification options (Phase 5)
    fedsparse_sparsify_upload: bool = False
    fedsparse_threshold: float = 1e-4

    # FedZip baseline options (Phase 5)
    fedzip_actual_mode: bool = False
    fedzip_z_ratio: float = 0.01
    fedzip_k_clusters: int = 3



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

    # ── Local validation split (diagnostic only) ─────────────────────────────
    # Fraction of each client's shard held out as a local validation set.
    # Purpose: measure whether high local train accuracy reflects genuine
    # generalisation or overfitting, without modifying any training logic.
    #
    # 0.0  → disabled.  get_client_datasets() is called directly and the
    #         execution path is *identical* to all pre-existing experiments.
    #         Use this value when reproducing any existing result.
    #
    # 0.1  → 90% train / 10% val split per client (recommended for diagnosis).
    #
    # !! COMPATIBILITY WARNING !!
    # Setting local_val_fraction > 0 creates a NEW experimental configuration.
    # Validation samples are *removed* from local training, so clients train on
    # fewer samples than before.  Do NOT enable this when resuming or continuing
    # a checkpoint from a run where local_val_fraction == 0.0 — doing so would
    # change the training-data distribution mid-run and invalidate the results.
    # ─────────────────────────────────────────────────────────────────────────
    local_val_fraction: float = 0.0

    # ── [DIAG] Routing Diagnostics ───────────────────────────────────────────
    enable_routing_diagnostics: bool = True
    # Routing score used for tier assignment.
    # "raw"  -> compare d_raw against tau (consistent: tau is also from d_raw)
    # "ema"  -> compare d_ema against tau (legacy behaviour: smooth but biased)
    # Aggregation weighting always uses d_ema regardless of this setting.
    routing_score: str = "raw"    # "raw" | "ema"
    # ─────────────────────────────────────────────────────────────────────────


    # ── Dataset / model selection (Phase 3 scalability) ──────────────────────
    # Defaults preserve the original CIFAR-10 / SimpleCNN behaviour exactly.
    dataset_name: str = "cifar10"    # "cifar10" | "cifar100"
    model_name:   str = "simplecnn"  # "simplecnn" | "resnet18"


    # ── [PART 8] Scientific Instrumentation Options ──────────────────────────
    # ALL flags default to False / "cosine" so that existing experiments are
    # completely unaffected when these fields are absent or at their defaults.

    # PART 1: Gradient Quality Diagnostics
    # When True, compute and log per-client update norms, relative update norms,
    # compression reconstruction error, retention ratio, and cosine similarity
    # between original and compressed updates.  Print every `diag_print_interval`
    # rounds and write to the JSON log.
    enable_gradient_diagnostics: bool = False

    # PART 2: Alternative Divergence Metric
    # Selects the metric used to measure client drift from the global model.
    # "cosine"        - existing implementation (1 - cosine_sim).  Default.
    # "l2"            - raw L2 norm of the parameter update.
    # "relative_l2"   - L2 norm normalised by global model norm.
    # "layerwise_cosine" - mean cosine divergence over all trainable layers.
    # Changing this requires NO modification to aggregation / compression logic.
    divergence_metric: str = "cosine"   # "cosine" | "l2" | "relative_l2" | "layerwise_cosine"

    # PART 3: Routing Quality Evaluation
    # When True, log per-client (tier, divergence, local_loss, update_norm,
    # aggregation_weight, compressed_norm) and compute Pearson/Spearman
    # correlations between divergence and local accuracy / loss / update norm.
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
    # ─────────────────────────────────────────────────────────────────────────


def get_recommended_divroute_config(**overrides) -> Config:
    """
    Returns a Config pre-set to the validated DivRoute operating point:

        Accuracy : ~64.1%  (CIFAR-10, alpha=0.9, 20 rounds)
        Comm saving: ~84%  bidirectional vs FedAvg

    Key findings that shaped these defaults:
        - use_server_momentum=False : β=0.9 / η=1.0 creates 10× effective step → −15.9pp
        - use_error_feedback=False  : EF at k=0.05 accumulates stale residuals → −3.8pp
        - use_adaptive_k=False      : late-round ratio decay harms convergence → −3.5pp
        - use_tier3_sync=False      : heartbeat adds 0.00pp over 20 rounds

    Any keyword argument in `overrides` is forwarded to Config(), allowing
    individual flags to be overridden for ablation experiments.
    """
    base = dict(
        use_server_momentum       = False,
        use_error_feedback        = False,
        use_tier3_sync            = False,
        use_adaptive_k            = False,
        include_tier3_in_aggregation = False,
        use_divergence_weighting  = True,
        use_adaptive_tau          = True,
        use_epoch_warmup          = True,
        divergence_weight_mode    = "sqrt",
        ntd_beta                  = 0.1,
        k_ratio_tier1             = 0.20,
        k_ratio_tier2             = 0.05,
        use_layerwise_topk        = True,
        ema_beta                  = 0.85,
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
        uniform_top5_mode         = True,
        use_server_momentum       = False,
        use_error_feedback        = False,
        use_tier3_sync            = False,
        use_adaptive_k            = False,
        use_divergence_weighting  = False,   # no divergence scores to weight by
        use_adaptive_tau          = False,   # no tier assignment to threshold
        use_epoch_warmup          = False,
        k_ratio_tier1             = 0.05,    # unused — all clients routed as Tier-2
        k_ratio_tier2             = 0.05,    # the single uniform compression ratio
    )
    base.update(overrides)
    return Config(**base)