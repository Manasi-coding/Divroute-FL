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

    # Rolling-window adaptive tau (Phase-5 revised).
    # Uses scores from the most recent tau_window rounds for statistics,
    # then applies exponential smoothing so thresholds do not jump.
    # Only active when use_adaptive_tau=True.
    tau_window: int = 5           # number of past rounds to keep in rolling window
    tau_smoothing: float = 0.80  # EMA coefficient applied to candidate tau each round
                                  # 0 = instant (no smoothing), 1 = frozen

    # EMA smoothing for divergence scores
    ema_beta: float = 0.85
    # When True, the EMA-smoothed routing score is the actual signal used for
    # tier assignment AND aggregation weighting (Phase-5 revised).
    # When False, routing uses d_raw (legacy behaviour).
    use_divergence_ema: bool = False

    # Directional divergence (Phase-5 revised).
    # When True, replaces the weight-space cosine divergence with a delta-space
    # cosine distance: div = 1 - cos(client_delta, previous_global_delta).
    # This decouples the routing signal from raw update magnitude.
    # When False, the original implementation is preserved exactly.
    use_directional_divergence: bool = False

    # Hybrid routing score weights (Phase-5 revised).
    # routing_score = dir_weight * normalised_directional_div
    #               + loss_weight * normalised_loss_improvement
    # Only active when use_directional_divergence=True AND local_val_fraction>0.
    # Must sum to 1.0; enforced at runtime.
    directional_div_weight:   float = 0.50
    loss_improvement_weight:  float = 0.50

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
    error_feedback_momentum: float = 0.9 # momentum factor for residual accumulation
    # Only meaningful when use_error_feedback=True. Default False preserves
    # the documented -3.8pp EF finding above exactly as measured (that
    # finding predates this flag and used the plain accumulate-always
    # behavior). When True, a client's residual buffer is reset to zero
    # whenever its tier (and therefore its k_ratio) changes since the
    # residual was last updated, instead of blending a residual accumulated
    # under a different compression target into the new one. FL literature
    # identifies this "stale error compensation" as a specific, documented
    # failure mode of error feedback under partial client participation;
    # DivRoute compounds it with round-to-round tier reassignment on top of
    # partial participation. See compression.py's apply_tiered_compression()
    # docstring for the full rationale.
    error_feedback_tier_aware: bool = False
    use_layerwise_topk: bool = False   # applies top-k independently per layer

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

    # ── [ABLATION] Experiment isolation flags ────────────────────────────────────
    # These flags implement the four ablation conditions (A–D) that isolate
    # the source of the CIFAR-100 accuracy loss.  They are applied at startup
    # in main.py and override routing/compression decisions ONLY — they do not
    # modify divergence formulas, adaptive-tau logic, client training, aggregation
    # mathematics, compression implementation, EMA behaviour, or checkpointing.
    #
    # Experiment A — FedAvg-equivalent sanity baseline
    #   Use fedavg_baseline_mode=True (existing validated path).  No new flag needed.
    #
    # Experiment B — DivRoute routing ON, compression OFF
    #   When True: tier assignment + divergence weighting run normally; however
    #   k_ratio is forced to 1.0 for EVERY tier so no coordinates are discarded.
    #   Purpose: isolate whether routing/weighting alone causes the accuracy drop.
    ablation_routing_no_compression: bool = False

    # Experiment C — Uniform compression, routing OFF
    #   When True: divergence scoring still runs (for diagnostic logging), but
    #   every selected client is assigned Tier-1 and the same k_ratio is used
    #   for all (ablation_uniform_k_ratio).  Divergence weighting is disabled.
    #   Purpose: test whether uniform compression at the same budget is safer
    #   than DivRoute's adaptive routing.
    ablation_uniform_compression: bool = False
    ablation_uniform_k_ratio:     float = 0.05   # mirrors DivRoute's Tier-2 ratio

    # Experiment D — Full DivRoute + dense per-client forensic logging
    #   When True: no algorithmic change; adds a detailed per-client table every
    #   round logging client_id, d_raw, d_ema, tier, k_ratio, selection weight,
    #   aggregation weight, local_loss, and update norm (where available).
    ablation_per_client_logging: bool = False

    # Experiment E — inverted tier-bandwidth polarity
    #   The live tier-assignment loop (main.py) gives the MOST bandwidth
    #   (tier=1, k_ratio_tier1) to the LEAST-divergent clients and compresses
    #   the MOST-divergent clients hardest (tier=3, k_ratio_tier2) -- the
    #   opposite of the plain-language design intent ("drifted clients need a
    #   strong correction signal" -> more bandwidth). See
    #   DIVROUTE_ACCURACY_MASTER_PLAN.md §3.1 (which flagged this mismatch but
    #   only fixed the aggregation-exclusion symptom, never this direction)
    #   and §14 (the fourth companion-run result this flag exists to test).
    #   When True: swaps which extreme gets tier=1 vs tier=3. Tier=2 (the
    #   middle band) and the NaN/Inf safety fallback are unaffected.
    #   Default False preserves all existing/validated behaviour exactly.
    invert_tier_polarity: bool = False

    # run_label — optional suffix that overrides the checkpoint sub-folder name.
    # When non-empty, main.py uses this label instead of the auto-derived method
    # name so each ablation experiment writes to its own isolated directory.
    # Example: "ablation_b_no_compression" → checkpoints/ablation_b_no_compression/
    # Leave empty ("") for normal operation (no behavioural change).
    run_label: str = ""
    # ─────────────────────────────────────────────────────────────────────────────



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

    # ── [Phase-5] Rolling-Window Adaptive Thresholds ─────────────────────────
    # If True, triggers threshold_mode="adaptive_tau" logic (legacy compat).
    # Modes: "fixed", "adaptive_tau", "percentile", "rolling_percentile"
    threshold_mode: str = "adaptive_tau"
    rolling_window_size: int = 5
    convergence_floor: float = 1e-4
    # bn_mode controls BatchNormalization behaviour. 
    # "default": Standard nn.BatchNorm2d, stats aggregated by server.
    # "local_bn": nn.BatchNorm2d, but running stats are excluded from global model.
    # "groupnorm": Replace nn.BatchNorm2d with nn.GroupNorm.
    # "ws_groupnorm": Replace nn.Conv2d with WSConv2d and nn.BatchNorm2d with nn.GroupNorm.
    bn_mode: str = "default"
    # Routing score used for tier assignment.
    # "raw"  -> compare d_raw against tau (consistent: tau is also from d_raw)
    # "ema"  -> compare d_ema against tau (legacy behaviour: smooth but biased)
    # Aggregation weighting always uses d_ema regardless of this setting.
    routing_score: str = "raw"    # "raw" | "ema"
    # ─────────────────────────────────────────────────────────────────────────


    # ── Dataset / model selection (Phase 3 scalability) ──────────────────────
    # Defaults preserve the original CIFAR-10 / SimpleCNN behaviour exactly.
    dataset_name: str = "cifar10"    # "cifar10" | "cifar100"
    model_name:   str = "simplecnn"  # "simplecnn" | "resnet18" | "efficientnet_b0_pretrained"

    # ── Pretrained-backbone fine-tuning support ───────────────────────────────
    # Only relevant when model_name="efficientnet_b0_pretrained". Image resize
    # and ImageNet normalisation are derived automatically from model_name
    # inside data.py — no separate flag needed for those, so they cannot drift
    # out of sync. This field controls the local-training LR schedule's floor:
    # local_lr decays via a half-cosine schedule from local_lr (round 0) down
    # to local_lr_min (final round) — see get_local_lr() in client.py. The
    # default (0.001) exactly matches client.train()'s previously-hardcoded
    # floor, so existing runs that don't reference this field are unaffected.
    # When fine-tuning a pretrained backbone, pair a lower local_lr (e.g. 0.01
    # instead of the from-scratch default of 0.1) with this floor — 0.1 is
    # tuned for random-init training and will damage pretrained features.
    local_lr_min: float = 0.001


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


def get_pretrained_finetune_config(**overrides) -> Config:
    """
    Config for DivRoute's pretrained-backbone fine-tuning run — the "main
    result" condition described in DIVROUTE_ACCURACY_MASTER_PLAN.md, targeting
    80-85% accuracy on CIFAR-100 (current from-scratch ceiling: 32.8% at round
    274/300 on ResNet-18; this project's own best validated from-scratch
    ceiling anywhere is ~64.1%, on the easier CIFAR-10, per
    get_recommended_divroute_config's docstring — see the plan doc for why
    from-scratch training cannot realistically reach 80-85% on CIFAR-100
    under this protocol, and why a pretrained backbone is the recommended fix).

    Starts from get_recommended_divroute_config()'s validated base — same
    mechanism (divergence formula, tier routing, top-k dispatch, byte
    accounting, gamma-decay), same k_ratio_tier1/tier2 — and changes only:
        - model_name: ResNet-18-from-scratch -> ImageNet-pretrained
          EfficientNet-B0 ("efficientnet_b0_pretrained"). data.py automatically
          resizes CIFAR-100 images and switches to ImageNet normalisation for
          this model_name — no separate flag needed.
        - bn_mode="default": explicitly preserves pretrained BatchNorm running
          statistics (do not set "groupnorm"/"ws_groupnorm" for this run —
          fine as a secondary ablation, but not the main result; see
          DIVROUTE_ACCURACY_MASTER_PLAN.md §4).
        - include_tier3_in_aggregation=True: the live tier-assignment loop
          (main.py) routes HIGH-divergence clients to Tier 3 and EXCLUDES
          them (0 bytes, 0 aggregation weight) — verified directly against
          main.py, see plan doc §3.1. This stops silently discarding the most
          locally-distinct clients' updates every round.
        - local_lr / local_lr_min: lowered for fine-tuning a pretrained
          backbone instead of training a randomly-initialised one from
          scratch — 0.1 (the from-scratch default) will damage pretrained
          features.

    Deliberately UNCHANGED from the validated base, despite general FL
    literature suggesting otherwise — do not flip these on without a
    dedicated re-validation run (see DIVROUTE_ALGORITHMS_MASTER_DOC.md §14):
        - use_error_feedback=False   (documented finding: -3.8pp at k=0.05)
        - use_server_momentum=False  (documented finding: -15.9pp, β=0.9/η=1.0
          effective step size bug)

    Caller should set run_label (via overrides) to something unique so this
    run writes to its own fresh checkpoint directory. The checkpoint/resume
    compatibility guard in main.py will correctly refuse to resume this run
    from the existing ResNet-18-from-scratch checkpoint (model_name mismatch)
    — that refusal is expected behaviour, not a bug to work around.

    NOT guaranteed by this config alone: a specific accuracy number. See
    DIVROUTE_ACCURACY_MASTER_PLAN.md §7 (guarantees vs. non-guarantees) and §9
    — a centralized (non-federated) sanity check with this same model/resize/
    normalisation pipeline, run before committing to the full federated
    experiment, is recommended there and is not automated by this function.

    Any keyword argument in `overrides` is forwarded to Config().
    """
    base = get_recommended_divroute_config().__dict__.copy()
    base.update(
        dataset_name                  = "cifar100",
        model_name                    = "efficientnet_b0_pretrained",
        bn_mode                       = "default",
        include_tier3_in_aggregation  = True,
        local_lr                      = 0.01,
        local_lr_min                  = 0.001,
        # tau_low/tau_high default to 0.01/0.02 and threshold_mode defaults
        # to "adaptive_tau" with tau_smoothing=0.8 -- tuned for from-scratch
        # ResNet-18 divergence scores, which run ~100x larger than a
        # pretrained-backbone fine-tune's (observed ~1e-4 vs the 0.01/0.02
        # tuned scale). adaptive_tau's EMA only decays 20%/round toward the
        # real distribution, so tau_low never gets within an order of
        # magnitude of the actual scores inside a 10- or even 20-round
        # budget -- every client's score stays below tau_low the whole run,
        # so every client is classified Tier 1 (confirmed empirically: a
        # real 10-round run showed "15/0/0" tier counts every single
        # round). Tier 2/3 compression never activates, defeating the
        # entire routing comparison this config exists to run.
        # rolling_percentile recomputes tau_low/tau_high directly from the
        # actual score distribution's shape every round (p33/p67 over a
        # rolling window) rather than decaying toward it from a stale
        # from-scratch default, so it is scale-independent.
        threshold_mode                = "rolling_percentile",
        # convergence_floor (default 1e-4) freezes tau updates once the
        # score spread drops below it -- also tuned for the from-scratch
        # scale. Pretrained-regime spreads observed so far are ~1e-5..1e-4,
        # i.e. already below the default floor, which would freeze tau
        # after round 1 even with rolling_percentile active. Lowered two
        # orders of magnitude so genuine convergence (not just a smaller
        # backbone) is what triggers the freeze.
        convergence_floor              = 1e-6,
        # use_k_warmup defaults to True with k_warmup_rounds=30 -- a from-
        # scratch-tuned schedule that ramps k_ratio_tier1/tier2 UP from
        # 0.70/0.30 to their final values over the first 30 rounds.
        # get_adaptive_k_ratios() (compression.py) checks use_k_warmup
        # BEFORE it ever looks at k_ratio_tier1/tier2, and main.py calls it
        # unconditionally every round regardless of fedavg_baseline_mode --
        # so for any companion run at <=20 rounds (the entire run falls
        # inside the warmup window), every round silently used k=0.70/0.30
        # instead of this config's real k_ratio_tier1=0.20/k_ratio_tier2=0.05,
        # for the ENTIRE run. Confirmed empirically: a real 20-round run's
        # byte accounting only makes sense under k=0.70/0.30 (at k>0.5 the
        # per-coordinate value+index overhead of top-k storage makes
        # compressed size EXCEED dense, matching upload numbers that were
        # sometimes larger than uncompressed FedAvg's reference). This
        # doesn't just corrupt byte accounting -- k_ratio truncates what
        # actually gets aggregated into the global model, so it silently
        # changed training dynamics (and therefore every accuracy number)
        # too. A 10-30 round warmup is reasonable for a from-scratch model
        # training for hundreds of rounds; it makes no sense for a
        # pretrained backbone fine-tuning for 10-20 rounds total, where the
        # "warmup" would consume the entire run.
        use_k_warmup                   = False,
    )
    base.update(overrides)
    return Config(**base)


def get_fedavg_pretrained_config(**overrides) -> Config:
    """
    Config for the "Full FedAvg" companion baseline (master plan §6) under the
    SAME pretrained-backbone fine-tuning condition as
    get_pretrained_finetune_config() — the iso-condition accuracy reference
    that the DivRoute pretrained run must be compared against for
    comm_vs_accuracy.png to be a valid claim.

    fedavg_baseline_mode=True triggers main.py's override block (forces
    use_divergence_weighting/use_server_momentum/use_error_feedback/
    use_adaptive_tau/use_tier3_sync/use_adaptive_k off, k_ratio_tier1=
    k_ratio_tier2=1.0, tau_low=tau_high=-1.0, gamma=1.0 — see main.py's
    run(), the block gated on `if config.fedavg_baseline_mode`).

    include_tier3_in_aggregation=True is NOT optional here — it is required
    for this to be a working FedAvg baseline at all, not just a safety net.
    With tau_low=tau_high=-1.0 and divergence scores that are always >= 0,
    main.py's tier-assignment (`d <= tau_low -> Tier 1`, `d <= tau_high ->
    Tier 2`, else Tier 3`) puts every single client in Tier 3 every round.
    server.aggregate() drops Tier-3 clients from aggregation unless
    include_tier3_in_aggregation=True (server.py: `if r["tier"] != 3 or
    include_tier3_in_aggregation`) -- so without this flag, EVERY client
    would be excluded and the global model would never update (verified
    directly against server.py and compression.py; also matches the
    corrected-FedAvg fix already validated in
    run_diag20_fedavg_corrected.py / logs/diag20_fedavg_corrected_cifar100.json,
    where client aggregation_weight is non-zero despite tier=3 specifically
    because this flag is set). With k_ratio_tier2 forced to 1.0 by the
    fedavg_baseline_mode override above, compression.py's tier-3-included
    path (`k_ratio = k1 if tier==1 else k2`) still resolves to an
    uncompressed (k=1.0) update -- i.e. genuine, undamaged FedAvg, not
    FedAvg-shaped-but-secretly-compressed.

    threshold_mode="fixed" is set for defense in depth, matching the same
    precedent run: main.py's current tier-assignment block already skips tau
    recomputation unconditionally whenever fedavg_baseline_mode=True (the
    `if not config.fedavg_baseline_mode` guard), so this is redundant with
    that guard today, but costs nothing and protects against that guard
    being weakened later without this config being re-checked.

    dataset_name / model_name / bn_mode / local_lr / local_lr_min mirror
    get_pretrained_finetune_config() exactly, so the only difference between
    the two runs is the routing/compression mechanism being tested, not the
    backbone, data pipeline, or optimisation regime.

    Caller MUST pass an identical seed, num_rounds, num_clients, alpha, and
    clients_per_round to get_pretrained_finetune_config() and
    get_uniform_top5_pretrained_config() (master plan §6) -- this function
    does not enforce that; it is the caller's responsibility. Caller should
    also set run_label (via overrides) to something unique so this run
    writes to its own checkpoint directory.

    Any keyword argument in `overrides` is forwarded to Config().
    """
    base = dict(
        fedavg_baseline_mode          = True,
        include_tier3_in_aggregation  = True,
        threshold_mode                = "fixed",
        dataset_name                  = "cifar100",
        model_name                    = "efficientnet_b0_pretrained",
        bn_mode                       = "default",
        local_lr                      = 0.01,
        local_lr_min                  = 0.001,
        # Critical, not cosmetic: use_k_warmup defaults to True
        # (k_warmup_rounds=30), and get_adaptive_k_ratios() checks it BEFORE
        # k_ratio_tier1/tier2 -- so for a <=20-round run the whole thing
        # falls inside the warmup window and every round silently used
        # k=0.70/0.30 instead of the k_ratio_tier1=k_ratio_tier2=1.0 this
        # override block sets above. Without this, "FedAvg" isn't dense at
        # all -- it's secretly compressed at the same ratio a from-scratch
        # warmup would use this early, which both understates its true
        # (dense) byte cost and, more importantly, changes what actually
        # gets aggregated into the global model. See
        # get_pretrained_finetune_config()'s matching note for the full
        # byte-math confirmation.
        use_k_warmup                   = False,
        # Methodology parity, not a correctness bug like the two above:
        # get_pretrained_finetune_config() inherits ntd_beta=0.1 (Not-True
        # Distillation, a local-training regularizer) from
        # get_recommended_divroute_config()'s base; this factory previously
        # left it at Config's default 0.0, giving DivRoute an unearned
        # regularization advantage no baseline shared (flagged as an open
        # question in DIVROUTE_ACCURACY_MASTER_PLAN.md §10, resolved here by
        # giving every method the same regularizer rather than removing it
        # from DivRoute -- either resolves the asymmetry, but this direction
        # doesn't also reduce DivRoute's own accuracy).
        ntd_beta                       = 0.1,
    )
    base.update(overrides)
    return Config(**base)


def get_uniform_top5_pretrained_config(**overrides) -> Config:
    """
    Config for the "Uniform Top-5%" companion baseline (master plan §6) under
    the SAME pretrained-backbone fine-tuning condition as
    get_pretrained_finetune_config() -- the communication-matched control
    that isolates DivRoute's routing intelligence from compression itself,
    at the same pretrained init as get_pretrained_finetune_config() and
    get_fedavg_pretrained_config().

    Starts from get_uniform_top5_config()'s validated base (uniform_top5_mode
    =True; main.py short-circuits the whole divergence/tau/tier-assignment
    block in this mode and hardcodes every client to tier=2, so -- unlike
    get_fedavg_pretrained_config() -- there is no Tier-3-exclusion hazard
    here and include_tier3_in_aggregation is left at its default) and
    changes only the backbone/dataset and fine-tuning LR regime, mirroring
    get_pretrained_finetune_config():
        - dataset_name="cifar100", model_name="efficientnet_b0_pretrained"
        - bn_mode="default" (preserve pretrained BatchNorm stats)
        - local_lr / local_lr_min: same lowered fine-tuning regime -- 0.1
          (get_uniform_top5_config's implicit from-scratch default) would
          damage pretrained features just as it would for DivRoute.

    Caller MUST pass an identical seed, num_rounds, num_clients, alpha, and
    clients_per_round to get_pretrained_finetune_config() and
    get_fedavg_pretrained_config() (master plan §6). Caller should also set
    run_label (via overrides) to something unique.

    Any keyword argument in `overrides` is forwarded to Config().
    """
    base = get_uniform_top5_config().__dict__.copy()
    base.update(
        dataset_name  = "cifar100",
        model_name    = "efficientnet_b0_pretrained",
        bn_mode       = "default",
        local_lr      = 0.01,
        local_lr_min  = 0.001,
        # Critical, not cosmetic -- see get_pretrained_finetune_config()'s
        # matching note. use_k_warmup defaults to True (k_warmup_rounds=30);
        # for a <=20-round run every round falls inside that warmup window,
        # so every client (always tier=2 in uniform_top5_mode) was silently
        # compressed at k_warmup_tier2=0.30 instead of this config's real
        # k_ratio_tier2=0.05 -- roughly 6x more retained per client than
        # "Uniform Top-5%" is supposed to mean, for the entire run.
        use_k_warmup  = False,
        # Training-budget parity, not a correctness bug: get_uniform_top5_config()
        # sets use_epoch_warmup=False, so every round trains at the full
        # local_epochs=5 from round 1 -- while DivRoute/FedAvg ramp 2->4->5
        # (75 "epoch-rounds" over a 20-round run vs. Uniform's 100, ~33% more
        # total local SGD steps for Uniform). This was an unresolved confound
        # in the fourth companion-run result (DIVROUTE_ACCURACY_MASTER_PLAN.md
        # §13): Uniform's accuracy edge over DivRoute could partly or wholly be
        # this extra training rather than compression-strategy quality.
        # Matching DivRoute/FedAvg's schedule here removes that confound so a
        # future comparison isolates the compression strategy itself.
        ntd_beta      = 0.1,   # see get_fedavg_pretrained_config()'s matching note
        use_epoch_warmup = True,
    )
    base.update(overrides)
    return Config(**base)