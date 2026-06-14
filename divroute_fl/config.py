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
    tau_low_pct: float = 20.0       # bottom 20% -> tier 3 (converged)
    tau_high_pct: float = 75.0      # top 25% -> tier 1 (drifted)

    # EMA smoothing for divergence scores
    ema_beta: float = 0.6

    # top-k compression ratios per tier
    k_ratio_tier1: float = 0.20
    k_ratio_tier2: float = 0.05
    use_adaptive_k: bool = True

    # error feedback (Phase 1.1)
    use_error_feedback: bool = True

    # Tier-3 staleness sync (Phase 1.2, Option A — periodic heartbeat)
    use_tier3_sync: bool = True
    tier3_sync_interval: int = 5     # every Nth round, Tier-3 clients get a tier-2 heartbeat

    # aggregation
    use_divergence_weighting: bool = True
    # weighting formula: "inverse" (1/d), "sqrt" (1/sqrt(d)), "exp" (exp(-d)), "softmax"
    divergence_weight_mode: str = "sqrt"   # sqrt is more stable than raw inverse
    use_server_momentum: bool = True
    server_momentum: float = 0.9
    server_lr: float = 1.0
    grad_clip_norm: float = 10.0

    # selection weight decay
    gamma: float = 0.85

    # Local epoch warmup: ramps from E//2 to E over the first 15 rounds.
    # Set False for baselines/ablations to use fixed local_epochs every round.
    use_epoch_warmup: bool = True

    # Baseline mode: disables all DivRoute features to reproduce vanilla FedAvg.
    # k_ratio=1.0, tau thresholds below any real score, gamma=1, no momentum/EF/sync.
    fedavg_baseline_mode: bool = False

    # Set True to suppress the interactive "Generate plots?" prompt.
    # Required for automated / multi-run scripts.
    skip_plot_prompt: bool = False

    # parallelism (set to 0 to disable multiprocessing — sequential on GPU)
    num_workers: int = 0

    seed: int = 42
    log_path: str = "logs/run.json"