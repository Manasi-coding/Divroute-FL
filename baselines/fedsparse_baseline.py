"""FedSparse configuration with shifted proximal updates and sparse upload accounting."""

from divroute_fl.config import Config


def get_fedsparse_config(
    fedsparse_lambda: float = 0.01,
    local_epochs: int = 5,
    batch_size: int = 32,
    **overrides,
) -> Config:
    """
    Returns a Config for the FedSparse baseline.

    FedSparse runs vanilla FedAvg download (full deltas to all clients) but
    each client's training loss is augmented with the L1 proximity term.
    No divergence scoring, no tiering, no top-k compression on downloads.

    Args:
        fedsparse_lambda : L1 regularisation strength.
                           0.01 → mild sparsification (FedSparse default)
                           0.04 → aggressive sparsification
        **overrides      : any Config field to override (seed, log_path, etc.)
    """
    base = dict(
        # Vanilla FedAvg download side — no tiering or compression
        fedavg_baseline_mode=True,
        use_server_momentum=False,
        use_error_feedback=False,
        use_tier3_sync=False,
        use_adaptive_k=False,
        use_divergence_weighting=False,
        use_adaptive_tau=False,
        use_epoch_warmup=False,
        fedsparse_sparsify_upload=True,
        fedsparse_threshold=1e-4,
        fedsparse_lambda=fedsparse_lambda,
        local_epochs=local_epochs,
        batch_size=batch_size,
        skip_plot_prompt=True,
    )
    base.update(overrides)
    return Config(**base)
