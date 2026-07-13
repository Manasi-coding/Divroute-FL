"""
baselines/fedsparse_baseline.py
================================
FedSparse baseline config factory for Phase 5 comparison.

FedSparse (Ruan et al.) adds an L1 proximity regularisation term to every
client's local training loss:

    L_total = CrossEntropy(w_local) + lambda * ||w_local - w_global||_1

This encourages clients to produce sparse gradient updates, making the
uploads more compressible.  The full DivRoute-FL download pipeline is kept
intact — clients still receive the global delta from the server normally.

Under Option 3 of the Phase 5 implementation plan:
  * We use standard sparse value + index encoding (float32 value [4 bytes] +
    int32 index [4 bytes] = 8 bytes per non-zero/retained parameter).
  * We do not retune threshold values or introduce bitmask/run-length encoding.
  * Because each retained non-zero coordinate costs 8 bytes instead of the
    4 bytes per parameter in dense FedAvg, the communication volume will
    exceed FedAvg if the retained fraction exceeds 50% (retaining >50%
    parameters at 8 bytes each is larger than 100% at 4 bytes each).
  * This is expected, correct, and intentional behavior to evaluate raw
    FedSparse thresholding against the baselines.

The actual L1 regularisation is implemented in client.py (gated behind
fedsparse_lambda > 0). The server-side upload sparsification is gated behind
config.fedsparse_sparsify_upload in server.py.
"""

from divroute_fl.config import Config


def get_fedsparse_config(
    fedsparse_lambda: float = 0.01,
    local_epochs: int = 5,
    batch_size: int = 32,
    **overrides
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
        fedavg_baseline_mode      = True,
        use_server_momentum       = False,
        use_error_feedback        = False,
        use_tier3_sync            = False,
        use_adaptive_k            = False,
        use_divergence_weighting  = False,
        use_adaptive_tau          = False,
        use_epoch_warmup          = False,
        # FedSparse upload sparsification options (Phase 5)
        fedsparse_sparsify_upload = True,
        fedsparse_threshold       = 1e-4,
        fedsparse_lambda          = fedsparse_lambda,
        local_epochs              = local_epochs,
        batch_size                = batch_size,
        skip_plot_prompt          = True,
    )
    base.update(overrides)
    return Config(**base)
