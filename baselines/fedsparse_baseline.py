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

The actual L1 regularisation is implemented in client.py (6 lines, gated
behind fedsparse_lambda > 0) and wired through config.fedsparse_lambda.
This file only provides the config factory that external scripts use.

Recommended lambda values (from the FedSparse paper): 0.01 and 0.04.
Usage (called from run_phase5_comparison.py — do not call directly):
    from baselines.fedsparse_baseline import get_fedsparse_config
"""

from divroute_fl.config import Config


def get_fedsparse_config(fedsparse_lambda: float = 0.01, **overrides) -> Config:
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
        # FedSparse L1 regularisation strength
        fedsparse_lambda          = fedsparse_lambda,
        skip_plot_prompt          = True,
    )
    base.update(overrides)
    return Config(**base)
