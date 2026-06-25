"""
baselines/fedzip_baseline.py
=============================
FedZip baseline for Phase 5 comparison.

FedZip (Xu et al., 2022) compresses server-to-client updates using:
  1. Top-z sparsification  — keep only the top-z fraction of delta values
  2. k-means quantization  — cluster the retained values into k_clusters groups
  3. Encoding             — transmit cluster indices + codebook

Because our FL loop runs on the server side, we do NOT reimplement the full
FedZip client-side quantization pipeline. Instead we:
  - Run the existing DivRoute-FL training loop with full (uncompressed) deltas
    so that the accuracy trajectory is realistic.
  - Post-hoc compute what FedZip's byte cost WOULD HAVE BEEN for the same
    delta tensors, using the exact FedZip byte formula.
  - This gives a fair *communication-matched* comparison: same accuracy signal,
    FedZip's true bandwidth footprint.

This approach is explicitly endorsed by the roadmap:
  "The cleanest approach is to compare at matched total MB."

Usage (called from run_phase5_comparison.py — do not call directly):
    from baselines.fedzip_baseline import fedzip_bytes_for_delta, get_fedzip_config
"""

import math
import torch

from divroute_fl.config import Config


# ── FedZip byte-cost formula ──────────────────────────────────────────────────

def fedzip_bytes_for_delta(
    delta: torch.Tensor,
    z_ratio: float = 0.01,
    k_clusters: int = 3,
) -> int:
    """
    Compute the byte cost FedZip would incur to transmit `delta`.

    FedZip byte formula (from the paper):
        - Top-z nonzero values : z × 4 bytes (float32 values)
        - Codebook             : k_clusters × 4 bytes (cluster centroids, float32)
        - Cluster indices      : z × ceil(log2(k_clusters)) bits, rounded up to bytes

    Args:
        delta      : flat or multi-dim parameter delta tensor
        z_ratio    : fraction of parameters to retain (default 0.01 = 1%)
        k_clusters : number of k-means quantization clusters (default 3)

    Returns:
        Byte cost as an integer.
    """
    numel = delta.numel()
    z = max(1, int(z_ratio * numel))

    bits_per_index = math.ceil(math.log2(max(k_clusters, 2)))
    index_bytes    = math.ceil(z * bits_per_index / 8)
    value_bytes    = z * 4                    # float32 retained values
    codebook_bytes = k_clusters * 4           # float32 cluster centroids

    return value_bytes + codebook_bytes + index_bytes


def fedzip_round_bytes(
    delta: torch.Tensor,
    num_clients: int,
    z_ratio: float = 0.01,
    k_clusters: int = 3,
) -> int:
    """
    Total FedZip download bytes for one round, given the global delta
    and the number of clients receiving it.
    """
    per_client = fedzip_bytes_for_delta(delta, z_ratio, k_clusters)
    return per_client * num_clients


# ── Config factory ────────────────────────────────────────────────────────────

def get_fedzip_config(**overrides) -> Config:
    """
    Returns a Config for the FedZip post-hoc accounting run.

    We run vanilla FedAvg (full deltas, all clients receive same update) so
    the model accuracy is unaffected by the compression.  Byte costs are
    recalculated in run_phase5_comparison.py using fedzip_bytes_for_delta().

    z_ratio=0.01 and k_clusters=3 match the FedZip paper's default settings.
    These values are stored in the Config as custom attributes so the runner
    can retrieve them easily.
    """
    base = dict(
        fedavg_baseline_mode = True,   # full delta, uncompressed training
        skip_plot_prompt     = True,
    )
    base.update(overrides)
    cfg = Config(**{k: v for k, v in base.items()
                    if k in Config.__dataclass_fields__})

    # Store FedZip-specific params as plain attributes (not dataclass fields)
    cfg.fedzip_z_ratio    = overrides.get("fedzip_z_ratio",    0.01)
    cfg.fedzip_k_clusters = overrides.get("fedzip_k_clusters", 3)
    return cfg
