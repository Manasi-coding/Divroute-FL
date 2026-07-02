"""
baselines/fedzip_baseline.py
=============================
FedZip baseline byte calculation for Phase 5 comparison.

This file contains the shared byte calculator formula `fedzip_bytes_for_delta`
which computes the true bandwidth footprint of FedZip for any given delta tensor.
The actual compression, quantization, and reconstruction of updates are
implemented dynamically in server.py and baselines/fedzip_actual.py.
"""

import math
import torch




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



