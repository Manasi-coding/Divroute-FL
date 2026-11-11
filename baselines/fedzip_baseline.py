"""Shared simulated byte accounting for the FedZip baseline."""

import math
import torch


def fedzip_bytes_for_delta(
    delta: torch.Tensor,
    z_ratio: float = 0.01,
    k_clusters: int = 3,
) -> int:
    """Compute the byte cost FedZip would incur to transmit `delta`."""
    numel = delta.numel()
    z = max(1, int(z_ratio * numel))

    bits_per_index = math.ceil(math.log2(max(k_clusters, 2)))
    index_bytes = math.ceil(z * bits_per_index / 8)
    value_bytes = z * 4  # float32 retained values
    codebook_bytes = k_clusters * 4  # float32 cluster centroids

    return value_bytes + codebook_bytes + index_bytes
