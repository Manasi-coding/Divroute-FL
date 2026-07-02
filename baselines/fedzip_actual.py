"""
baselines/fedzip_actual.py
==========================
True implementation of the FedZip baseline (Phase 5).

This baseline implements a real compression path integrated directly into the
server aggregation pipeline:
  1. Top-z sparsification: Keeps the top-z fraction of largest-magnitude updates.
  2. MiniBatchKMeans quantization: Clusters the selected values into k_clusters.
  3. Reconstruction: Builds the sparse quantized update back to the model's parameter shape.
  4. Server-side byte accounting: Instead of post-hoc log parsing, the true byte cost
     is computed on the fly by calling the shared `fedzip_bytes_for_delta` function.
"""

import torch
from sklearn.cluster import MiniBatchKMeans
from divroute_fl.config import Config
from baselines.fedzip_baseline import fedzip_bytes_for_delta


def fedzip_compress_delta(
    delta: torch.Tensor,
    z_ratio: float = 0.01,
    k_clusters: int = 3,
    random_state: int = 42,
) -> tuple[torch.Tensor, int]:
    """
    Compresses delta using Top-z sparsification + MiniBatchKMeans quantization,
    and returns (reconstructed_delta, bytes_cost).
    """
    assert delta.dim() == 1, f"delta must be 1D, got dim={delta.dim()}"

    numel = delta.numel()
    k = max(1, int(z_ratio * numel))

    # Top-z sparsification
    abs_delta = delta.abs()
    _, indices = torch.topk(abs_delta, k)
    retained_values = delta[indices]

    # MiniBatchKMeans quantization
    values_np = retained_values.detach().cpu().numpy().reshape(-1, 1)
    n_clusters = min(k_clusters, k)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(1024, len(values_np)),
        n_init="auto"
    )
    kmeans.fit(values_np)

    centroids = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    quantized_values_np = centroids[labels]

    quantized_values = torch.from_numpy(quantized_values_np).to(delta.device).to(delta.dtype)

    # Reconstruct sparse update
    reconstructed = torch.zeros_like(delta)
    reconstructed[indices] = quantized_values

    # Compute bytes_cost using the shared byte calculator
    bytes_cost = fedzip_bytes_for_delta(delta, z_ratio, k_clusters)

    return reconstructed, bytes_cost


def get_fedzip_actual_config(z_ratio: float = 0.01, k_clusters: int = 3, **overrides) -> Config:
    """
    Returns a Config for the actual FedZip baseline.
    """
    base = dict(
        fedzip_actual_mode        = True,
        fedzip_z_ratio            = z_ratio,
        fedzip_k_clusters         = k_clusters,
        # FedZip baseline runs vanilla FedAvg downloads on the routing side
        fedavg_baseline_mode      = True,
        use_server_momentum       = False,
        use_error_feedback        = False,
        use_tier3_sync            = False,
        use_adaptive_k            = False,
        use_divergence_weighting  = False,
        use_adaptive_tau          = False,
        use_epoch_warmup          = False,
        skip_plot_prompt          = True,
    )
    base.update(overrides)
    return Config(**base)
