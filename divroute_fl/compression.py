"""
DivRoute-FL Lite — Compression
Top-k sparsification + tiered dispatcher for adaptive delta compression.

Standalone module — only imports torch. No dependency on any other project file.
"""
import torch


def compress_delta(delta: torch.Tensor, k_ratio: float) -> dict:
    """
    Compress a flat 1-D delta tensor by keeping only the top-k parameters
    with the largest absolute magnitude.

    Parameters
    ----------
    delta : torch.Tensor
        Flat 1-D tensor representing the global model update.
    k_ratio : float
        Fraction of parameters to keep (e.g. 0.20 = top 20%).

    Returns
    -------
    dict with keys:
        indices           – int64 tensor, shape [k]
        values            – float32 tensor, shape [k] (signed values)
        bytes_transmitted – int (honest cost: values + indices)
        total_params      – int (original delta size, needed for reconstruction)
    """
    k = max(1, int(k_ratio * delta.numel()))

    # Select top-k by absolute magnitude
    _, indices = torch.topk(delta.abs(), k)

    # Extract the actual signed values (NOT abs values)
    values = delta[indices]

    # Honest byte cost: float32 values + int32 indices
    values_bytes = k * 4
    indices_bytes = k * 4
    total_bytes = values_bytes + indices_bytes

    return {
        "indices": indices,
        "values": values,
        "bytes_transmitted": total_bytes,
        "total_params": delta.numel(),
    }


def reconstruct_delta(payload: dict) -> torch.Tensor:
    """
    Reconstruct an approximate full-size delta from a compressed payload.

    Tier-3 null packets (values=None) produce an all-zero tensor.
    Tier-1/2 payloads scatter values back into a zero tensor.
    """
    # Handle tier-3 null packet
    if payload["values"] is None:
        return torch.zeros(payload["total_params"])

    # Scatter values back into a zero tensor
    delta = torch.zeros(payload["total_params"])
    delta[payload["indices"]] = payload["values"]
    return delta


def apply_tiered_compression(delta: torch.Tensor, tier: int, config) -> dict:
    """
    Dispatcher: routes the delta to the correct compression level based on
    the client's tier assignment.

    Tier 1 → top-k with k_ratio_tier1 (e.g. 20%)
    Tier 2 → top-k with k_ratio_tier2 (e.g. 5%)
    Tier 3 → null packet (0 bytes, client is skipped)
    """
    if tier == 1:
        return compress_delta(delta, k_ratio=config.k_ratio_tier1)

    elif tier == 2:
        return compress_delta(delta, k_ratio=config.k_ratio_tier2)

    elif tier == 3:
        return {
            "indices": None,
            "values": None,
            "bytes_transmitted": 0,
            "total_params": delta.numel(),
        }

    else:
        raise ValueError(f"Unknown tier: {tier}")


def compute_round_bytes(tier_list: list, delta_numel: int, config) -> int:
    """
    Compute the total bytes transmitted in one round given the list of
    tier assignments for all selected clients.

    Parameters
    ----------
    tier_list : list[int]
        Tier assignment for each selected client.
    delta_numel : int
        Number of parameters in the global delta.
    config : Config
        Provides k_ratio_tier1 and k_ratio_tier2.

    Returns
    -------
    int — total bytes transmitted across all clients this round.
    """
    total = 0
    for tier in tier_list:
        if tier == 1:
            k = int(config.k_ratio_tier1 * delta_numel)
            total += 2 * k * 4
        elif tier == 2:
            k = int(config.k_ratio_tier2 * delta_numel)
            total += 2 * k * 4
        elif tier == 3:
            total += 0  # null packet
    return total
