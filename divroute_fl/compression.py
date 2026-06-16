import torch


def compress_delta(delta: torch.Tensor, k_ratio: float) -> dict:
    """
    Top-k sparsification.

    Byte accounting (Phase 0.2 fix): indices are stored as int32 (4 bytes each),
    values as float32 (4 bytes each) -> bytes_transmitted = k * (4 + 4) = 8k.
    """
    k = max(1, int(k_ratio * delta.numel()))
    _, indices = torch.topk(delta.abs(), k)
    indices = indices.to(torch.int32)          # int32: 4 bytes each, covers up to 2B params
    values = delta[indices.long()]             # int64 required for indexing
    return {
        "indices": indices,
        "values": values,
        "bytes_transmitted": k * 8,            # float32 values (4) + int32 indices (4)
        "total_params": delta.numel(),
    }


def reconstruct_delta(payload: dict) -> torch.Tensor:
    if payload["values"] is None:
        return torch.zeros(payload["total_params"])
    device = payload["values"].device          # infer device from payload — no CPU/GPU mismatch
    delta = torch.zeros(payload["total_params"], device=device)
    delta[payload["indices"].long()] = payload["values"]   # int32 -> int64 for indexing
    return delta


def apply_tiered_compression(delta: torch.Tensor, tier: int, config,
                              error_buffers: dict = None, client_id: int = None) -> dict:
    """
    Tier 1/2 -> top-k sparsification with error feedback accumulation (Phase 1.1).
    Tier 3    -> null packet (0 bytes); error buffer cleared to prevent stale residuals.

    Error feedback: compression residual is accumulated in error_buffers[client_id]
    and added to the next round's delta before compression.

        corrected_delta   = delta + error_buffers[client_id]
        payload           = top_k(corrected_delta, k)
        residual          = corrected_delta - reconstruct(payload)
        error_buffers[client_id] = residual
    """
    if tier == 3 and not getattr(config, "include_tier3_in_aggregation", False):
        # clear buffer — stale residuals from prior active rounds must not persist
        if error_buffers is not None and client_id is not None:
            error_buffers[client_id] = torch.zeros_like(delta)
        return {"indices": None, "values": None, "bytes_transmitted": 0,
                "total_params": delta.numel()}

    k_ratio = config.k_ratio_tier1 if tier == 1 else config.k_ratio_tier2

    if config.use_error_feedback and error_buffers is not None and client_id is not None:
        buf = error_buffers.get(client_id, torch.zeros_like(delta))
        effective_delta = delta + buf
        payload = compress_delta(effective_delta, k_ratio)
        # residual = what was dropped; carry it to next round
        residual = effective_delta.clone()
        residual[payload["indices"].long()] = 0.0    # int32 -> int64 for indexing
        error_buffers[client_id] = residual
    else:
        payload = compress_delta(delta, k_ratio)

    return payload


def get_adaptive_k_ratios(config, rnd: int) -> tuple:
    """Shrink k_ratios linearly over training — bigger updates early, smaller late."""
    if not config.use_adaptive_k:
        return config.k_ratio_tier1, config.k_ratio_tier2
    decay = 1.0 - 0.5 * (rnd / max(config.num_rounds - 1, 1))
    return (
        max(0.05, config.k_ratio_tier1 * decay),
        max(0.01, config.k_ratio_tier2 * decay),
    )