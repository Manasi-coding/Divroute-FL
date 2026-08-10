import torch

def compress_delta_layerwise(delta_flat: torch.Tensor, layer_slices: list, importances: list, k_ratio: float, use_fp16: bool = False) -> dict:
    """
    Allocate k_total indices across layers proportionally to importances.
    Guarantees: sum(k_l) == k_total, 0 <= k_l <= layer_size.
    Uses largest-remainder (Hamilton) rounding.
    """
    numel = delta_flat.numel()
    k_total = max(1, int(k_ratio * numel))
    
    if k_total >= numel:
        values = delta_flat.clone()
        if use_fp16:
            values = values.half()
        return {
            "indices":           None,
            "values":            values,
            "bytes_transmitted": numel * 2 if use_fp16 else numel * 4,
            "total_params":      numel,
        }

    L = len(importances)
    sigma = sum(importances)
    
    # Degenerate case: all importances are zero — distribute uniformly
    if sigma == 0.0:
        base = k_total // L
        k_l = [base] * L
        remainder = k_total - base * L
        for i in range(remainder):
            k_l[i] += 1
    else:
        # Proportional allocation
        raw_k = [k_total * s / sigma for s in importances]
        
        # Integer floor — NO max(1, ...) floor
        k_l = [int(r) for r in raw_k]
        
        # Largest-remainder correction to restore exact budget
        deficit = k_total - sum(k_l)
        fracs = sorted(range(L), key=lambda l: raw_k[l] - k_l[l], reverse=True)
        for l in fracs[:deficit]:
            k_l[l] += 1

    all_indices = []
    all_values = []
    
    for k, (name, offset, size) in zip(k_l, layer_slices):
        k = min(k, size) # cap at layer size
        if k == 0:
            continue
        
        layer_chunk = delta_flat[offset : offset + size]
        if k >= size:
            # Full layer
            all_indices.append(torch.arange(offset, offset + size, dtype=torch.int32, device=delta_flat.device))
            all_values.append(layer_chunk)
        else:
            _, local_idx = torch.topk(layer_chunk.abs(), k)
            all_indices.append(local_idx.to(torch.int32) + offset)
            all_values.append(layer_chunk[local_idx.long()])

    if not all_indices:
        # Extreme fallback
        values = delta_flat.clone()
        if use_fp16:
            values = values.half()
        return {"indices": None, "values": values, "bytes_transmitted": numel * 2 if use_fp16 else numel * 4, "total_params": numel}
        
    indices = torch.cat(all_indices)
    values = torch.cat(all_values)
    
    if use_fp16:
        values = values.half()
    
    return {
        "indices": indices,
        "values": values,
        "bytes_transmitted": len(indices) * 6 if use_fp16 else len(indices) * 8,
        "total_params": numel,
    }


def compress_delta(delta: torch.Tensor, k_ratio: float, use_fp16: bool = False) -> dict:
    """
    Top-k sparsification.

    Byte accounting:
      k_ratio < 1.0  → sparse top-k format: int32 indices (4B) + float32 values (4B)
                        → bytes_transmitted = k * 8
      k_ratio = 1.0  → full model, no indices needed (plain float32)
                        → bytes_transmitted = numel * 4
    This avoids double-counting the FedAvg baseline (which uses k=1.0) against a
    4-bytes/param reference in the saving% calculation.
    """
    numel = delta.numel()
    k = max(1, int(k_ratio * numel))

    if k >= numel:
        # Full model — send as plain float32, no index overhead
        values = delta.clone()
        if use_fp16:
            values = values.half()
        return {
            "indices":           None,           # signal: full-model (no sparse indices)
            "values":            values,
            "bytes_transmitted": numel * 2 if use_fp16 else numel * 4,      # float32 only, no indices
            "total_params":      numel,
        }

    _, indices = torch.topk(delta.abs(), k)
    indices = indices.to(torch.int32)          # int32: 4 bytes each, covers up to 2B params
    values = delta[indices.long()]             # int64 required for indexing
    if use_fp16:
        values = values.half()
    return {
        "indices":           indices,
        "values":            values,
        "bytes_transmitted": k * 6 if use_fp16 else k * 8,            # float32 values (4) + int32 indices (4)
        "total_params":      numel,
    }


def reconstruct_delta(payload: dict) -> torch.Tensor:
    if payload["values"] is None:
        # Tier 3 null packet
        return torch.zeros(payload["total_params"])
        
    values = payload["values"].float()
    
    if payload["indices"] is None:
        # Full-model payload (k=1.0) — values IS the full delta, no scatter needed
        return values
    device = values.device          # infer device from payload — no CPU/GPU mismatch
    delta = torch.zeros(payload["total_params"], device=device)
    delta[payload["indices"].long()] = values   # int32 -> int64 for indexing
    return delta


def apply_tiered_compression(delta: torch.Tensor, tier: int, config,
                              error_buffers: dict = None, client_id: int = None,
                              layer_slices: list = None, layer_importances: list = None) -> dict:
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

    k1 = getattr(config, "active_k_ratio_tier1", config.k_ratio_tier1)
    k2 = getattr(config, "active_k_ratio_tier2", config.k_ratio_tier2)
    k_ratio = k1 if tier == 1 else k2

    use_fp16 = getattr(config, "use_fp16_upload", False)

    if getattr(config, "use_layerwise_topk", False) and layer_slices is not None and layer_importances is not None:
        compress_fn = lambda d, k: compress_delta_layerwise(d, layer_slices, layer_importances, k, use_fp16)
    else:
        compress_fn = lambda d, k: compress_delta(d, k, use_fp16)

    if config.use_error_feedback and error_buffers is not None and client_id is not None:
        buf = error_buffers.get(client_id, torch.zeros_like(delta))
        effective_delta = delta + buf
        payload = compress_fn(effective_delta, k_ratio)
        # residual = what was dropped + quantization error; carry it to next round
        residual = effective_delta - reconstruct_delta(payload)
        error_buffers[client_id] = residual
    else:
        payload = compress_fn(delta, k_ratio)

    return payload


def get_adaptive_k_ratios(config, rnd: int) -> tuple:
    """Shrink k_ratios linearly over training, optionally applying an early warm-up."""
    # Determine base k_ratios for this round
    if getattr(config, "use_k_warmup", False) and rnd < getattr(config, "k_warmup_rounds", 30):
        base_k1 = config.k_warmup_tier1
        base_k2 = config.k_warmup_tier2
    else:
        base_k1 = config.k_ratio_tier1
        base_k2 = config.k_ratio_tier2

    if not config.use_adaptive_k:
        return base_k1, base_k2
        
    decay = 1.0 - 0.5 * (rnd / max(config.num_rounds - 1, 1))
    return (
        max(0.05, base_k1 * decay),
        max(0.01, base_k2 * decay),
    )