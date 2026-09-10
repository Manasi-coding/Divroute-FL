import torch

def compress_layerwise_independent(delta_flat: torch.Tensor, layer_slices: list, k_ratio: float, use_fp16: bool = False, client_id=None, tier=None, round_num=None) -> dict:
    """
    Selects k = max(1, int(numel * k_ratio)) elements independently for every layer
    using magnitude-based Top-k.
    """
    numel = delta_flat.numel()
    
    if k_ratio >= 1.0:
        values = delta_flat.clone()
        if use_fp16:
            values = values.half()
        return {
            "indices": None,
            "values": values,
            "bytes_transmitted": numel * 2 if use_fp16 else numel * 4,
            "total_params": numel,
        }

    all_indices = []
    all_values = []
    
    for name, offset, size in layer_slices:
        k = max(1, int(size * k_ratio))
        k = min(k, size) # cap at layer size
        if k == 0:
            continue
            
        layer_chunk = delta_flat[offset : offset + size]
        if k >= size:
            all_indices.append(torch.arange(offset, offset + size, dtype=torch.int32, device=delta_flat.device))
            all_values.append(layer_chunk)
        else:
            _, local_idx = torch.topk(layer_chunk.abs(), k)
            all_indices.append(local_idx.to(torch.int32) + offset)
            all_values.append(layer_chunk[local_idx.long()])

    if not all_indices:
        values = delta_flat.clone()
        if use_fp16:
            values = values.half()
        return {"indices": None, "values": values, "bytes_transmitted": numel * 2 if use_fp16 else numel * 4, "total_params": numel}
        
    indices = torch.cat(all_indices)
    values = torch.cat(all_values)
    
    if use_fp16:
        values = values.half()
    


    if round_num is not None and client_id is not None and tier is not None and layer_slices is not None:
        import os
        import csv
        log_file = "b3_layerwise_audit.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["round_num", "client_id", "tier", "target_k_ratio", 
                                 "parameter_name", "parameter_count", "selected_count", "selected_pct", 
                                 "pre_mean_abs", "pre_max_abs", "post_mean_abs", "post_max_abs"])
            
            for name, offset, size in layer_slices:
                layer_chunk = delta_flat[offset : offset + size]
                k = max(1, int(size * k_ratio))
                k = min(k, size)
                
                selected_pct = (k / size) * 100
                pre_mean = layer_chunk.abs().mean().item()
                pre_max = layer_chunk.abs().max().item()
                
                if k >= size:
                    post_mean = pre_mean
                    post_max = pre_max
                else:
                    _, local_idx = torch.topk(layer_chunk.abs(), k)
                    selected_vals = layer_chunk[local_idx.long()]
                    post_mean = selected_vals.abs().mean().item()
                    post_max = selected_vals.abs().max().item()
                
                writer.writerow([round_num, client_id, tier, k_ratio, 
                                 name, size, k, selected_pct, 
                                 pre_mean, pre_max, post_mean, post_max])

    return {
        "indices": indices,
        "values": values,
        "bytes_transmitted": len(indices) * 6 if use_fp16 else len(indices) * 8,
        "total_params": numel,
    }


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


def compress_delta(delta: torch.Tensor, k_ratio: float, use_fp16: bool = False,
                   layer_slices: list = None, client_id: int = None, 
                   tier: int = None, round_num: int = None) -> dict:
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
        
    # [B3 AUDIT] Forensic instrumentation for global Top-k
    if layer_slices is not None and round_num is not None:
        import os, csv
        csv_path = "b3_layer_starvation_audit.csv"
        file_exists = os.path.isfile(csv_path)
        
        # Create a set of selected indices for O(1) lookup
        selected_set = set(indices.cpu().tolist())
        
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "round", "client_id", "tier", "parameter_name", "parameter_count",
                    "selected_count", "selected_pct", "parameter_share", "selected_share",
                    "allocation_ratio", "pre_mean_abs", "pre_max_abs", "post_mean_abs", "post_max_abs"
                ])
                
            for name, offset, size in layer_slices:
                layer_chunk = delta[offset : offset + size]
                pre_mean = layer_chunk.abs().mean().item()
                pre_max = layer_chunk.abs().max().item()
                
                # Count selected indices in this layer
                # Indices in layer are [offset, offset + size - 1]
                # To be efficient, we could do intersection or tensor ops
                layer_indices = torch.arange(offset, offset + size, device=indices.device)
                mask = torch.isin(layer_indices, indices)
                selected_count = mask.sum().item()
                
                # Reconstruct post-compression values for this layer
                post_chunk = torch.zeros_like(layer_chunk)
                if selected_count > 0:
                    local_selected_indices = layer_indices[mask] - offset
                    # The values are retrieved from the original delta
                    post_chunk[local_selected_indices] = layer_chunk[local_selected_indices]
                post_mean = post_chunk.abs().mean().item()
                post_max = post_chunk.abs().max().item() if selected_count > 0 else 0.0
                
                selected_pct = (selected_count / size) * 100.0
                param_share = size / numel
                sel_share = selected_count / k if k > 0 else 0.0
                alloc_ratio = sel_share / param_share if param_share > 0 else 0.0
                
                writer.writerow([
                    round_num, client_id, tier, name, size,
                    selected_count, selected_pct, param_share, sel_share,
                    alloc_ratio, pre_mean, pre_max, post_mean, post_max
                ])

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
                             layer_slices: list = None, layer_importances: list = None, round_num: int = None,
                             error_buffer_tiers: dict = None) -> dict:
    """
    Tier 1/2 -> top-k sparsification with error feedback accumulation (Phase 1.1).
    Tier 3    -> null packet (0 bytes); error buffer cleared to prevent stale residuals.

    Error feedback: compression residual is accumulated in error_buffers[client_id]
    and added to the next round's delta before compression.

        corrected_delta   = delta + error_buffers[client_id]
        payload           = top_k(corrected_delta, k)
        residual          = corrected_delta - reconstruct(payload)
        error_buffers[client_id] = residual

    error_buffer_tiers / config.error_feedback_tier_aware (default False,
    preserves the exact EF behaviour above and its documented -3.8pp finding
    at k_ratio_tier2=0.05 unchanged): when a client's tier -- and therefore
    its k_ratio -- changes between rounds, a residual accumulated under the
    OLD k_ratio is being blended into a delta about to be compressed at a
    DIFFERENT retention rate. This is a known failure mode in the FL
    literature: error feedback assumes a stable per-client compression
    target, and under partial participation (only a subset of clients
    selected per round) plus round-to-round tier reassignment (confirmed via
    this project's own logs: clients routinely flip tiers several times per
    round), that assumption is violated -- the residual is stale relative to
    the new target, not corrective. When error_feedback_tier_aware=True and
    a tier change is detected for this client, the residual is reset to zero
    (a fresh start under the new target) instead of reused, mirroring the
    existing tier==3 buffer-clear below rather than inventing new semantics.
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

    if getattr(config, "use_layerwise_topk", False) and layer_slices is not None:
        compress_fn = lambda d, k: compress_layerwise_independent(d, layer_slices, k, use_fp16, client_id=client_id, tier=tier, round_num=round_num)
    else:
        compress_fn = lambda d, k: compress_delta(d, k, use_fp16, layer_slices=layer_slices, client_id=client_id, tier=tier, round_num=round_num)

    if config.use_error_feedback and error_buffers is not None and client_id is not None:
        tier_aware = getattr(config, "error_feedback_tier_aware", False)
        if (tier_aware and error_buffer_tiers is not None
                and error_buffer_tiers.get(client_id) is not None
                and error_buffer_tiers.get(client_id) != tier):
            # this client's compression target changed since its residual was
            # accumulated -- start fresh rather than blend in a stale target
            error_buffers[client_id] = torch.zeros_like(delta)

        buf = error_buffers.get(client_id, torch.zeros_like(delta))
        effective_delta = delta + 0.9 * buf
        payload = compress_fn(effective_delta, k_ratio)
        # residual = what was dropped + quantization error; carry it to next round
        residual = effective_delta - reconstruct_delta(payload)
        error_buffers[client_id] = residual
        if tier_aware and error_buffer_tiers is not None:
            error_buffer_tiers[client_id] = tier
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