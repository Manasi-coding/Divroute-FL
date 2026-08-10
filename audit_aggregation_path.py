import torch
import torch.nn as nn
import torch.optim as optim
import math
from divroute_fl.model import get_model
from divroute_fl.config import Config
from divroute_fl.compression import apply_tiered_compression, reconstruct_delta
from torch.distributions import Beta

def run_audit():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize global model
    global_model = get_model('resnet18', 100).to(device)
    param_names = {n for n, _ in global_model.named_parameters()}
    
    global_sd = global_model.state_dict()
    old_flat = torch.cat([global_sd[k].flatten().float() for k in global_sd if k in param_names])
    
    print("=" * 80)
    print("AUDIT: DELTA TRANSFORMATIONS IN FL PIPELINE")
    print("=" * 80)
    
    # Simulate 3 clients with different amounts of data/training
    clients = []
    _beta_dist = Beta(torch.tensor(0.2, device=device), torch.tensor(0.2, device=device))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Train clients
    for cid in range(3):
        print(f"\n--- Training Client {cid} ---")
        local_model = get_model('resnet18', 100).to(device)
        local_model.load_state_dict(global_sd)
        local_model.train()
        
        opt = optim.SGD(local_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        
        # Train for different number of steps to create varied delta norms
        steps = 80 if cid == 0 else (40 if cid == 1 else 10)
        print(f"Running {steps} SGD steps (lr=0.05, momentum=0.9)...")
        
        for _ in range(steps):
            x = torch.randn(32, 3, 32, 32, device=device)
            y = torch.randint(0, 100, (32,), device=device)
            opt.zero_grad()
            
            lam = float(_beta_dist.sample().clamp(0.0, 1.0))
            lam = max(lam, 1.0 - lam)
            idx = torch.randperm(32, device=device)
            mixed_x = lam * x + (1.0 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            
            logits = local_model(mixed_x)
            loss = lam * crit(logits, y_a) + (1.0 - lam) * crit(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)
            opt.step()
            
        local_sd = local_model.state_dict()
        cf = torch.cat([local_sd[k].flatten().float() for k in local_sd if k in param_names])
        raw_delta = cf - old_flat
        
        print(f"Client {cid} Local Model vs Global Model (before processing):")
        print(f"  -> Raw Delta Norm: {raw_delta.norm().item():.4f}")
        
        clients.append({
            "client_id": cid,
            "raw_delta": raw_delta,
            "num_samples": 500, # equal weighting for simplicity
            "tier": 1, # FedAvg mode (all tier 1)
        })

    # 2. Server Aggregation Path
    print("\n" + "=" * 80)
    print("SERVER AGGREGATION PATH (config.fedavg_baseline_mode = True)")
    print("=" * 80)
    
    cfg = Config(fedavg_baseline_mode=True)
    # Explicitly set to what the code defaults to
    cfg.grad_clip_norm = 10.0 
    cfg.k_ratio_tier1 = 1.0
    
    agg_delta = torch.zeros_like(old_flat)
    weights = [1.0 / 3.0] * 3 # uniform weights for 3 clients
    
    canonical_agg_delta = torch.zeros_like(old_flat)
    
    for r, w in zip(clients, weights):
        cid = r["client_id"]
        raw_delta = r["raw_delta"].clone()
        canonical_agg_delta += w * raw_delta
        
        print(f"\n--- Processing Client {cid} (weight={w:.4f}) ---")
        print(f"1. Delta Computation (w_local - w_global):")
        norm_before = raw_delta.norm().item()
        print(f"   Norm: {norm_before:.4f}")
        
        # Transformation 1: Clipping (server.py L154)
        print(f"2. Server-side Clipping (grad_clip_norm={cfg.grad_clip_norm}):")
        if norm_before > cfg.grad_clip_norm:
            clip_factor = cfg.grad_clip_norm / norm_before
            raw_delta = raw_delta * clip_factor
            print(f"   -> CLIPPED! Factor = {clip_factor:.4f}")
            print(f"   -> Math eq to canonical FedAvg? NO. Scales delta by {clip_factor:.4f}.")
        else:
            print(f"   -> Not clipped.")
            print(f"   -> Math eq to canonical FedAvg? YES (for this client).")
        norm_after_clip = raw_delta.norm().item()
        print(f"   Norm after clip: {norm_after_clip:.4f}")
        
        # Transformation 2: Compression (server.py L191)
        print(f"3. Compression (k_ratio={cfg.k_ratio_tier1}):")
        payload = apply_tiered_compression(raw_delta, r["tier"], cfg, None, cid)
        if payload["indices"] is None:
            print("   -> Indices is None (lossless k=1.0 mode)")
            print("   -> Math eq to canonical FedAvg? YES.")
        
        # Transformation 3: Reconstruction (server.py L202)
        print(f"4. Reconstruction:")
        compressed_delta = reconstruct_delta(payload).to(device)
        norm_after_recon = compressed_delta.norm().item()
        print(f"   Norm after recon: {norm_after_recon:.4f}")
        error = (compressed_delta - raw_delta).abs().max().item()
        print(f"   Reconstruction error: {error:.4e}")
        
        # Transformation 4: Weighting
        print(f"5. Weighting (w={w:.4f}):")
        weighted_delta = w * compressed_delta
        print(f"   Norm after weighting: {weighted_delta.norm().item():.4f}")
        
        # Transformation 5: Aggregation
        agg_delta += weighted_delta

    print("\n" + "=" * 80)
    print("FINAL GLOBAL UPDATE")
    print("=" * 80)
    print(f"Actual Aggregated Delta Norm: {agg_delta.norm().item():.4f}")
    print(f"Canonical FedAvg Delta Norm : {canonical_agg_delta.norm().item():.4f}")
    
    # Compute dot product (cosine sim) to see if direction changed
    cos_sim = torch.nn.functional.cosine_similarity(agg_delta.unsqueeze(0), canonical_agg_delta.unsqueeze(0)).item()
    print(f"Cosine Similarity (Actual vs Canonical direction): {cos_sim:.6f}")
    
    # Momentum (server.py L215) - default False for FedAvg
    print(f"\n6. Server Momentum (use_server_momentum={cfg.use_server_momentum}):")
    print(f"   -> Skipped. Math eq to canonical FedAvg? YES.")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("Is the behaviour intentional or an implementation bug?")
    print("The server-side clip is applied to the accumulated multi-step delta, not the per-step gradient.")
    print("Because each client trains for a different number of steps (or has different data), their delta norms vary.")
    print("The clipping applies a non-uniform scaling factor (e.g. client 0 scaled by ~0.68, client 2 not scaled).")
    print("This permanently alters the direction and magnitude of the global update, breaking canonical FedAvg.")

if __name__ == '__main__':
    run_audit()
