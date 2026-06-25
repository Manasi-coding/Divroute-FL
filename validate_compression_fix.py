import torch
from divroute_fl.compression import compress_delta, reconstruct_delta

delta = torch.randn(11_229_752)  # ResNet-18 param count

# k=1.0 (FedAvg): should use 4 bytes/param, no indices
p_full = compress_delta(delta, k_ratio=1.0)
assert p_full["indices"] is None, "indices should be None for full-model"
assert p_full["bytes_transmitted"] == delta.numel() * 4
rec_full = reconstruct_delta(p_full)
assert torch.allclose(rec_full, delta), "full-model reconstruct mismatch"
mb_actual   = p_full["bytes_transmitted"] / 1e6
mb_expected = delta.numel() * 4 / 1e6
print(f"k=1.0  bytes: {mb_actual:.2f} MB  (expected {mb_expected:.2f} MB)  OK")

# k=0.05 (Tier 2): should use 8 bytes/param (values + indices)
p_sparse = compress_delta(delta, k_ratio=0.05)
assert p_sparse["indices"] is not None
k = max(1, int(0.05 * delta.numel()))
assert p_sparse["bytes_transmitted"] == k * 8
rec_sparse = reconstruct_delta(p_sparse)
assert rec_sparse.shape == delta.shape
mb_s = p_sparse["bytes_transmitted"] / 1e6
print(f"k=0.05 bytes: {mb_s:.2f} MB  (expected {k*8/1e6:.2f} MB)  OK")

# Verify saving% for FedAvg vs FedAvg is now 0%
fedavg_ref  = 10 * delta.numel() * 4
fedavg_real = 10 * p_full["bytes_transmitted"]
saving = 1.0 - fedavg_real / fedavg_ref
print(f"FedAvg saving vs reference: {saving*100:.1f}%  (expected 0.0%)")
print()
print("All checks passed.")
