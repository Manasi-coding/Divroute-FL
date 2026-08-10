"""
DivRoute-FL Forensic Analysis Script
Reads divroute_seed42.json and extracts all available round data.
"""
import json
import os

log_file = r"c:\Users\l\Desktop\GDG-implementation\logs\phase5\cifar100\divroute_seed42.json"
with open(log_file, "r") as f:
    data = json.load(f)

print(f"=== JSON LOG: {len(data)} rounds ===\n")

# Per-round analysis
print(f"{'Rnd':>4} {'Acc':>6} {'T1':>3} {'T2':>3} {'T3':>3} "
      f"{'AvgD(ema)':>12} {'TotalUp(MB)':>12} {'TotalDown(MB)':>13} "
      f"{'tau_low':>10} {'tau_high':>10}")

for rnd_data in data:
    rnd = rnd_data.get('round', 0) + 1  # 0-indexed in JSON
    acc = rnd_data.get('test_accuracy', 0.0)
    clients = rnd_data['clients']
    t1 = sum(1 for c in clients if c['tier'] == 1)
    t2 = sum(1 for c in clients if c['tier'] == 2)
    t3 = sum(1 for c in clients if c['tier'] == 3)
    
    avg_d = sum(c.get('divergence_score', 0) for c in clients) / len(clients)
    
    total_up   = rnd_data.get('total_upload_bytes',   0)
    total_down = rnd_data.get('total_download_bytes', 0)
    tau_low    = rnd_data.get('tau_low',  0)
    tau_high   = rnd_data.get('tau_high', 0)
    
    print(f"{rnd:>4} {acc:>6.4f} {t1:>3} {t2:>3} {t3:>3} "
          f"{avg_d:>12.8f} {total_up/1e6:>12.3f} {total_down/1e6:>13.3f} "
          f"{tau_low:>10.7f} {tau_high:>10.7f}")

print("\n=== COMMUNICATION ACCOUNTING AUDIT ===\n")

# Full model size
delta_numel = data[0].get('delta_numel', 11220132)
full_model_bytes = delta_numel * 4
print(f"Full model size: {delta_numel:,} params = {full_model_bytes/1e6:.3f} MB")
print(f"FedAvg baseline per round (20 clients):")
fedavg_up   = 20 * full_model_bytes
fedavg_down = 20 * full_model_bytes
fedavg_bidir = fedavg_up + fedavg_down
print(f"  Upload:   {fedavg_up/1e6:.3f} MB")
print(f"  Download: {fedavg_down/1e6:.3f} MB")
print(f"  Bidir:    {fedavg_bidir/1e6:.3f} MB\n")

for rnd_data in data:
    rnd = rnd_data.get('round', 0) + 1
    clients = rnd_data['clients']
    t1 = sum(1 for c in clients if c['tier'] == 1)
    t2 = sum(1 for c in clients if c['tier'] == 2)
    t3 = sum(1 for c in clients if c['tier'] == 3)
    
    # Per-client upload bytes
    per_client_uploads = [(c['client_id'], c['tier'], c.get('upload_bytes',0), c.get('bytes_received',0))
                          for c in clients]
    unique_uploads = {}
    for cid, tier, up, recv in per_client_uploads:
        key = (tier, up)
        if key not in unique_uploads:
            unique_uploads[key] = []
        unique_uploads[key].append(cid)
    
    total_up   = rnd_data.get('total_upload_bytes',   0)
    total_down = rnd_data.get('total_download_bytes', 0)
    total_bidir = total_up + total_down
    saving = 100.0 * (1.0 - total_bidir / fedavg_bidir)
    
    print(f"Round {rnd}: T1={t1}/T2={t2}/T3={t3}")
    for (tier, up_bytes), cids in sorted(unique_uploads.items()):
        pct = (up_bytes / full_model_bytes) * 100
        print(f"  Tier {tier}: {len(cids)} clients, upload={up_bytes/1e6:.3f}MB ({pct:.1f}% of model)")
    print(f"  Total upload={total_up/1e6:.3f}MB, download={total_down/1e6:.3f}MB, saving={saving:.1f}%")
    print()

print("\n=== KEY FINDING: k_ratio SCHEDULE OVERRIDE ===")
print("main.py lines 551-557:")
print("  if rnd < 30:")
print("      config.k_ratio_tier1 = 0.50")
print("      config.k_ratio_tier2 = 0.20")
print("  else:")
print("      config.k_ratio_tier1 = 0.35")
print("      config.k_ratio_tier2 = 0.10")
print()
print("This means for rounds 1-30:")
print(f"  T1 uploads {0.5 * full_model_bytes / 1e6:.3f} MB (50% of model)")
print(f"  T2 uploads {0.2 * full_model_bytes / 1e6:.3f} MB (20% of model)")
print()
print("Verifying against round 1 (T1=6, T2=14):")
r1_t1_up = 6 * int(0.50 * delta_numel) * 4
r1_t2_up = 14 * int(0.20 * delta_numel) * 4
r1_total  = r1_t1_up + r1_t2_up
r1_down   = 20 * full_model_bytes
print(f"  Expected T1 upload: {r1_t1_up/1e6:.3f} MB (6 clients * 50%)")
print(f"  Expected T2 upload: {r1_t2_up/1e6:.3f} MB (14 clients * 20%)")
print(f"  Expected total upload: {r1_total/1e6:.3f} MB")
print(f"  Expected total bidir:  {(r1_total+r1_down)/1e6:.3f} MB")
print(f"  Expected saving: {100*(1-(r1_total+r1_down)/fedavg_bidir):.1f}%")
print(f"  Log shows: up=520.614 MB, saving=21.0%")
print()

# The log text shows up=520.614MB for round 1. Let's verify:
# With T1=6 and k=0.50: 6 * 0.50 * 44.88 = 134.6 MB upload
# With T2=14 and k=0.20: 14 * 0.20 * 44.88 = 125.7 MB upload
# Total upload = 260.3 MB
# Total download = 20 * 44.88 = 897.6 MB
# Bidir = 260.3 + 897.6 = 1157.9 MB
# Saving vs 1795.2 = (1795.2-1157.9)/1795.2 = 35.5%
# But log shows 21.0%... so T1=6 with T2=14 doesn't add up

# Check with actual round 1 tiers from inline log: T1=6, T2=14, T3=0
# (warmup = all T3->T2, but original had no T3 in round 1)
# So the actual T1=6 comes from routing, T2=14, T3=0 (from inline log round 1)

print("=== Checking round 1 upload bytes from JSON ===")
r0 = data[0]
for c in r0['clients']:
    print(f"  Client {c['client_id']}: tier={c['tier']}, upload_bytes={c.get('upload_bytes',0)}, bytes_received={c.get('bytes_received',0)}")
