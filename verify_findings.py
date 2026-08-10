"""
Proof script for both findings.
"""

# ============================================================================
# FINDING 1: natural_tier assignment timing
# ============================================================================

def assign_tier_stub(d, tau_low=0.00020, tau_high=0.00025):
    if d > tau_high:
        return 1
    elif d > tau_low:
        return 2
    else:
        return 3

print("=" * 70)
print("FINDING 1: natural_tier assigned AFTER warmup override")
print("=" * 70)
print()
print("Exact code from main.py:341-352:")
print("""
    for r in results:
        d_for_tier = (r.get("d_raw", r["divergence_score"])
                      if config.routing_score == "raw"
                      else r["divergence_score"])
        tier = server.assign_tier(d_for_tier)        # line 347
        if rnd < 15 and tier == 3:                   # line 349
            tier = 2                                  # line 350
        r["natural_tier"] = tier                     # line 351  <-- set AFTER override
        r["tier"] = tier                             # line 352
""")

print("Exact code from mechanism.py:114-120 (update_selection_weights):")
print("""
    for r in results:
        cid = r["client_id"]
        tier = r.get("natural_tier", r["tier"])      # line 116  <-- reads natural_tier
        if tier == 3:
            weights[cid] *= gamma
        else:
            weights[cid] = 1.0
""")

print("Simulation of the bug for rnd=0..14 (d=0.000185, below tau_low=0.00020):")
print()
print(f"{'rnd':>4}  {'assign_tier':>12}  {'override?':>9}  {'natural_tier':>12}  {'decayed?':>9}")
print("-" * 60)

d = 0.000185  # a client that would be Tier-3
for rnd in range(16):
    raw_tier = assign_tier_stub(d)  # what server.assign_tier returns
    tier = raw_tier
    if rnd < 15 and tier == 3:
        tier = 2                    # warmup override
    natural_tier = tier             # AFTER override
    decayed = (natural_tier == 3)
    print(f"{rnd:>4}  {raw_tier:>12}  {'YES' if rnd < 15 and raw_tier == 3 else 'no':>9}  "
          f"{natural_tier:>12}  {'YES' if decayed else 'NO':>9}")

print()
print("CONCLUSION:")
print("  For rnd 0-14: assign_tier returns 3, override sets tier=2,")
print("  natural_tier=2 -> update_selection_weights sees 2 -> weight stays 1.0.")
print("  The client is NEVER penalised during the 15-round warmup.")
print()
print("  For rnd 15: override condition (rnd < 15) is False -> tier stays 3,")
print("  natural_tier=3 -> decay fires correctly from round 16 onwards.")
print()

# Practical impact: selection weight after N rounds of no-decay
gamma = 0.85  # typical value
print("Selection weight comparison (gamma=0.85):")
print(f"  With correct behaviour (natural_tier=3 fires from round 1):")
w = 1.0
for rnd in range(15):
    w *= gamma
print(f"    After 15 rounds of decay: weight = {w:.6f}")
print(f"  With buggy behaviour (natural_tier never 3 during warmup):")
print(f"    After 15 rounds of no-decay: weight = 1.000000")
print(f"  Difference: {1.0 - w:.4f} (the naturally-T3 client is {1/w:.2f}x more likely to be selected than it should be)")

# ============================================================================
# FINDING 2: k_ratio override
# ============================================================================

print()
print("=" * 70)
print("FINDING 2: k_ratio schedule override vs get_adaptive_k_ratios")
print("=" * 70)
print()

print("Complete sequence of k_ratio_tier1/tier2 assignments per round:")
print()
print("Step A: config.py defaults (line 32-33):")
print("  k_ratio_tier1 = 0.20, k_ratio_tier2 = 0.05")
print()
print("Step B: run_phase5_comparison.py overrides at config construction:")
print("  k_ratio_tier1 = 0.35, k_ratio_tier2 = 0.10")
print()
print("Step C: main.py:551-557 (hardcoded schedule, runs FIRST each round):")
print("  if rnd < 30:  k_ratio_tier1=0.50, k_ratio_tier2=0.20")
print("  else:         k_ratio_tier1=0.35, k_ratio_tier2=0.10")
print()
print("Step D: main.py:561-562 get_adaptive_k_ratios (runs SECOND each round):")
print("  if not config.use_adaptive_k:  return config.k_ratio_tier1, config.k_ratio_tier2")
print("  (adaptive_k=False in preset -> returns whatever Step C just set, unchanged)")
print()
print("Step E: server.aggregate() -> compression.py:73:")
print("  k_ratio = config.k_ratio_tier1 if tier==1 else config.k_ratio_tier2")
print()
print("PROOF: Since use_adaptive_k=False, get_adaptive_k_ratios() at line 95:")
print("  return config.k_ratio_tier1, config.k_ratio_tier2")
print("which returns exactly what Step C wrote. Step C is the FINAL determinant.")
print()

# Quantify
delta_numel = 11220132
full_bytes = delta_numel * 4
print("Communication impact on a round with T1=4, T2=16, T3=0:")
print()
for label, k1, k2 in [("Config preset (0.35/0.10)", 0.35, 0.10),
                       ("Step C warmup (0.50/0.20)", 0.50, 0.20)]:
    t1_up = 4 * int(k1 * delta_numel) * 4
    t2_up = 16 * int(k2 * delta_numel) * 4
    t3_up = 0
    total_up = t1_up + t2_up + t3_up
    total_down = 20 * full_bytes
    bidir = total_up + total_down
    fedavg_bidir = 2 * 20 * full_bytes
    saving = 100.0 * (1.0 - bidir / fedavg_bidir)
    print(f"  {label}:")
    print(f"    T1 upload (4 clients): {t1_up/1e6:.2f} MB  (k={k1})")
    print(f"    T2 upload (16 clients): {t2_up/1e6:.2f} MB (k={k2})")
    print(f"    Total bidir: {bidir/1e6:.2f} MB | saving: {saving:.1f}%")
    print()

print("Effect of override on cumulative savings (rounds 1-29):")
preset_savings = 0.0
schedule_savings = 0.0
# T1/T2 averages from log: approx T1=4, T2=16, T3=0 for rounds 1-15 warmup
fedavg_bidir_total = 0.0
for rnd in range(30):
    t1, t2, t3 = 4, 16, 0
    down = 20 * full_bytes
    fedavg_bidir_r = 2 * 20 * full_bytes
    fedavg_bidir_total += fedavg_bidir_r

    for label, k1, k2 in [("preset", 0.35, 0.10), ("schedule", 0.50, 0.20)]:
        up = t1 * int(k1 * delta_numel) * 4 + t2 * int(k2 * delta_numel) * 4
        bidir = up + down
        if label == "preset":
            preset_savings += (fedavg_bidir_r - bidir)
        else:
            schedule_savings += (fedavg_bidir_r - bidir)

print(f"  Cumulative bytes saved over rounds 1-30:")
print(f"    With 0.35/0.10 (intended):   {preset_savings/1e9:.4f} GB")
print(f"    With 0.50/0.20 (actual):     {schedule_savings/1e9:.4f} GB")
print(f"    Difference:                  {(preset_savings - schedule_savings)/1e9:.4f} GB")
print(f"  The schedule wastes {(preset_savings - schedule_savings)/1e9:.4f} GB of communication budget")
print(f"  compared to the intended config for the first 30 rounds.")

print()
print("Is this intentional design?")
print("  The comment at line 551 says: '-- communication schedule --'")
print("  There is no docstring, config flag, or mention in run_phase5_comparison.py.")
print("  The preset config explicitly sets k_ratio_tier1=0.35, k_ratio_tier2=0.10.")
print("  The schedule silently overwrites those values without the caller knowing.")
print("  Verdict: this is NOT declared as an intentional warm-up schedule.")
print("  It would need to be explicitly documented and match the paper's description.")
