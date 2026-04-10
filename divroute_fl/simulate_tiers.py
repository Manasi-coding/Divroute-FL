import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from divroute_fl.mechanism import assign_tier

rounds, n = 100, 20
print(f"{'Round':>6}  {'mean_d':>6}  T1  T2  T3")
print("-" * 35)

for r in range(0, rounds + 1, 10):
    mean_d = 0.5 + (r / rounds) * 0.4
    scores = np.clip(np.random.normal(mean_d, 0.1, n), 0.0, 1.0)
    tiers  = [assign_tier(float(d)) for d in scores]
    t1, t2, t3 = tiers.count(1), tiers.count(2), tiers.count(3)
    bar = "█" * t1 + "▒" * t2 + "░" * t3
    print(f"{r:>6}  {mean_d:.2f}    {t1:>2}  {t2:>2}  {t3:>2}  {bar}")
