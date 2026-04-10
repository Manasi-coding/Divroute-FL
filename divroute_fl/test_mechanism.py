import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from divroute_fl.mechanism import compute_divergence, assign_tier, update_selection_weights
from divroute_fl.model import SimpleCNN


def test_identical_models():
    m = SimpleCNN()
    d = compute_divergence(m, copy.deepcopy(m))
    assert d > 0.999, f"got {d:.6f}"
    print(f"[PASS] identical models: d={d:.6f}")


def test_random_models():
    # two fresh random inits are nearly orthogonal in ~200k-dim space
    d = compute_divergence(SimpleCNN(), SimpleCNN())
    assert d < 0.5, f"got {d:.6f}"
    print(f"[PASS] random models: d={d:.6f}")


def test_tier_boundaries():
    cases = [
        (0.50, 1),
        (0.75, 2),
        (0.95, 3),
        (0.70, 2),  # exactly at tau_low → Tier 2
        (0.90, 3),  # exactly at tau_high → Tier 3
    ]
    for d, expected in cases:
        got = assign_tier(d)
        assert got == expected, f"assign_tier({d})={got}, expected {expected}"
    print(f"[PASS] tier boundaries: {len(cases)}/5 correct")


def test_weight_decay():
    weights = np.ones(3)
    results = [
        {"client_id": 0, "tier": 3},
        {"client_id": 1, "tier": 1},
        {"client_id": 2, "tier": 3},
    ]
    update_selection_weights(weights, results, gamma=0.85)
    assert abs(weights[0] - 0.85) < 1e-6, f"got {weights[0]}"
    assert abs(weights[1] - 1.00) < 1e-6, f"got {weights[1]}"
    assert abs(weights[2] - 0.85) < 1e-6, f"got {weights[2]}"
    print(f"[PASS] weight decay: skipped={weights[0]:.2f}, active={weights[1]:.2f}, skipped={weights[2]:.2f}")


def test_cumulative_decay():
    weights = np.ones(1)
    result = [{"client_id": 0, "tier": 3}]
    for _ in range(5):
        update_selection_weights(weights, result, gamma=0.85)
    expected = 0.85 ** 5
    assert abs(weights[0] - expected) < 1e-6, f"got {weights[0]:.6f}, expected {expected:.6f}"
    print(f"[PASS] cumulative decay (5 skips): {weights[0]:.6f}")


if __name__ == "__main__":
    test_identical_models()
    test_random_models()
    test_tier_boundaries()
    test_weight_decay()
    test_cumulative_decay()
    print("\nAll tests passed.")
