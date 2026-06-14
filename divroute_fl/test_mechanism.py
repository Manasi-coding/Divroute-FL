import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from divroute_fl.mechanism import (compute_divergence, assign_tier,
                                    update_selection_weights, compute_adaptive_taus,
                                    update_ema)
from divroute_fl.model import SimpleCNN


def test_identical_models():
    """
    Phase 0.1 fix: identical models -> cosine_sim ~ 1.0 -> d = 1 - cos_sim ~ 0.0.
    The original assertion `d > 0.999` was backwards.
    """
    m = SimpleCNN()
    d = compute_divergence(m, copy.deepcopy(m))
    assert d < 0.001, f"Expected near-zero divergence for identical models, got {d:.6f}"
    print(f"[PASS] identical models: d={d:.6f}")


def test_random_models():
    # two fresh random inits are nearly orthogonal in ~200k-dim parameter space,
    # so cos_sim ~ 0 and d = 1 - cos_sim ~ 1.0
    d = compute_divergence(SimpleCNN(), SimpleCNN())
    assert 0.5 < d < 1.5, f"got {d:.6f}, expected d close to 1.0 (near-orthogonal random inits)"
    print(f"[PASS] random models: d={d:.6f}")


def test_tier_boundaries():
    """
    assign_tier(d, tau_low, tau_high):
        d > tau_high -> 1
        d > tau_low  -> 2
        otherwise    -> 3
    """
    tau_low, tau_high = 0.01, 0.02
    cases = [
        (0.05, 1),    # well above tau_high -> Tier 1
        (0.015, 2),   # between tau_low and tau_high -> Tier 2
        (0.001, 3),   # below tau_low -> Tier 3
        (0.02, 2),    # exactly at tau_high -> NOT > tau_high -> Tier 2
        (0.01, 3),    # exactly at tau_low -> NOT > tau_low -> Tier 3
    ]
    for d, expected in cases:
        got = assign_tier(d, tau_low, tau_high)
        assert got == expected, f"assign_tier({d}, {tau_low}, {tau_high})={got}, expected {expected}"
    print(f"[PASS] tier boundaries: {len(cases)}/{len(cases)} correct")


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


def test_adaptive_taus():
    """
    Phase 6.1: percentile-based tau thresholds should split a distribution
    into roughly the requested proportions, and the min-separation guard
    should kick in for degenerate (all-equal) distributions.
    """
    # 100 evenly spaced scores from 0.00 to 0.99
    scores = [i / 100.0 for i in range(100)]
    tau_low, tau_high = compute_adaptive_taus(scores, tau_low_pct=20.0, tau_high_pct=75.0)
    assert 0.19 <= tau_low <= 0.20
    assert 0.74 <= tau_high <= 0.75
    assert tau_low < tau_high

    # degenerate case: all scores identical -> guard must prevent tau_low == tau_high
    flat_scores = [0.5] * 10
    tau_low2, tau_high2 = compute_adaptive_taus(flat_scores, tau_low_pct=20.0, tau_high_pct=75.0)
    assert tau_low2 < tau_high2, "min-separation guard failed on degenerate input"
    print(f"[PASS] adaptive taus: normal=({tau_low:.4f},{tau_high:.4f}), "
          f"degenerate=({tau_low2:.4f},{tau_high2:.4f})")


def test_ema_smoothing():
    ema_scores = {}
    beta = 0.6
    # first call: no prior value -> smoothed == d_current
    d1 = update_ema(ema_scores, client_id=0, d_current=1.0, beta=beta)
    assert abs(d1 - 1.0) < 1e-9, f"got {d1}"
    # second call: smoothed = beta*prev + (1-beta)*current = 0.6*1.0 + 0.4*0.0
    d2 = update_ema(ema_scores, client_id=0, d_current=0.0, beta=beta)
    assert abs(d2 - 0.6) < 1e-9, f"got {d2}"
    print(f"[PASS] EMA smoothing: d1={d1:.4f}, d2={d2:.4f}")


if __name__ == "__main__":
    test_identical_models()
    test_random_models()
    test_tier_boundaries()
    test_weight_decay()
    test_cumulative_decay()
    test_adaptive_taus()
    test_ema_smoothing()
    print("\nAll tests passed.")