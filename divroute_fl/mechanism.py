import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_divergence(local_model: nn.Module, global_model: nn.Module) -> float:
    """
    Divergence = 1 − cosine_similarity(local, global).
      0  → identical (no drift)
      1  → orthogonal (maximum drift)
    """
    local_vec  = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    global_vec = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    cos_sim = F.cosine_similarity(local_vec.unsqueeze(0), global_vec.unsqueeze(0), eps=1e-8).item()
    return 1.0 - cos_sim


def assign_tier(d: float, tau_low: float = 0.01, tau_high: float = 0.05) -> int:
    """
    Now d = 1 − cos_sim, so HIGHER d = MORE drifted.
      d > tau_high  → tier 1 (drifted far, send high-fidelity delta)
      tau_low < d ≤ tau_high → tier 2 (moderate, send low-fidelity delta)
      d ≤ tau_low   → tier 3 (converged, skip this round)
    """
    if d > tau_high:
        return 1   # high fidelity — client has drifted, needs top-20% delta
    elif d > tau_low:
        return 2   # low fidelity — mostly aligned, top-5% is enough
    else:
        return 3   # skip — already converged, no update this round


def update_selection_weights(weights: np.ndarray, results: list, gamma: float = 0.85) -> np.ndarray:
    for r in results:
        cid = r["client_id"]
        if r["tier"] == 3:
            weights[cid] *= gamma  # decay — skipped clients get deprioritized
        else:
            weights[cid] = 1.0    # reset — active clients always get a fair shot
    return weights
