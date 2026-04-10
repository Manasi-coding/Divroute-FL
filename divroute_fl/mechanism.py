import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_divergence(local_model: nn.Module, global_model: nn.Module) -> float:
    local_vec  = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    global_vec = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    # eps=1e-8 avoids div-by-zero in round 0 when weights are still near-zero
    return F.cosine_similarity(local_vec.unsqueeze(0), global_vec.unsqueeze(0), eps=1e-8).item()


def assign_tier(d: float, tau_low: float = 0.70, tau_high: float = 0.90) -> int:
    if d < tau_low:
        return 1   # high fidelity — client has drifted, needs the full update
    elif d < tau_high:
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
