import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_divergence(local_model: nn.Module, global_model: nn.Module) -> float:
    """
    Cosine-distance divergence between two models' flattened parameter vectors.
    d = 1 - cos_sim. d=0 -> identical models. d=1 -> orthogonal. d=2 -> opposite.
    """
    local_vec = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    global_vec = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    cos_sim = F.cosine_similarity(local_vec.unsqueeze(0), global_vec.unsqueeze(0), eps=1e-8).item()
    return 1.0 - cos_sim


def assign_tier(d: float, tau_low: float, tau_high: float) -> int:
    """
    Tier assignment from a divergence score and a pair of thresholds.

    d > tau_high  -> Tier 1 (high divergence -> high-fidelity update)
    d > tau_low   -> Tier 2 (moderate divergence -> lightweight update)
    otherwise     -> Tier 3 (converged -> skip)
    """
    if d > tau_high:
        return 1
    elif d > tau_low:
        return 2
    else:
        return 3


def compute_divergence_weight(d: float, mode: str = "sqrt") -> float:
    """
    Convert a divergence score to an aggregation weight multiplier.
    Lower divergence = more aligned = higher weight.

    Modes:
        inverse  - 1/(d+eps)         - aggressive, unstable at small d
        sqrt     - 1/sqrt(d+eps)     - moderate, recommended
        exp      - exp(-10*d)        - smooth decay
        softmax  - use compute_softmax_weights() for the full client list
    """
    eps = 1e-6
    if mode == "inverse":
        return 1.0 / (d + eps)
    elif mode == "sqrt":
        return 1.0 / (d ** 0.5 + eps)
    elif mode == "exp":
        return float(np.exp(-10.0 * d))
    else:
        # fallback — caller should use the softmax variant for the "softmax" mode
        return 1.0 / (d ** 0.5 + eps)


def compute_softmax_weights(d_scores: list, temperature: float = 50.0) -> list:
    """
    Softmax over negative divergences — sum always equals 1, no normalisation needed.
    Higher temperature = more uniform; lower = winner-takes-all.
    """
    arr = np.array(d_scores, dtype=np.float64)
    neg = -arr * temperature
    neg -= neg.max()   # numerical stability
    exp = np.exp(neg)
    return (exp / exp.sum()).tolist()


def update_ema(ema_scores: dict, client_id: int, d_current: float, beta: float) -> float:
    """Exponential moving average of a client's divergence score across rounds."""
    prev = ema_scores.get(client_id, d_current)
    smoothed = beta * prev + (1 - beta) * d_current
    ema_scores[client_id] = smoothed
    return smoothed


def compute_adaptive_taus(all_d_scores: list, tau_low_pct: float, tau_high_pct: float):
    """
    Percentile-based tau thresholds (Phase 6.1 — "Make Adaptive Real").

    Computed fresh each round from the distribution of selected clients'
    divergence scores:
        tau_low  = tau_low_pct-th  percentile  -> clients below this are Tier 3
        tau_high = tau_high_pct-th percentile  -> clients above this are Tier 1

    A minimum-separation guard prevents tau_low == tau_high (which would make
    Tier 2 empty) when the distribution is degenerate (e.g. all scores equal).
    """
    arr = np.array(all_d_scores, dtype=np.float64)
    tau_low = float(np.percentile(arr, tau_low_pct))
    tau_high = float(np.percentile(arr, tau_high_pct))
    if tau_low >= tau_high:
        tau_high = tau_low * 1.5 + 1e-6
    return tau_low, tau_high


def update_selection_weights(weights: np.ndarray, results: list, gamma: float) -> None:
    """
    Tier-3 (converged) clients have their selection probability decayed by gamma.
    Any client that is NOT Tier 3 has its weight reset to 1.0 (full priority).
    """
    for r in results:
        cid = r["client_id"]
        if r["tier"] == 3:
            weights[cid] *= gamma
        else:
            weights[cid] = 1.0