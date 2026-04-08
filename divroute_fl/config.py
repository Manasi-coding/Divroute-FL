"""
DivRoute-FL Lite — Configuration
All hyperparameters live here in a single dataclass.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- Federation layout ---
    num_clients: int = 20
    clients_per_round: int = 10
    num_rounds: int = 100

    # --- Local training ---
    local_epochs: int = 3
    local_lr: float = 0.01
    batch_size: int = 32

    # --- Non-IID data split ---
    alpha: float = 0.5  # Dirichlet concentration (lower = more heterogeneous)

    # --- Divergence-aware routing (Person B) ---
    tau_low: float = 0.70   # cosine-sim threshold: below → tier 1 (high fidelity)
    tau_high: float = 0.90  # cosine-sim threshold: above → tier 3 (skip)

    # --- Top-k compression ratios (Person C) ---
    k_ratio_tier1: float = 0.20  # top-k ratio for tier-1 (drifted) clients
    k_ratio_tier2: float = 0.05  # top-k ratio for tier-2 (aligned) clients

    # --- Selection weight decay for skipped clients ---
    gamma: float = 0.85

    # --- Reproducibility ---
    seed: int = 42

    # --- Logging ---
    log_path: str = "logs/run.json"
