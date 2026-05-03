from dataclasses import dataclass


@dataclass
class Config:
    # federation setup
    num_clients: int = 25
    clients_per_round: int = 15
    num_rounds: int = 20

    # local training
    local_epochs: int = 5
    local_lr: float = 0.1
    batch_size: int = 32

    # Dirichlet alpha — lower = more non-IID (e.g. 0.1 is very skewed, 100 is basically IID)
    alpha: float = 0.9      # moderate non-IID — skewed but no empty shards

    # divergence thresholds for tiering (d = 1 − cos_sim; higher = more drifted)
    tau_low: float = 0.01    # below this → tier 3 (converged, skip)
    tau_high: float = 0.02   # above this → tier 1 (drifted, high-fidelity delta)

    # top-k ratios per tier (Person C)
    k_ratio_tier1: float = 0.20
    k_ratio_tier2: float = 0.05

    # weight decay for clients that get skipped too many rounds
    gamma: float = 0.85

    seed: int = 42
    log_path: str = "logs/run.json"
