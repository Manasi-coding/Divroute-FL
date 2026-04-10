from dataclasses import dataclass


@dataclass
class Config:
    # federation setup
    num_clients: int = 20
    clients_per_round: int = 10
    num_rounds: int = 100

    # local training
    local_epochs: int = 3
    local_lr: float = 0.01
    batch_size: int = 32

    # Dirichlet alpha — lower = more non-IID (e.g. 0.1 is very skewed, 100 is basically IID)
    alpha: float = 0.5

    # cosine-sim thresholds for tiering (Person B fills this in properly)
    tau_low: float = 0.70
    tau_high: float = 0.90

    # top-k ratios per tier (Person C)
    k_ratio_tier1: float = 0.20
    k_ratio_tier2: float = 0.05

    # weight decay for clients that get skipped too many rounds
    gamma: float = 0.85

    seed: int = 42
    log_path: str = "logs/run.json"
