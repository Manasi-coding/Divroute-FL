"""
DivRoute-FL Lite — Main training loop
Runs the baseline FedAvg simulation and logs results.
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import get_client_datasets, get_test_dataset
from .model import SimpleCNN
from .client import FLClient
from .server import FLServer
from .logger import FLLogger


def _seed_everything(seed: int) -> None:
    """Pin all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config: Config | None = None) -> None:
    """Execute the full federated-learning experiment."""
    if config is None:
        config = Config()

    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    # ---- Data -----------------------------------------------------------
    print("[init] Loading CIFAR-10 and creating non-IID splits …")
    client_datasets = get_client_datasets(
        num_clients=config.num_clients,
        alpha=config.alpha,
        seed=config.seed,
    )
    test_dataset = get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Report shard sizes so we can sanity-check the Dirichlet split
    shard_sizes = [len(ds) for ds in client_datasets]
    print(f"[init] Client shard sizes: min={min(shard_sizes)}, "
          f"max={max(shard_sizes)}, mean={np.mean(shard_sizes):.0f}")

    # ---- Model + Server + Clients ----------------------------------------
    global_model = SimpleCNN()
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(
            client_id=i,
            dataset=client_datasets[i],
            local_epochs=config.local_epochs,
            local_lr=config.local_lr,
            batch_size=config.batch_size,
            device=device,
        )
        for i in range(config.num_clients)
    ]

    logger = FLLogger(config.log_path)
    all_client_ids = list(range(config.num_clients))

    # ---- Training loop ---------------------------------------------------
    print(f"[train] Starting {config.num_rounds} rounds "
          f"({config.clients_per_round}/{config.num_clients} clients/round)")

    for round_idx in range(config.num_rounds):
        # 1. Select clients
        selected = server.select_clients(all_client_ids)

        # 2. Each selected client trains locally
        results = [clients[cid].train(server.global_model) for cid in selected]

        # 3. Compute divergence score + assign tier (stubs for now)
        for r in results:
            local_model = clients[r["client_id"]].get_local_model()
            r["divergence_score"] = server.compute_divergence(
                local_model, server.global_model,
            )
            r["tier"] = server.assign_tier(r["divergence_score"])

        # 4. Aggregate → compute global delta
        server.aggregate(results)

        # 5. Compress and simulate transmission (stub: full delta each time)
        for r in results:
            payload = server.compress_delta(server.global_delta, r["tier"])
            r["bytes_received"] = payload["bytes_transmitted"]

        # 6. Evaluate + log
        acc = server.evaluate(test_loader)
        logger.log(round_idx, acc, results)

        total_bytes = sum(r["bytes_received"] for r in results)
        print(
            f"Round {round_idx + 1:>3}/{config.num_rounds} | "
            f"Accuracy: {acc:.4f} | "
            f"Bytes: {total_bytes:>12,}"
        )

    print(f"\n[done] Log saved to {config.log_path}")


# Allow: python -m divroute_fl.main
if __name__ == "__main__":
    run()
