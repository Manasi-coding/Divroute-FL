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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config: Config | None = None) -> None:
    if config is None:
        config = Config()

    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] using device: {device}")

    print("[init] loading CIFAR-10 + building non-IID splits...")
    client_datasets = get_client_datasets(config.num_clients, config.alpha, config.seed)
    test_loader = DataLoader(get_test_dataset(), batch_size=256, shuffle=False)

    # sanity check for how skewed the Dirichlet split actually is
    shard_sizes = [len(ds) for ds in client_datasets]
    print(f"[init] shard sizes — min: {min(shard_sizes)}, max: {max(shard_sizes)}, mean: {np.mean(shard_sizes):.0f}")

    global_model = SimpleCNN()
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(i, client_datasets[i], config.local_epochs, config.local_lr, config.batch_size, device)
        for i in range(config.num_clients)
    ]

    logger = FLLogger(config.log_path)
    all_ids = list(range(config.num_clients))

    print(f"[train] {config.num_rounds} rounds, {config.clients_per_round}/{config.num_clients} clients per round")

    for rnd in range(config.num_rounds):
        selected = server.select_clients(all_ids)

        results = [clients[cid].train(server.global_model) for cid in selected]

        # divergence + tier (stubs for now, Person B will make these real)
        for r in results:
            r["divergence_score"] = server.compute_divergence(clients[r["client_id"]].get_local_model(), server.global_model)
            r["tier"] = server.assign_tier(r["divergence_score"])

        server.aggregate(results)

        # compression stub — Person C will replace this with actual top-k
        for r in results:
            payload = server.compress_delta(server.global_delta, r["tier"])
            r["bytes_received"] = payload["bytes_transmitted"]

        acc = server.evaluate(test_loader)
        logger.log(rnd, acc, results)

        # update selection weights so Tier 3 clients are deprioritized next round
        server.update_selection_weights(results)

        total_bytes = sum(r["bytes_received"] for r in results)
        t1 = sum(1 for r in results if r["tier"] == 1)
        t2 = sum(1 for r in results if r["tier"] == 2)
        t3 = sum(1 for r in results if r["tier"] == 3)
        print(f"  round {rnd + 1:>3}/{config.num_rounds} | acc: {acc:.4f} | bytes: {total_bytes:,} | tiers: {t1}/{t2}/{t3}")

    print(f"\n[done] log written to {config.log_path}")


if __name__ == "__main__":
    run()
