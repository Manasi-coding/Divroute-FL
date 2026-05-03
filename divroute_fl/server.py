import copy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .mechanism import compute_divergence, assign_tier, update_selection_weights
from .compression import apply_tiered_compression


class FLServer:
    def __init__(self, global_model: nn.Module, config: Config, device: torch.device):
        self.global_model = global_model.to(device)
        self.config = config
        self.device = device

        # uniform to start; Person B will adjust these based on divergence/skip history
        self.selection_weights = np.ones(config.num_clients, dtype=np.float64)
        self.global_delta: torch.Tensor | None = None

        self._rng = np.random.default_rng(config.seed)

    def select_clients(self, all_ids: List[int]) -> List[int]:
        weights = self.selection_weights[all_ids].copy()
        prob = weights / weights.sum()
        selected = self._rng.choice(
            all_ids,
            size=self.config.clients_per_round,
            replace=False,
            p=prob,
        )
        return selected.tolist()

    def aggregate(self, client_results: List[dict]) -> None:
        # snapshot old params before updating so we can compute the delta
        old_flat = self._flatten(self.global_model.state_dict())

        total_samples = sum(r["num_samples"] for r in client_results)

        new_state: OrderedDict = OrderedDict()
        for key in self.global_model.state_dict():
            new_state[key] = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
            for r in client_results:
                w = r["num_samples"] / total_samples
                new_state[key] += w * r["state_dict"][key].float()

        self.global_model.load_state_dict(new_state)
        self.global_delta = self._flatten(new_state) - old_flat

    def evaluate(self, test_loader: DataLoader) -> float:
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.global_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    # --- Person B's mechanism (real implementations, not stubs) ---

    def compute_divergence(self, local_model: nn.Module, global_model: nn.Module) -> float:
        return compute_divergence(local_model, global_model)

    def assign_tier(self, d: float) -> int:
        return assign_tier(d, self.config.tau_low, self.config.tau_high)

    def update_selection_weights(self, tier_results: list) -> None:
        update_selection_weights(self.selection_weights, tier_results, self.config.gamma)

    # --- Person C's compression (real implementation) ---

    def compress_delta(self, delta: torch.Tensor, tier: int) -> dict:
        return apply_tiered_compression(delta, tier, self.config)

    @staticmethod
    def _flatten(state_dict: OrderedDict) -> torch.Tensor:
        return torch.cat([v.flatten().float() for v in state_dict.values()])
