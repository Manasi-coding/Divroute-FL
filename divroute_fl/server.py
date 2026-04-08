"""
DivRoute-FL Lite — Server
Handles client selection, FedAvg aggregation, global-delta computation,
and stub methods for divergence / tiering / compression.
"""
import copy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config


class FLServer:
    """Central server that orchestrates federated training rounds."""

    def __init__(self, global_model: nn.Module, config: Config, device: torch.device):
        self.global_model = global_model.to(device)
        self.config = config
        self.device = device

        # Selection weights — uniform to start, Person B will modulate them.
        self.selection_weights = np.ones(config.num_clients, dtype=np.float64)

        # Will be populated after each aggregate() call.
        self.global_delta: torch.Tensor | None = None

        # Internal RNG for reproducible client selection
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def select_clients(self, all_ids: List[int]) -> List[int]:
        """
        Weighted random sample of ``clients_per_round`` clients.
        Weights are uniform for now — Person B will update them based on
        skip history (gamma decay).
        """
        weights = self.selection_weights[all_ids].copy()
        prob = weights / weights.sum()
        selected = self._rng.choice(
            all_ids,
            size=self.config.clients_per_round,
            replace=False,
            p=prob,
        )
        return selected.tolist()

    # ------------------------------------------------------------------
    # FedAvg aggregation + delta computation
    # ------------------------------------------------------------------

    def aggregate(self, client_results: List[dict]) -> None:
        """
        Standard FedAvg: weighted average of client state_dicts by
        ``num_samples``.  Before averaging, we snapshot the old global
        model as a flat vector so we can compute the global delta
        afterwards.
        """
        # 1. Snapshot old global params
        old_flat = self._flatten(self.global_model.state_dict())

        # 2. Compute total samples
        total_samples = sum(r["num_samples"] for r in client_results)

        # 3. Weighted average of state_dicts
        new_state: OrderedDict = OrderedDict()
        for key in self.global_model.state_dict():
            new_state[key] = torch.zeros_like(
                self.global_model.state_dict()[key], dtype=torch.float32,
            )
            for r in client_results:
                weight = r["num_samples"] / total_samples
                new_state[key] += weight * r["state_dict"][key].float()

        self.global_model.load_state_dict(new_state)

        # 4. Compute global delta
        new_flat = self._flatten(new_state)
        self.global_delta = new_flat - old_flat

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_loader: DataLoader) -> float:
        """Return top-1 accuracy of the current global model on *test_loader*."""
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.global_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Stubs — Person B and C will replace these
    # ------------------------------------------------------------------

    def compute_divergence(self, local_model: nn.Module, global_model: nn.Module) -> float:
        """Placeholder: always returns 1.0 (max divergence)."""
        return 1.0

    def assign_tier(self, d: float) -> int:
        """Placeholder: always returns tier 1 (high-fidelity delta)."""
        return 1

    def compress_delta(self, delta: torch.Tensor, tier: int) -> dict:
        """
        Placeholder: no compression — transmits the full delta.
        Returns a dict matching the interface Person C will implement:
            tier              – int
            values            – Tensor of selected values
            indices           – Tensor of corresponding indices
            bytes_transmitted – estimated bytes (float32 → 4 bytes per param)
        """
        return {
            "tier": tier,
            "values": delta.clone(),
            "indices": torch.arange(delta.numel()),
            "bytes_transmitted": delta.numel() * 4,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(state_dict: OrderedDict) -> torch.Tensor:
        """Concatenate all parameter tensors into a single 1-D vector."""
        return torch.cat([v.flatten().float() for v in state_dict.values()])
