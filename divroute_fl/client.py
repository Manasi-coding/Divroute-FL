"""
DivRoute-FL Lite — Client
Each FLClient owns a private dataset shard and performs local SGD.
"""
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FLClient:
    """Simulates a single federated-learning client."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        local_epochs: int,
        local_lr: float,
        batch_size: int,
        device: torch.device,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self.device = device

        # Keeps a reference to the last-trained local model
        # so that Person B can compute divergence later.
        self._local_model: nn.Module | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, global_model: nn.Module) -> dict:
        """
        Deep-copy the global model, train it on local data for E epochs.

        Returns
        -------
        dict with keys:
            client_id       – int
            state_dict      – OrderedDict of trained weights
            num_samples     – size of local dataset
            divergence_score – None (filled by Person B)
            tier             – None (filled by Person B)
            bytes_received   – None (filled by Person C)
        """
        # Deep-copy so training doesn't mutate the server's model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(local_model.parameters(), lr=self.local_lr)

        for _epoch in range(self.local_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimiser.zero_grad()
                loss = criterion(local_model(images), labels)
                loss.backward()
                optimiser.step()

        # Cache the trained local model for later divergence checks
        self._local_model = local_model

        return {
            "client_id": self.client_id,
            "state_dict": copy.deepcopy(local_model.state_dict()),
            "num_samples": len(self.dataset),
            "divergence_score": None,   # placeholder – Person B
            "tier": None,               # placeholder – Person B
            "bytes_received": None,     # placeholder – Person C
        }

    def get_local_model(self) -> nn.Module | None:
        """Return the most-recently-trained local model (or None)."""
        return self._local_model
