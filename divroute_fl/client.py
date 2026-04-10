import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FLClient:
    def __init__(self, client_id, dataset, local_epochs, local_lr, batch_size, device):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self.device = device

        # stash the trained model so Person B can compute divergence later
        self._local_model: nn.Module | None = None

    def train(self, global_model: nn.Module) -> dict:
        # deep-copy so we don't accidentally mutate the server's model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(local_model.parameters(), lr=self.local_lr)

        for _ in range(self.local_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = criterion(local_model(images), labels)
                loss.backward()
                opt.step()

        self._local_model = local_model

        return {
            "client_id": self.client_id,
            "state_dict": copy.deepcopy(local_model.state_dict()),
            "num_samples": len(self.dataset),
            "divergence_score": None,  # Person B fills this
            "tier": None,              # Person B fills this
            "bytes_received": None,    # Person C fills this
        }

    def get_local_model(self) -> nn.Module | None:
        return self._local_model
