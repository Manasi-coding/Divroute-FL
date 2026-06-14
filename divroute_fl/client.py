import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SimpleCNN


class FLClient:
    def __init__(self, client_id, dataset, local_epochs, local_lr, batch_size, device):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self.device = device
        self._local_model: nn.Module | None = None

        # persistent loader — created once, reused every round
        self._loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )

    def train(self, global_state_dict: dict, local_epochs: int | None = None) -> dict:
        """
        Accepts a state_dict (serialisable) instead of the model object —
        required for multiprocessing (model objects can't cross process boundaries).

        The returned dict includes `upload_bytes` (Phase 1.3): the cost of
        sending this client's full local model back to the server. Uploads are
        currently uncompressed float32 — this is measured and reported, not
        yet compressed (see paper limitations section).
        """
        epochs = local_epochs or self.local_epochs

        local_model = SimpleCNN().to(self.device)
        local_model.load_state_dict(global_state_dict)   # fast, no deepcopy
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(local_model.parameters(), lr=self.local_lr)

        for _ in range(epochs):
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = criterion(local_model(images), labels)
                loss.backward()
                opt.step()

        self._local_model = local_model

        

        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
            "num_samples": len(self.dataset),
            "divergence_score": None,
            "tier": None,
            "bytes_received": None,
            "upload_bytes": 0,  # placeholder; overwritten by server.aggregate()
        }

    def get_local_model(self) -> nn.Module | None:
        return self._local_model