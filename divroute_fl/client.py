import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import get_model


class FLClient:
    def __init__(self, client_id, dataset, local_epochs, local_lr, batch_size, device, model_name, num_classes):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self._local_model: nn.Module | None = None

        # persistent loader — created once, reused every round
        self._loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0, pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )

    def train(self, global_state_dict: dict, local_epochs: int | None = None,
              fedsparse_lambda: float = 0.0) -> dict:
        """
        Accepts a state_dict (serialisable) instead of the model object —
        required for multiprocessing (model objects can't cross process boundaries).

        The returned dict includes `upload_bytes` (Phase 1.3): the cost of
        sending this client's full local model back to the server. Uploads are
        currently uncompressed float32 — this is measured and reported, not
        yet compressed (see paper limitations section).

        fedsparse_lambda > 0 activates FedSparse (Phase 5 baseline): adds an
        L1 proximity term ||w_local - w_global||_1 to the cross-entropy loss,
        encouraging sparser gradient updates. Default 0.0 = standard training.
        """
        epochs = local_epochs or self.local_epochs

        local_model = get_model(self.model_name, self.num_classes).to(self.device)
        local_model.load_state_dict(global_state_dict)   # fast, no deepcopy
        local_model.train()

        # snapshot of the global weights for FedSparse proximity term
        if fedsparse_lambda > 0.0:
            global_params = torch.cat(
                [p.detach().flatten() for p in local_model.parameters()]
            )

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(local_model.parameters(), lr=self.local_lr)

        for _ in range(epochs):
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = criterion(local_model(images), labels)

                # ── FedSparse L1 regularisation ──────────────────────────────
                if fedsparse_lambda > 0.0:
                    local_params = torch.cat(
                        [p.flatten() for p in local_model.parameters()]
                    )
                    loss = loss + fedsparse_lambda * torch.norm(
                        local_params - global_params, p=1
                    )
                # ─────────────────────────────────────────────────────────────

                loss.backward()
                # Gradient clipping: prevents NaN/Inf weight explosions on
                # large models (ResNet-18) with SGD. Clip norm matches the
                # server-side grad_clip_norm default (10.0).
                nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)
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
