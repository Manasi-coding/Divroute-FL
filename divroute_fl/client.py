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

        # ── FedSparse Stage 3: IRW persistent buffer ─────────────────────────
        # Stores ‖w_j^local − w_j^global‖₂ for every named parameter after
        # each round this client participates in.  On the NEXT round these norms
        # are normalised and scaled by fedsparse_lambda to produce per-parameter
        # threshold coefficients λ_j (Algorithm 2, Eq. 4).  Empty until the
        # client has completed at least one FedSparse round (bootstrap: uniform).
        self._irw_norms: dict = {}

        # persistent loader — created once, reused every round
        if self.num_classes == 100:
            effective_batch_size = min(batch_size, len(dataset))
            self._loader = DataLoader(
                dataset, batch_size=effective_batch_size, shuffle=True,
                drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )
        else:
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

        # ── FedSparse Stage 1: snapshot global weights before local training ──
        # Stored as a named-parameter dict so each tensor can be paired with
        # its corresponding current parameter by name during the L1 penalty
        # computation.  detach() + clone() guarantees no gradient linkage and
        # no shared storage with the live model parameters.
        if fedsparse_lambda > 0.0:
            global_snapshot = {
                name: param.detach().clone()
                for name, param in local_model.named_parameters()
            }

            # ── FedSparse Stage 3: compute per-parameter λ_j ─────────────────
            # Uses AU norms stored from the PREVIOUS round this client took part
            # in.  This resolves Algorithm 2's ordering ("initialise λ_j before
            # training") without a second training pass: the norms are available
            # immediately because they were saved when the previous round ended.
            #
            # Bootstrap (round 0 or first participation): _irw_norms is empty,
            # so max_raw == min_raw == 0.  The guard below sets sigma_j = 1.0
            # for every parameter, making λ_j = fedsparse_lambda uniformly —
            # identical to the Stage 2 uniform behaviour.
            raw = {name: self._irw_norms.get(name, 0.0)
                   for name, _ in local_model.named_parameters()}
            raw_vals = list(raw.values())
            min_raw  = min(raw_vals)
            max_raw  = max(raw_vals)
            if max_raw == min_raw:          # bootstrap: all equal (incl. all 0)
                lambda_j = {name: fedsparse_lambda for name in raw}
            else:
                range_raw = max_raw - min_raw
                lambda_j  = {
                    name: fedsparse_lambda * (
                        1.0 - (v - min_raw) / range_raw
                    )
                    for name, v in raw.items()
                }
            # ─────────────────────────────────────────────────────────────────

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(local_model.parameters(), lr=self.local_lr)

        for _ in range(epochs):
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = criterion(local_model(images), labels)



                loss.backward()
                # Gradient clipping: prevents NaN/Inf weight explosions on
                # large models (ResNet-18) with SGD. Clip norm matches the
                # server-side grad_clip_norm default (10.0).
                nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)
                opt.step()

                # ── FedSparse Stage 2 + 3: shifted proximal operator ──────
                # Derived from the paper's objective h_k(w) = F_k(w) + λ‖w−w_t‖₁.
                # Threshold is lr * λ_j[name] — per-parameter (Stage 3 IRW).
                # On bootstrap (_irw_norms empty) λ_j == fedsparse_lambda uniformly,
                # giving identical behaviour to Stage 2 alone.
                # δ = w_after_SGD − w_global  →  soft-threshold  →  w ← w_global + δ
                if fedsparse_lambda > 0.0:
                    lr = opt.param_groups[0]["lr"]
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            threshold = lr * lambda_j[name]         # per-param λ_j
                            w_t   = global_snapshot[name]           # frozen anchor
                            delta = param.data - w_t                # post-SGD delta
                            delta = delta.sign() * (
                                delta.abs() - threshold
                            ).clamp(min=0.0)
                            param.data.copy_(w_t + delta)
                # ─────────────────────────────────────────────────────────────

        # ── FedSparse Stage 3: refresh IRW norm buffer ───────────────────────
        # After all local epochs are done, record ‖w_j^local − w_j^global‖₂
        # for every named parameter.  These scalars are used as AU_j on the
        # NEXT round this client is selected (Algorithm 2 Eq. 4 / Section 4.3).
        # Computed under no_grad; .item() moves each scalar to CPU so the buffer
        # contains plain Python floats regardless of device (CUDA-safe).
        if fedsparse_lambda > 0.0:
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    self._irw_norms[name] = (
                        (param.data - global_snapshot[name])
                        .norm(p=2).item()
                    )
        # ─────────────────────────────────────────────────────────────────────

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
