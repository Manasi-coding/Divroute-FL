import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data import DataLoader

from .model import get_model


# MixUp alpha — kept as a module-level constant so every client and every
# method uses the same value without any config changes.
_MIXUP_ALPHA = 0.4


DEBUG_BN = False
_DEBUG_BN_CLIENT = 0  # only log this client_id
_DEBUG_BN_LAYER = "bn1"  # only log this BN layer name


def _bn_snapshot(sd: dict, layer: str) -> str:
    """Return a compact multi-line string showing BN buffer values for `layer`."""
    keys = [
        f"{layer}.weight",
        f"{layer}.bias",
        f"{layer}.running_mean",
        f"{layer}.running_var",
        f"{layer}.num_batches_tracked",
    ]
    lines = []
    for k in keys:
        if k not in sd:
            lines.append(f"  {k}: MISSING")
            continue
        v = sd[k].float().cpu()
        if v.numel() == 1:
            lines.append(f"  {k}: {v.item():.6f}")
        else:
            # show first 4 values + norm
            lines.append(
                f"  {k}: [{', '.join(f'{x:.6f}' for x in v[:4].tolist())} ...] "
                f"norm={v.norm().item():.6f}"
            )
    return "\n".join(lines)


_EMA_DECAY = 0.95


def _update_ema(ema_model: nn.Module, live_model: nn.Module, decay: float) -> None:
    """Update all shadow-model parameters and buffers with an exponential average."""
    with torch.no_grad():
        ema_sd = ema_model.state_dict()
        live_sd = live_model.state_dict()
        for key in ema_sd:
            # Cast to float for the lerp computation, then cast back to the
            # original dtype (e.g. int64 for num_batches_tracked).
            ema_val = ema_sd[key].float()
            live_val = live_sd[key].float()
            updated = decay * ema_val + (1.0 - decay) * live_val
            ema_sd[key].copy_(updated.to(ema_sd[key].dtype))

        ema_model.load_state_dict(ema_sd)


def _mixup_loss(
    criterion: nn.CrossEntropyLoss,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute cross-entropy for the two MixUp targets."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def _ntd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor = None,
    tau: float = 3.0,
) -> torch.Tensor:
    """Distill the teacher distribution with both MixUp target classes masked."""
    s_nt = student_logits.clone()
    t_nt = teacher_logits.clone()

    # Mask out true classes by setting logits to a large negative number.
    # This forces their softmax probabilities to exactly 0.0 in both distributions,
    # effectively removing them from the KL divergence without changing the tensor shape.
    s_nt.scatter_(1, y_a.unsqueeze(1), -1e9)
    t_nt.scatter_(1, y_a.unsqueeze(1), -1e9)
    if y_b is not None:
        s_nt.scatter_(1, y_b.unsqueeze(1), -1e9)
        t_nt.scatter_(1, y_b.unsqueeze(1), -1e9)

    return F.kl_div(
        F.log_softmax(s_nt / tau, dim=1),
        F.softmax(t_nt / tau, dim=1),
        reduction="batchmean",
    ) * (tau**2)


def get_local_lr(
    base_lr: float, min_lr: float, round_num: int, total_rounds: int
) -> float:
    """Compute the cosine-annealed learning rate for a given global round."""
    if total_rounds <= 1:
        return base_lr
    t = round_num + 1
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * t / total_rounds)
    )


class FLClient:
    def __init__(
        self,
        client_id,
        dataset,
        local_epochs,
        local_lr,
        batch_size,
        device,
        model_name,
        num_classes,
        val_dataset=None,
        bn_mode="default",
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self.bn_mode = bn_mode
        self._local_model: nn.Module | None = None

        # Buffer for local_bn ablation
        self._local_bn_stats = None

        self._irw_norms: dict = {}

        if self.num_classes in (62, 100):
            effective_batch_size = min(batch_size, len(dataset))
            self._loader = DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )
        else:
            self._loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )

        if val_dataset is not None and len(val_dataset) > 0:
            val_bs = min(batch_size * 4, len(val_dataset))
            self._val_loader: DataLoader | None = DataLoader(
                val_dataset,
                batch_size=val_bs,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )
        else:
            self._val_loader = None

    def _prepare_sparse_update(self, local_model, fedsparse_lambda):
        global_snapshot = {
            name: param.detach().clone()
            for name, param in local_model.named_parameters()
        }

        raw = {
            name: self._irw_norms.get(name, 0.0)
            for name, _ in local_model.named_parameters()
        }
        raw_vals = list(raw.values())
        min_raw = min(raw_vals)
        max_raw = max(raw_vals)
        if max_raw == min_raw:  # bootstrap: all equal (incl. all 0)
            lambda_j = {name: fedsparse_lambda for name in raw}
        else:
            range_raw = max_raw - min_raw
            lambda_j = {
                name: fedsparse_lambda * (1.0 - (v - min_raw) / range_raw)
                for name, v in raw.items()
            }
        return global_snapshot, lambda_j

    def _evaluate_local_model(self, local_model):
        local_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = local_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        local_train_acc = correct / total if total > 0 else 0.0

        if self._val_loader is not None:
            val_correct, val_total = 0, 0
            val_loss_sum = 0.0
            val_criterion = nn.CrossEntropyLoss()  # no label smoothing for val
            with torch.no_grad():
                for images, labels in self._val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = local_model(images)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    val_loss_sum += val_criterion(logits, labels).item() * labels.size(
                        0
                    )
            local_val_acc: float | None = (
                val_correct / val_total if val_total > 0 else None
            )
            local_val_loss: float | None = (
                val_loss_sum / val_total if val_total > 0 else None
            )
        else:
            local_val_acc = None
            local_val_loss = None
        return local_train_acc, local_val_acc, local_val_loss

    def _log_trained_batchnorm(self, local_model, global_state_dict, round_num):
        if DEBUG_BN and self.client_id == _DEBUG_BN_CLIENT:
            raw_sd = local_model.state_dict()
            uploaded_sd = {
                k: v.cpu() for k, v in raw_sd.items()
            }  # matches return below

            print(
                f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                f"RAW LOCAL MODEL AFTER training ({_DEBUG_BN_LAYER}):"
            )
            print(_bn_snapshot(raw_sd, _DEBUG_BN_LAYER))

            print(
                f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                f"STATE_DICT UPLOADED TO SERVER ({_DEBUG_BN_LAYER}) "
                f"[note: raw model, NOT EMA]:"
            )
            print(_bn_snapshot(uploaded_sd, _DEBUG_BN_LAYER))

            # Check whether BN buffers differ from the incoming global state_dict
            bn_buffer_keys = [
                k
                for k in uploaded_sd
                if any(
                    s in k
                    for s in ("running_mean", "running_var", "num_batches_tracked")
                )
            ]
            present = [k for k in bn_buffer_keys if k in uploaded_sd]
            changed = [
                k
                for k in present
                if not torch.allclose(
                    uploaded_sd[k].float(),
                    global_state_dict[k].float().cpu(),
                    atol=1e-7,
                )
            ]
            print(
                f"\n[DEBUG_BN] BN buffers present in uploaded state_dict: "
                f"{len(present)} / {len(bn_buffer_keys)}"
            )
            print(
                f"[DEBUG_BN] BN buffers that changed vs global:          "
                f"{len(changed)} of {len(present)}"
            )
            if changed:
                print(f"[DEBUG_BN] Changed keys: {changed}")

    def _train_epochs(
        self,
        epochs,
        local_model,
        ema_model,
        opt,
        criterion,
        _beta_dist,
        ntd_beta,
        ntd_tau,
        teacher_model,
        fedprox_mu,
        fedprox_global_snapshot,
        fedsparse_lambda,
        global_snapshot,
        lambda_j,
    ):
        for _ in range(epochs):
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()

                lam = float(_beta_dist.sample().clamp(0.0, 1.0))
                lam = max(lam, 1.0 - lam)  # enforce lam ≥ 0.5

                # Shuffle indices within the current mini-batch only.
                # randperm is placed on CPU then used for indexing; no
                # GPU-allocated index tensor is needed.
                batch_size = images.size(0)
                index = torch.randperm(batch_size, device=self.device)

                # Linear interpolation of inputs — no extra GPU copy:
                # images[index] is a view-based gather, and the in-place
                # mul_ / add_ pair avoids allocating a third full tensor.
                mixed_x = lam * images + (1.0 - lam) * images[index]
                y_a, y_b = labels, labels[index]

                student_logits = local_model(mixed_x)
                loss = _mixup_loss(criterion, student_logits, y_a, y_b, lam)

                if ntd_beta > 0:
                    with torch.no_grad():
                        teacher_logits = teacher_model(mixed_x)
                    # Mask both MixUp labels to prevent gradient conflict.
                    loss = loss + ntd_beta * _ntd_loss(
                        student_logits, teacher_logits, y_a, y_b=y_b, tau=ntd_tau
                    )

                if fedprox_mu > 0.0:
                    prox_term = sum(
                        (param - fedprox_global_snapshot[name]).pow(2).sum()
                        for name, param in local_model.named_parameters()
                    )
                    loss = loss + (fedprox_mu / 2.0) * prox_term

                loss.backward()
                # Gradient clipping: prevents NaN/Inf weight explosions on
                # large models (ResNet-18) with SGD. Clip norm matches the
                # server-side grad_clip_norm default (10.0).
                nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)
                opt.step()

                if fedsparse_lambda > 0.0:
                    lr = opt.param_groups[0]["lr"]
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            threshold = lr * lambda_j[name]  # per-param λ_j
                            w_t = global_snapshot[name]  # frozen anchor
                            delta = param.data - w_t  # post-SGD delta
                            delta = delta.sign() * (delta.abs() - threshold).clamp(
                                min=0.0
                            )
                            param.data.copy_(w_t + delta)

                if (
                    fedsparse_lambda > 0.0
                ):  # EMA is consumed only by the FedSparse mask below
                    _update_ema(ema_model, local_model, _EMA_DECAY)
        return loss if "loss" in locals() else None

    def train(
        self,
        global_state_dict: dict,
        local_epochs: int | None = None,
        fedsparse_lambda: float = 0.0,
        ntd_beta: float = 0.0,
        ntd_tau: float = 3.0,
        fedprox_mu: float = 0.0,
        round_num: int = 0,
        total_rounds: int = 1,
        min_local_lr: float = 0.001,
    ) -> dict:
        """Train a local model and return its raw weights, metrics, and upload metadata."""
        epochs = local_epochs or self.local_epochs
        teacher_model = fedprox_global_snapshot = global_snapshot = lambda_j = None

        local_model = get_model(self.model_name, self.num_classes, self.bn_mode).to(
            self.device
        )
        local_model.load_state_dict(global_state_dict)  # fast, no deepcopy

        if self.bn_mode == "local_bn" and self._local_bn_stats is not None:
            local_model.load_state_dict(self._local_bn_stats, strict=False)

        local_model.train()

        if DEBUG_BN and self.client_id == _DEBUG_BN_CLIENT:
            print(
                f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                f"GLOBAL MODEL BEFORE training ({_DEBUG_BN_LAYER}):"
            )
            print(_bn_snapshot(global_state_dict, _DEBUG_BN_LAYER))

        if fedsparse_lambda > 0.0:
            global_snapshot, lambda_j = self._prepare_sparse_update(
                local_model, fedsparse_lambda
            )

        if fedprox_mu > 0.0:
            fedprox_global_snapshot = {
                name: param.detach().clone()
                for name, param in local_model.named_parameters()
            }

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        effective_lr = get_local_lr(
            self.local_lr, min_local_lr, round_num, total_rounds
        )

        opt = torch.optim.SGD(
            local_model.parameters(),
            lr=effective_lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        # MixUp: one Beta(α,α) sampler reused across all mini-batches.
        # Kept outside the epoch/batch loops to avoid re-allocating the
        # distribution object on every iteration.
        _beta_dist = Beta(
            torch.tensor(_MIXUP_ALPHA, device=self.device),
            torch.tensor(_MIXUP_ALPHA, device=self.device),
        )

        ema_model = get_model(self.model_name, self.num_classes, self.bn_mode).to(
            self.device
        )
        ema_model.load_state_dict(local_model.state_dict())
        ema_model.eval()  # EMA model is never trained; eval() disables dropout etc.

        if ntd_beta > 0:
            teacher_model = get_model(
                self.model_name, self.num_classes, self.bn_mode
            ).to(self.device)
            teacher_model.load_state_dict(global_state_dict)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)

        loss = self._train_epochs(
            epochs,
            local_model,
            ema_model,
            opt,
            criterion,
            _beta_dist,
            ntd_beta,
            ntd_tau,
            teacher_model,
            fedprox_mu,
            fedprox_global_snapshot,
            fedsparse_lambda,
            global_snapshot,
            lambda_j,
        )

        if fedsparse_lambda > 0.0:
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    self._irw_norms[name] = (
                        (param.data - global_snapshot[name]).norm(p=2).item()
                    )

        if fedsparse_lambda > 0.0:
            with torch.no_grad():
                live_sd = local_model.state_dict()
                for name, param in ema_model.named_parameters():
                    w_t = global_snapshot[name]
                    mask = live_sd[name] == w_t
                    param.data[mask] = w_t[mask]

        local_train_acc, local_val_acc, local_val_loss = self._evaluate_local_model(
            local_model
        )

        self._local_model = local_model

        self._log_trained_batchnorm(local_model, global_state_dict, round_num)

        if self.bn_mode == "local_bn":
            self._local_bn_stats = {
                k: v.cpu().clone()
                for k, v in local_model.state_dict().items()
                if "running" in k or "num_batches_tracked" in k
            }

        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
            "num_samples": len(
                self.dataset
            ),  # training shard size only — feeds aggregation weights
            "local_loss": loss.item() if loss is not None else 0.0,
            "local_train_acc": local_train_acc,
            "local_val_acc": local_val_acc,
            "local_val_loss": local_val_loss,
            "local_lr": effective_lr,
            "divergence_score": None,
            "tier": None,
            "bytes_received": None,
            "upload_bytes": 0,  # placeholder; overwritten by server.aggregate()
        }

    def get_local_model(self) -> nn.Module | None:
        return self._local_model
