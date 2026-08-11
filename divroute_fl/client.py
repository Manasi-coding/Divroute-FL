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

# ── BatchNorm propagation diagnostic ─────────────────────────────────────────
# Set True to print BN state at key moments for the first participating client.
# Has zero effect on any computation — all prints are gated on this flag.
# Set False (default) for all production / ablation runs.
DEBUG_BN = False
_DEBUG_BN_CLIENT  = 0      # only log this client_id
_DEBUG_BN_LAYER   = "bn1"  # only log this BN layer name


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
# ─────────────────────────────────────────────────────────────────────────────

# EMA decay — 0.999 gives a half-life of ~693 steps, providing a smooth
# trailing average that is robust to noisy mini-batch gradient updates
# while tracking the training trajectory closely enough to benefit from
# late-round convergence.  Chosen to match the standard torchvision EMA
# recipe used in the official ResNet training scripts.
_EMA_DECAY = 0.95


def _update_ema(ema_model: nn.Module, live_model: nn.Module, decay: float) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * live.

    Operates on all tensors in state_dict (parameters AND buffers), so
    BatchNorm running statistics are also smoothed.  All operations are
    performed under torch.no_grad() to guarantee that EMA tensors never
    accumulate gradients.

    Parameters
    ----------
    ema_model  : the shadow model whose weights are updated in-place
    live_model : the model being trained (source of the new values)
    decay      : EMA decay coefficient (e.g. 0.999)
    """
    with torch.no_grad():
        ema_sd  = ema_model.state_dict()
        live_sd = live_model.state_dict()
        for key in ema_sd:
            # Cast to float for the lerp computation, then cast back to the
            # original dtype (e.g. int64 for num_batches_tracked).
            ema_val  = ema_sd[key].float()
            live_val = live_sd[key].float()
            updated  = decay * ema_val + (1.0 - decay) * live_val
            ema_sd[key].copy_(updated.to(ema_sd[key].dtype))
        # state_dict() returns copies — we must push the mutated dict back.
        ema_model.load_state_dict(ema_sd)


def _mixup_loss(
    criterion: nn.CrossEntropyLoss,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixed cross-entropy loss: λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b).

    Works correctly alongside label_smoothing because CrossEntropyLoss
    applies smoothing independently to each call; the two weighted terms
    are then summed, which is algebraically equivalent to smoothing the
    linearly-interpolated soft label directly.

    Parameters
    ----------
    criterion : nn.CrossEntropyLoss instance (with label_smoothing=0.1)
    pred      : raw logits from the model, shape (B, C)
    y_a       : integer class labels for the original batch, shape (B,)
    y_b       : integer class labels for the shuffled batch, shape (B,)
    lam       : scalar mixing coefficient drawn from Beta(alpha, alpha)
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def _ntd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
              y_a: torch.Tensor, y_b: torch.Tensor = None, tau: float = 3.0) -> torch.Tensor:
    """Not-True Distillation loss (NeurIPS 2022).

    Computes the KL divergence between student and teacher softmax
    distributions over the non-true classes only.  This preserves the
    global model's knowledge about inter-class relationships without
    conflicting with the cross-entropy loss on the true class(es).

    Parameters
    ----------
    student_logits : (B, C) raw logits from the local model
    teacher_logits : (B, C) raw logits from the frozen global model
    y_a            : (B,)   primary integer class labels
    y_b            : (B,)   optional secondary integer class labels (for MixUp)
    tau            : temperature for soft probabilities
    """
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
        reduction='batchmean',
    ) * (tau ** 2)


def _cosine_lr(base_lr: float, round_num: int, total_rounds: int) -> float:
    """Compute the cosine-annealed learning rate for a given global round.

    Decays from ``base_lr`` at round 0 to exactly ``base_lr * 0.01`` at the
    final round (``total_rounds - 1``) following the standard cosine schedule:

        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / (T - 1)))

    The denominator is ``T - 1`` (not ``T``) because the loop in main.py runs
    ``for rnd in range(total_rounds)``, so ``round_num`` ranges over the closed
    integer set {0, 1, …, T-1}.  Using T-1 as denominator maps this domain
    exactly onto the cosine interval [0, π]:

        t = 0   → cos(0)   = +1  → lr = base_lr          (maximum)
        t = T-1 → cos(π)   = -1  → lr = lr_min = base_lr * 0.01  (minimum)

    Using T instead (the original formula) maps the domain onto [0, π(T-1)/T],
    which reaches only ~π - π/T at the final round.  With T=100 this leaves
    the LR ~2.5 % above lr_min on the last round — never reaching the target
    minimum.

    Parameters
    ----------
    base_lr      : the configured initial learning rate (``Config.local_lr``)
    round_num    : 0-indexed current global communication round
    total_rounds : total number of communication rounds (``Config.num_rounds``)

    Design note
    -----------
    A pure function of the global round is the correct pattern for federated
    learning.  PyTorch's ``CosineAnnealingLR`` requires a persistent
    ``(optimizer, scheduler)`` pair that is stepped each call.  Because the
    client optimizer is recreated fresh inside ``FLClient.train()`` on every
    round, attaching a scheduler would restart the cosine curve every round.
    Computing the LR analytically from ``round_num`` and setting it directly
    avoids this restart and ensures a single, monotone decay across the full
    training run.
    """
    lr_min = base_lr * 0.05
    # Guard: T-1 == 0 when total_rounds == 1, which would cause ZeroDivisionError.
    # With a single round there is nothing to anneal; return the full base_lr.
    if total_rounds <= 1:
        return base_lr
    return lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * round_num / (total_rounds - 1)))


class FLClient:
    def __init__(self, client_id, dataset, local_epochs, local_lr, batch_size, device, model_name, num_classes,
                 val_dataset=None, bn_mode="default"):
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

        # ── FedSparse Stage 3: IRW persistent buffer ─────────────────────────
        # Stores ‖w_j^local − w_j^global‖₂ for every named parameter after
        # each round this client participates in.  On the NEXT round these norms
        # are normalised and scaled by fedsparse_lambda to produce per-parameter
        # threshold coefficients λ_j (Algorithm 2, Eq. 4).  Empty until the
        # client has completed at least one FedSparse round (bootstrap: uniform).
        self._irw_norms: dict = {}

        # persistent training loader — created once, reused every round
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

        # ── Local validation loader (diagnostic only) ─────────────────────────
        # Created only when a held-out val_dataset is provided (local_val_fraction > 0).
        # Validation samples are backed by the test-transform (no-augmentation) Dataset
        # so they are never affected by RandomCrop, RandomHorizontalFlip, or RandomErasing.
        # This loader is NEVER used in the training loop, loss computation, optimiser
        # step, divergence scoring, EMA update, or aggregation.
        if val_dataset is not None and len(val_dataset) > 0:
            val_bs = min(batch_size * 4, len(val_dataset))
            self._val_loader: DataLoader | None = DataLoader(
                val_dataset, batch_size=val_bs, shuffle=False,
                drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )
        else:
            self._val_loader = None
        # ─────────────────────────────────────────────────────────────────────

    def train(self, global_state_dict: dict, local_epochs: int | None = None,
              fedsparse_lambda: float = 0.0,
              ntd_beta: float = 0.0, ntd_tau: float = 3.0,
              round_num: int = 0, total_rounds: int = 1) -> dict:
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

        round_num / total_rounds drive the global cosine LR schedule.  All
        clients selected in the same communication round receive the same
        round_num, so they are guaranteed to train with an identical LR.
        """
        epochs = local_epochs or self.local_epochs

        local_model = get_model(self.model_name, self.num_classes, self.bn_mode).to(self.device)
        local_model.load_state_dict(global_state_dict)   # fast, no deepcopy
        
        if self.bn_mode == "local_bn" and self._local_bn_stats is not None:
            local_model.load_state_dict(self._local_bn_stats, strict=False)
            
        local_model.train()

        # ── [DEBUG_BN] Log global BN state BEFORE local training ──────────────
        if DEBUG_BN and self.client_id == _DEBUG_BN_CLIENT:
            print(f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                  f"GLOBAL MODEL BEFORE training ({_DEBUG_BN_LAYER}):")
            print(_bn_snapshot(global_state_dict, _DEBUG_BN_LAYER))
        # ──────────────────────────────────────────────────────────────────────

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

        # Label smoothing (ε=0.1): replaces hard one-hot targets with a soft
        # distribution (0.9 on the true class, 0.1/C spread over all C classes).
        # Applied only during local client training; evaluation in server.py uses
        # argmax-based top-1 accuracy with hard labels and is completely unaffected.
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Global cosine LR schedule: computed as a pure function of the current
        # communication round so the LR decays smoothly from local_lr (round 0)
        # to local_lr*0.01 (round total_rounds) across the full training run.
        # Because the formula depends only on round_num (not on any accumulated
        # optimizer state), recreating the optimizer every round is correct and
        # intentional — there is no "restart" of the cosine curve.
        # All clients selected in a given round share the same round_num, so
        # they train with an identical LR, preserving experimental fairness.
        effective_lr = _cosine_lr(self.local_lr, round_num, total_rounds)
        # Standard ResNet-18 / CIFAR-100 optimiser recipe (momentum=0.9, WD=5e-4,
        # Nesterov=True).  The optimiser is constructed fresh on every call to
        # train(), so its momentum buffers are scoped to this local_model instance
        # and are garbage-collected when train() returns.  There is no mechanism
        # by which momentum state can bleed between clients or across communication
        # rounds — each invocation starts from a clean buffer.
        # Weight decay is applied solely during local SGD steps; it does not
        # affect the server aggregation, compression, or communication accounting.
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

        # ── EMA shadow model ──────────────────────────────────────────────
        # Initialised from the global state_dict (same as local_model) so
        # the EMA starts exactly at the global checkpoint and immediately
        # begins tracking the per-step SGD trajectory.
        #
        # requires_grad is False for all EMA parameters by construction:
        # get_model() returns a freshly initialised model and we never pass
        # ema_model.parameters() to any optimiser.  _update_ema() operates
        # solely through state_dict copies under torch.no_grad().
        #
        # ema_model is scoped to this train() call — it is not stored on
        # self, so there is no possibility of EMA state leaking across
        # clients or across communication rounds.
        ema_model = get_model(self.model_name, self.num_classes).to(self.device)
        ema_model.load_state_dict(local_model.state_dict())
        ema_model.eval()   # EMA model is never trained; eval() disables dropout etc.
        # ─────────────────────────────────────────────────────────────────

        # ── FedNTD: frozen teacher model ──────────────────────────────────
        # The global model serves as the teacher for Not-True Distillation.
        # It is frozen (no grad) and kept in eval mode.  The teacher sees
        # the same mixed inputs as the student so the NTD loss is computed
        # on the correct input distribution.
        if ntd_beta > 0:
            teacher_model = get_model(self.model_name, self.num_classes, self.bn_mode).to(self.device)
            teacher_model.load_state_dict(global_state_dict)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
        # ─────────────────────────────────────────────────────────────────

        for _ in range(epochs):
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt.zero_grad()

                # ── MixUp data augmentation ───────────────────────────────
                # Sample λ from Beta(α, α) and enforce λ ≥ 0.5 by taking
                # max(λ, 1−λ).  This is the standard convention (also used
                # in torchvision's MixUp implementation) that avoids the
                # symmetry between y_a and y_b being broken by a very small λ.
                lam = float(_beta_dist.sample().clamp(0.0, 1.0))
                lam = max(lam, 1.0 - lam)        # enforce lam ≥ 0.5

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
                # ─────────────────────────────────────────────────────────

                student_logits = local_model(mixed_x)
                loss = _mixup_loss(criterion, student_logits, y_a, y_b, lam)

                # ── FedNTD: Not-True Distillation loss ────────────────────
                if ntd_beta > 0:
                    with torch.no_grad():
                        teacher_logits = teacher_model(mixed_x)
                    # Mask both MixUp labels to prevent gradient conflict.
                    loss = loss + ntd_beta * _ntd_loss(
                        student_logits, teacher_logits, y_a, y_b=y_b, tau=ntd_tau)
                # ─────────────────────────────────────────────────────────

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

                # ── EMA step ─────────────────────────────────────────────────
                # Updated AFTER opt.step() AND after the FedSparse proximal
                # operator so that the EMA tracks the final post-prox parameter
                # values at every step, not the intermediate pre-prox values.
                _update_ema(ema_model, local_model, _EMA_DECAY)
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

        # ── EMA + FedSparse Sparsity Fix ──────────────────────────────────────────
        # If FedSparse is active, the EMA model contains exponentially decaying
        # non-zero residuals for coordinates that were active during training but
        # were later zeroed by the proximal operator. This destroys the exact
        # structural sparsity required for communication savings.
        # We restore the exact sparsity pattern of the final local model by
        # masking the EMA model: if the final raw parameter equals the frozen
        # global anchor, we force the EMA parameter to equal the global anchor.
        if fedsparse_lambda > 0.0:
            with torch.no_grad():
                live_sd = local_model.state_dict()
                for name, param in ema_model.named_parameters():
                    w_t = global_snapshot[name]
                    mask = (live_sd[name] == w_t)
                    param.data[mask] = w_t[mask]
        # ─────────────────────────────────────────────────────────────────────────

        # ── Evaluate Local Train Accuracy ────────────────────────────────────────
        # Evaluated on the *training* loader to measure how well the model fits
        # its local training data.  Named "local_train_acc" to distinguish it from
        # the held-out validation accuracy below.  This metric was previously
        # labelled "local_accuracy" and is unchanged in computation.
        local_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self._loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = local_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        local_train_acc = correct / total if total > 0 else 0.0
        # ─────────────────────────────────────────────────────────────────────────

        # ── Evaluate Local Val Accuracy (diagnostic only) ─────────────────────────
        # Evaluated on the held-out validation subset (test transform, no augmentation).
        # None when no val loader was provided (local_val_fraction == 0.0).
        # This block does NOT affect training, routing, aggregation, or any other
        # DivRoute logic — it is purely diagnostic.
        if self._val_loader is not None:
            val_correct, val_total = 0, 0
            val_loss_sum = 0.0
            val_criterion = nn.CrossEntropyLoss()   # no label smoothing for val
            with torch.no_grad():
                for images, labels in self._val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = local_model(images)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)
                    val_loss_sum += val_criterion(logits, labels).item() * labels.size(0)
            local_val_acc: float | None  = val_correct / val_total if val_total > 0 else None
            local_val_loss: float | None = val_loss_sum / val_total if val_total > 0 else None
        else:
            local_val_acc  = None
            local_val_loss = None

        # ─────────────────────────────────────────────────────────────────────────

        # Keep the raw trained model for divergence scoring in main.py.
        # main.py computes cosine similarity between the client's trained
        # flat parameter vector and the global flat parameter vector to
        # assign divergence scores and tiers.  Using the raw model (which
        # reflects the actual gradient signal) is correct here — the EMA
        # weights would understate the divergence because they lag behind
        # the training trajectory.
        self._local_model = local_model

        # ── [DEBUG_BN] Log raw post-training and uploaded BN state ────────────
        if DEBUG_BN and self.client_id == _DEBUG_BN_CLIENT:
            raw_sd      = local_model.state_dict()
            uploaded_sd = {k: v.cpu() for k, v in raw_sd.items()}  # matches return below

            print(f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                  f"RAW LOCAL MODEL AFTER training ({_DEBUG_BN_LAYER}):")
            print(_bn_snapshot(raw_sd, _DEBUG_BN_LAYER))

            print(f"\n[DEBUG_BN] client={self.client_id} round={round_num}  "
                  f"STATE_DICT UPLOADED TO SERVER ({_DEBUG_BN_LAYER}) "
                  f"[note: raw model, NOT EMA]:")
            print(_bn_snapshot(uploaded_sd, _DEBUG_BN_LAYER))

            # Check whether BN buffers differ from the incoming global state_dict
            bn_buffer_keys = [k for k in uploaded_sd
                              if any(s in k for s in
                                     ("running_mean", "running_var", "num_batches_tracked"))]
            present = [k for k in bn_buffer_keys if k in uploaded_sd]
            changed = [k for k in present
                       if not torch.allclose(
                           uploaded_sd[k].float(),
                           global_state_dict[k].float().cpu(),
                           atol=1e-7)]
            print(f"\n[DEBUG_BN] BN buffers present in uploaded state_dict: "
                  f"{len(present)} / {len(bn_buffer_keys)}")
            print(f"[DEBUG_BN] BN buffers that changed vs global:          "
                  f"{len(changed)} of {len(present)}")
            if changed:
                print(f"[DEBUG_BN] Changed keys: {changed}")
        # ──────────────────────────────────────────────────────────────────────

        if self.bn_mode == "local_bn":
            self._local_bn_stats = {
                k: v.cpu().clone()
                for k, v in local_model.state_dict().items()
                if "running" in k or "num_batches_tracked" in k
            }

        return {
            "client_id": self.client_id,
            # Upload the raw locally-trained model, not the EMA shadow model.
            # EMA is retained (ema_model is still in scope) for potential local
            # evaluation or diagnostics, but is not transmitted to the server.
            # The server must receive w_T (final post-SGD weights) to compute
            # the true gradient signal Δ = w_T - w_global.  Transmitting the
            # EMA model ê_T instead causes a systematic dampening of every
            # aggregated update by a factor of ~0.64 (with β=0.95, T=48 steps).
            "state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
            "num_samples": len(self.dataset),  # training shard size only — feeds aggregation weights
            "local_loss": loss.item() if 'loss' in locals() else 0.0,
            "local_train_acc": local_train_acc,
            "local_val_acc":   local_val_acc,
            "local_val_loss":  local_val_loss,
            "divergence_score": None,
            "tier": None,
            "bytes_received": None,
            "upload_bytes": 0,  # placeholder; overwritten by server.aggregate()
        }


    def get_local_model(self) -> nn.Module | None:
        return self._local_model
