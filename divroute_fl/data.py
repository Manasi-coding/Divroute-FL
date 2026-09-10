from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# ── Dataset normalisation stats ───────────────────────────────────────────────
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ImageNet normalisation stats — required when feeding a pretrained backbone.
# Pretrained features are calibrated to this specific normalisation; using the
# CIFAR-specific stats above with a pretrained model would hurt transfer
# accuracy independent of anything federation-related.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Target input resolution for pretrained backbones. CIFAR's native 32x32 is
# too small for these architectures' stems/downsampling stages (the same
# class of problem model.py's ResNet-18 stem fix addresses for that
# architecture — here the fix is upsampling the input instead of modifying
# the pretrained stem, since modifying it would invalidate the pretrained
# weights in those layers). 128 is a practical compute/accuracy compromise;
# raise to e.g. 224 if wall-clock budget allows — this is a single-line change.
_PRETRAINED_IMAGE_SIZE = {
    "efficientnet_b0_pretrained": 128,
}


def _resolve_transform_params(dataset_name: str, model_name: str = None):
    """Return (mean, std, resize_to) for a (dataset, model) combination.

    resize_to is None for CIFAR-native models (unchanged behaviour — no Resize
    step, CIFAR-specific normalisation) or an int (target square resolution)
    for pretrained backbones, which also switches normalisation to ImageNet
    statistics. Resize/normalisation are derived from model_name rather than
    a separate config flag so the two settings can never drift out of sync.
    model_name=None (the default) always preserves CIFAR-native behaviour,
    so every existing caller that doesn't pass it is completely unaffected.
    """
    name = dataset_name.lower().strip()
    if name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Supported values: 'cifar10', 'cifar100'."
        )

    resize_to = _PRETRAINED_IMAGE_SIZE.get((model_name or "").lower().strip())
    if resize_to is not None:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return mean, std, resize_to


def _make_train_transform(dataset_name: str, model_name: str = None) -> transforms.Compose:
    """Return the training transform for the requested dataset (+ optional model).

    Includes standard data augmentation (RandomCrop + RandomHorizontalFlip)
    applied before tensor conversion.  Applied only to client training data;
    the test transform (_make_test_transform) is kept augmentation-free.

    Augmentation is identical for every client because all clients share the
    same torchvision Dataset object (partitioned via index Subsets).  Every
    FL method (FedAvg, FedSparse, FedZip, Uniform, DivRoute) therefore sees
    the same augmented distribution, preserving experimental fairness.

    When model_name selects a pretrained backbone, a Resize step is inserted
    AFTER RandomCrop/RandomHorizontalFlip (so crop/flip augmentation strength
    is unchanged from the CIFAR-native pipeline — it still operates on the
    32x32 image — and only the final upsampled resolution differs) and BEFORE
    ToTensor/Normalize (which switch to ImageNet statistics).
    """
    mean, std, resize_to = _resolve_transform_params(dataset_name, model_name)
    ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if resize_to is not None:
        ops.append(transforms.Resize(resize_to))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]
    return transforms.Compose(ops)


def _make_test_transform(dataset_name: str, model_name: str = None) -> transforms.Compose:
    """Return the evaluation/test transform for the requested dataset (+ optional model).

    No augmentation — deterministic Resize (if applicable) + ToTensor + Normalize only.
    """
    mean, std, resize_to = _resolve_transform_params(dataset_name, model_name)
    ops = []
    if resize_to is not None:
        ops.append(transforms.Resize(resize_to))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def _num_classes(dataset_name: str) -> int:
    return 10 if dataset_name.lower().strip() == "cifar10" else 100


def _load_train(dataset_name: str, transform) -> Dataset:
    name = dataset_name.lower().strip()
    if name == "cifar10":
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        return datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)


def _load_test(dataset_name: str, transform) -> Dataset:
    name = dataset_name.lower().strip()
    if name == "cifar10":
        return datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        return datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)


# ── Public API ────────────────────────────────────────────────────────────────

def get_client_datasets(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    model_name: str = None,
) -> List[Subset]:
    """
    Split a dataset's training set across clients using Dirichlet(alpha) partitioning.
    Lower alpha = more heterogeneous (each client ends up with mostly one or two classes).

    Supported dataset_name values: "cifar10", "cifar100"

    The Dirichlet partitioning logic is identical for both datasets;
    the only difference is num_classes (10 or 100) and the normalisation transform.

    model_name : optional. When set to a pretrained-backbone model_name (see
        model.py's get_model()), the training transform resizes images and
        switches to ImageNet normalisation to match that backbone's pretrained
        calibration. None (default) preserves exact CIFAR-native behaviour.
    """
    transform    = _make_train_transform(dataset_name, model_name)   # augmented training transform
    full_train   = _load_train(dataset_name, transform)
    targets      = np.array(full_train.targets)
    num_classes  = _num_classes(dataset_name)

    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)

        # draw Dirichlet proportions — this is the core of the non-IID split
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        if dataset_name.lower().strip() == "cifar100":
            exact = proportions * len(cls_idx)
            counts = np.floor(exact).astype(int)
            remainders = exact - counts
            leftover = len(cls_idx) - counts.sum()

            # Largest-remainder apportionment: give the leftover units to whichever
            # clients were closest to rounding up, instead of always client 0, 1, 2...
            top_leftover_clients = np.argsort(-remainders)[:leftover]
            counts[top_leftover_clients] += 1
        else:
            counts = (proportions * len(cls_idx)).astype(int)

            # fix any off-by-one from rounding
            leftover = len(cls_idx) - counts.sum()
            for i in range(leftover):
                counts[i % num_clients] += 1

        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end

    return [Subset(full_train, idxs) for idxs in client_indices]


def get_client_datasets_with_val(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    val_fraction: float = 0.1,
    model_name: str = None,
) -> tuple:
    """Split training data across clients (Dirichlet) then carve a held-out
    local validation set per client.

    Returns
    -------
    train_subsets : List[Subset]
        90% (or 1 - val_fraction) of each client's shard, backed by the
        augmented training Dataset.  These are the *only* samples ever seen
        during forward/backward passes, optimiser steps, or divergence scoring.

    val_subsets : List[Subset | None]
        10% (or val_fraction) of each client's shard, backed by a *separate*
        Dataset object loaded with the test transform (no augmentation).
        ``None`` for clients whose shard is too small (< 10 samples total) to
        yield at least 1 validation sample after rounding.

    Design guarantees
    -----------------
    * The Dirichlet partitioning is byte-for-byte identical to
      ``get_client_datasets()`` for the same (dataset_name, num_clients, alpha,
      seed) arguments — only the final Subset slicing differs.
    * The val/train split uses a second RNG seeded with
      ``seed + client_id + 99999`` so the shuffle is independent of the
      Dirichlet RNG and reproducible across runs with the same seed.
    * Validation samples are backed by the test-transform Dataset, so they
      are never subject to RandomCrop, RandomHorizontalFlip, or RandomErasing.

    Compatibility
    -------------
    Clients with ``val_fraction > 0`` train on fewer samples than they would
    under ``get_client_datasets()``.  Do NOT mix results from this function
    with checkpoints produced when ``val_fraction == 0``.
    """
    _MIN_SHARD_FOR_VAL = 10   # shards smaller than this get no validation set

    # ── Step 1: identical Dirichlet partitioning ──────────────────────────────
    # Re-uses the same logic as get_client_datasets(); the full_train Dataset
    # here uses the *training* (augmented) transform so train Subsets share it.
    train_transform = _make_train_transform(dataset_name, model_name)
    full_train      = _load_train(dataset_name, train_transform)
    targets         = np.array(full_train.targets)
    num_classes     = _num_classes(dataset_name)

    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        if dataset_name.lower().strip() == "cifar100":
            exact = proportions * len(cls_idx)
            counts = np.floor(exact).astype(int)
            remainders = exact - counts
            leftover = len(cls_idx) - counts.sum()
            top_leftover_clients = np.argsort(-remainders)[:leftover]
            counts[top_leftover_clients] += 1
        else:
            counts = (proportions * len(cls_idx)).astype(int)
            leftover = len(cls_idx) - counts.sum()
            for i in range(leftover):
                counts[i % num_clients] += 1

        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end

    # ── Step 2: load a second Dataset for val (test transform, no augmentation)
    test_transform = _make_test_transform(dataset_name, model_name)
    full_train_noaug = _load_train(dataset_name, test_transform)

    # ── Step 3: split each client's indices into train / val ─────────────────
    train_subsets: list = []
    val_subsets:   list = []

    for cid, idxs in enumerate(client_indices):
        n = len(idxs)

        if n < _MIN_SHARD_FOR_VAL:
            # Shard too small to safely carve a val set — keep all for training.
            train_subsets.append(Subset(full_train, idxs))
            val_subsets.append(None)
            continue

        val_size = max(1, int(n * val_fraction))
        if val_size >= n:
            # Pathological: val_fraction ≥ 1.0 — keep at least 1 train sample.
            val_size = n - 1

        # Deterministic per-client shuffle, independent of the Dirichlet RNG.
        # Seed: seed + client_id + 99999 avoids any overlap with the main seed.
        client_rng = np.random.default_rng(seed + cid + 99999)
        shuffled = np.array(idxs, dtype=np.int64)
        client_rng.shuffle(shuffled)

        train_idxs = shuffled[val_size:].tolist()   # trailing 90%
        val_idxs   = shuffled[:val_size].tolist()   # leading 10%

        train_subsets.append(Subset(full_train, train_idxs))
        val_subsets.append(Subset(full_train_noaug, val_idxs))

    return train_subsets, val_subsets


def get_test_dataset(dataset_name: str = "cifar10", model_name: str = None) -> Dataset:
    """Return the test split for the requested dataset.

    Supported dataset_name values: "cifar10", "cifar100"
    Default is "cifar10" for backward compatibility.

    model_name : optional, see get_client_datasets() — must match whatever was
        passed there so train and test data go through identical resize/
        normalisation (mismatched preprocessing between train/test would
        silently corrupt evaluation).
    """
    transform = _make_test_transform(dataset_name, model_name)   # no augmentation on test data
    return _load_test(dataset_name, transform)