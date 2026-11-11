from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _make_train_transform(
    dataset_name: str, model_name: str = None
) -> transforms.Compose:
    """Return the training transform for the requested dataset (+ optional model)."""
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


def _make_test_transform(
    dataset_name: str, model_name: str = None
) -> transforms.Compose:
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
    name = dataset_name.lower().strip()
    if name == "cifar10":
        return 10
    if name == "femnist":
        return 62
    return 100


def _is_femnist(dataset_name: str) -> bool:
    return dataset_name.lower().strip() == "femnist"


def _check_partition_mode(dataset_name: str, partition_mode: str) -> None:
    """FEMNIST only has a natural (per-writer) partition; the CIFAR datasets
    only have simulated ones. Refuse mismatches instead of silently ignoring
    the requested mode, so a run's label can never disagree with its data."""
    if _is_femnist(dataset_name) and partition_mode != "natural":
        raise ValueError(
            f"dataset 'femnist' requires partition_mode='natural' (got '{partition_mode}'): "
            f"its clients are real writers; alpha / shards_per_client do not apply."
        )
    if not _is_femnist(dataset_name) and partition_mode == "natural":
        raise ValueError(
            f"partition_mode='natural' is only defined for dataset 'femnist' "
            f"(got '{dataset_name}')."
        )


class _TensorCache(Dataset):
    """In-memory copy of a torchvision image dataset with its (deterministic)
    transform applied once. Same samples/labels/order as the wrapped dataset, so
    results are unchanged; it only removes the per-step PIL->tensor conversion
    (the transform has no random augmentation)."""

    def __init__(self, base: Dataset):
        self.targets = list(base.targets)
        self.x = torch.stack([base[i][0] for i in range(len(base))])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _cacheable(transform) -> bool:
    """Cache only when the transform is fixed and keeps images small (no Resize)."""
    ops = getattr(transform, "transforms", None)
    return ops is not None and not any(isinstance(o, transforms.Resize) for o in ops)


def _load_train(dataset_name: str, transform) -> Dataset:
    name = dataset_name.lower().strip()
    if name == "cifar10":
        ds = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        ds = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
    return _TensorCache(ds) if _cacheable(transform) else ds


def _load_test(dataset_name: str, transform) -> Dataset:
    name = dataset_name.lower().strip()
    if name == "cifar10":
        ds = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        ds = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    return _TensorCache(ds) if _cacheable(transform) else ds


def _dirichlet_partition_indices(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    dataset_name: str,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Core Dirichlet(alpha) split, factored out so both get_client_datasets()
    and get_client_datasets_with_val() call the exact same logic."""
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

    return client_indices


def _shard_partition_indices(
    targets: np.ndarray,
    num_clients: int,
    shards_per_client: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Pathological non-IID split (McMahan et al., 2017-style): sort every
    training index by label, cut the sorted sequence into
    num_clients * shards_per_client equal-size contiguous shards (each shard
    drawn from a narrow label range), then deal shards_per_client shards to
    each client at random without replacement."""
    order = np.argsort(targets, kind="stable")  # every index, sorted by label
    num_shards = num_clients * shards_per_client
    shards = np.array_split(order, num_shards)  # each shard: a narrow label range

    shard_ids = np.arange(num_shards)
    rng.shuffle(shard_ids)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for i, shard_id in enumerate(shard_ids):
        cid = i // shards_per_client
        client_indices[cid].extend(shards[shard_id].tolist())

    return client_indices


def get_client_datasets(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    model_name: str = None,
    partition_mode: str = "dirichlet",
    shards_per_client: int = 2,
) -> List[Subset]:
    """Split a dataset's training set across clients."""
    _check_partition_mode(dataset_name, partition_mode)
    if _is_femnist(dataset_name):
        from .femnist_data import get_femnist_client_datasets

        return get_femnist_client_datasets(num_clients, seed)

    transform = _make_train_transform(
        dataset_name, model_name
    )  # augmented training transform
    full_train = _load_train(dataset_name, transform)
    targets = np.array(full_train.targets)
    num_classes = _num_classes(dataset_name)

    rng = np.random.default_rng(seed)

    if partition_mode == "pathological":
        client_indices = _shard_partition_indices(
            targets, num_clients, shards_per_client, rng
        )
    else:
        client_indices = _dirichlet_partition_indices(
            targets, num_clients, alpha, num_classes, dataset_name, rng
        )

    return [Subset(full_train, idxs) for idxs in client_indices]


def get_client_datasets_with_val(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    val_fraction: float = 0.1,
    model_name: str = None,
    partition_mode: str = "dirichlet",
    shards_per_client: int = 2,
) -> tuple:
    """Split training data across clients (Dirichlet, or "pathological" —
    see get_client_datasets()) then carve a held-out local validation set
    per client."""
    _check_partition_mode(dataset_name, partition_mode)
    if _is_femnist(dataset_name):
        raise NotImplementedError(
            "local_val_fraction > 0 is not implemented for 'femnist'."
        )
    _MIN_SHARD_FOR_VAL = 10  # shards smaller than this get no validation set

    # Re-uses the same logic; the full_train Dataset here uses the *training*
    # (augmented) transform so train Subsets share it.
    train_transform = _make_train_transform(dataset_name, model_name)
    full_train = _load_train(dataset_name, train_transform)
    targets = np.array(full_train.targets)
    num_classes = _num_classes(dataset_name)

    rng = np.random.default_rng(seed)

    if partition_mode == "pathological":
        client_indices = _shard_partition_indices(
            targets, num_clients, shards_per_client, rng
        )
    else:
        client_indices = _dirichlet_partition_indices(
            targets, num_clients, alpha, num_classes, dataset_name, rng
        )

    test_transform = _make_test_transform(dataset_name, model_name)
    full_train_noaug = _load_train(dataset_name, test_transform)

    train_subsets: list = []
    val_subsets: list = []

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

        train_idxs = shuffled[val_size:].tolist()  # trailing 90%
        val_idxs = shuffled[:val_size].tolist()  # leading 10%

        train_subsets.append(Subset(full_train, train_idxs))
        val_subsets.append(Subset(full_train_noaug, val_idxs))

    return train_subsets, val_subsets


def get_test_dataset(dataset_name: str = "cifar10", model_name: str = None) -> Dataset:
    """Return the test split for the requested dataset.

    Supported dataset_name values: "cifar10", "cifar100", "femnist" (fixed
    held-out-writers test set, see femnist_data.py).
    Default is "cifar10" for backward compatibility.

    model_name : optional, see get_client_datasets() — must match whatever was
        passed there so train and test data go through identical resize/
        normalisation (mismatched preprocessing between train/test would
        silently corrupt evaluation).
    """
    if _is_femnist(dataset_name):
        from .femnist_data import get_femnist_test_dataset

        return get_femnist_test_dataset()
    transform = _make_test_transform(
        dataset_name, model_name
    )  # no augmentation on test data
    return _load_test(dataset_name, transform)
