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


def _make_train_transform(dataset_name: str) -> transforms.Compose:
    """Return the training transform for the requested dataset.

    Includes standard data augmentation (RandomCrop + RandomHorizontalFlip)
    applied before tensor conversion.  Applied only to client training data;
    the test transform (_make_test_transform) is kept augmentation-free.

    Augmentation is identical for every client because all clients share the
    same torchvision Dataset object (partitioned via index Subsets).  Every
    FL method (FedAvg, FedSparse, FedZip, Uniform, DivRoute) therefore sees
    the same augmented distribution, preserving experimental fairness.
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
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _make_test_transform(dataset_name: str) -> transforms.Compose:
    """Return the evaluation/test transform for the requested dataset.

    No augmentation — deterministic ToTensor + Normalize only.
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
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


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
) -> List[Subset]:
    """
    Split a dataset's training set across clients using Dirichlet(alpha) partitioning.
    Lower alpha = more heterogeneous (each client ends up with mostly one or two classes).

    Supported dataset_name values: "cifar10", "cifar100"

    The Dirichlet partitioning logic is identical for both datasets;
    the only difference is num_classes (10 or 100) and the normalisation transform.
    """
    transform    = _make_train_transform(dataset_name)   # augmented training transform
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


def get_test_dataset(dataset_name: str = "cifar10") -> Dataset:
    """Return the test split for the requested dataset.

    Supported dataset_name values: "cifar10", "cifar100"
    Default is "cifar10" for backward compatibility.
    """
    transform = _make_test_transform(dataset_name)   # no augmentation on test data
    return _load_test(dataset_name, transform)