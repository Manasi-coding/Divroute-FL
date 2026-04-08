"""
DivRoute-FL Lite — Data
CIFAR-10 download + Dirichlet-based non-IID partitioning across clients.
"""
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# Standard CIFAR-10 normalisation (ImageNet-style stats used widely)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])


def get_client_datasets(
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[Subset]:
    """
    Split the CIFAR-10 training set across *num_clients* using a
    symmetric Dirichlet(alpha) distribution over labels.

    Low alpha (e.g. 0.5) → each client sees mostly 1–2 classes (non-IID).
    High alpha (e.g. 100) → near-uniform split (IID).

    Returns a list of ``torch.utils.data.Subset`` — one per client.
    """
    full_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=_train_transform,
    )
    targets = np.array(full_train.targets)
    num_classes = 10

    rng = np.random.default_rng(seed)

    # For every class, draw a Dirichlet vector of length num_clients
    # that decides what fraction of that class each client gets.
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)

        # Dirichlet proportions for this class
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions → cumulative counts
        proportions = proportions / proportions.sum()
        counts = (proportions * len(cls_idx)).astype(int)
        # Distribute any leftover samples due to rounding
        remainder = len(cls_idx) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1

        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(cls_idx[start:end].tolist())
            start = end

    subsets = [Subset(full_train, idxs) for idxs in client_indices]
    return subsets


def get_test_dataset() -> Dataset:
    """Return the full CIFAR-10 test set (10 000 images) for global evaluation."""
    return datasets.CIFAR10(
        root="./data", train=False, download=True, transform=_test_transform,
    )
