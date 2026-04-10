from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# standard CIFAR-10 stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_client_datasets(num_clients: int, alpha: float, seed: int) -> List[Subset]:
    """
    Split CIFAR-10 training set across clients using Dirichlet(alpha) partitioning.
    Lower alpha = more heterogeneous (each client ends up with mostly one or two classes).
    """
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=_transform)
    targets = np.array(full_train.targets)
    num_classes = 10

    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)

        # draw Dirichlet proportions — this is the core of the non-IID split
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
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


def get_test_dataset() -> Dataset:
    return datasets.CIFAR10(root="./data", train=False, download=True, transform=_transform)
