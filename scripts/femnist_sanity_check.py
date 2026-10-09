"""
FEMNIST sanity check: (1) how heterogeneous is the natural per-writer
partition, and (2) what does a centralised (non-federated) run of the
from-scratch CNN reach on the held-out-writers test set?

Heterogeneity is summarised by the total-variation (TV) distance between each
client's label distribution and the pooled label distribution
(TV = 0.5 * sum_c |p_client(c) - p_pooled(c)|; 0 = identical, 1 = disjoint),
averaged over clients. It is reported for (a) the real FEMNIST writers,
(b) an IID reference with the same client sizes (a random split of the same
pooled data -- the value TV takes from sampling noise alone), and (c) the
simulated CIFAR-100 partitions used elsewhere in this project, computed on
synthetic CIFAR-100-shaped labels with the project's own partition functions,
as a rough yardstick. (b) and (c) are context, not claims about FEMNIST.

Usage
-----
    python scripts/femnist_sanity_check.py                  # stats + centralised run
    python scripts/femnist_sanity_check.py --no-train       # stats only
    python scripts/femnist_sanity_check.py --lr 0.03 --epochs 20
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divroute_fl import femnist_data as F
from divroute_fl.data import _dirichlet_partition_indices, _shard_partition_indices
from divroute_fl.model import get_model


def mean_tv(label_lists, num_classes):
    pooled = np.bincount(np.concatenate(label_lists), minlength=num_classes).astype(float)
    pooled /= pooled.sum()
    tvs = []
    for lab in label_lists:
        p = np.bincount(lab, minlength=num_classes).astype(float)
        p /= p.sum()
        tvs.append(0.5 * np.abs(p - pooled).sum())
    return float(np.mean(tvs))


def simulated_cifar100_tv(num_clients=25, seed=42):
    """Reference TVs on synthetic CIFAR-100-shaped labels (500 per class)."""
    targets = np.repeat(np.arange(100), 500)
    out = {}
    for name, fn in [
        ("Dirichlet(0.9)", lambda r: _dirichlet_partition_indices(targets, num_clients, 0.9, 100, "cifar100", r)),
        ("Dirichlet(0.1)", lambda r: _dirichlet_partition_indices(targets, num_clients, 0.1, 100, "cifar100", r)),
        ("pathological (2 shards/client)", lambda r: _shard_partition_indices(targets, num_clients, 2, r)),
    ]:
        idx = fn(np.random.default_rng(seed))
        out[name] = mean_tv([targets[np.array(i)] for i in idx], 100)
    return out


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-train", action="store_true")
    args = p.parse_args()

    clients = F.get_femnist_client_datasets(args.clients, args.seed)
    test = F.get_femnist_test_dataset()

    sizes = np.array([len(c) for c in clients])
    labels = [c.targets for c in clients]
    distinct = np.array([len(np.unique(l)) for l in labels])
    print(f"\nclients: {len(clients)} | samples/client min={sizes.min()} median={int(np.median(sizes))} "
          f"mean={sizes.mean():.1f} max={sizes.max()} | total train={sizes.sum()} | test={len(test)}")
    print(f"distinct classes per client (of {F.NUM_CLASSES}): min={distinct.min()} "
          f"median={int(np.median(distinct))} mean={distinct.mean():.1f} max={distinct.max()}")

    tv_real = mean_tv(labels, F.NUM_CLASSES)
    pooled = np.concatenate(labels)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pooled))
    iid, start = [], 0
    for s in sizes:
        iid.append(pooled[perm[start:start + s]])
        start += s
    tv_iid = mean_tv(iid, F.NUM_CLASSES)
    print(f"\nmean TV(label dist. vs pooled): FEMNIST writers = {tv_real:.3f} | "
          f"IID reference, same sizes = {tv_iid:.3f}")
    print("yardsticks (synthetic CIFAR-100-shaped labels, 25 clients):")
    for k, v in simulated_cifar100_tv().items():
        print(f"   {k:<32s} {v:.3f}")

    if args.no_train:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = get_model("femnist_cnn", F.NUM_CLASSES).to(device)
    xs = torch.cat([c.x for c in clients])
    ys = torch.cat([c.y for c in clients])
    train_loader = DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=1024, shuffle=False)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"\ncentralised run on the pooled training writers: lr={args.lr}, batch={args.batch_size}, "
          f"{args.epochs} epochs, device={device}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        acc = evaluate(model, test_loader, device)
        print(f"  epoch {ep:>2}/{args.epochs}  held-out-writer test acc = {acc:.4f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
