"""
Centralized (non-federated) fine-tuning sanity check for the pretrained
EfficientNet-B0 + CIFAR-100 pipeline.

Why this script exists
-----------------------
DIVROUTE_ACCURACY_MASTER_PLAN.md targets 80-85% accuracy on CIFAR-100 by
switching DivRoute's backbone from a from-scratch ResNet-18 to an
ImageNet-pretrained EfficientNet-B0 (see get_pretrained_finetune_config() in
divroute_fl/config.py). That target is sourced from *centralized* transfer-
learning literature (~82-88% band), not from anything federated. Before
spending compute on the three required federated companion runs (DivRoute /
FedAvg / Uniform Top-5%, all needing identical pretrained init), this script
answers a cheaper, load-bearing question first: with no client partitioning,
no partial participation, no compression, and no federated rounds at all --
does this exact backbone + resize/normalisation pipeline even reach a
reasonable accuracy on CIFAR-100? If it doesn't, the bug is in the pipeline
(most likely the resize step -- see master plan doc, "resize step is a
bigger risk than a tradeoff"), not in anything federation-related, and no
amount of federated tuning will fix it.

Reuses divroute_fl/model.py (get_model) and divroute_fl/data.py's private
transform/loader helpers so preprocessing is byte-for-byte identical to what
the federated runs will use -- this script does not re-derive its own
transform pipeline.

Usage
-----
    python scripts/centralized_finetune_sanity_check.py --smoke
        Fast end-to-end pipeline check: 1 epoch over a 512-image subset.
        Confirms the model/data/training loop runs on this machine (a few
        seconds to ~1 minute) before committing to a full run.

    python scripts/centralized_finetune_sanity_check.py
        Full run over the entire CIFAR-100 train set with default
        hyperparameters (see --help for all options).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divroute_fl.data import _make_train_transform, _load_train, get_test_dataset
from divroute_fl.model import get_model
from divroute_fl.client import get_local_lr


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_full_train_dataset(dataset_name: str, model_name: str):
    """Full train split, no client partitioning -- same transform pipeline
    (augmentation + resize + normalisation) that get_client_datasets() gives
    every federated run, so this sanity check's preprocessing can't drift
    from what DivRoute/FedAvg/Uniform will actually train on.
    """
    transform = _make_train_transform(dataset_name, model_name)
    return _load_train(dataset_name, transform)


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--model", default="efficientnet_b0_pretrained")
    p.add_argument("--bn-mode", default="default",
                   choices=["default", "local_bn", "groupnorm", "ws_groupnorm"],
                   help="'default' preserves pretrained BatchNorm stats -- "
                        "matches get_pretrained_finetune_config()'s recommended setting.")
    p.add_argument("--epochs", type=int, default=15,
                   help="literature-grounded default (TDS EfficientNet-B0/CIFAR-100 "
                        "replication used 15 epochs)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01,
                   help="matches get_pretrained_finetune_config()'s local_lr")
    p.add_argument("--lr-min", type=float, default=0.001,
                   help="matches get_pretrained_finetune_config()'s local_lr_min "
                        "(cosine floor, via client.py's get_local_lr)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--log-path", default=None,
                   help="default: logs/centralized_finetune_sanity_check_<dataset>.json")
    p.add_argument("--checkpoint-path", default=None,
                   help="optional path to save the best-test-accuracy model state_dict "
                        "(off by default -- this script is a sanity check, not a training "
                        "run whose weights are meant to be reused)")
    p.add_argument("--smoke", action="store_true",
                   help="fast pipeline check: forces 1 epoch over a small fixed subset, "
                        "never writes a checkpoint")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def main():
    args = _parse_args()
    if args.smoke:
        args.epochs = 1

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}")
    if device.type == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}")

    num_classes = 10 if args.dataset == "cifar10" else 100

    print(f"[init] loading full {args.dataset.upper()} train/test sets "
          f"(model={args.model}, bn_mode={args.bn_mode})...")
    train_ds = _make_full_train_dataset(args.dataset, args.model)
    test_ds = get_test_dataset(args.dataset, model_name=args.model)

    if args.smoke:
        # Small deterministic subsets -- proves the pipeline runs end to end
        # without waiting through a full epoch over 50k images.
        train_ds = Subset(train_ds, list(range(min(512, len(train_ds)))))
        test_ds = Subset(test_ds, list(range(min(512, len(test_ds)))))

    print(f"[init] train samples: {len(train_ds)}, test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = get_model(args.model, num_classes, args.bn_mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] model params: {n_params:,}")

    # Same recipe as divroute_fl/client.py's local training (label smoothing,
    # SGD+momentum+nesterov, grad clipping) so this ceiling reflects the same
    # optimisation regime the federated runs will use -- only the federation
    # machinery (partitioning, partial participation, compression, rounds)
    # is absent here.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=5e-4, nesterov=True,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    log_path = args.log_path or f"logs/centralized_finetune_sanity_check_{args.dataset}.json"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    history = []
    best_acc = 0.0
    t_start = time.time()

    print(f"[train] {args.epochs} epoch(s), batch_size={args.batch_size}, "
          f"lr={args.lr}->{args.lr_min} (cosine), amp={use_amp}")

    for epoch in range(args.epochs):
        model.train()
        epoch_lr = get_local_lr(args.lr, args.lr_min, epoch, args.epochs)
        for g in optimizer.param_groups:
            g["lr"] = epoch_lr

        t_epoch = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - t_epoch

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)

        print(f"[epoch {epoch + 1:>3}/{args.epochs}] lr={epoch_lr:.5f} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
              f"best={best_acc:.4f} ({epoch_time:.1f}s)"
              + ("  *" if is_best else ""))

        history.append({
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_sec": epoch_time,
        })

        if is_best and args.checkpoint_path and not args.smoke:
            os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint_path)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "dataset": args.dataset, "model": args.model, "bn_mode": args.bn_mode,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "lr_min": args.lr_min, "seed": args.seed,
                    "smoke": args.smoke, "num_train_samples": len(train_ds),
                    "num_test_samples": len(test_ds), "num_params": n_params,
                },
                "history": history,
                "best_test_acc": best_acc,
            }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[done] best test accuracy: {best_acc:.4f}  "
          f"total wall-clock: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"[done] results written to {log_path}")


if __name__ == "__main__":
    main()
