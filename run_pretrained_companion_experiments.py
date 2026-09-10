"""
Master plan §6 — required companion runs for the pretrained EfficientNet-B0
CIFAR-100 comparison: DivRoute (main result) / Full FedAvg / Uniform Top-5%,
all sharing an identical fresh pretrained init, dataset, and round budget.
This is what makes comm_vs_accuracy.png valid for the new pretrained-backbone
setting (see DIVROUTE_ACCURACY_MASTER_PLAN.md §6 and
DIVROUTE_REMAINING_WORK.md item 4).

All three configs come from divroute_fl/config.py's
get_pretrained_finetune_config() / get_fedavg_pretrained_config() /
get_uniform_top5_pretrained_config() and are launched here with identical
seed/num_rounds/num_clients/clients_per_round/alpha (each factory's own
inherited defaults -- this script does not override those) so the three
runs are a valid comparison.

Usage
-----
    python run_pretrained_companion_experiments.py
        Full run: all three configs back to back, num_rounds=20 (the shared
        default -- see DIVROUTE_REMAINING_WORK.md item 4).

    python run_pretrained_companion_experiments.py --only divroute --num-rounds 2 --fresh
        Calibration mode: one config, a couple of rounds, ignoring any
        existing checkpoint -- for measuring real wall-clock time before
        committing to the full budget.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from divroute_fl.config import (
    get_pretrained_finetune_config, get_fedavg_pretrained_config,
    get_uniform_top5_pretrained_config,
)
from divroute_fl.main import run

SEED = 42
LOG_PREFIX = "logs/pretrained_companion"

CONFIG_FACTORIES = {
    "divroute": (get_pretrained_finetune_config,     "divroute_pretrained_cifar100"),
    "fedavg":   (get_fedavg_pretrained_config,       "fedavg_pretrained_cifar100"),
    "uniform":  (get_uniform_top5_pretrained_config, "uniform_pretrained_cifar100"),
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--only", choices=list(CONFIG_FACTORIES), default=None,
                   help="run a single config only (for calibration)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoint and start this run over")
    p.add_argument("--invert-polarity", action="store_true",
                   help="DivRoute only: run with invert_tier_polarity=True as a "
                        "separately labeled experiment (requires --only divroute) -- "
                        "see DIVROUTE_ACCURACY_MASTER_PLAN.md §14")
    args = p.parse_args()
    if args.invert_polarity and args.only != "divroute":
        p.error("--invert-polarity requires --only divroute "
                 "(the flag is a no-op for fedavg/uniform)")
    return args


def main():
    args = _parse_args()
    names = [args.only] if args.only else list(CONFIG_FACTORIES)

    for name in names:
        factory, run_label = CONFIG_FACTORIES[name]
        extra_kwargs = {}
        log_suffix = ""
        if args.invert_polarity:
            run_label = f"{run_label}_inverted_polarity"
            log_suffix = "_inverted_polarity"
            extra_kwargs["invert_tier_polarity"] = True

        cfg = factory(
            run_label=run_label,
            seed=SEED,
            num_rounds=args.num_rounds,
            log_path=f"{LOG_PREFIX}_{name}{log_suffix}.json",
            skip_plot_prompt=True,
            **extra_kwargs,
        )
        if args.fresh:
            cfg.fresh = True

        print(f"\n{'=' * 70}\nRUNNING: {name}  (num_rounds={args.num_rounds})\n{'=' * 70}")
        t0 = time.time()
        run(cfg)
        elapsed = time.time() - t0
        print(f"[{name}] wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
        with open(f"{LOG_PREFIX}_{name}.walltime.txt", "w") as f:
            f.write(f"{elapsed:.2f} sec ({elapsed / 60:.2f} min)\n")


if __name__ == "__main__":
    main()
