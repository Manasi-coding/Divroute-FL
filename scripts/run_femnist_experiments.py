"""
FEMNIST from-scratch experiments (DIVROUTE_ACCURACY_MASTER_PLAN.md sections 31-34).
One client = one real writer; test set = fixed held-out writers. Run from the repo root.

    python scripts/run_femnist_experiments.py                      # fedavg/uniform/divroute x seeds 42-44
    python scripts/run_femnist_experiments.py --variant pure matched purerandom
    python scripts/run_femnist_experiments.py --variant uniform --kscale 0.1 0.2 0.4 --seeds 42
    python scripts/run_femnist_experiments.py --variant divroute uniform --lr 0.06 --seeds 42
    python scripts/run_femnist_experiments.py --variant pure purerandom matched uniform --kscale 0.1 --no-ef

Variants (identical data/model/optimisation; DivRoute differs from Uniform in tiered k,
per-layer top-k, divergence-weighted aggregation and selection decay, so the ablations
remove those one at a time):
  fedavg, uniform, divroute   the three main methods
  matched     Uniform at k=0.105 (about DivRoute's bytes: bytes = 2*k of dense)
  nolayer     DivRoute with global top-k (use_layerwise_topk=False)
  noweight    DivRoute without divergence-weighted aggregation
  nodecay     DivRoute without tier-3 selection decay (gamma=1)
  pure        DivRoute with all three removed: tiered k only
  purerandom  `pure` with tier labels shuffled (random_tier_assignment)
  invert      DivRoute with inverted tier polarity
  pureinvert  `pure` with inverted tier polarity (most bandwidth to most-divergent clients)
  diag        DivRoute logging gradient/routing diagnostics (divergence vs update norm etc.)

--kscale s multiplies every compression ratio of the variant (uniform 0.05, matched 0.105,
DivRoute family 0.20/0.05) by s, to move to a tighter byte budget while keeping variants
byte-matched to each other (pure*s vs matched*s). Not applicable to fedavg.
Finished runs are skipped; interrupted runs auto-resume; a finished run's checkpoint
folder is deleted unless --keep-checkpoints.
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divroute_fl.config import (
    get_femnist_fedavg_config, get_femnist_uniform_config, get_femnist_divroute_config,
)
from divroute_fl.main import run

_F = {"fedavg": get_femnist_fedavg_config, "uniform": get_femnist_uniform_config,
      "divroute": get_femnist_divroute_config}
_BASE_K = {"uniform": (0.05, 0.05), "divroute": (0.20, 0.05)}   # factory default tier ratios
_PURE = dict(use_layerwise_topk=False, use_divergence_weighting=False, gamma=1.0)

VARIANTS = {   # name -> (factory, overrides)
    "fedavg": ("fedavg", {}), "uniform": ("uniform", {}), "divroute": ("divroute", {}),
    "matched": ("uniform", dict(k_ratio_tier1=0.105, k_ratio_tier2=0.105)),
    "nolayer": ("divroute", dict(use_layerwise_topk=False)),
    "noweight": ("divroute", dict(use_divergence_weighting=False)),
    "nodecay": ("divroute", dict(gamma=1.0)),
    "pure": ("divroute", dict(_PURE)),
    "purerandom": ("divroute", dict(_PURE, random_tier_assignment=True)),
    "invert": ("divroute", dict(invert_tier_polarity=True)),
    "pureinvert": ("divroute", dict(_PURE, invert_tier_polarity=True)),   # `pure` with tier polarity swapped
    "pure_l2": ("divroute", dict(_PURE, divergence_metric="l2")),                # routing score = update L2 norm
    "pure_layercos": ("divroute", dict(_PURE, divergence_metric="layerwise_cosine")),
    "diag": ("divroute", dict(enable_gradient_diagnostics=True, enable_routing_quality_analysis=True)),
}


def _rounds_logged(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return 0
    return len(raw["history"] if isinstance(raw, dict) else raw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", nargs="+", choices=list(VARIANTS), default=["fedavg", "uniform", "divroute"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--num-rounds", type=int, default=None, help="default: factories' value (150)")
    p.add_argument("--lr", type=float, default=None, help="local_lr (local_lr_min = lr/10); label gets _lr<lr>")
    p.add_argument("--kscale", type=float, nargs="+", default=None, help="scale compression ratios (see above)")
    p.add_argument("--no-ef", action="store_true", help="disable error feedback (control); label gets _noef")
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args()
    rounds = args.num_rounds or get_femnist_fedavg_config().num_rounds

    done = {}   # label-without-seed -> [log paths]
    for seed, v, ks in itertools.product(args.seeds, args.variant, args.kscale or [None]):
        factory, ov = VARIANTS[v]
        ov, tag = dict(ov), ""
        if args.lr is not None:
            ov.update(local_lr=args.lr, local_lr_min=args.lr / 10)
            tag += f"_lr{args.lr:g}"
        if args.no_ef:
            if factory == "fedavg":
                print(f"[skip] {v}: no error feedback in dense FedAvg")
                continue
            ov.update(use_error_feedback=False)
            tag += "_noef"
        if ks is not None:
            if factory == "fedavg":
                print(f"[skip] {v}: --kscale does not apply to dense FedAvg")
                continue
            k1, k2 = _BASE_K[factory]
            ov.update(k_ratio_tier1=ov.get("k_ratio_tier1", k1) * ks, k_ratio_tier2=ov.get("k_ratio_tier2", k2) * ks)
            tag += f"_ks{ks:g}"
        label = f"femnist_{v}{tag}_r{rounds}_s{seed}"
        path = f"logs/{label}.json"
        done.setdefault(v + tag, []).append(path)
        if _rounds_logged(path) >= rounds:
            print(f"[skip] {label}: already holds {rounds} rounds")
            continue
        cfg = _F[factory](run_label=label, seed=seed, num_rounds=rounds, log_path=path,
                          skip_plot_prompt=True, **ov)
        print(f"\n{'=' * 70}\nRUNNING: {label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        run(cfg)
        print(f"[{label}] wall-clock: {(time.time() - t0) / 60:.1f} min", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(os.path.join("checkpoints", f"{label}_femnist_seed{seed}"), ignore_errors=True)

    cmd = "python scripts/summarize_runs.py --last 20"
    for name, paths in done.items():
        cmd += f" \\\n    --group {name} " + " ".join(paths)
    print("\nSummarize (plateau = mean of the last 20 rounds):\n  " + cmd)
    print("Check each run's last two 10-round windows differ by well under 0.5pp before comparing.")


if __name__ == "__main__":
    main()
