"""
scratch/sparsity_probe_c0.py
============================
Re-runs FedSparse (CIFAR-10, seed=42, lambda=0.01, 100 rounds) and
reports end-of-training sparsity metrics for client 0 at rounds
1, 20, 40, 60, 80, and 100.

No existing source files are modified.
"""

import random
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from divroute_fl.data import get_client_datasets, get_test_dataset
from divroute_fl.model import get_model
from divroute_fl.client import FLClient
from divroute_fl.server import FLServer
from baselines.fedsparse_baseline import get_fedsparse_config

PROBE_ROUNDS   = {1, 20, 40, 60, 80, 100}
TARGET_CLIENT  = 0
SEED           = 42
NUM_ROUNDS     = 100


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[init] device:", device, flush=True)

    cfg = get_fedsparse_config(
        fedsparse_lambda  = 0.01,
        num_clients       = 25,
        clients_per_round = 15,
        num_rounds        = NUM_ROUNDS,
        local_lr          = 0.1,
        seed              = SEED,
        dataset_name      = "cifar10",
        model_name        = "simplecnn",
        skip_plot_prompt  = True,
        log_path          = "scratch/sparsity_probe_c0_run.json",
    )

    print(f"[cfg] lambda={cfg.fedsparse_lambda}  sparsify_upload={cfg.fedsparse_sparsify_upload}  lr={cfg.local_lr}", flush=True)

    print("[init] loading CIFAR-10...", flush=True)
    client_datasets = get_client_datasets("cifar10", 25, cfg.alpha, SEED)
    test_loader = DataLoader(get_test_dataset("cifar10"), batch_size=256, shuffle=False, num_workers=0)

    num_classes  = 10
    global_model = get_model("simplecnn", num_classes)
    server       = FLServer(global_model, cfg, device)

    clients = [
        FLClient(i, client_datasets[i], cfg.local_epochs, cfg.local_lr,
                 cfg.batch_size, device, "simplecnn", num_classes)
        for i in range(cfg.num_clients)
    ]

    all_ids       = list(range(cfg.num_clients))
    error_buffers = {}

    records         = {}
    last_c0_metrics = None
    last_c0_round   = None

    print(f"\n[train] {NUM_ROUNDS} rounds | probe rounds: {sorted(PROBE_ROUNDS)}", flush=True)

    for rnd in range(NUM_ROUNDS):
        rnd_1indexed = rnd + 1

        selected  = server.select_clients(all_ids)
        global_sd = server.global_model.state_dict()

        results = []
        for cid in selected:
            result = clients[cid].train(global_sd, cfg.local_epochs,
                                        fedsparse_lambda=cfg.fedsparse_lambda)
            results.append(result)

        for r in results:
            r["divergence_score"] = 0.0
            r["tier"] = 1
            r["bytes_received"] = 0

        # Probe client 0 before aggregate
        c0_result = next((r for r in results if r["client_id"] == TARGET_CLIENT), None)

        if c0_result is not None:
            param_names = {n for n, _ in server.global_model.named_parameters()}

            global_flat = torch.cat([
                v.flatten().float().to(device)
                for k, v in global_sd.items()
                if k in param_names
            ]).cpu()

            c0_param_flat = torch.cat([
                v.flatten().float()
                for k, v in c0_result["state_dict"].items()
                if k in param_names
            ])

            raw_delta = c0_param_flat - global_flat
            norm = torch.norm(raw_delta)
            if norm > cfg.grad_clip_norm:
                raw_delta = raw_delta * (cfg.grad_clip_norm / norm)

            total_params = raw_delta.numel()
            abs_delta    = raw_delta.abs()

            mask_1e12 = abs_delta > 1e-12
            nnz       = int(mask_1e12.sum().item())
            nnz_pct   = 100.0 * nnz / total_params
            pct_1e4   = 100.0 * (abs_delta > 1e-4).float().sum().item() / total_params

            c0_client = clients[TARGET_CLIENT]
            irw       = c0_client._irw_norms
            raw_irw   = {name: irw.get(name, 0.0)
                         for name, _ in server.global_model.named_parameters()}
            raw_vals  = list(raw_irw.values())
            min_raw, max_raw = min(raw_vals), max(raw_vals)
            if max_raw == min_raw:
                lambda_j = {name: cfg.fedsparse_lambda for name in raw_irw}
            else:
                r_range  = max_raw - min_raw
                lambda_j = {
                    name: cfg.fedsparse_lambda * (1.0 - (v - min_raw) / r_range)
                    for name, v in raw_irw.items()
                }

            lr = cfg.local_lr
            prox_thresh_vec = torch.cat([
                torch.full((p.numel(),), lr * lambda_j[name])
                for name, p in server.global_model.named_parameters()
            ])
            pct_prox = 100.0 * (abs_delta > prox_thresh_vec).float().sum().item() / total_params

            last_c0_metrics = {
                "total_params": total_params,
                "l2":           raw_delta.norm(p=2).item(),
                "l1":           raw_delta.norm(p=1).item(),
                "nnz":          nnz,
                "nnz_pct":      nnz_pct,
                "pct_1e12":     100.0 * mask_1e12.float().sum().item() / total_params,
                "pct_1e4":      pct_1e4,
                "pct_prox":     pct_prox,
                "prox_thresh":  lr * cfg.fedsparse_lambda,
            }
            last_c0_round = rnd_1indexed

        server.aggregate(results, error_buffers)

        if rnd_1indexed in PROBE_ROUNDS:
            if c0_result is not None:
                records[rnd_1indexed] = {"actual_round": rnd_1indexed, "metrics": last_c0_metrics}
            else:
                records[rnd_1indexed] = {
                    "actual_round": last_c0_round,
                    "metrics":      last_c0_metrics,
                    "note": f"c0 NOT selected at rnd {rnd_1indexed}; using rnd {last_c0_round}",
                }

        if rnd_1indexed in PROBE_ROUNDS or rnd_1indexed % 10 == 0:
            acc = server.evaluate(test_loader)
            m = last_c0_metrics
            if m:
                print(f"  rnd {rnd_1indexed:>3}/{NUM_ROUNDS} acc={acc:.4f} | "
                      f"c0 last@{last_c0_round}: L2={m['l2']:.4f} nnz={m['nnz']:,} ({m['nnz_pct']:.1f}%)",
                      flush=True)
            else:
                print(f"  rnd {rnd_1indexed:>3}/{NUM_ROUNDS} acc={acc:.4f} | c0 not seen yet", flush=True)

    # Final table
    print("\n" + "=" * 100, flush=True)
    print("  SPARSITY INSTRUMENTATION — Client 0 — FedSparse CIFAR-10 seed=42 lambda=0.01", flush=True)
    print("=" * 100, flush=True)
    print(f"  Proximal threshold (uniform) = lr x lambda = {cfg.local_lr} x {cfg.fedsparse_lambda} = {cfg.local_lr * cfg.fedsparse_lambda:.6f}", flush=True)
    print(flush=True)

    hdr = (f"  {'Target':>6}  {'Actual':>6}  {'L2':>10}  {'L1':>12}  "
           f"{'nnz':>8}  {'nnz%':>7}  {'>1e-12%':>8}  {'>1e-4%':>8}  {'>prox%':>8}")
    print(hdr, flush=True)
    print("  " + "-" * 84, flush=True)

    for target_rnd in sorted(PROBE_ROUNDS):
        rec = records.get(target_rnd)
        if rec is None or rec["metrics"] is None:
            print(f"  {target_rnd:>6}  {'N/A':>6}  (client 0 never selected)", flush=True)
            continue
        m   = rec["metrics"]
        act = rec.get("actual_round", "?")
        note = f"  [NOTE: {rec['note']}]" if "note" in rec else ""
        print(
            f"  {target_rnd:>6}  {act:>6}  {m['l2']:>10.4f}  {m['l1']:>12.2f}  "
            f"{m['nnz']:>8,}  {m['nnz_pct']:>7.2f}  {m['pct_1e12']:>8.2f}  "
            f"{m['pct_1e4']:>8.2f}  {m['pct_prox']:>8.2f}{note}", flush=True
        )

    print(flush=True)
    print("  Columns:", flush=True)
    print("    Target  = requested communication round (1-indexed)", flush=True)
    print("    Actual  = round in which c0 was last selected (may differ)", flush=True)
    print("    L2      = ||delta||_2 (after gradient clip)", flush=True)
    print("    L1      = ||delta||_1 (after gradient clip)", flush=True)
    print("    nnz     = |{i : |delta_i| > 1e-12}|", flush=True)
    print("    nnz%    = nnz / total_params * 100", flush=True)
    print("    >1e-12% = % coords > 1e-12 (server upload mask)", flush=True)
    print("    >1e-4%  = % coords > 1e-4  (fedsparse_threshold config)", flush=True)
    print("    >prox%  = % coords > lr*lambda_j (proximal threshold)", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
