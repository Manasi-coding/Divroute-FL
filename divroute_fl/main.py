import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import get_client_datasets, get_test_dataset
from .model import SimpleCNN
from .client import FLClient
from .server import FLServer
from .logger import FLLogger
from .mechanism import update_ema, compute_adaptive_taus
from .compression import get_adaptive_k_ratios


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_local_epochs(rnd: int, config: Config) -> int:
    if not config.use_epoch_warmup:
        return config.local_epochs
    if rnd < 5:
        return max(2, config.local_epochs // 2)
    elif rnd < 15:
        return max(3, config.local_epochs - 1)
    return config.local_epochs


def _fedavg_bytes_per_round(num_clients: int, delta_numel: int) -> dict:
    """Baseline: full float32 delta for every selected client, both directions."""
    per_direction = num_clients * delta_numel * 4
    return {
        "download": per_direction,
        "upload":   per_direction,   # FedAvg upload = full model, same size as download
        "bidir":    per_direction * 2,
    }



def run(config: Config | None = None) -> None:
    if config is None:
        config = Config()

    # ── Baseline mode: force all settings to vanilla FedAvg ─────────────────────
    if config.fedavg_baseline_mode:
        config.use_divergence_weighting = False
        config.use_server_momentum      = False
        config.use_error_feedback       = False
        config.use_adaptive_tau         = False
        config.use_tier3_sync           = False
        config.use_adaptive_k           = False
        # k_ratio=1.0 → top-k keeps ALL params → no effective compression
        config.k_ratio_tier1            = 1.0
        config.k_ratio_tier2            = 1.0
        # tau below any real divergence score → every client is Tier 1 (no skipping)
        config.tau_low                  = -1.0
        config.tau_high                 = -1.0
        # gamma=1 → no selection-weight decay
        config.gamma                    = 1.0
        print("[init] *** FEDAVG BASELINE MODE — all DivRoute features disabled ***")
    # ─────────────────────────────────────────────────────────────────────

    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"[init] using device: {device}")

    print("[init] loading CIFAR-10 + building non-IID splits...")
    client_datasets = get_client_datasets(config.num_clients, config.alpha, config.seed)
    test_loader = DataLoader(get_test_dataset(), batch_size=256, shuffle=False, num_workers=0)

    shard_sizes = [len(ds) for ds in client_datasets]
    print(f"[init] shard sizes — min: {min(shard_sizes)}, max: {max(shard_sizes)}, "
          f"mean: {np.mean(shard_sizes):.0f}")

    global_model = SimpleCNN()
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(i, client_datasets[i], config.local_epochs, config.local_lr,
                 config.batch_size, device)
        for i in range(config.num_clients)
    ]

    logger = FLLogger(config.log_path)
    all_ids = list(range(config.num_clients))

    ema_scores: dict = {}
    error_buffers: dict = {}

    _baseline_bpr: dict | None = None   # FedAvg bytes-per-round (set once after round 0)

    print(f"[train] {config.num_rounds} rounds, "
          f"{config.clients_per_round}/{config.num_clients} clients/round")
    print(f"[train] div-weighted: {config.use_divergence_weighting} "
          f"({config.divergence_weight_mode}) | "
          f"momentum: {config.use_server_momentum} | "
          f"error-feedback: {config.use_error_feedback} | "
          f"adaptive-tau: {config.use_adaptive_tau} "
          f"(pct: {config.tau_low_pct}/{config.tau_high_pct}) | "
          f"tier3-sync: {config.use_tier3_sync} "
          f"(every {config.tier3_sync_interval} rounds) | "
          f"epoch-warmup: {config.use_epoch_warmup} | "
          f"fedavg-mode: {config.fedavg_baseline_mode}")

    cumulative_divroute_download  = 0
    cumulative_baseline_download  = 0
    cumulative_upload             = 0
    cumulative_baseline_upload    = 0

    for rnd in range(config.num_rounds):
        selected = server.select_clients(all_ids)
        local_epochs = _get_local_epochs(rnd, config)

        # -- client training -------------------------------------------------
        results = []
        global_sd = server.global_model.state_dict()

        for cid in selected:
            result = clients[cid].train(global_sd, local_epochs)
            results.append(result)

        # -- divergence: EMA-smoothed cosine distance -------------------------
        global_flat = torch.cat([
            v.flatten().float().cpu() for v in global_sd.values()
        ])

        raw_d_scores = []
        for r in results:
            client_flat = torch.cat([
                v.flatten().float().cpu() for v in r["state_dict"].values()
            ])
            cos = F.cosine_similarity(
                client_flat.unsqueeze(0), global_flat.unsqueeze(0),
                dim=1, eps=1e-8
            ).item()
            d_raw = 1.0 - cos
            d_ema = update_ema(ema_scores, r["client_id"], d_raw, config.ema_beta)
            r["divergence_score"] = d_ema
            raw_d_scores.append(d_ema)

        # -- adaptive tau (Phase 6.1) ------------------------------------------
        if config.use_adaptive_tau and len(raw_d_scores) >= 3:
            config.tau_low, config.tau_high = compute_adaptive_taus(
                raw_d_scores, config.tau_low_pct, config.tau_high_pct)

        # -- tier assignment -----------------------------------------------------
        for r in results:
            r["tier"] = server.assign_tier(r["divergence_score"])

        # -- adaptive k_ratio (updated before aggregate so compression uses
        #    the correct ratios for this round) --------------------------------
        k1, k2 = get_adaptive_k_ratios(config, rnd)
        config.k_ratio_tier1, config.k_ratio_tier2 = k1, k2

        # -- initialise bytes_received; aggregate applies tiered compression
        #    to each client's upload delta and sets r["bytes_received"] --------
        for r in results:
            r["bytes_received"] = 0

        server.aggregate(results, error_buffers)

        # -- Tier-3 staleness sync (Phase 1.2, Option A) ------------------------
        # Every `tier3_sync_interval` rounds, Tier-3 clients receive a
        # Tier-2-compressed "heartbeat" of the global delta instead of nothing,
        # so they don't train on an arbitrarily stale model when re-selected.
        tier3_sync_count = 0
        if config.use_tier3_sync and (rnd + 1) % config.tier3_sync_interval == 0:
            for r in results:
                if r["tier"] == 3:
                    heartbeat = server.tier3_heartbeat_payload()
                    r["bytes_received"] += heartbeat["bytes_transmitted"]
                    tier3_sync_count += 1

        # -- baseline byte accounting (dict with download/upload/bidir) -----------
        delta_numel = server.global_delta.numel()
        if _baseline_bpr is None:
            _baseline_bpr = _fedavg_bytes_per_round(config.clients_per_round, delta_numel)

        # -- evaluation + logging --------------------------------------------------
        acc = server.evaluate(test_loader)
        logger.log(rnd, acc, results, delta_numel=delta_numel)
        server.update_selection_weights(results)

        total_download = sum(r["bytes_received"] for r in results)
        total_upload   = sum(r["upload_bytes"]   for r in results)
        cumulative_divroute_download += total_download
        cumulative_baseline_download += _baseline_bpr["download"]
        cumulative_upload            += total_upload
        cumulative_baseline_upload   += _baseline_bpr["upload"]

        saving_pct = 100.0 * (1.0 - total_download / _baseline_bpr["download"])
        cumulative_saving_pct = 100.0 * (
            1.0 - cumulative_divroute_download / cumulative_baseline_download)

        t1 = sum(1 for r in results if r["tier"] == 1)
        t2 = sum(1 for r in results if r["tier"] == 2)
        t3 = sum(1 for r in results if r["tier"] == 3)

        d_vals = [r["divergence_score"] for r in results]
        sync_note = f" | t3-sync: {tier3_sync_count}" if tier3_sync_count > 0 else ""
        print(
            f"  round {rnd + 1:>3}/{config.num_rounds} | acc: {acc:.4f} | "
            f"tiers: {t1}/{t2}/{t3} | "
            f"down: {total_download/1e6:.3f}MB (save {saving_pct:.1f}%, "
            f"cum {cumulative_saving_pct:.1f}%) | "
            f"up: {total_upload/1e6:.3f}MB | "
            f"tau: [{config.tau_low:.4f}, {config.tau_high:.4f}] | "
            f"d: [{min(d_vals):.4f}, {max(d_vals):.4f}] | "
            f"epochs: {local_epochs}"
            f"{sync_note}"
        )

    cumulative_bidirectional         = cumulative_divroute_download + cumulative_upload
    cumulative_baseline_bidirectional = cumulative_baseline_download + cumulative_baseline_upload
    bidir_saving = 100.0 * (1.0 - cumulative_bidirectional / cumulative_baseline_bidirectional)

    print(f"\n[done] log written to {config.log_path}")
    print(f"[summary] final acc     : {acc:.4f}")
    print(f"[summary] download      : {cumulative_divroute_download/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_download/1e6:.2f} MB, "
          f"saving {cumulative_saving_pct:.1f}%)")
    print(f"[summary] upload        : {cumulative_upload/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_upload/1e6:.2f} MB)")
    print(f"[summary] bidirectional : {cumulative_bidirectional/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_bidirectional/1e6:.2f} MB, "
          f"saving {bidir_saving:.1f}%)")

    if not config.skip_plot_prompt and input("\nGenerate plots? (y/n): ").strip().lower() == "y":
        from .visualize import generate_all_plots
        generate_all_plots(config.log_path)


if __name__ == "__main__":
    run()