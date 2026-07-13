import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import get_client_datasets, get_test_dataset
from .model import get_model
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

    print(f"[init] loading {config.dataset_name.upper()} + building non-IID splits...")
    client_datasets = get_client_datasets(
        config.dataset_name, config.num_clients, config.alpha, config.seed)
    test_loader = DataLoader(
        get_test_dataset(config.dataset_name), batch_size=256, shuffle=False, num_workers=0)

    shard_sizes = [len(ds) for ds in client_datasets]
    print(f"[init] shard sizes — min: {min(shard_sizes)}, max: {max(shard_sizes)}, "
          f"mean: {np.mean(shard_sizes):.0f}")

    num_classes  = 10 if config.dataset_name.lower() == "cifar10" else 100
    global_model = get_model(config.model_name, num_classes)
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(i, client_datasets[i], config.local_epochs, config.local_lr,
                 config.batch_size, device, config.model_name, num_classes)
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
    print(f"[train] adaptive-k: {config.use_adaptive_k} | "
          f"k_ratio: tier1={config.k_ratio_tier1:.2f} tier2={config.k_ratio_tier2:.2f} | "
          f"include-tier3: {config.include_tier3_in_aggregation} | "
          f"uniform-top5: {config.uniform_top5_mode}")

    cumulative_divroute_download  = 0
    cumulative_baseline_download  = 0
    cumulative_upload             = 0
    cumulative_baseline_upload    = 0

    # ── Checkpoint and Resume System ─────────────────────────────────────────
    import os
    def _get_method_name(cfg: Config) -> str:
        if cfg.fedavg_baseline_mode:
            if getattr(cfg, "fedzip_actual_mode", False):
                return "fedzip"
            if getattr(cfg, "fedsparse_lambda", 0.0) > 0.0:
                return "fedsparse"
            return "fedavg"
        if getattr(cfg, "uniform_top5_mode", False):
            return "uniform"
        return "divroute"

    method_name = _get_method_name(config)
    checkpoint_dir = os.path.join("checkpoints", f"{method_name}_{config.dataset_name}_seed{config.seed}")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    
    start_round = 0
    should_resume = False
    
    if config.fresh:
        should_resume = False
    elif config.resume:
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"Checkpoint not found at {latest_path} but --resume was specified.")
        should_resume = True
    else:
        # Default: auto-resume if checkpoint exists
        if os.path.exists(latest_path):
            should_resume = True
            
    if should_resume:
        print("[checkpoint]")
        print(f"Loaded checkpoint: {latest_path}")
        print(f"Method: {method_name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Seed: {config.seed}")
        
        checkpoint = torch.load(latest_path, map_location=device)
        
        # Safety/compatibility checks
        if checkpoint["method_name"] != method_name:
            raise ValueError(f"Incompatible checkpoint: method mismatch. Checkpoint={checkpoint['method_name']}, Expected={method_name}")
        if checkpoint["dataset_name"] != config.dataset_name:
            raise ValueError(f"Incompatible checkpoint: dataset mismatch. Checkpoint={checkpoint['dataset_name']}, Expected={config.dataset_name}")
        if checkpoint["model_name"] != config.model_name:
            raise ValueError(f"Incompatible checkpoint: model mismatch. Checkpoint={checkpoint['model_name']}, Expected={config.model_name}")
        if checkpoint["seed"] != config.seed:
            raise ValueError(f"Incompatible checkpoint: seed mismatch. Checkpoint={checkpoint['seed']}, Expected={config.seed}")
            
        checkpoint_config = checkpoint["config"]
        for field in ["num_clients", "clients_per_round", "local_lr", "alpha"]:
            checkpoint_val = getattr(checkpoint_config, field, None)
            expected_val = getattr(config, field, None)
            if checkpoint_val != expected_val:
                raise ValueError(f"Incompatible checkpoint: config field '{field}' mismatch. Checkpoint={checkpoint_val}, Expected={expected_val}")
                
        # Restore dynamically adapted configuration state
        config.tau_low = checkpoint_config.tau_low
        config.tau_high = checkpoint_config.tau_high
        config.k_ratio_tier1 = checkpoint_config.k_ratio_tier1
        config.k_ratio_tier2 = checkpoint_config.k_ratio_tier2
                
        # Restore RNG states
        random.setstate(checkpoint["rng_state"]["python"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(checkpoint["rng_state"]["torch_cpu"])
        if torch.cuda.is_available() and checkpoint["rng_state"].get("torch_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["torch_cuda"])
            
        # Restore server state
        server.global_model.load_state_dict(checkpoint["global_model_state_dict"])
        server.selection_weights = checkpoint["server_state"]["selection_weights"]
        server.global_delta = checkpoint["server_state"]["global_delta"]
        if server.global_delta is not None:
            server.global_delta = server.global_delta.to(device)
        server._rng.bit_generator.state = checkpoint["server_state"]["_rng_state"]
        server._momentum_buf = checkpoint["server_state"]["_momentum_buf"]
        if server._momentum_buf is not None:
            server._momentum_buf = server._momentum_buf.to(device)
            
        # Restore client states
        checkpoint_client_irw = checkpoint["client_states"]["irw_norms"]
        for client in clients:
            if client.client_id in checkpoint_client_irw:
                client._irw_norms = checkpoint_client_irw[client.client_id]
                
        # Restore main loop state
        ema_scores = checkpoint["main_loop_state"]["ema_scores"]
        error_buffers = {k: v.to(device) for k, v in checkpoint["main_loop_state"]["error_buffers"].items()}
        _baseline_bpr = checkpoint["main_loop_state"]["_baseline_bpr"]
        cumulative_divroute_download = checkpoint["main_loop_state"]["cumulative_divroute_download"]
        cumulative_baseline_download = checkpoint["main_loop_state"]["cumulative_baseline_download"]
        cumulative_upload = checkpoint["main_loop_state"]["cumulative_upload"]
        cumulative_baseline_upload = checkpoint["main_loop_state"]["cumulative_baseline_upload"]
        
        # Restore logger history
        logger.history = checkpoint["logger_history"]
        logger._flush()
        
        start_round = checkpoint["round_num"] + 1
        print(f"Resuming from round {start_round}/{config.num_rounds}")
    # ─────────────────────────────────────────────────────────────────────────

    for rnd in range(start_round, config.num_rounds):
        selected = server.select_clients(all_ids)
        local_epochs = _get_local_epochs(rnd, config)

        # -- client training -------------------------------------------------
        results = []
        global_sd = server.global_model.state_dict()

        for cid in selected:
            result = clients[cid].train(global_sd, local_epochs,
                                        fedsparse_lambda=config.fedsparse_lambda,
                                        round_num=rnd,
                                        total_rounds=config.num_rounds)
            results.append(result)

        # -- divergence, adaptive tau, and tier assignment ---------------------
        # Skipped entirely in uniform_top5_mode: every client is treated as
        # Tier-2 (top-5% compression) with a neutral divergence score of 0.0.
        if config.uniform_top5_mode:
            for r in results:
                r["divergence_score"] = 0.0
                r["tier"] = 2          # Tier-2 path → k_ratio_tier2=0.05
        else:
            # -- divergence: EMA-smoothed cosine distance -------------------------
            param_names = [n for n, _ in server.global_model.named_parameters()]
            global_flat = torch.cat([
                global_sd[k].flatten().float().cpu() for k in param_names
            ])

            raw_d_scores = []
            for r in results:
                # Skip divergence update for clients with invalid updates.
                # Writing NaN into ema_scores would permanently poison all future rounds.
                if any(torch.isnan(v).any() or torch.isinf(v).any()
                       for v in r["state_dict"].values()):
                    fallback_val = ema_scores.get(r["client_id"], config.tau_low)
                    if rnd == 15:
                        print(f"  [DEBUG-NAN] client {r['client_id']:>3} triggered NaN/Inf fallback. Score set to {fallback_val:.8f}")
                    r["divergence_score"] = fallback_val
                    continue

                local_model = clients[r["client_id"]].get_local_model()
                client_flat = torch.cat([
                    p.flatten().float().cpu() for n, p in local_model.named_parameters()
                ])
                cos = F.cosine_similarity(
                    client_flat.double().unsqueeze(0), global_flat.double().unsqueeze(0),
                    dim=1, eps=1e-8
                ).item()
                d_raw = max(0.0, 1.0 - cos)
                d_ema = update_ema(ema_scores, r["client_id"], d_raw, config.ema_beta)
                r["divergence_score"] = d_ema
                raw_d_scores.append(d_ema)

            # -- adaptive tau (Phase 6.1) ------------------------------------------
            if config.use_adaptive_tau and len(raw_d_scores) >= 3:
                config.tau_low, config.tau_high = compute_adaptive_taus(
                    raw_d_scores, config.tau_low_pct, config.tau_high_pct)

            # -- tier assignment ---------------------------------------------------
            for r in results:
                tier = server.assign_tier(r["divergence_score"])
                # Tier-3 warm-up period (first 15 rounds)
                if rnd < 15 and tier == 3:
                    tier = 2
                r["tier"] = tier

            # -- Progress Guarantee ------------------------------------------------
            p_count = sum(1 for r in results if r["tier"] in (1, 2))
            if p_count == 0 and results:
                highest_div_client = max(results, key=lambda x: x["divergence_score"])
                highest_div_client["tier"] = 2
                print(f"  [Progress Guarantee] Promoted client {highest_div_client['client_id']} to Tier 2")

            # -- [DEBUG] per-client detail on the first post-warmup round ----------
            if rnd == 15:
                print(f"  [DEBUG] round 16 tier assignment — tau_low={config.tau_low:.8f} tau_high={config.tau_high:.8f}")
                for r in results:
                    print(
                        f"  [DEBUG] client {r['client_id']:>3} "
                        f"d={r['divergence_score']:.8f} "
                        f"tier={r['tier']} "
                        f"tau_low={config.tau_low:.8f} "
                        f"tau_high={config.tau_high:.8f}"
                    )
            # ----------------------------------------------------------------------

        # -- communication schedule --------------------------------------------
        if rnd < 30:
            config.k_ratio_tier1 = 0.50
            config.k_ratio_tier2 = 0.20
        else:
            config.k_ratio_tier1 = 0.35
            config.k_ratio_tier2 = 0.10

        # -- adaptive k_ratio (updated before aggregate so compression uses
        #    the correct ratios for this round) --------------------------------
        k1, k2 = get_adaptive_k_ratios(config, rnd)
        config.k_ratio_tier1, config.k_ratio_tier2 = k1, k2

        # -- initialise bytes_received; aggregate applies tiered compression
        #    to each client's upload delta and sets r["bytes_received"] --------
        for r in results:
            r["bytes_received"] = 0

        server.aggregate(results, error_buffers, round_num=rnd + 1)

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
            f"tau: [{config.tau_low:.8f}, {config.tau_high:.8f}] | "
            f"d: [{min(d_vals):.8f}, {max(d_vals):.8f}] | "
            f"epochs: {local_epochs}"
            f"{sync_note}"
        )

        # ── Checkpoint and Resume System: Save Checkpoint ───────────────────────
        completed_round = rnd + 1
        if completed_round % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            client_irw_norms = {client.client_id: client._irw_norms for client in clients}
            
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            
            checkpoint = {
                "round_num": rnd,
                "num_rounds": config.num_rounds,
                "seed": config.seed,
                "config": config,
                "method_name": method_name,
                "dataset_name": config.dataset_name,
                "model_name": config.model_name,
                "global_model_state_dict": {k: v.cpu() for k, v in server.global_model.state_dict().items()},
                "server_state": {
                    "selection_weights": server.selection_weights.copy(),
                    "global_delta": server.global_delta.cpu() if server.global_delta is not None else None,
                    "_rng_state": server._rng.bit_generator.state,
                    "_momentum_buf": server._momentum_buf.cpu() if server._momentum_buf is not None else None,
                },
                "client_states": {
                    "irw_norms": client_irw_norms,
                },
                "main_loop_state": {
                    "ema_scores": ema_scores.copy(),
                    "error_buffers": {k: v.cpu() for k, v in error_buffers.items()},
                    "_baseline_bpr": _baseline_bpr.copy() if _baseline_bpr is not None else None,
                    "cumulative_divroute_download": cumulative_divroute_download,
                    "cumulative_baseline_download": cumulative_baseline_download,
                    "cumulative_upload": cumulative_upload,
                    "cumulative_baseline_upload": cumulative_baseline_upload,
                },
                "logger_history": logger.history.copy(),
                "rng_state": rng_state,
            }
            
            tmp_path = latest_path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, latest_path)
            print(f"[checkpoint] Saved checkpoint Round {completed_round}")
            
            if completed_round % 50 == 0:
                milestone_path = os.path.join(checkpoint_dir, f"round_{completed_round:03d}.pt")
                torch.save(checkpoint, milestone_path)
        # ─────────────────────────────────────────────────────────────────────────

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