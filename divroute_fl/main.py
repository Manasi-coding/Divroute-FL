import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import get_client_datasets, get_client_datasets_with_val, get_test_dataset
from .model import get_model
from .client import FLClient
from .server import FLServer
from .logger import FLLogger
from .mechanism import update_ema, compute_adaptive_taus
from .compression import get_adaptive_k_ratios
from . import diagnostics as _diag


def _spearman(x, y):
    if len(x) < 2: return float('nan')
    x_r = np.argsort(np.argsort(x))
    y_r = np.argsort(np.argsort(y))
    corr = np.corrcoef(x_r, y_r)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def _pearson(x, y):
    if len(x) < 2: return float('nan')
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

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
    if config.local_val_fraction > 0.0:
        # ── Diagnostic mode: 90/10 train/val split per client ──────────────────
        # !! COMPATIBILITY WARNING: clients train on fewer samples than usual !!
        # Do NOT use this when resuming a checkpoint from a run where
        # local_val_fraction == 0.0.  Start a fresh run with --fresh.
        print(f"[init] local_val_fraction={config.local_val_fraction:.2f} — "
              f"holding out {config.local_val_fraction*100:.0f}% of each shard "
              f"for local validation (NEW experimental configuration)")
        client_datasets, val_datasets = get_client_datasets_with_val(
            config.dataset_name, config.num_clients, config.alpha, config.seed,
            val_fraction=config.local_val_fraction)
    else:
        # ── Standard mode: exact original execution path ───────────────────────
        # get_client_datasets() is called directly; nothing is reshuffled.
        client_datasets = get_client_datasets(
            config.dataset_name, config.num_clients, config.alpha, config.seed)
        val_datasets = [None] * config.num_clients
    test_loader = DataLoader(
        get_test_dataset(config.dataset_name), batch_size=256, shuffle=False, num_workers=0)

    shard_sizes = [len(ds) for ds in client_datasets]  # training shard sizes only
    print(f"[init] shard sizes — min: {min(shard_sizes)}, max: {max(shard_sizes)}, "
          f"mean: {np.mean(shard_sizes):.0f}")

    num_classes  = 10 if config.dataset_name.lower() == "cifar10" else 100
    global_model = get_model(config.model_name, num_classes)
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(i, client_datasets[i], config.local_epochs, config.local_lr,
                 config.batch_size, device, config.model_name, num_classes,
                 val_dataset=val_datasets[i])
        for i in range(config.num_clients)
    ]

    logger = FLLogger(config.log_path)
    all_ids = list(range(config.num_clients))

    ema_scores: dict = {}
    _pg_last_round: dict = {}
    error_buffers: dict = {}

    _baseline_bpr: dict | None = None   # FedAvg bytes-per-round (set once after round 0)

    # ── [DIAG] Routing-dynamics tracking ─────────────────────────────────────
    _diag_prev_tiers: dict = {}
    _diag_prev_ema_ranks: dict = {}
    _diag_prev_raw_ranks: dict = {}
    _diag_t1_seen: set = set()
    _diag_hist = {
        "raw_sigma": [], "ema_sigma": [], "raw_spearman": [], "ema_spearman": [],
        "pearson_div_norm": [], "pearson_div_loss": [], "tier_changes": []
    }
    # ─────────────────────────────────────────────────────────────────────────

    # ── [PART 1/3/4/5] Per-round diagnostic accumulators ─────────────────────
    # These are reset each round and only populated when the corresponding
    # Config flag is True.  Zero cost when all flags are False.
    _grad_diags: dict = {}          # {client_id: grad_diag_dict}  (Part 1)
    _raw_deltas_buf: dict = {}      # {client_id: Tensor}           (Parts 4+5)
    _comp_deltas_buf: dict = {}     # {client_id: Tensor}           (Parts 4+5)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"[train] {config.num_rounds} rounds, "
          f"{config.clients_per_round}/{config.num_clients} clients/round")
    print(f"[train] div-weighted: {config.use_divergence_weighting} "
          f"({config.divergence_weight_mode}) | "
          f"momentum: {config.use_server_momentum} | "
          f"error-feedback: {config.use_error_feedback} | "
          f"adaptive-tau: {config.use_adaptive_tau} "
          f"(alpha={config.tau_alpha}/beta={config.tau_beta}) | "
          f"tier3-sync: {config.use_tier3_sync} "
          f"(every {config.tier3_sync_interval} rounds) | "
          f"epoch-warmup: {config.use_epoch_warmup} | "
          f"fedavg-mode: {config.fedavg_baseline_mode}")
    print(f"[train] adaptive-k: {config.use_adaptive_k} | "
          f"k_ratio: tier1={config.k_ratio_tier1:.2f} tier2={config.k_ratio_tier2:.2f} | "
          f"include-tier3: {config.include_tier3_in_aggregation} | "
          f"uniform-top5: {config.uniform_top5_mode}")
          
    print("\n[Transmission Precision]")
    print(f"  Upload   : {'FP16' if getattr(config, 'use_fp16_upload', False) else 'FP32'}")
    print(f"  Download : {'FP16' if getattr(config, 'use_fp16_download', False) else 'FP32'}\n")

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
    save_checkpoint_dir = os.path.join("checkpoints", f"{method_name}_{config.dataset_name}_seed{config.seed}")
    latest_save_path = os.path.join(save_checkpoint_dir, "latest.pt")

    load_checkpoint_dir = os.path.join(getattr(config, "checkpoint_dir", "checkpoints"), f"{method_name}_{config.dataset_name}_seed{config.seed}")
    latest_load_path = os.path.join(load_checkpoint_dir, "latest.pt")
    
    start_round = 0
    should_resume = False
    
    if config.fresh:
        should_resume = False
    elif config.resume:
        if not os.path.exists(latest_load_path):
            raise FileNotFoundError(f"Checkpoint not found at {latest_load_path} but --resume was specified.")
        should_resume = True
    else:
        # Default: auto-resume if checkpoint exists
        if os.path.exists(latest_load_path):
            should_resume = True
            
    if should_resume:
        print("[checkpoint]")
        print(f"Resuming from checkpoint directory:\n{config.checkpoint_dir}")
        print(f"Loaded checkpoint: {latest_load_path}")
        print(f"Method: {method_name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Seed: {config.seed}")
        
        checkpoint = torch.load(latest_load_path, map_location=device, weights_only=False)
        
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
                
        # Restore RNG states (robust across PyTorch versions)
        try:
            random.setstate(checkpoint["rng_state"]["python"])
        except Exception as e:
            print(f"[resume] Warning: could not restore Python RNG ({e})")

        try:
            np.random.set_state(checkpoint["rng_state"]["numpy"])
        except Exception as e:
            print(f"[resume] Warning: could not restore NumPy RNG ({e})")

        try:
            cpu_state = checkpoint["rng_state"]["torch_cpu"]

            if not isinstance(cpu_state, torch.ByteTensor):
                cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)

            torch.set_rng_state(cpu_state)
        except Exception as e:
            print(f"[resume] Warning: could not restore CPU RNG ({e}), continuing...")

        try:
            if torch.cuda.is_available():
                cuda_state = checkpoint["rng_state"].get("torch_cuda", [])

                if cuda_state:
                    fixed_states = []
                    for state in cuda_state:
                        if not isinstance(state, torch.ByteTensor):
                            state = torch.tensor(state, dtype=torch.uint8)
                        fixed_states.append(state)

                    torch.cuda.set_rng_state_all(fixed_states)
        except Exception as e:
            print(f"[resume] Warning: could not restore CUDA RNG ({e}), continuing...")
            
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
        _pg_last_round = checkpoint["main_loop_state"].get("_pg_last_round", {})
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
        
        if getattr(config, "use_fp16_download", False):
            # simulate transmission by casting to fp16 then back to fp32
            global_sd = {k: v.half().float() for k, v in global_sd.items()}

        for cid in selected:
            result = clients[cid].train(global_sd, local_epochs,
                                        fedsparse_lambda=config.fedsparse_lambda,
                                        ntd_beta=config.ntd_beta,
                                        ntd_tau=config.ntd_tau,
                                        round_num=rnd,
                                        total_rounds=config.num_rounds)
            results.append(result)

        # -- divergence, adaptive tau, and tier assignment ---------------------
        # Skipped entirely in uniform_top5_mode: every client is treated as
        # Tier-2 (top-5% compression) with a neutral divergence score of 0.0.

        # Reset per-round diagnostic accumulators
        _grad_diags = {}
        _raw_deltas_buf = {}
        _comp_deltas_buf = {}
        _tau_diag_payload = None
        raw_d_scores = []

        if config.uniform_top5_mode:
            for r in results:
                r["divergence_score"] = 0.0
                r["tier"] = 2          # Tier-2 path → k_ratio_tier2=0.05
        else:
            # -- divergence: EMA-smoothed distance (metric is configurable) ------
            # Default: "cosine" reproduces the original implementation exactly.
            # Alternative metrics (l2, relative_l2, layerwise_cosine) are used
            # when config.divergence_metric != "cosine".
            param_names = [n for n, _ in server.global_model.named_parameters()]
            global_flat = torch.cat([
                global_sd[k].flatten().float().cpu() for k in param_names
            ])

            # Build layer_slices once per run if layerwise_cosine is requested
            _div_layer_slices = None
            if getattr(config, "divergence_metric", "cosine") == "layerwise_cosine":
                offset = 0
                _div_layer_slices = []
                for k in param_names:
                    numel = global_sd[k].numel()
                    _div_layer_slices.append((k, offset, numel))
                    offset += numel

            # [DIAG] per-client raw d captured BEFORE EMA update this round
            _diag_raw_per_client: dict = {}
            _diag_norm_per_client: dict = {}
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

                # ── PART 2: divergence metric dispatch ─────────────────────────────
                # When divergence_metric=="cosine" the path is identical to the
                # original implementation.  Other metrics are computed by
                # _diag.compute_divergence_metric() which has no side-effects.
                _div_metric = getattr(config, "divergence_metric", "cosine")
                if _div_metric == "cosine":
                    # Preserve original FP64 cosine path exactly
                    cos = F.cosine_similarity(
                        client_flat.double().unsqueeze(0),
                        global_flat.double().unsqueeze(0),
                        dim=1, eps=1e-8,
                    ).item()
                    d_raw = max(0.0, 1.0 - cos)
                else:
                    d_raw = _diag.compute_divergence_metric(
                        client_flat, global_flat,
                        metric=_div_metric,
                        layer_slices=_div_layer_slices,
                    )
                # ─────────────────────────────────────────────────────────────────────

                # ── PART 1: gradient quality diagnostics (pre-compression) ───────
                # Compression error fields are filled after server.aggregate()
                # has computed the compressed delta.
                if getattr(config, "enable_gradient_diagnostics", False):
                    _grad_diags[r["client_id"]] = _diag.compute_gradient_diagnostics(
                        client_flat, global_flat, compressed_flat=None
                    )
                # ─────────────────────────────────────────────────────────────────────

                if getattr(config, "enable_routing_diagnostics", False):
                    _diag_raw_per_client[r["client_id"]] = d_raw
                    _diag_norm_per_client[r["client_id"]] = float(torch.norm(client_flat - global_flat).item())
                    
                d_ema = update_ema(ema_scores, r["client_id"], d_raw, config.ema_beta)
                r["divergence_score"] = d_ema   # d_ema — always used for aggregation weighting
                r["d_raw"] = d_raw              # stored for tier routing when routing_score="raw"
                raw_d_scores.append(d_raw)      # tau estimation uses raw divergence, not EMA

                # -- Memory Optimisation (Change 1) --
                # Release the client's GPU model cache immediately to prevent OOM
                # across rounds.
                clients[r["client_id"]]._local_model = None
                del local_model
                del client_flat

            # -- adaptive tau (Phase 6.1) ------------------------------------------
            if config.use_adaptive_tau and len(raw_d_scores) >= 3:
                config.tau_low, config.tau_high, _mu, _sigma = compute_adaptive_taus(
                    raw_d_scores, config.tau_alpha, config.tau_beta)
            else:
                _mu, _sigma = 0.0, 0.0

            # ── PART 6: adaptive tau diagnostics ────────────────────────────────────
            if getattr(config, "enable_tau_diagnostics", False) and raw_d_scores:
                _tau_diag_payload = _diag.compute_tau_diagnostics(
                    raw_d_scores, _mu, _sigma,
                    config.tau_low, config.tau_high, results
                )
                if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0:
                    _diag.print_tau_diagnostics(rnd + 1, _tau_diag_payload)
            # ──────────────────────────────────────────────────────────────────────

            # -- tier assignment ---------------------------------------------------
            # routing_score="raw": compare d_raw against tau (consistent, since tau
            #   is derived from the raw distribution via compute_adaptive_taus).
            # routing_score="ema": compare d_ema against tau (legacy, smooth but biased
            #   upward during convergence because EMA lags behind declining d_raw).
            # Aggregation weighting uses r["divergence_score"] (d_ema) in both modes.
            for r in results:
                d_for_tier = (
                    r.get("d_raw", r["divergence_score"])   # fall back for NaN clients
                    if config.routing_score == "raw"
                    else r["divergence_score"]
                )
                tier = server.assign_tier(d_for_tier)
                r["natural_tier"] = tier
                # Tier-3 warm-up period (first 15 rounds)
                if rnd < 15 and tier == 3:
                    tier = 2
                r["tier"] = tier

            # -- Progress Guarantee ------------------------------------------------
            p_count = sum(1 for r in results if r["tier"] in (1, 2))
            if p_count == 0 and results:
                ranked = sorted(
                    results,
                    key=lambda r: (r["divergence_score"], -_pg_last_round.get(r["client_id"], -1)),
                    reverse=True
                )
                for winner in ranked[:2]:
                    winner["tier"] = 2
                    _pg_last_round[winner["client_id"]] = rnd
                    print(f"  [Progress Guarantee] Promoted client {winner['client_id']} to Tier 2")

            # ── PART 3: routing quality recording (accumulated every round) ───────
            # Correlation computations happen after aggregation (when agg weights
            # are available).  Recording here is just appending raw values.
            # ─────────────────────────────────────────────────────────────────────────

            # -- Routing diagnostic table -------------------------------------------
            # Only prints at specific rounds to keep logs clean
            _diag_rounds = {1, 2, 3, 4, 5, 25, 50, 100, 150, 200, config.num_rounds}
            if (rnd + 1) in _diag_rounds:
                _rout_diff = 0
                _rout_lines = []
                _raw_counts = {1: 0, 2: 0, 3: 0}
                _ema_counts = {1: 0, 2: 0, 3: 0}
                _act_counts = {1: 0, 2: 0, 3: 0}
                
                _valid_d_raws = []
                _valid_d_emas = []
                
                for r in results:
                    _dr = r.get("d_raw", None)
                    _de = r["divergence_score"]
                    _act_counts[r["tier"]] += 1
                    
                    if _dr is None:                          # NaN-guarded client
                        _tr = _te = r["tier"]
                    else:
                        _tr = server.assign_tier(_dr)       # tier using raw score
                        _te = server.assign_tier(_de)       # tier using ema score
                        _valid_d_raws.append(_dr)
                        _valid_d_emas.append(_de)
                        
                    _raw_counts[_tr] += 1
                    _ema_counts[_te] += 1
                    
                    if _tr != _te:
                        _rout_diff += 1
                    _rout_lines.append((r["client_id"], _dr, _de, _tr, _te, r["tier"]))
                    
                print(f"\n  [ROUTING] round={rnd+1} mode={config.routing_score!r} "
                      f"tau=[{config.tau_low:.6f},{config.tau_high:.6f}]")
                print(f"  {'CID':>4}  {'d_raw':>10}  {'d_ema':>10}  "
                      f"{'t(raw)':>7}  {'t(ema)':>7}  {'chosen':>7}")
                for _cid, _dr2, _de2, _tr2, _te2, _tc2 in _rout_lines:
                    _dr_str = f"{_dr2:.6f}" if _dr2 is not None else "     NaN"
                    _mk = " *" if _tr2 != _te2 else ""
                    print(f"  {_cid:>4}  {_dr_str:>10}  {_de2:>10.6f}  "
                          f"{_tr2:>7}  {_te2:>7}  {_tc2:>7}{_mk}")
                
                print(f"\n  [ROUTING SUMMARY]")
                print(f"  Routing disagreement: Different assignments: {_rout_diff} / {len(results)}")
                print(f"  Tier counts (raw)   : {_raw_counts[1]}/{_raw_counts[2]}/{_raw_counts[3]}")
                print(f"  Tier counts (ema)   : {_ema_counts[1]}/{_ema_counts[2]}/{_ema_counts[3]}")
                print(f"  Actual tier counts  : {_act_counts[1]}/{_act_counts[2]}/{_act_counts[3]}")
                
                if _valid_d_raws:
                    _avg_raw = sum(_valid_d_raws) / len(_valid_d_raws)
                    _avg_ema = sum(_valid_d_emas) / len(_valid_d_emas)
                    _avg_diff = sum(e - r for e, r in zip(_valid_d_emas, _valid_d_raws)) / len(_valid_d_raws)
                    _max_abs = max(abs(e - r) for e, r in zip(_valid_d_emas, _valid_d_raws))
                    print(f"  Average d_raw                   : {_avg_raw:.8f}")
                    print(f"  Average d_ema                   : {_avg_ema:.8f}")
                    print(f"  Average (d_ema - d_raw)         : {_avg_diff:.8f}")
                    print(f"  Maximum absolute difference     : {_max_abs:.8f}")
                    print(f"  Clients whose routing changes   : {_rout_diff}")
                print()
            # -----------------------------------------------------------------------

            # -- [DIAG] Full routing-dynamics instrumentation ──────────────────────
            if getattr(config, "enable_routing_diagnostics", False) and len(raw_d_scores) >= 2:
                # [BUG FIX 2] _valid_for_diag restricts to clients that passed the NaN-guard
                # and have a valid raw score.  Iterating all `results` previously raised KeyError
                # for NaN-guarded clients absent from _diag_raw_per_client.
                _valid_for_diag = [r for r in results if r["client_id"] in _diag_raw_per_client]
                _raw_arr = np.array([_diag_raw_per_client[r["client_id"]] for r in _valid_for_diag], dtype=np.float64)
                # [BUG FIX 2] _ema_arr was previously built from raw_d_scores (raw divergences),
                # making it byte-identical to _raw_arr and causing EMA stats to appear the same
                # as raw stats.  Now correctly uses r["divergence_score"] (d_ema).
                _ema_arr = np.array([r["divergence_score"] for r in _valid_for_diag], dtype=np.float64)
                
                _r_mu = float(np.mean(_raw_arr))
                _r_med = float(np.median(_raw_arr))
                _r_sigma = float(np.std(_raw_arr))
                _r_min = float(np.min(_raw_arr))
                _r_max = float(np.max(_raw_arr))
                _r_spread = _r_max - _r_min
                
                _e_mu = float(np.mean(_ema_arr))
                _e_med = float(np.median(_ema_arr))
                _e_sigma = float(np.std(_ema_arr))
                _e_min = float(np.min(_ema_arr))
                _e_max = float(np.max(_ema_arr))
                _e_spread = _e_max - _e_min
                
                _diag_hist["raw_sigma"].append(_r_sigma)
                _diag_hist["ema_sigma"].append(_e_sigma)
                
                print("\n  [DIAG] 1. Raw vs EMA statistics")
                print(f"  raw : mu={_r_mu:.8f} med={_r_med:.8f} std={_r_sigma:.8f} min={_r_min:.8f} max={_r_max:.8f} spread={_r_spread:.8f}")
                print(f"  EMA : mu={_e_mu:.8f} med={_e_med:.8f} std={_e_sigma:.8f} min={_e_min:.8f} max={_e_max:.8f} spread={_e_spread:.8f}")
                
                # [BUG FIX 3] Rankings computed only over valid (non-NaN-guarded) clients.
                # Rank 0 = highest divergence within the selected cohort this round.
                _cur_raw_ranks = {
                    r["client_id"]: rank for rank, r in enumerate(
                        sorted(_valid_for_diag, key=lambda x: _diag_raw_per_client[x["client_id"]], reverse=True))
                }
                _cur_ema_ranks = {
                    r["client_id"]: rank for rank, r in enumerate(
                        sorted(_valid_for_diag, key=lambda x: x["divergence_score"], reverse=True))
                }
                _cur_tiers = {r["client_id"]: r["tier"] for r in results}

                # [BUG FIX 3] Spearman is now WITHIN-ROUND: measures agreement between raw and
                # EMA rankings for the same set of clients in the current round.  The previous
                # implementation compared prev-round ranks vs curr-round ranks (cross-round
                # stability), which could produce spurious ±1.0 when only 2 clients overlapped.
                # Aligned by client_id: same _aligned_cids list for both ranking arrays.
                _aligned_cids = sorted(_cur_raw_ranks.keys())
                if len(_aligned_cids) >= 2:
                    _rve_spearman = _spearman(
                        [_cur_raw_ranks[c] for c in _aligned_cids],
                        [_cur_ema_ranks[c] for c in _aligned_cids]
                    )
                    # rank_changes: clients whose raw rank ≠ EMA rank within this round
                    _rank_changes = sum(1 for c in _aligned_cids if _cur_raw_ranks[c] != _cur_ema_ranks[c])
                else:
                    _rve_spearman = float('nan')
                    _rank_changes = 0

                # Cross-round tier stability (separate metric from within-round Spearman)
                _common_cid = [c for c in _cur_tiers if c in _diag_prev_tiers]
                _tier_changes = sum(1 for c in _common_cid if _cur_tiers[c] != _diag_prev_tiers[c])
                
                if _common_cid:
                    _t1_p = {c for c, t in _diag_prev_tiers.items() if t == 1}
                    _t2_p = {c for c, t in _diag_prev_tiers.items() if t == 2}
                    _t3_p = {c for c, t in _diag_prev_tiers.items() if t == 3}
                    _t1_c = {c for c, t in _cur_tiers.items() if t == 1}
                    _t2_c = {c for c, t in _cur_tiers.items() if t == 2}
                    _t3_c = {c for c, t in _cur_tiers.items() if t == 3}
                    _t1_ret = len(_t1_p & _t1_c) / max(len(_t1_p), 1)
                    _t2_ret = len(_t2_p & _t2_c) / max(len(_t2_p), 1)
                    _t3_ret = len(_t3_p & _t3_c) / max(len(_t3_p), 1)
                else:
                    _t1_ret = _t2_ret = _t3_ret = float('nan')
                
                # Store within-round raw↔EMA Spearman in both hist slots
                if not np.isnan(_rve_spearman): _diag_hist["raw_spearman"].append(_rve_spearman)
                if not np.isnan(_rve_spearman): _diag_hist["ema_spearman"].append(_rve_spearman)
                _diag_hist["tier_changes"].append(_tier_changes)

                print("  [DIAG] 2. Ranking stability (within-round raw\u2194EMA)")
                print(f"  raw\u2194ema spearman={_rve_spearman:.4f}  rank_changes(raw\u2260ema)={_rank_changes}  tier_changes_vs_prev={_tier_changes}")
                print(f"  retention: T1={_t1_ret:.2f} T2={_t2_ret:.2f} T3={_t3_ret:.2f}")
                
                print("  [DIAG] 3. Optimisation correlation")
                # [BUG FIX 4] _norms and _losses now aligned to _valid_for_diag (same client list as
                # _raw_arr / _ema_arr) to prevent KeyError for NaN-guarded clients and misalignment.
                _norms = np.array([_diag_norm_per_client[r["client_id"]] for r in _valid_for_diag])
                _losses = np.array([r.get("local_loss", 0.0) for r in _valid_for_diag])
                _p_div_norm = _pearson(_raw_arr, _norms)
                _s_div_norm = _spearman(_raw_arr, _norms)
                _p_div_loss = _pearson(_raw_arr, _losses)
                _s_div_loss = _spearman(_raw_arr, _losses)
                if not np.isnan(_p_div_norm): _diag_hist["pearson_div_norm"].append(_p_div_norm)
                if not np.isnan(_p_div_loss): _diag_hist["pearson_div_loss"].append(_p_div_loss)
                for r in _valid_for_diag:
                    cid = r["client_id"]
                    print(f"  c{cid:>3}: raw={_diag_raw_per_client[cid]:.6f} ema={r['divergence_score']:.6f} norm={_diag_norm_per_client[cid]:.4f} loss={r.get('local_loss', 0.0):.4f}")
                # [ISSUE 5 NOTE] For cosine divergence, div = 1 - cos(w_client, w_global) measures
                # DIRECTIONAL drift only; update_norm = ||w_client - w_global||_2 measures both
                # direction and magnitude.  They are NOT mathematically equivalent.  High
                # Pearson/Spearman (e.g. +1.0) here indicates clients with larger parameter
                # displacements also tend to drift more directionally in this run.  This may be
                # coincidental.  Monitor across rounds — it does NOT imply the routing signal
                # is merely a transformed norm.
                print(f"  correlation div<->norm: pearson={_p_div_norm:+.4f} spearman={_s_div_norm:+.4f}")
                print(f"  correlation div<->loss: pearson={_p_div_loss:+.4f} spearman={_s_div_loss:+.4f}")
                
                print("  [DIAG] 4. Threshold diagnostics")
                _tau_diff = config.tau_high - config.tau_low
                _ratio = _r_spread / _tau_diff if _tau_diff > 0 else 0.0
                print(f"  mu={_mu:.6f} sigma={_sigma:.6f} tau_low={config.tau_low:.6f} tau_high={config.tau_high:.6f}")
                print(f"  high-low={_tau_diff:.6f}  spread/diff_ratio={_ratio:.4f}")
                
                print("  [DIAG] 5. Routing diversity")
                _t1_ids = [r["client_id"] for r in results if r["tier"] == 1]
                _t2_ids = [r["client_id"] for r in results if r["tier"] == 2]
                _t3_ids = [r["client_id"] for r in results if r["tier"] == 3]
                # [BUG FIX 1] T1 entering/leaving now use proper set-difference semantics.
                # OLD _t1_entered: counted clients absent from prev round as "entering" (wrong).
                # OLD _t1_left:    required c in _cur_tiers, hiding clients that left the selected
                #                  set entirely (they were correctly T1-leaving but not counted).
                # CORRECT: entering = current_T1 - previous_T1  (new arrivals to T1)
                #          leaving  = previous_T1 - current_T1  (clients no longer in T1)
                _t1_prev_set = {c for c, t in _diag_prev_tiers.items() if t == 1}
                _t1_curr_set = set(_t1_ids)
                _t1_entered = _t1_curr_set - _t1_prev_set   # in T1 now but NOT in T1 last round
                _t1_left    = _t1_prev_set - _t1_curr_set   # was in T1 last round but NOT in T1 now
                for c in _t1_ids: _diag_t1_seen.add(c)
                print(f"  T1 ({len(_t1_ids)}): {sorted(_t1_ids)}")
                print(f"  T2 ({len(_t2_ids)}): {sorted(_t2_ids)}")
                print(f"  T3 ({len(_t3_ids)}): {sorted(_t3_ids)}")
                print(f"  T1 entering={len(_t1_entered)} leaving={len(_t1_left)} unique_seen={len(_diag_t1_seen)}")
                # [ISSUE 7] Internal consistency checks — emit warnings, never crash training.
                _t2_set = set(_t2_ids)
                _t3_set = set(_t3_ids)
                _all_selected_ids = {r["client_id"] for r in results}
                if _t1_curr_set & _t2_set:
                    print(f"  [DIAG WARNING] T1\u2229T2 not empty: {_t1_curr_set & _t2_set}")
                if _t1_curr_set & _t3_set:
                    print(f"  [DIAG WARNING] T1\u2229T3 not empty: {_t1_curr_set & _t3_set}")
                if _t2_set & _t3_set:
                    print(f"  [DIAG WARNING] T2\u2229T3 not empty: {_t2_set & _t3_set}")
                if _t1_curr_set | _t2_set | _t3_set != _all_selected_ids:
                    _missing = _all_selected_ids - (_t1_curr_set | _t2_set | _t3_set)
                    print(f"  [DIAG WARNING] T1\u222aT2\u222aT3 \u2260 selected clients. Missing: {_missing}")
                if _t1_entered & _t1_left:
                    print(f"  [DIAG WARNING] T1 entering\u2229leaving not empty: {_t1_entered & _t1_left}")
                if len(_raw_arr) != len(_ema_arr):
                    print(f"  [DIAG WARNING] raw_arr length ({len(_raw_arr)}) \u2260 ema_arr length ({len(_ema_arr)})")
                
                _diag_prev_raw_ranks = _cur_raw_ranks
                _diag_prev_ema_ranks = _cur_ema_ranks
                _diag_prev_tiers = _cur_tiers
            # ── end [DIAG] block ──────────────────────────────────────────────────

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
            
            # -- Memory Optimisation (Change 1) --
            # Explicitly release the global parameters tensor before server aggregation
            # to keep peak GPU memory down.
            del global_flat



        # -- adaptive k_ratio (updated before aggregate so compression uses
        #    the correct ratios for this round) --------------------------------
        k1, k2 = get_adaptive_k_ratios(config, rnd)
        config.active_k_ratio_tier1, config.active_k_ratio_tier2 = k1, k2

        # -- initialise bytes; aggregate() overwrites these for Tier-1/2 clients.
        #    Tier-3 clients send 0 upload bytes and receive no server-side payload
        #    here (heartbeat is counted separately below if tier3_sync is active).
        for r in results:
            r["bytes_received"] = 0
            r["upload_bytes"]   = 0    # compressed upload — set by server.aggregate()
            r["download_bytes"] = 0    # full-model download — set by server.aggregate()

        server.aggregate(results, error_buffers, round_num=rnd + 1)

        # ── PART 1 (post-compression) + PARTS 4+5: capture per-client deltas ───────
        # global_delta is the weighted agg update; we need per-client compressed
        # norms.  We read upload_bytes set by aggregate() as a proxy for the
        # compressed norm (values are already freed).  For exact reconstruction
        # error the raw delta tensors are needed — those are computed during the
        # aggregation loop inside server.aggregate().  We capture them here via
        # a post-hoc per-client recompute if either flag is True.
        _do_delta_capture = (
            getattr(config, "enable_gradient_diagnostics", False)
            or getattr(config, "enable_compression_analysis", False)
        )
        if _do_delta_capture:
            _global_sd_after = server.global_model.state_dict()
            _gf_after = torch.cat([
                _global_sd_after[k].flatten().float().cpu()
                for k in [n for n, _ in server.global_model.named_parameters()]
            ])
            for r in results:
                cid = r.get("client_id")
                if cid is None:
                    continue
                # Re-flatten client state to get raw delta (client is still in memory)
                try:
                    _cf = torch.cat([
                        v.flatten().float().cpu()
                        for v in r["state_dict"].values()
                        if v.dtype.is_floating_point and v.dim() >= 1
                    ])
                    # Align length to global_flat in case of BN buffer size diff
                    _numel = _gf_after.numel()
                    if _cf.numel() >= _numel:
                        _raw = _cf[:_numel] - _gf_after
                    else:
                        _raw = None
                except Exception:
                    _raw = None

                if _raw is not None:
                    _raw_deltas_buf[cid] = _raw
                    # Approximate compressed delta from upload_bytes
                    _ub = r.get("upload_bytes", 0) or 0
                    _k_approx = _ub / (_numel * 8) if _numel > 0 and _ub > 0 else 0.0
                    # Use global_delta * weight as compressed approximation when available
                    if server.global_delta is not None and r.get("aggregation_weight", 0) > 0:
                        # Scale global delta inversely by weight to approximate single-client
                        _comp = server.global_delta.cpu() / max(r["aggregation_weight"], 1e-9)
                        _comp_deltas_buf[cid] = _comp[:_numel]
                    else:
                        _comp_deltas_buf[cid] = torch.zeros_like(_raw)

                # Update Part 1 grad_diag with compression fields
                if getattr(config, "enable_gradient_diagnostics", False) and cid in _grad_diags:
                    _compressed_flat = _comp_deltas_buf.get(cid)
                    if _compressed_flat is not None and cid in _raw_deltas_buf:
                        _gd_raw = _raw_deltas_buf[cid]
                        _orig_norm = float(_gd_raw.norm(p=2).item())
                        _comp_norm = float(_compressed_flat.norm(p=2).item())
                        _err = float((_gd_raw - _compressed_flat).norm(p=2).item())
                        _grad_diags[cid]["compression_error"] = _err
                        _grad_diags[cid]["retention_ratio"] = _comp_norm / max(_orig_norm, 1e-12)
                        import torch.nn.functional as _F
                        if _orig_norm > 1e-12 and _comp_norm > 1e-12:
                            _cs = _F.cosine_similarity(
                                _gd_raw.unsqueeze(0), _compressed_flat.unsqueeze(0),
                                dim=1, eps=1e-8
                            ).item()
                            _grad_diags[cid]["compressed_cosine"] = float(_cs)
        # ─────────────────────────────────────────────────────────────────────────

        # ── PART 3: routing quality correlations (post-aggregation) ─────────────
        _routing_corr_payload = None
        if getattr(config, "enable_routing_quality_analysis", False):
            _diag.record_routing_quality(rnd + 1, results, _grad_diags)
            _routing_corr_payload = _diag.compute_routing_correlations(
                _diag._routing_quality_rows, rnd + 1
            )
            if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0:
                _diag.print_routing_quality(rnd + 1, _routing_corr_payload)
        # ─────────────────────────────────────────────────────────────────────────

        # ── PARTS 4+5: tier contribution + compression analysis ─────────────────
        _tier_contrib_payload = None
        _compress_analysis_payload = None
        if getattr(config, "enable_compression_analysis", False) and _raw_deltas_buf:
            _tier_contrib_payload = _diag.compute_tier_contributions(
                results, _raw_deltas_buf, _comp_deltas_buf
            )
            _compress_analysis_payload = _diag.compute_compression_analysis(
                results, _raw_deltas_buf, _comp_deltas_buf
            )
            if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0:
                _diag.print_tier_contributions(rnd + 1, _tier_contrib_payload)
                _diag.print_compression_analysis(rnd + 1, _compress_analysis_payload)
        # ─────────────────────────────────────────────────────────────────────────

        # ── PART 1: print gradient diagnostics summary ──────────────────────────
        _agg_norm_payload = None
        _global_norm_payload = None
        if getattr(config, "enable_gradient_diagnostics", False):
            _agg_norm = float(server.global_delta.norm(p=2).item()) if server.global_delta is not None else 0.0
            _global_norm = float(
                torch.cat([p.data.flatten().float().cpu()
                           for p in server.global_model.parameters()]).norm(p=2).item()
            )
            _agg_norm_payload = _agg_norm
            _global_norm_payload = _global_norm
            if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0 and _grad_diags:
                _diag.print_gradient_diagnostics_summary(
                    rnd + 1, _grad_diags, _agg_norm, _global_norm
                )
        # ─────────────────────────────────────────────────────────────────────────

        # -- BN Ablation Overhead ----------------------------------------------
        if getattr(config, "bn_ablation_mode", False):
            # ResNet-18 BN buffers take exactly 38,480 bytes in FP32
            for r in results:
                r["upload_bytes"] += 38480

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
        logger.log(
            rnd, acc, results, delta_numel=delta_numel,
            # ── PART 7: pass diagnostic payloads (None when disabled = no overhead) ──
            grad_diagnostics     = _grad_diags if _grad_diags else None,
            agg_update_norm      = _agg_norm_payload,
            global_update_norm   = _global_norm_payload,
            routing_correlations = _routing_corr_payload,
            tier_contributions   = _tier_contrib_payload,
            compression_analysis = _compress_analysis_payload,
            tau_diagnostics      = _tau_diag_payload if not config.uniform_top5_mode else None,
        )
        server.update_selection_weights(results)

        # -- communication accounting (corrected) ---------------------------------
        # upload  = compressed bytes each client sent to the server (set in aggregate())
        # download = full global model the server broadcast to every selected client
        #            (Tier-3 clients receive the full model too, even if not aggregated)
        bytes_per_param = 2 if getattr(config, "use_fp16_download", False) else 4
        full_model_bytes = delta_numel * bytes_per_param   # bytes in the global parameter vector
        total_upload   = sum(r["upload_bytes"]   for r in results)
        total_download = config.clients_per_round * full_model_bytes

        # Heartbeat (Tier-3 sync): counts as additional upload for those clients
        # (already added to r["bytes_received"] below; not re-counted here)

        cumulative_upload            += total_upload
        cumulative_baseline_upload   += _baseline_bpr["upload"]
        cumulative_divroute_download += total_download
        cumulative_baseline_download += _baseline_bpr["download"]

        total_bidir    = total_upload + total_download
        baseline_bidir = _baseline_bpr["bidir"]
        saving_pct = 100.0 * (1.0 - total_bidir / baseline_bidir)

        cumulative_saving_pct = 100.0 * (
            1.0
            - (cumulative_upload + cumulative_divroute_download)
            / (cumulative_baseline_upload + cumulative_baseline_download)
        )

        t1 = sum(1 for r in results if r["tier"] == 1)
        t2 = sum(1 for r in results if r["tier"] == 2)
        t3 = sum(1 for r in results if r["tier"] == 3)

        d_vals = [r["divergence_score"] for r in results]
        sync_note = f" | t3-sync: {tier3_sync_count}" if tier3_sync_count > 0 else ""
        print(
            f"  round {rnd + 1:>3}/{config.num_rounds} | acc: {acc:.4f} | "
            f"tiers: {t1}/{t2}/{t3} | "
            f"up: {total_upload/1e6:.3f}MB "
            f"down: {total_download/1e6:.3f}MB "
            f"(bidir save {saving_pct:.1f}%, cum {cumulative_saving_pct:.1f}%) | "
            f"tau: [{config.tau_low:.8f}, {config.tau_high:.8f}] | "
            f"d: [{min(d_vals):.8f}, {max(d_vals):.8f}] | "
            f"epochs: {local_epochs}"
            f"{sync_note}"
        )
        for r in sorted(results, key=lambda x: x["client_id"]):
            _train_acc = r.get('local_train_acc', 0.0)
            _val_acc   = r.get('local_val_acc')
            _val_str   = f"{_val_acc:.4f}" if _val_acc is not None else "  N/A  "
            print(f"    Client {r['client_id']:>3} | Local Train Acc: {_train_acc:.4f} | "
                  f"Local Val Acc: {_val_str} | "
                  f"Agg Weight: {r.get('aggregation_weight', 0.0):.4f} | Tier: {r.get('tier')}")

        # ── Checkpoint and Resume System: Save Checkpoint ───────────────────────
        completed_round = rnd + 1
        if completed_round % 5 == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            
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
                    "_pg_last_round": _pg_last_round.copy(),
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
            
            tmp_path = latest_save_path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, latest_save_path)
            print(f"[checkpoint] Saved checkpoint Round {completed_round}")
            
            if completed_round % 50 == 0:
                milestone_path = os.path.join(save_checkpoint_dir, f"round_{completed_round:03d}.pt")
                torch.save(checkpoint, milestone_path)
        # ─────────────────────────────────────────────────────────────────────────

    if getattr(config, "enable_routing_diagnostics", False):
        print("\n=== ROUTING DIAGNOSTICS SUMMARY ===")
        print(f"Average raw sigma: {np.mean(_diag_hist['raw_sigma']):.8f}" if _diag_hist["raw_sigma"] else "Average raw sigma: N/A")
        print(f"Average EMA sigma: {np.mean(_diag_hist['ema_sigma']):.8f}" if _diag_hist["ema_sigma"] else "Average EMA sigma: N/A")
        if _diag_hist["raw_sigma"] and _diag_hist["ema_sigma"]:
            print(f"Average raw/EMA ratio: {np.mean(np.array(_diag_hist['raw_sigma']) / np.array(_diag_hist['ema_sigma'])):.4f}")
        # Both 'raw_spearman' and 'ema_spearman' slots now store the within-round raw<->EMA Spearman
        print(f"Average raw\u2194EMA Spearman (within-round): {np.mean(_diag_hist['raw_spearman']):.4f}" if _diag_hist["raw_spearman"] else "Average raw\u2194EMA Spearman: N/A")
        print(f"Average div<->norm Pearson: {np.mean(_diag_hist['pearson_div_norm']):.4f}" if _diag_hist["pearson_div_norm"] else "Average div<->norm Pearson: N/A")
        print(f"Average div<->loss Pearson: {np.mean(_diag_hist['pearson_div_loss']):.4f}" if _diag_hist["pearson_div_loss"] else "Average div<->loss Pearson: N/A")
        print(f"Average tier changes/round: {np.mean(_diag_hist['tier_changes']):.2f}" if _diag_hist["tier_changes"] else "Average tier changes/round: N/A")
        print(f"Unique T1 clients observed: {len(_diag_t1_seen)}")
        print("===================================")

    cumulative_upload_total   = cumulative_upload
    cumulative_download_total = cumulative_divroute_download
    cumulative_bidirectional  = cumulative_upload_total + cumulative_download_total
    cumulative_baseline_bidirectional = cumulative_baseline_upload + cumulative_baseline_download
    bidir_saving = 100.0 * (1.0 - cumulative_bidirectional / cumulative_baseline_bidirectional)
    # Upload-only saving (compression saving on client→server direction)
    upload_saving = 100.0 * (1.0 - cumulative_upload_total / cumulative_baseline_upload)

    print(f"\n[done] log written to {config.log_path}")
    print(f"[summary] final acc     : {acc:.4f}")
    print(f"[summary] upload (C->S) : {cumulative_upload_total/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_upload/1e6:.2f} MB, "
          f"saving {upload_saving:.1f}%)")
    print(f"[summary] download (S->C): {cumulative_download_total/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_download/1e6:.2f} MB, saving 0.0%)")
    print(f"[summary] bidirectional : {cumulative_bidirectional/1e6:.2f} MB "
          f"(FedAvg: {cumulative_baseline_bidirectional/1e6:.2f} MB, "
          f"saving {bidir_saving:.1f}%)"
          f"  [upload-only compression]")

    if not config.skip_plot_prompt and input("\nGenerate plots? (y/n): ").strip().lower() == "y":
        from .visualize import generate_all_plots
        generate_all_plots(config.log_path)


if __name__ == "__main__":
    run()