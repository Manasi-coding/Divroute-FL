from .checkpointing import _restore_checkpoint, _save_checkpoint
from .reporting import (
    _print_training_config,
    _print_routing_table,
    _detailed_routing_diagnostics,
    _print_run_summary,
    _log_routing_dynamics,
    _post_aggregation_diagnostics,
    _print_client_details,
)
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    get_client_datasets,
    get_client_datasets_with_val,
    get_test_dataset,
    _num_classes,
)
from .model import get_model
from .client import FLClient
from .server import FLServer
from .logger import FLLogger
from .mechanism import update_ema, compute_decoupled_divergence
from . import mechanism as _mech
from .compression import get_adaptive_k_ratios
from . import diagnostics as _diag


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
        return min(config.local_epochs, max(2, config.local_epochs // 2))
    elif rnd < 15:
        return min(config.local_epochs, max(3, config.local_epochs - 1))
    return config.local_epochs


def _fedavg_bytes_per_round(num_clients: int, delta_numel: int) -> dict:
    """Baseline: full float32 delta for every selected client, both directions."""
    per_direction = num_clients * delta_numel * 4
    return {
        "download": per_direction,
        "upload": per_direction,  # FedAvg upload = full model, same size as download
        "bidir": per_direction * 2,
    }


def _configure_fedavg(config):
    if config.fedavg_baseline_mode:
        config.use_divergence_weighting = False
        config.use_server_momentum = False
        config.use_error_feedback = False
        config.use_adaptive_tau = False
        config.use_tier3_sync = False
        config.use_adaptive_k = False
        # k_ratio=1.0 → top-k keeps ALL params → no effective compression
        config.k_ratio_tier1 = 1.0
        config.k_ratio_tier2 = 1.0
        # tau below any real divergence score → every client is Tier 1 (no skipping)
        config.tau_low = -1.0
        config.tau_high = -1.0
        # gamma=1 → no selection-weight decay
        config.gamma = 1.0
        print("[init] *** FEDAVG BASELINE MODE — all DivRoute features disabled ***")


def _build_training_components(config, device):
    print(f"[init] loading {config.dataset_name.upper()} + building non-IID splits...")
    if config.local_val_fraction > 0.0:
        print(
            f"[init] local_val_fraction={config.local_val_fraction:.2f} — "
            f"holding out {config.local_val_fraction * 100:.0f}% of each shard "
            f"for local validation (NEW experimental configuration)"
        )
        client_datasets, val_datasets = get_client_datasets_with_val(
            config.dataset_name,
            config.num_clients,
            config.alpha,
            config.seed,
            val_fraction=config.local_val_fraction,
            model_name=config.model_name,
            partition_mode=getattr(config, "partition_mode", "dirichlet"),
            shards_per_client=getattr(config, "shards_per_client", 2),
        )
    else:
        # get_client_datasets() is called directly; nothing is reshuffled.
        client_datasets = get_client_datasets(
            config.dataset_name,
            config.num_clients,
            config.alpha,
            config.seed,
            model_name=config.model_name,
            partition_mode=getattr(config, "partition_mode", "dirichlet"),
            shards_per_client=getattr(config, "shards_per_client", 2),
        )
        val_datasets = [None] * config.num_clients
    test_loader = DataLoader(
        get_test_dataset(config.dataset_name, model_name=config.model_name),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    shard_sizes = [len(ds) for ds in client_datasets]  # training shard sizes only
    print(
        f"[init] shard sizes — min: {min(shard_sizes)}, max: {max(shard_sizes)}, "
        f"mean: {np.mean(shard_sizes):.0f}"
    )

    num_classes = _num_classes(config.dataset_name)
    global_model = get_model(
        config.model_name, num_classes, getattr(config, "bn_mode", "default")
    )
    server = FLServer(global_model, config, device)

    clients = [
        FLClient(
            i,
            client_datasets[i],
            config.local_epochs,
            config.local_lr,
            config.batch_size,
            device,
            config.model_name,
            num_classes,
            val_dataset=val_datasets[i],
            bn_mode=getattr(config, "bn_mode", "default"),
        )
        for i in range(config.num_clients)
    ]
    return test_loader, server, clients


def _configure_ablation(config):
    if getattr(config, "ablation_routing_no_compression", False):
        # Experiment B: full update for every tier — routing+weighting still active.
        config.k_ratio_tier1 = 1.0
        config.k_ratio_tier2 = 1.0
        print(
            "[init] *** ABLATION B — ROUTING ON / COMPRESSION OFF "
            "(k=1.0 for all tiers) ***"
        )

    elif getattr(config, "ablation_uniform_compression", False):
        # Experiment C: uniform k for every client; divergence weighting OFF.
        uk = getattr(config, "ablation_uniform_k_ratio", 0.05)
        config.k_ratio_tier1 = uk
        config.k_ratio_tier2 = uk
        config.use_divergence_weighting = False
        print(
            f"[init] *** ABLATION C — UNIFORM COMPRESSION k={uk:.3f} / ROUTING OFF ***"
        )

    elif getattr(config, "ablation_per_client_logging", False):
        print("[init] *** ABLATION D — FULL DIVROUTE + PER-CLIENT FORENSIC LOGGING ***")


def _train_selected_clients(config, server, clients, selected, local_epochs, rnd):
    results = []
    global_sd = server.global_model.state_dict()

    if getattr(config, "use_fp16_download", False):
        # simulate transmission by casting to fp16 then back to fp32
        global_sd = {k: v.half().float() for k, v in global_sd.items()}

    for cid in selected:
        result = clients[cid].train(
            global_sd,
            local_epochs,
            fedsparse_lambda=config.fedsparse_lambda,
            ntd_beta=config.ntd_beta,
            ntd_tau=config.ntd_tau,
            fedprox_mu=(
                config.fedprox_mu if getattr(config, "use_fedprox", False) else 0.0
            ),
            round_num=rnd,
            total_rounds=config.num_rounds,
            min_local_lr=getattr(config, "local_lr_min", 0.001),
        )
        results.append(result)
    return results, global_sd


def _update_hybrid_scores(
    config, results, rnd, ema_scores, raw_d_scores, _use_directional
):
    if _use_directional:
        _dir_vals = [
            r.get("_d_dir_raw_unnorm")
            for r in results
            if r.get("_d_dir_raw_unnorm") is not None
        ]
        _improv_vals = [
            r.get("_loss_improv_raw")
            for r in results
            if r.get("_loss_improv_raw") is not None
        ]

        # Normalise directional divergence to [0, 1] over cohort
        _dir_min = min(_dir_vals) if _dir_vals else 0.0
        _dir_max = max(_dir_vals) if _dir_vals else 1.0
        _dir_range = _dir_max - _dir_min

        # Normalise loss improvement to [0, 1] over cohort
        _has_any_improv = len(_improv_vals) >= 2
        if _has_any_improv:
            _imp_min = min(_improv_vals)
            _imp_max = max(_improv_vals)
            _imp_range = _imp_max - _imp_min
        else:
            _imp_min = _imp_max = _imp_range = 0.0

        _n_hybrid_fallback = 0
        valid_results = [r for r in results if r.get("_d_dir_raw_unnorm") is not None]
        div_scores = [r.get("_d_dir_raw_unnorm") for r in valid_results]

        loss_scores = []
        fallback = False
        for r in valid_results:
            _uimp = r.get("_loss_improv_raw")
            if _uimp is not None and _has_any_improv:
                loss_scores.append(_uimp)
            else:
                fallback = True

        from divroute_fl.mechanism import compute_dynamic_hybrid_scores

        if not fallback and len(div_scores) > 0 and len(loss_scores) == len(div_scores):
            hybrid_scores = compute_dynamic_hybrid_scores(
                div_scores, loss_scores, rnd, config.num_rounds
            )
            for r, _hybrid in zip(valid_results, hybrid_scores):
                r["routing_score_hybrid"] = _hybrid
                r["d_raw"] = _hybrid
                raw_d_scores.append(_hybrid)
        else:
            for r in valid_results:
                _udir = r.get("_d_dir_raw_unnorm")
                _dir_norm = (
                    (_udir - _dir_min) / _dir_range if _dir_range > 1e-12 else 0.5
                )
                _hybrid = _dir_norm
                _n_hybrid_fallback += 1
                r["routing_score_hybrid"] = _hybrid
                r["d_raw"] = _hybrid
                raw_d_scores.append(_hybrid)

        # raw_d_scores will have been double-appended for directional clients;
        # rebuild cleanly from final d_raw values.
        raw_d_scores = [
            r["d_raw"]
            for r in results
            if r.get("d_raw") is not None and not np.isnan(r["d_raw"])
        ]

        if _n_hybrid_fallback > 0 and rnd < 5:
            print(
                f"  [Phase-5] {_n_hybrid_fallback} clients used directional-only fallback "
                f"(no val loss available) — set local_val_fraction>0 to enable hybrid score"
            )

        # Recompute EMA from hybrid d_raw
        for r in results:
            _cid = r["client_id"]
            _hr = r.get("d_raw")
            if _hr is not None and not np.isnan(_hr):
                d_ema = update_ema(ema_scores, _cid, _hr, config.ema_beta)
                r["d_ema"] = d_ema
    return raw_d_scores


def _update_routing_thresholds(
    config,
    active_ema_scores,
    _rolling_ema_history,
    _last_tau_low,
    _last_tau_high,
    _tau_score_history,
):
    if not config.fedavg_baseline_mode:
        _t_mode = getattr(config, "threshold_mode", "rolling_percentile")

        if _t_mode in ("percentile", "rolling_percentile"):
            if _t_mode == "rolling_percentile":
                _rolling_ema_history.append(active_ema_scores)
                _history_flat = [
                    s for round_scores in _rolling_ema_history for s in round_scores
                ]
            else:
                _history_flat = active_ema_scores

            # Convergence Protection
            _spread = float(np.std(_history_flat)) if _history_flat else 0.0
            _conv_floor = getattr(config, "convergence_floor", 1e-4)

            if _spread < _conv_floor and _last_tau_low is not None:
                config.tau_low = _last_tau_low
                config.tau_high = _last_tau_high
                _convergence_frozen = True
            else:
                _p33, _p67 = _mech.compute_percentile_taus(_history_flat)
                config.tau_low = _p33
                config.tau_high = _p67
                _last_tau_low = config.tau_low
                _last_tau_high = config.tau_high
                _convergence_frozen = False
            _mu, _sigma = 0.0, 0.0
        else:
            # Rolling window logic for adaptive_tau
            _tau_win = getattr(config, "tau_window", 1)
            _tau_smooth = getattr(config, "tau_smoothing", 0.0)
            if active_ema_scores:
                _tau_score_history.append(list(active_ema_scores))
                if len(_tau_score_history) > _tau_win:
                    _tau_score_history = _tau_score_history[-_tau_win:]
            _rolled_scores = [
                s for rnd_scores in _tau_score_history for s in rnd_scores
            ]
            if _t_mode == "adaptive_tau" and len(_rolled_scores) >= 3:
                _cand_low, _cand_high, _mu, _sigma = _mech.compute_adaptive_taus(
                    _rolled_scores, config.tau_alpha, config.tau_beta
                )
                if _tau_smooth > 0.0:
                    _new_low = (
                        _tau_smooth * config.tau_low + (1.0 - _tau_smooth) * _cand_low
                    )
                    _new_high = (
                        _tau_smooth * config.tau_high + (1.0 - _tau_smooth) * _cand_high
                    )
                    config.tau_low = min(_new_low, _new_high)
                    config.tau_high = max(_new_low, _new_high)
                else:
                    config.tau_low = _cand_low
                    config.tau_high = _cand_high
            else:
                _mu, _sigma = 0.0, 0.0
            _convergence_frozen = False
    else:
        # fedavg_baseline_mode: tau_low/tau_high stay exactly as set by the
        # fedavg override block above (-1.0/-1.0) — no recomputation of any kind.
        _mu, _sigma = 0.0, 0.0
        _convergence_frozen = False
    return (
        _last_tau_low,
        _last_tau_high,
        _tau_score_history,
        _convergence_frozen,
        _mu,
        _sigma,
    )


def _score_clients(
    config,
    server,
    clients,
    global_sd,
    device,
    results,
    rnd,
    ema_scores,
    raw_d_scores,
    _grad_diags,
    _previous_global_delta,
):
    param_names = [n for n, _ in server.global_model.named_parameters()]
    global_flat = torch.cat([global_sd[k].flatten().float().cpu() for k in param_names])

    _use_directional = getattr(config, "use_directional_divergence", False)

    # Flatten the stored previous global delta (for directional mode)
    # Using CPU float32 to match client_flat.
    _prev_delta_flat: torch.Tensor | None = None
    if _use_directional and _previous_global_delta is not None:
        _prev_delta_flat = _previous_global_delta.float().cpu()
        _prev_delta_norm = _prev_delta_flat.norm(p=2).item()
    else:
        _prev_delta_norm = 0.0

    # Build layer_slices once per run if layerwise_cosine is requested (legacy)
    _div_layer_slices = None
    if (
        not _use_directional
        and getattr(config, "divergence_metric", "cosine") == "layerwise_cosine"
    ):
        offset = 0
        _div_layer_slices = []
        for k in param_names:
            numel = global_sd[k].numel()
            _div_layer_slices.append((k, offset, numel))
            offset += numel

    # [DIAG] per-client raw d captured BEFORE EMA update this round
    _diag_raw_per_client: dict = {}
    _diag_norm_per_client: dict = {}

    # Global model eval loss (needed for loss-improvement hybrid score).
    # Computed once per round, shared across all clients this round.
    # Only computed when directional_divergence AND val splits are active.
    _global_val_loss_per_client = _global_validation_losses(
        config, server, clients, results, device, _use_directional
    )

    for r in results:
        # Skip divergence update for clients with invalid updates.
        # Writing NaN into ema_scores would permanently poison all future rounds.
        if any(
            torch.isnan(v).any() or torch.isinf(v).any()
            for v in r["state_dict"].values()
        ):
            fallback_val = ema_scores.get(r["client_id"], config.tau_low)
            if rnd == 15:
                print(
                    f"  [DEBUG-NAN] client {r['client_id']:>3} triggered NaN/Inf fallback. Score set to {fallback_val:.8f}"
                )
            r["divergence_score"] = fallback_val
            continue

        local_model = clients[r["client_id"]].get_local_model()
        client_flat = torch.cat(
            [p.flatten().float().cpu() for n, p in local_model.named_parameters()]
        )
        client_delta = client_flat - global_flat  # w_local - w_global

        d_raw = _raw_client_score(
            config,
            server,
            r,
            rnd,
            client_flat,
            client_delta,
            global_flat,
            _use_directional,
            _global_val_loss_per_client,
            _div_layer_slices,
        )

        if getattr(config, "enable_gradient_diagnostics", False):
            _grad_diags[r["client_id"]] = _diag.compute_gradient_diagnostics(
                client_flat, global_flat, compressed_flat=None
            )

        if getattr(config, "enable_routing_diagnostics", False):
            _diag_raw_per_client[r["client_id"]] = d_raw
            _diag_norm_per_client[r["client_id"]] = float(client_delta.norm(p=2).item())

        r["d_raw"] = d_raw
        if not _use_directional:
            d_ema = update_ema(ema_scores, r["client_id"], d_raw, config.ema_beta)
            r["d_ema"] = d_ema
            raw_d_scores.append(d_raw)

        clients[r["client_id"]]._local_model = None
        del local_model
        del client_flat
        del client_delta

    raw_d_scores = _update_hybrid_scores(
        config, results, rnd, ema_scores, raw_d_scores, _use_directional
    )

    active_ema_scores = []
    for r in results:
        _cid = r["client_id"]
        _hr = r.get("d_raw")
        if _hr is None or np.isnan(_hr):
            # NaN fallback
            d_for_tier = ema_scores.get(_cid, config.tau_low)
        else:
            d_for_tier = r["d_ema"]

        # ENFORCE: Active routing score is strictly the EMA score
        r["divergence_score"] = d_for_tier
        active_ema_scores.append(d_for_tier)
    return (
        global_flat,
        raw_d_scores,
        active_ema_scores,
        _diag_raw_per_client,
        _diag_norm_per_client,
    )


def _assign_client_tiers(config, results, rnd, _previous_client_tiers, _pg_last_round):
    _invert_polarity = getattr(config, "invert_tier_polarity", False)
    _tier_flips = 0
    for r in results:
        _cid = r["client_id"]
        d_for_tier = r["divergence_score"]

        tier = _mech.assign_tier(
            d_for_tier, config.tau_low, config.tau_high, _invert_polarity
        )

        r["natural_tier"] = tier
        r["tier"] = tier

        if _cid in _previous_client_tiers and _previous_client_tiers[_cid] != tier:
            _tier_flips += 1
        _previous_client_tiers[_cid] = tier

    # Random-tier control: shuffle tier labels among this round's clients
    # (same tier counts and bytes), using a dedicated RNG so no other
    # random stream is touched.
    if getattr(config, "random_tier_assignment", False):
        _perm = np.random.default_rng(config.seed * 100003 + rnd).permutation(
            len(results)
        )
        _orig = [r["tier"] for r in results]
        for r, _i in zip(results, _perm):
            r["tier"] = r["natural_tier"] = _orig[_i]

    # DAGC-style per-client k (continuous, size-driven); see mechanism.dagc_k_ratios.
    if getattr(config, "dagc_mean_k", 0.0) > 0.0:
        _ks = _mech.dagc_k_ratios(
            [r["num_samples"] for r in results],
            config.dagc_mean_k,
            direction=getattr(config, "dagc_direction", 1),
        )
        if getattr(config, "dagc_shuffle", False):
            _ks = [
                _ks[_i]
                for _i in np.random.default_rng(
                    config.seed * 100003 + rnd + 7
                ).permutation(len(_ks))
            ]
        for r, _k in zip(results, _ks):
            r["k_override"] = _k

    p_count = sum(1 for r in results if r["tier"] in (1, 2))
    if p_count == 0 and results:
        ranked = sorted(
            results,
            key=lambda r: (
                r["divergence_score"],
                -_pg_last_round.get(r["client_id"], -1),
            ),
            reverse=True,
        )
        for winner in ranked[:2]:
            winner["tier"] = 2
            _pg_last_round[winner["client_id"]] = rnd
            print(
                f"  [Progress Guarantee] Promoted client {winner['client_id']} to Tier 2"
            )
    return _tier_flips


def _global_validation_losses(
    config, server, clients, results, device, _use_directional
):
    _global_val_loss_per_client: dict = {}  # {cid: float}
    _need_global_val_loss = (
        _use_directional
        and getattr(config, "local_val_fraction", 0.0) > 0.0
        and getattr(config, "loss_improvement_weight", 0.0) > 0.0
    )
    if _need_global_val_loss:
        # Evaluate the *current* global model on every selected client's val set
        _global_model_eval = server.global_model
        _global_model_eval.eval()
        _global_ce = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for r in results:
                _cid = r["client_id"]
                _val_loader = clients[_cid]._val_loader
                if _val_loader is None:
                    continue
                _gloss_sum, _gloss_n = 0.0, 0
                for _imgs, _lbls in _val_loader:
                    _imgs, _lbls = _imgs.to(device), _lbls.to(device)
                    _logits = _global_model_eval(_imgs)
                    _gloss_sum += _global_ce(_logits, _lbls).item() * _lbls.size(0)
                    _gloss_n += _lbls.size(0)
                if _gloss_n > 0:
                    _global_val_loss_per_client[_cid] = _gloss_sum / _gloss_n
    return _global_val_loss_per_client


def _raw_client_score(
    config,
    server,
    r,
    rnd,
    client_flat,
    client_delta,
    global_flat,
    _use_directional,
    _global_val_loss_per_client,
    _div_layer_slices,
):
    if _use_directional:
        # Use server momentum as the reference vector
        ref_vec = getattr(server, "_momentum_buf", None)
        if ref_vec is None or rnd == 0:
            d_dir = 0.5  # safe fallback for round 1 / zero momentum
            if rnd == 0:
                r["_round1_fallback"] = True
        else:
            d_dir = compute_decoupled_divergence(client_delta, ref_vec.cpu())
        d_raw = d_dir  # d_raw is the directional score
        r["d_dir"] = d_dir

        # Only computed when val data is available AND hybrid is requested.
        _cid = r["client_id"]
        _local_val_loss = r.get("local_val_loss")  # from client.train()
        _global_val_loss = _global_val_loss_per_client.get(_cid)
        _has_val = _local_val_loss is not None and _global_val_loss is not None
        r["_has_val_for_hybrid"] = _has_val
        if _has_val:
            # loss improvement: positive when local model is better
            _loss_improv_raw = _global_val_loss - _local_val_loss
            r["_loss_improv_raw"] = _loss_improv_raw
        else:
            r["_loss_improv_raw"] = None
        # Defer hybrid normalisation until all clients are scored
        # so we can normalise across the cohort.  Store raw dir score.
        r["_d_dir_raw_unnorm"] = d_dir
    else:
        _div_metric = getattr(config, "divergence_metric", "cosine")
        if _div_metric == "cosine":
            cos = F.cosine_similarity(
                client_flat.double().unsqueeze(0),
                global_flat.double().unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()
            d_raw = max(0.0, 1.0 - cos)
        elif _div_metric == "num_samples":
            # Routing score = local dataset size (control: is the divergence
            # score acting as a proxy for client data size?).
            d_raw = float(r["num_samples"])
        else:
            d_raw = _diag.compute_divergence_metric(
                client_flat,
                global_flat,
                metric=_div_metric,
                layer_slices=_div_layer_slices,
            )
    return d_raw


def run(config: Config | None = None) -> None:
    if config is None:
        config = Config()

    _configure_fedavg(config)

    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"[init] using device: {device}")

    test_loader, server, clients = _build_training_components(config, device)

    logger_meta = {
        "alpha": config.alpha,
        "partition_mode": getattr(config, "partition_mode", "dirichlet"),
        "shards_per_client": getattr(config, "shards_per_client", 2),
        "dataset": config.dataset_name,
        "model": config.model_name,
        "seed": config.seed,
        "local_epochs": config.local_epochs,
        "clients_per_round": config.clients_per_round,
        "num_clients": config.num_clients,
    }
    logger = FLLogger(config.log_path, metadata=logger_meta)
    all_ids = list(range(config.num_clients))

    ema_scores: dict = {}
    _pg_last_round: dict = {}
    error_buffers: dict = {}
    error_buffer_tiers: dict = {}  # {client_id: last tier its EF residual was accumulated under}

    _baseline_bpr: dict | None = None  # FedAvg bytes-per-round (set once after round 0)

    _previous_global_delta: torch.Tensor | None = None

    # Rolling window of routing scores per client for smoothed adaptive tau.
    # Key: round index (int), Value: list of routing scores that round.
    # Only the most recent `config.tau_window` entries are kept.
    _tau_score_history: list = []  # list of per-round score lists

    from collections import deque

    _rolling_ema_history = deque(maxlen=getattr(config, "rolling_window_size", 5))
    _last_tau_low = None
    _last_tau_high = None
    _previous_client_tiers = {}
    _convergence_frozen = False

    _diag_prev_tiers: dict = {}
    _diag_prev_ema_ranks: dict = {}
    _diag_prev_raw_ranks: dict = {}
    _diag_t1_seen: set = set()
    _diag_hist = {
        "raw_sigma": [],
        "ema_sigma": [],
        "raw_spearman": [],
        "ema_spearman": [],
        "pearson_div_norm": [],
        "pearson_div_loss": [],
        "tier_changes": [],
    }

    # These are reset each round and only populated when the corresponding
    # Config flag is True.  Zero cost when all flags are False.
    _grad_diags: dict = {}  # {client_id: grad_diag_dict}  (Part 1)
    _raw_deltas_buf: dict = {}  # {client_id: Tensor}           (Parts 4+5)
    _comp_deltas_buf: dict = {}  # {client_id: Tensor}           (Parts 4+5)

    _print_training_config(config)

    cumulative_divroute_download = 0
    cumulative_baseline_download = 0
    cumulative_upload = 0
    cumulative_baseline_upload = 0

    import os

    def _get_method_name(cfg: Config) -> str:
        # run_label (if set) fully overrides the auto-derived name so each
        # ablation experiment writes to its own isolated checkpoint directory.
        if getattr(cfg, "run_label", ""):
            return cfg.run_label
        if cfg.fedavg_baseline_mode:
            if getattr(cfg, "fedzip_actual_mode", False):
                return "fedzip"
            if getattr(cfg, "fedsparse_lambda", 0.0) > 0.0:
                return "fedsparse"
            return "fedavg"
        if getattr(cfg, "uniform_top5_mode", False):
            return "uniform"
        return "divroute"

    _configure_ablation(config)

    method_name = _get_method_name(config)
    save_checkpoint_dir = os.path.join(
        "checkpoints", f"{method_name}_{config.dataset_name}_seed{config.seed}"
    )
    latest_save_path = os.path.join(save_checkpoint_dir, "latest.pt")

    load_checkpoint_dir = os.path.join(
        getattr(config, "checkpoint_dir", "checkpoints"),
        f"{method_name}_{config.dataset_name}_seed{config.seed}",
    )
    latest_load_path = os.path.join(load_checkpoint_dir, "latest.pt")

    start_round = 0
    should_resume = False

    if config.fresh:
        should_resume = False
    elif config.resume:
        if not os.path.exists(latest_load_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {latest_load_path} but --resume was specified."
            )
        should_resume = True
    else:
        # Default: auto-resume if checkpoint exists
        if os.path.exists(latest_load_path):
            should_resume = True

    if should_resume:
        (
            start_round,
            ema_scores,
            _pg_last_round,
            error_buffers,
            error_buffer_tiers,
            _baseline_bpr,
            cumulative_divroute_download,
            cumulative_baseline_download,
            cumulative_upload,
            cumulative_baseline_upload,
            _previous_global_delta,
            _tau_score_history,
        ) = _restore_checkpoint(
            config, device, latest_load_path, method_name, server, clients, logger
        )

    acc: float | None = None  # set each round; guards summary when 0 rounds run

    for rnd in range(start_round, config.num_rounds):
        selected = server.select_clients(all_ids)
        local_epochs = _get_local_epochs(rnd, config)

        results, global_sd = _train_selected_clients(
            config, server, clients, selected, local_epochs, rnd
        )

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
                r["tier"] = 2  # Tier-2 path → k_ratio_tier2=0.05
        else:
            (
                global_flat,
                raw_d_scores,
                active_ema_scores,
                _diag_raw_per_client,
                _diag_norm_per_client,
            ) = _score_clients(
                config,
                server,
                clients,
                global_sd,
                device,
                results,
                rnd,
                ema_scores,
                raw_d_scores,
                _grad_diags,
                _previous_global_delta,
            )

            (
                _last_tau_low,
                _last_tau_high,
                _tau_score_history,
                _convergence_frozen,
                _mu,
                _sigma,
            ) = _update_routing_thresholds(
                config,
                active_ema_scores,
                _rolling_ema_history,
                _last_tau_low,
                _last_tau_high,
                _tau_score_history,
            )

            if getattr(config, "enable_tau_diagnostics", False) and active_ema_scores:
                _tau_diag_payload = _diag.compute_tau_diagnostics(
                    active_ema_scores,
                    _mu,
                    _sigma,
                    config.tau_low,
                    config.tau_high,
                    results,
                )
                if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0:
                    _diag.print_tau_diagnostics(rnd + 1, _tau_diag_payload)

            _tier_flips = _assign_client_tiers(
                config, results, rnd, _previous_client_tiers, _pg_last_round
            )

            _print_routing_table(
                config, results, server, rnd, _tier_flips, _convergence_frozen
            )

            _diag_prev_raw_ranks, _diag_prev_ema_ranks, _diag_prev_tiers = (
                _log_routing_dynamics(
                    config,
                    results,
                    rnd,
                    raw_d_scores,
                    _diag_raw_per_client,
                    _diag_norm_per_client,
                    _diag_hist,
                    _diag_prev_tiers,
                    _diag_prev_raw_ranks,
                    _diag_prev_ema_ranks,
                    _diag_t1_seen,
                    _mu,
                    _sigma,
                )
            )

            if rnd == 15:
                print(
                    f"  [DEBUG] round 16 tier assignment — tau_low={config.tau_low:.8f} tau_high={config.tau_high:.8f}"
                )
                for r in results:
                    print(
                        f"  [DEBUG] client {r['client_id']:>3} "
                        f"d={r['divergence_score']:.8f} "
                        f"tier={r['tier']} "
                        f"tau_low={config.tau_low:.8f} "
                        f"tau_high={config.tau_high:.8f}"
                    )

            # Explicitly release the global parameters tensor before server aggregation
            # to keep peak GPU memory down.
            del global_flat

        #    the correct ratios for this round) --------------------------------
        k1, k2 = get_adaptive_k_ratios(config, rnd)
        config.active_k_ratio_tier1, config.active_k_ratio_tier2 = k1, k2

        # use_adaptive_k (if somehow True) cannot silently restore compression.
        if getattr(config, "ablation_routing_no_compression", False):
            config.active_k_ratio_tier1 = 1.0
            config.active_k_ratio_tier2 = 1.0

        # Tier-1 path so apply_tiered_compression uses active_k_ratio_tier1.
        # Divergence scores are still computed above for diagnostic logging.
        elif getattr(config, "ablation_uniform_compression", False):
            uk = getattr(config, "ablation_uniform_k_ratio", 0.05)
            config.active_k_ratio_tier1 = uk
            config.active_k_ratio_tier2 = uk
            for r in results:
                r["tier"] = 1  # route all clients through Tier-1 path

        #    Tier-3 clients send 0 upload bytes and receive no server-side payload
        #    here (heartbeat is counted separately below if tier3_sync is active).
        for r in results:
            r["bytes_received"] = 0
            r["upload_bytes"] = 0  # compressed upload — set by server.aggregate()
            r["download_bytes"] = 0  # full-model download — set by server.aggregate()

        server.aggregate(
            results,
            error_buffers,
            round_num=rnd + 1,
            error_buffer_tiers=error_buffer_tiers,
        )

        if (
            getattr(config, "use_directional_divergence", False)
            and server.global_delta is not None
        ):
            _previous_global_delta = server.global_delta.detach().cpu().clone()

        (
            _routing_corr_payload,
            _tier_contrib_payload,
            _compress_analysis_payload,
            _agg_norm_payload,
            _global_norm_payload,
        ) = _post_aggregation_diagnostics(
            config, server, results, rnd, _grad_diags, _raw_deltas_buf, _comp_deltas_buf
        )

        if getattr(config, "bn_ablation_mode", False):
            # ResNet-18 BN buffers take exactly 38,480 bytes in FP32
            for r in results:
                r["upload_bytes"] += 38480

        tier3_sync_count = 0
        if config.use_tier3_sync and (rnd + 1) % config.tier3_sync_interval == 0:
            for r in results:
                if r["tier"] == 3:
                    heartbeat = server.tier3_heartbeat_payload()
                    r["bytes_received"] += heartbeat["bytes_transmitted"]
                    tier3_sync_count += 1

        delta_numel = server.global_delta.numel()
        if _baseline_bpr is None:
            _baseline_bpr = _fedavg_bytes_per_round(
                config.clients_per_round, delta_numel
            )

        t1 = sum(1 for r in results if r["tier"] == 1)
        t2 = sum(1 for r in results if r["tier"] == 2)
        t3 = sum(1 for r in results if r["tier"] == 3)

        _detailed_diag = _detailed_routing_diagnostics(results, _grad_diags, t1, t2, t3)

        acc = server.evaluate(test_loader)

        # Communication tracking for logging
        bytes_per_param = 2 if getattr(config, "use_fp16_download", False) else 4
        full_model_bytes = delta_numel * bytes_per_param
        total_upload = sum(r["upload_bytes"] for r in results)
        total_download = config.clients_per_round * full_model_bytes
        total_bidir = total_upload + total_download

        # Compute temporary cumulative values for accurate logging of current state
        _temp_cumul_upload = cumulative_upload + total_upload
        _temp_cumul_download = cumulative_divroute_download + total_download
        _temp_cumul_total = _temp_cumul_upload + _temp_cumul_download

        baseline_bidir = _baseline_bpr["bidir"]
        saving_pct = (
            100.0 * (1.0 - total_bidir / baseline_bidir) if baseline_bidir > 0 else 0.0
        )

        logger.log(
            rnd,
            acc,
            results,
            delta_numel=delta_numel,
            grad_diagnostics=_grad_diags if _grad_diags else None,
            agg_update_norm=_agg_norm_payload,
            global_update_norm=_global_norm_payload,
            routing_correlations=_routing_corr_payload,
            tier_contributions=_tier_contrib_payload,
            compression_analysis=_compress_analysis_payload,
            tau_diagnostics=_tau_diag_payload
            if not getattr(config, "uniform_top5_mode", False)
            else None,
            detailed_routing_diagnostics=_detailed_diag,
            upload_mb=total_upload / 1e6,
            download_mb=total_download / 1e6,
            total_mb=total_bidir / 1e6,
            cumulative_total_mb=_temp_cumul_total / 1e6,
            communication_savings=saving_pct,
        )
        server.update_selection_weights(results)

        # Restore cumulative trackers
        cumulative_upload += total_upload
        cumulative_baseline_upload += _baseline_bpr["upload"]
        cumulative_divroute_download += total_download
        cumulative_baseline_download += _baseline_bpr["download"]

        cumulative_saving_pct = 100.0 * (
            1.0
            - (cumulative_upload + cumulative_divroute_download)
            / (cumulative_baseline_upload + cumulative_baseline_download)
        )

        d_vals = [r["divergence_score"] for r in results]
        sync_note = f" | t3-sync: {tier3_sync_count}" if tier3_sync_count > 0 else ""
        current_lr = results[0]["local_lr"] if results else config.local_lr
        print(
            f"  round {rnd + 1:>3}/{config.num_rounds} | acc: {acc:.4f} | "
            f"tiers: {t1}/{t2}/{t3} | "
            f"up: {total_upload / 1e6:.3f}MB "
            f"down: {total_download / 1e6:.3f}MB "
            f"(bidir save {saving_pct:.1f}%, cum {cumulative_saving_pct:.1f}%) | "
            f"tau: [{config.tau_low:.8f}, {config.tau_high:.8f}] | "
            f"d: [{min(d_vals):.8f}, {max(d_vals):.8f}] | "
            f"epochs: {local_epochs} | "
            f"local_lr={current_lr:.6f}"
            f"{sync_note}"
        )
        for r in sorted(results, key=lambda x: x["client_id"]):
            _train_acc = r.get("local_train_acc", 0.0)
            _val_acc = r.get("local_val_acc")
            _val_str = f"{_val_acc:.4f}" if _val_acc is not None else "  N/A  "
            print(
                f"    Client {r['client_id']:>3} | Local Train Acc: {_train_acc:.4f} | "
                f"Local Val Acc: {_val_str} | "
                f"Agg Weight: {r.get('aggregation_weight', 0.0):.4f} | Tier: {r.get('tier')}"
            )

        # Printed every round when ablation_per_client_logging=True.
        # No algorithmic change — reads values already computed above.
        _print_client_details(config, results, server, rnd, _grad_diags)

        completed_round = rnd + 1
        if completed_round % 5 == 0:
            _save_checkpoint(
                config,
                server,
                clients,
                logger,
                rnd,
                completed_round,
                method_name,
                save_checkpoint_dir,
                latest_save_path,
                {
                    "ema_scores": ema_scores.copy(),
                    "_pg_last_round": _pg_last_round.copy(),
                    "error_buffers": {k: v.cpu() for k, v in error_buffers.items()},
                    "error_buffer_tiers": error_buffer_tiers.copy(),
                    "_baseline_bpr": _baseline_bpr.copy()
                    if _baseline_bpr is not None
                    else None,
                    "cumulative_divroute_download": cumulative_divroute_download,
                    "cumulative_baseline_download": cumulative_baseline_download,
                    "cumulative_upload": cumulative_upload,
                    "cumulative_baseline_upload": cumulative_baseline_upload,
                    "_previous_global_delta": _previous_global_delta.cpu()
                    if _previous_global_delta is not None
                    else None,
                    "_tau_score_history": _tau_score_history,
                },
            )

    _print_run_summary(
        config,
        _diag_hist,
        _diag_t1_seen,
        cumulative_upload,
        cumulative_divroute_download,
        cumulative_baseline_upload,
        cumulative_baseline_download,
        acc,
    )

    if (
        not config.skip_plot_prompt
        and input("\nGenerate plots? (y/n): ").strip().lower() == "y"
    ):
        from .visualize import generate_all_plots

        generate_all_plots(config.log_path)


if __name__ == "__main__":
    run()
