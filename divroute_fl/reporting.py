import numpy as np
import torch

from . import diagnostics as _diag
from .diagnostics import _pearson, _spearman


def _print_training_config(config):
    print(
        f"[train] {config.num_rounds} rounds, "
        f"{config.clients_per_round}/{config.num_clients} clients/round"
    )
    _partition_mode = getattr(config, "partition_mode", "dirichlet")
    if _partition_mode == "pathological":
        _partition_detail = (
            str(getattr(config, "shards_per_client", 2)) + " shards/client"
        )
    elif _partition_mode == "natural":
        _partition_detail = "one client per real writer"
    else:
        _partition_detail = "alpha=" + str(config.alpha)
    print(f"[train] partition: {_partition_mode} ({_partition_detail})")
    print(
        f"[train] div-weighted: {config.use_divergence_weighting} "
        f"({config.divergence_weight_mode}) | "
        f"momentum: {config.use_server_momentum} | "
        f"error-feedback: {config.use_error_feedback} | "
        f"adaptive-tau: {config.use_adaptive_tau} "
        f"(alpha={config.tau_alpha}/beta={config.tau_beta}) | "
        f"tier3-sync: {config.use_tier3_sync} "
        f"(every {config.tier3_sync_interval} rounds) | "
        f"epoch-warmup: {config.use_epoch_warmup} | "
        f"fedavg-mode: {config.fedavg_baseline_mode} | "
        f"fedprox: {getattr(config, 'use_fedprox', False)} "
        f"(mu={getattr(config, 'fedprox_mu', 0.0)})"
    )
    print(
        f"[train] adaptive-k: {config.use_adaptive_k} | "
        f"k_ratio: tier1={config.k_ratio_tier1:.2f} tier2={config.k_ratio_tier2:.2f} | "
        f"include-tier3: {config.include_tier3_in_aggregation} | "
        f"uniform-top5: {config.uniform_top5_mode}"
    )

    _use_dir_div = getattr(config, "use_directional_divergence", False)
    _use_div_ema = getattr(config, "use_divergence_ema", False)
    _tau_win = getattr(config, "tau_window", 1)
    _tau_smooth = getattr(config, "tau_smoothing", 0.0)
    _ema_beta_val = getattr(config, "ema_beta", 0.6)
    print(
        f"[train] directional-div: {_use_dir_div} | "
        f"div-ema-routing: {_use_div_ema} (beta={_ema_beta_val}) | "
        f"tau-window: {_tau_win} | tau-smoothing: {_tau_smooth:.2f}"
    )

    print("\n[Transmission Precision]")
    print(
        f"  Upload   : {'FP16' if getattr(config, 'use_fp16_upload', False) else 'FP32'}"
    )
    print(
        f"  Download : {'FP16' if getattr(config, 'use_fp16_download', False) else 'FP32'}\n"
    )


def _print_routing_table(
    config, results, server, rnd, _tier_flips, _convergence_frozen
):
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

            if _dr is None:  # NaN-guarded client
                _tr = _te = r["tier"]
            else:
                _tr = server.assign_tier(_dr)  # tier using raw score
                _te = server.assign_tier(_de)  # tier using ema score
                _valid_d_raws.append(_dr)
                _valid_d_emas.append(_de)

            _raw_counts[_tr] += 1
            _ema_counts[_te] += 1

            if _tr != _te:
                _rout_diff += 1
            _rout_lines.append((r["client_id"], _dr, _de, _tr, _te, r["tier"]))

        print(
            f"\n  [ROUTING] round={rnd + 1} mode={config.routing_score!r} "
            f"tau=[{config.tau_low:.6f},{config.tau_high:.6f}]"
        )
        print(
            f"  {'CID':>4}  {'d_raw':>10}  {'d_ema':>10}  "
            f"{'t(raw)':>7}  {'t(ema)':>7}  {'chosen':>7}"
        )
        for _cid, _dr2, _de2, _tr2, _te2, _tc2 in _rout_lines:
            _dr_str = f"{_dr2:.6f}" if _dr2 is not None else "     NaN"
            _mk = " *" if _tr2 != _te2 else ""
            print(
                f"  {_cid:>4}  {_dr_str:>10}  {_de2:>10.6f}  "
                f"{_tr2:>7}  {_te2:>7}  {_tc2:>7}{_mk}"
            )

        print("\n  [ROUTING SUMMARY]")
        print(
            f"  Routing disagreement: Different assignments: {_rout_diff} / {len(results)}"
        )
        print(
            f"  Tier counts (raw)   : {_raw_counts[1]}/{_raw_counts[2]}/{_raw_counts[3]}"
        )
        print(
            f"  Tier counts (ema)   : {_ema_counts[1]}/{_ema_counts[2]}/{_ema_counts[3]}"
        )
        print(
            f"  Actual tier counts  : {_act_counts[1]}/{_act_counts[2]}/{_act_counts[3]}"
        )
        print(f"  Tier flips          : {_tier_flips}")
        print(f"  Thresholds frozen   : {_convergence_frozen}")
        _tratio = config.tau_high / max(config.tau_low, 1e-8)
        print(f"  Threshold ratio     : {_tratio:.4f}")

        if _valid_d_raws:
            _avg_raw = sum(_valid_d_raws) / len(_valid_d_raws)
            _avg_ema = sum(_valid_d_emas) / len(_valid_d_emas)
            _avg_diff = sum(e - r for e, r in zip(_valid_d_emas, _valid_d_raws)) / len(
                _valid_d_raws
            )
            _max_abs = max(abs(e - r) for e, r in zip(_valid_d_emas, _valid_d_raws))
            print(f"  Average d_raw                   : {_avg_raw:.8f}")
            print(f"  Average d_ema                   : {_avg_ema:.8f}")
            print(f"  Average (d_ema - d_raw)         : {_avg_diff:.8f}")
            print(f"  Maximum absolute difference     : {_max_abs:.8f}")
            print(f"  Clients whose routing changes   : {_rout_diff}")
        print()


def _detailed_routing_diagnostics(results, _grad_diags, t1, t2, t3):
    _detailed_diag = {}
    _dir_divs = [r.get("d_dir") for r in results if r.get("d_dir") is not None]
    _loss_imps = [
        r.get("_loss_improv_raw")
        for r in results
        if r.get("_loss_improv_raw") is not None
    ]
    _scores = [
        r.get("routing_score_hybrid", r.get("d_raw"))
        for r in results
        if r.get("routing_score_hybrid", r.get("d_raw")) is not None
    ]

    def _stats(arr):
        if not arr:
            return None
        a = np.array(arr)
        return {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    _detailed_diag["dir_div"] = _stats(_dir_divs)
    _detailed_diag["loss_improvement"] = _stats(_loss_imps)
    _detailed_diag["routing_score"] = _stats(_scores)
    _detailed_diag["tier_counts"] = {"1": t1, "2": t2, "3": t3}

    # correlations
    _update_norms = []
    for r in results:
        _cid = r.get("client_id")
        if _cid is not None and _grad_diags and _cid in _grad_diags:
            _update_norms.append(float(_grad_diags[_cid].get("update_l2_norm", np.nan)))
        else:
            _update_norms.append(None)

    import scipy.stats as stats

    def _corr(xs, ys):
        valid = [
            (x, y)
            for x, y in zip(xs, ys)
            if x is not None and y is not None and not np.isnan(x) and not np.isnan(y)
        ]
        if len(valid) < 2:
            return {"pearson": None, "spearman": None}
        vx, vy = zip(*valid)
        if np.std(vx) == 0.0 or np.std(vy) == 0.0:
            return {"pearson": None, "spearman": None}
        try:
            p, _ = stats.pearsonr(vx, vy)
            s, _ = stats.spearmanr(vx, vy)
            return {
                "pearson": float(p) if not np.isnan(p) else None,
                "spearman": float(s) if not np.isnan(s) else None,
            }
        except Exception:
            return {"pearson": None, "spearman": None}

    _detailed_diag["corr_dir_div_norm"] = _corr(_dir_divs, _update_norms)
    _detailed_diag["corr_score_norm"] = _corr(_scores, _update_norms)
    return _detailed_diag


def _print_run_summary(
    config,
    _diag_hist,
    _diag_t1_seen,
    cumulative_upload,
    cumulative_divroute_download,
    cumulative_baseline_upload,
    cumulative_baseline_download,
    acc,
):
    if getattr(config, "enable_routing_diagnostics", False):
        print("\n=== ROUTING DIAGNOSTICS SUMMARY ===")
        print(
            f"Average raw sigma: {np.mean(_diag_hist['raw_sigma']):.8f}"
            if _diag_hist["raw_sigma"]
            else "Average raw sigma: N/A"
        )
        print(
            f"Average EMA sigma: {np.mean(_diag_hist['ema_sigma']):.8f}"
            if _diag_hist["ema_sigma"]
            else "Average EMA sigma: N/A"
        )
        if _diag_hist["raw_sigma"] and _diag_hist["ema_sigma"]:
            print(
                f"Average raw/EMA ratio: {np.mean(np.array(_diag_hist['raw_sigma']) / np.array(_diag_hist['ema_sigma'])):.4f}"
            )
        # Both 'raw_spearman' and 'ema_spearman' slots now store the within-round raw<->EMA Spearman
        print(
            f"Average raw<->EMA Spearman (within-round): {np.mean(_diag_hist['raw_spearman']):.4f}"
            if _diag_hist["raw_spearman"]
            else "Average raw<->EMA Spearman: N/A"
        )
        print(
            f"Average div<->norm Pearson: {np.mean(_diag_hist['pearson_div_norm']):.4f}"
            if _diag_hist["pearson_div_norm"]
            else "Average div<->norm Pearson: N/A"
        )
        print(
            f"Average div<->loss Pearson: {np.mean(_diag_hist['pearson_div_loss']):.4f}"
            if _diag_hist["pearson_div_loss"]
            else "Average div<->loss Pearson: N/A"
        )
        print(
            f"Average tier changes/round: {np.mean(_diag_hist['tier_changes']):.2f}"
            if _diag_hist["tier_changes"]
            else "Average tier changes/round: N/A"
        )
        print(f"Unique T1 clients observed: {len(_diag_t1_seen)}")
        print("===================================")

    cumulative_upload_total = cumulative_upload
    cumulative_download_total = cumulative_divroute_download
    cumulative_bidirectional = cumulative_upload_total + cumulative_download_total
    cumulative_baseline_bidirectional = (
        cumulative_baseline_upload + cumulative_baseline_download
    )
    if cumulative_baseline_upload > 0:
        upload_saving = 100.0 * (
            1.0 - cumulative_upload_total / cumulative_baseline_upload
        )
    else:
        upload_saving = 0.0
    if cumulative_baseline_bidirectional > 0:
        bidir_saving = 100.0 * (
            1.0 - cumulative_bidirectional / cumulative_baseline_bidirectional
        )
    else:
        bidir_saving = 0.0

    print(f"\n[done] log written to {config.log_path}")
    if acc is None:
        print("[summary] final acc     : N/A (no rounds ran — already at target)")
    else:
        print(f"[summary] final acc     : {acc:.4f}")
    print(
        f"[summary] upload (C->S) : {cumulative_upload_total / 1e6:.2f} MB "
        f"(FedAvg: {cumulative_baseline_upload / 1e6:.2f} MB, "
        f"saving {upload_saving:.1f}%)"
    )
    print(
        f"[summary] download (S->C): {cumulative_download_total / 1e6:.2f} MB "
        f"(FedAvg: {cumulative_baseline_download / 1e6:.2f} MB, saving 0.0%)"
    )
    print(
        f"[summary] bidirectional : {cumulative_bidirectional / 1e6:.2f} MB "
        f"(FedAvg: {cumulative_baseline_bidirectional / 1e6:.2f} MB, "
        f"saving {bidir_saving:.1f}%)"
        f"  [upload-only compression]"
    )


def _log_routing_dynamics(
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
):
    if getattr(config, "enable_routing_diagnostics", False) and len(raw_d_scores) >= 2:
        # and have a valid raw score.  Iterating all `results` previously raised KeyError
        # for NaN-guarded clients absent from _diag_raw_per_client.
        _valid_for_diag = [r for r in results if r["client_id"] in _diag_raw_per_client]
        _raw_arr = np.array(
            [_diag_raw_per_client[r["client_id"]] for r in _valid_for_diag],
            dtype=np.float64,
        )

        # making it byte-identical to _raw_arr and causing EMA stats to appear the same
        # as raw stats.  Now correctly uses r["divergence_score"] (d_ema).
        _ema_arr = np.array(
            [r["divergence_score"] for r in _valid_for_diag], dtype=np.float64
        )

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
        print(
            f"  raw : mu={_r_mu:.8f} med={_r_med:.8f} std={_r_sigma:.8f} min={_r_min:.8f} max={_r_max:.8f} spread={_r_spread:.8f}"
        )
        print(
            f"  EMA : mu={_e_mu:.8f} med={_e_med:.8f} std={_e_sigma:.8f} min={_e_min:.8f} max={_e_max:.8f} spread={_e_spread:.8f}"
        )

        # Rank 0 = highest divergence within the selected cohort this round.
        _cur_raw_ranks = {
            r["client_id"]: rank
            for rank, r in enumerate(
                sorted(
                    _valid_for_diag,
                    key=lambda x: _diag_raw_per_client[x["client_id"]],
                    reverse=True,
                )
            )
        }
        _cur_ema_ranks = {
            r["client_id"]: rank
            for rank, r in enumerate(
                sorted(
                    _valid_for_diag, key=lambda x: x["divergence_score"], reverse=True
                )
            )
        }
        _cur_tiers = {r["client_id"]: r["tier"] for r in results}

        _aligned_cids = sorted(_cur_raw_ranks.keys())
        if len(_aligned_cids) >= 2:
            _rve_spearman = _spearman(
                [_cur_raw_ranks[c] for c in _aligned_cids],
                [_cur_ema_ranks[c] for c in _aligned_cids],
            )
            # rank_changes: clients whose raw rank ≠ EMA rank within this round
            _rank_changes = sum(
                1 for c in _aligned_cids if _cur_raw_ranks[c] != _cur_ema_ranks[c]
            )
        else:
            _rve_spearman = float("nan")
            _rank_changes = 0

        # Cross-round tier stability (separate metric from within-round Spearman)
        _common_cid = [c for c in _cur_tiers if c in _diag_prev_tiers]
        _tier_changes = sum(
            1 for c in _common_cid if _cur_tiers[c] != _diag_prev_tiers[c]
        )

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
            _t1_ret = _t2_ret = _t3_ret = float("nan")

        # Store within-round raw↔EMA Spearman in both hist slots
        if not np.isnan(_rve_spearman):
            _diag_hist["raw_spearman"].append(_rve_spearman)
        if not np.isnan(_rve_spearman):
            _diag_hist["ema_spearman"].append(_rve_spearman)
        _diag_hist["tier_changes"].append(_tier_changes)

        print("  [DIAG] 2. Ranking stability (within-round raw<->EMA)")
        print(
            f"  raw<->ema spearman={_rve_spearman:.4f}  rank_changes(raw!=ema)={_rank_changes}  tier_changes_vs_prev={_tier_changes}"
        )
        print(f"  retention: T1={_t1_ret:.2f} T2={_t2_ret:.2f} T3={_t3_ret:.2f}")

        print("  [DIAG] 3. Optimisation correlation")

        # _raw_arr / _ema_arr) to prevent KeyError for NaN-guarded clients and misalignment.
        _norms = np.array(
            [_diag_norm_per_client[r["client_id"]] for r in _valid_for_diag]
        )
        _losses = np.array([r.get("local_loss", 0.0) for r in _valid_for_diag])
        _p_div_norm = _pearson(_raw_arr, _norms)
        _s_div_norm = _spearman(_raw_arr, _norms)
        _p_div_loss = _pearson(_raw_arr, _losses)
        _s_div_loss = _spearman(_raw_arr, _losses)
        if not np.isnan(_p_div_norm):
            _diag_hist["pearson_div_norm"].append(_p_div_norm)
        if not np.isnan(_p_div_loss):
            _diag_hist["pearson_div_loss"].append(_p_div_loss)
        for r in _valid_for_diag:
            cid = r["client_id"]
            print(
                f"  c{cid:>3}: raw={_diag_raw_per_client[cid]:.6f} ema={r['divergence_score']:.6f} norm={_diag_norm_per_client[cid]:.4f} loss={r.get('local_loss', 0.0):.4f}"
            )

        print(
            f"  correlation div<->norm: pearson={_p_div_norm:+.4f} spearman={_s_div_norm:+.4f}"
        )
        print(
            f"  correlation div<->loss: pearson={_p_div_loss:+.4f} spearman={_s_div_loss:+.4f}"
        )

        print("  [DIAG] 4. Threshold diagnostics")
        _tau_diff = config.tau_high - config.tau_low
        _ratio = _r_spread / _tau_diff if _tau_diff > 0 else 0.0
        print(
            f"  mu={_mu:.6f} sigma={_sigma:.6f} tau_low={config.tau_low:.6f} tau_high={config.tau_high:.6f}"
        )
        print(f"  high-low={_tau_diff:.6f}  spread/diff_ratio={_ratio:.4f}")

        print("  [DIAG] 5. Routing diversity")
        _t1_ids = [r["client_id"] for r in results if r["tier"] == 1]
        _t2_ids = [r["client_id"] for r in results if r["tier"] == 2]
        _t3_ids = [r["client_id"] for r in results if r["tier"] == 3]

        _t1_prev_set = {c for c, t in _diag_prev_tiers.items() if t == 1}
        _t1_curr_set = set(_t1_ids)
        _t1_entered = _t1_curr_set - _t1_prev_set  # in T1 now but NOT in T1 last round
        _t1_left = _t1_prev_set - _t1_curr_set  # was in T1 last round but NOT in T1 now
        for c in _t1_ids:
            _diag_t1_seen.add(c)
        print(f"  T1 ({len(_t1_ids)}): {sorted(_t1_ids)}")
        print(f"  T2 ({len(_t2_ids)}): {sorted(_t2_ids)}")
        print(f"  T3 ({len(_t3_ids)}): {sorted(_t3_ids)}")
        print(
            f"  T1 entering={len(_t1_entered)} leaving={len(_t1_left)} unique_seen={len(_diag_t1_seen)}"
        )

        _t2_set = set(_t2_ids)
        _t3_set = set(_t3_ids)
        _all_selected_ids = {r["client_id"] for r in results}
        if _t1_curr_set & _t2_set:
            print(
                f"  [DIAG WARNING] T1 intersect T2 not empty: {_t1_curr_set & _t2_set}"
            )
        if _t1_curr_set & _t3_set:
            print(
                f"  [DIAG WARNING] T1 intersect T3 not empty: {_t1_curr_set & _t3_set}"
            )
        if _t2_set & _t3_set:
            print(f"  [DIAG WARNING] T2 intersect T3 not empty: {_t2_set & _t3_set}")
        if _t1_curr_set | _t2_set | _t3_set != _all_selected_ids:
            _missing = _all_selected_ids - (_t1_curr_set | _t2_set | _t3_set)
            print(
                f"  [DIAG WARNING] T1 union T2 union T3 != selected clients. Missing: {_missing}"
            )
        if _t1_entered & _t1_left:
            print(
                f"  [DIAG WARNING] T1 entering intersect leaving not empty: {_t1_entered & _t1_left}"
            )
        if len(_raw_arr) != len(_ema_arr):
            print(
                f"  [DIAG WARNING] raw_arr length ({len(_raw_arr)}) != ema_arr length ({len(_ema_arr)})"
            )

        _diag_prev_raw_ranks = _cur_raw_ranks
        _diag_prev_ema_ranks = _cur_ema_ranks
        _diag_prev_tiers = _cur_tiers
    return _diag_prev_raw_ranks, _diag_prev_ema_ranks, _diag_prev_tiers


def _post_aggregation_diagnostics(
    config, server, results, rnd, _grad_diags, _raw_deltas_buf, _comp_deltas_buf
):
    _do_delta_capture = getattr(
        config, "enable_gradient_diagnostics", False
    ) or getattr(config, "enable_compression_analysis", False)
    if _do_delta_capture:
        _global_sd_after = server.global_model.state_dict()
        _gf_after = torch.cat(
            [
                _global_sd_after[k].flatten().float().cpu()
                for k in [n for n, _ in server.global_model.named_parameters()]
            ]
        )
        for r in results:
            cid = r.get("client_id")
            if cid is None:
                continue
            # Re-flatten client state to get raw delta (client is still in memory)
            try:
                _cf = torch.cat(
                    [
                        v.flatten().float().cpu()
                        for v in r["state_dict"].values()
                        if v.dtype.is_floating_point and v.dim() >= 1
                    ]
                )
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
                if (
                    server.global_delta is not None
                    and r.get("aggregation_weight", 0) > 0
                ):
                    # Scale global delta inversely by weight to approximate single-client
                    _comp = server.global_delta.cpu() / max(
                        r["aggregation_weight"], 1e-9
                    )
                    _comp_deltas_buf[cid] = _comp[:_numel]
                else:
                    _comp_deltas_buf[cid] = torch.zeros_like(_raw)

            # Update Part 1 grad_diag with compression fields
            if (
                getattr(config, "enable_gradient_diagnostics", False)
                and cid in _grad_diags
            ):
                _compressed_flat = _comp_deltas_buf.get(cid)
                if _compressed_flat is not None and cid in _raw_deltas_buf:
                    _gd_raw = _raw_deltas_buf[cid]
                    _orig_norm = float(_gd_raw.norm(p=2).item())
                    _comp_norm = float(_compressed_flat.norm(p=2).item())
                    _err = float((_gd_raw - _compressed_flat).norm(p=2).item())
                    _grad_diags[cid]["compression_error"] = _err
                    _grad_diags[cid]["retention_ratio"] = _comp_norm / max(
                        _orig_norm, 1e-12
                    )
                    import torch.nn.functional as _F

                    if _orig_norm > 1e-12 and _comp_norm > 1e-12:
                        _cs = _F.cosine_similarity(
                            _gd_raw.unsqueeze(0),
                            _compressed_flat.unsqueeze(0),
                            dim=1,
                            eps=1e-8,
                        ).item()
                        _grad_diags[cid]["compressed_cosine"] = float(_cs)

    _routing_corr_payload = None
    if getattr(config, "enable_routing_quality_analysis", False):
        _diag.record_routing_quality(rnd + 1, results, _grad_diags)
        _routing_corr_payload = _diag.compute_routing_correlations(
            _diag._routing_quality_rows, rnd + 1
        )
        if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0:
            _diag.print_routing_quality(rnd + 1, _routing_corr_payload)

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

    _agg_norm_payload = None
    _global_norm_payload = None
    if getattr(config, "enable_gradient_diagnostics", False):
        _agg_norm = (
            float(server.global_delta.norm(p=2).item())
            if server.global_delta is not None
            else 0.0
        )
        _global_norm = float(
            torch.cat(
                [
                    p.data.flatten().float().cpu()
                    for p in server.global_model.parameters()
                ]
            )
            .norm(p=2)
            .item()
        )
        _agg_norm_payload = _agg_norm
        _global_norm_payload = _global_norm
        if (rnd + 1) % getattr(config, "diag_print_interval", 25) == 0 and _grad_diags:
            _diag.print_gradient_diagnostics_summary(
                rnd + 1, _grad_diags, _agg_norm, _global_norm
            )
    return (
        _routing_corr_payload,
        _tier_contrib_payload,
        _compress_analysis_payload,
        _agg_norm_payload,
        _global_norm_payload,
    )


def _print_client_details(config, results, server, rnd, _grad_diags):
    if getattr(config, "ablation_per_client_logging", False):
        _k1_active = getattr(config, "active_k_ratio_tier1", config.k_ratio_tier1)
        _k2_active = getattr(config, "active_k_ratio_tier2", config.k_ratio_tier2)
        print(f"\n  [ABLATION-D] round={rnd + 1} per-client forensic log")
        print(
            f"  {'CID':>4}  {'d_raw':>10}  {'d_ema':>10}  {'tier':>4}  "
            f"{'k_ratio':>7}  {'sel_w':>8}  {'agg_w':>8}  "
            f"{'loss':>8}  {'norm_before':>11}"
        )
        for _r in sorted(results, key=lambda x: x["client_id"]):
            _cid = _r["client_id"]
            _draw = _r.get("d_raw", float("nan"))
            _dema = _r.get("divergence_score", float("nan"))
            _tier = _r.get("tier", -1)
            _kr = _k1_active if _tier == 1 else _k2_active
            _selw = server.selection_weights[_cid]
            _aggw = _r.get("aggregation_weight", 0.0)
            _loss = _r.get("local_loss", float("nan"))
            # update_l2_norm populated by gradient diagnostics when enabled
            _norm = (
                _grad_diags[_cid]["update_l2_norm"]
                if _cid in _grad_diags
                else float("nan")
            )
            _draw_str = f"{_draw:>10.6f}" if not np.isnan(_draw) else "       nan"
            _norm_str = f"{_norm:>11.5f}" if not np.isnan(_norm) else "          nan"
            print(
                f"  {_cid:>4}  {_draw_str}  {_dema:>10.6f}  {_tier:>4}  "
                f"{_kr:>7.4f}  {_selw:>8.4f}  {_aggw:>8.4f}  "
                f"{_loss:>8.4f}  {_norm_str}"
            )
