"""
divroute_fl/diagnostics.py
==========================
Scientific instrumentation for the DivRoute-FL forensic investigation.

All public functions are PURELY DIAGNOSTIC: they read existing tensors and
produce log entries / console output.  They never modify model weights,
gradients, configuration state, or routing decisions.

Every function is individually gated by a Config flag so that existing
experiments remain byte-identical when all flags are False.

Parts implemented here
----------------------
  Part 1  — Gradient quality diagnostics
  Part 2  — Alternative divergence metrics (cosine / l2 / relative_l2 / layerwise_cosine)
  Part 3  — Routing quality evaluation (Pearson / Spearman correlations)
  Part 4  — Tier contribution analysis
  Part 5  — Compression analysis
  Part 6  — Adaptive tau diagnostics
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. Returns NaN when undefined."""
    if len(x) < 2:
        return float("nan")
    c = np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1]
    return float(c) if not np.isnan(c) else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN when undefined."""
    if len(x) < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    c = np.corrcoef(xr.astype(np.float64), yr.astype(np.float64))[0, 1]
    return float(c) if not np.isnan(c) else float("nan")


def _safe_fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "    nan"
    return f"{v:+.4f}"


# ---------------------------------------------------------------------------
# PART 2 — Alternative divergence metrics
# ---------------------------------------------------------------------------

def compute_divergence_metric(
    client_flat: torch.Tensor,
    global_flat: torch.Tensor,
    metric: str,
    layer_slices: Optional[List[Tuple[str, int, int]]] = None,
) -> float:
    """
    Compute the requested divergence metric between the client parameter
    vector and the global parameter vector.

    Parameters
    ----------
    client_flat : 1-D float tensor (CPU or GPU)
    global_flat : 1-D float tensor (same device / dtype)
    metric      : one of ``"cosine"`` | ``"l2"`` | ``"relative_l2"`` |
                  ``"layerwise_cosine"``
    layer_slices: list of (name, offset, numel) — required for
                  ``"layerwise_cosine"``, ignored for other metrics.

    Returns
    -------
    float  — scalar divergence score (always ≥ 0).

    Notes
    -----
    * ``"cosine"`` reproduces the existing implementation identically.
    * This function is called only when a non-default metric is configured,
      or when gradient diagnostics need the raw metric alongside the cosine
      score for comparison.
    * No side-effects on any tensor.
    """
    c = client_flat.double()
    g = global_flat.double()
    delta = c - g

    if metric == "cosine":
        cos = F.cosine_similarity(
            c.unsqueeze(0), g.unsqueeze(0), dim=1, eps=1e-8
        ).item()
        return max(0.0, 1.0 - cos)

    elif metric == "l2":
        return float(delta.norm(p=2).item())

    elif metric == "relative_l2":
        g_norm = float(g.norm(p=2).item())
        if g_norm < 1e-12:
            return 0.0
        return float(delta.norm(p=2).item()) / g_norm

    elif metric == "layerwise_cosine":
        if not layer_slices:
            # Fallback: treat the whole vector as one layer.
            cos = F.cosine_similarity(
                c.unsqueeze(0), g.unsqueeze(0), dim=1, eps=1e-8
            ).item()
            return max(0.0, 1.0 - cos)
        scores = []
        for (_name, offset, numel) in layer_slices:
            end = offset + numel
            lc = c[offset:end]
            lg = g[offset:end]
            if lc.norm(p=2) < 1e-12 or lg.norm(p=2) < 1e-12:
                scores.append(0.0)
                continue
            cos_l = F.cosine_similarity(
                lc.unsqueeze(0), lg.unsqueeze(0), dim=1, eps=1e-8
            ).item()
            scores.append(max(0.0, 1.0 - cos_l))
        return float(np.mean(scores)) if scores else 0.0

    else:
        raise ValueError(
            f"Unknown divergence_metric {metric!r}. "
            "Choose: 'cosine', 'l2', 'relative_l2', 'layerwise_cosine'."
        )


# ---------------------------------------------------------------------------
# PART 1 — Gradient quality diagnostics (per-client)
# ---------------------------------------------------------------------------

def compute_gradient_diagnostics(
    client_flat: torch.Tensor,
    global_flat: torch.Tensor,
    compressed_flat: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute gradient-quality statistics for one client.

    Parameters
    ----------
    client_flat    : 1-D float tensor — local model parameters after training.
    global_flat    : 1-D float tensor — global model parameters.
    compressed_flat: 1-D float tensor — reconstructed delta after top-k
                     compression.  Pass None if compression has not been
                     applied yet (reconstruction error fields will be NaN).

    Returns
    -------
    dict with keys:
      update_l2_norm      — ‖w_local − w_global‖₂
      relative_update_norm— ‖Δ‖ / ‖w_global‖
      compression_error   — ‖Δ_orig − Δ_compressed‖₂  (NaN if no compressed)
      retention_ratio     — ‖Δ_compressed‖₂ / ‖Δ_orig‖₂  (NaN if no compressed)
      compressed_cosine   — cos_sim(Δ_orig, Δ_compressed)  (NaN if no compressed)
    """
    c = client_flat.float()
    g = global_flat.float()
    delta = c - g

    update_l2 = float(delta.norm(p=2).item())
    g_norm = float(g.norm(p=2).item())
    rel_norm = update_l2 / max(g_norm, 1e-12)

    out: Dict[str, float] = {
        "update_l2_norm": update_l2,
        "relative_update_norm": rel_norm,
        "compression_error": float("nan"),
        "retention_ratio": float("nan"),
        "compressed_cosine": float("nan"),
    }

    if compressed_flat is not None:
        comp_delta = compressed_flat.float()
        err = float((delta - comp_delta).norm(p=2).item())
        comp_norm = float(comp_delta.norm(p=2).item())
        orig_norm = update_l2

        out["compression_error"] = err
        out["retention_ratio"] = (
            comp_norm / max(orig_norm, 1e-12)
        )
        # cosine between original and compressed delta
        if orig_norm > 1e-12 and comp_norm > 1e-12:
            cs = F.cosine_similarity(
                delta.unsqueeze(0),
                comp_delta.unsqueeze(0),
                dim=1, eps=1e-8,
            ).item()
            out["compressed_cosine"] = float(cs)

    return out


def print_gradient_diagnostics_summary(
    rnd: int,
    per_client_diags: Dict[int, Dict[str, float]],
    agg_update_norm: float,
    global_update_norm: float,
) -> None:
    """
    Print a compact summary table of gradient quality statistics.
    Called at most every `diag_print_interval` rounds from main.py.
    """
    print(f"\n  [GRAD-DIAG] round={rnd}")
    print(
        f"  {'CID':>4}  {'L2-norm':>10}  {'rel-norm':>9}  "
        f"{'cmp-err':>9}  {'retain':>7}  {'cmp-cos':>8}"
    )
    for cid, d in sorted(per_client_diags.items()):
        print(
            f"  {cid:>4}  {d['update_l2_norm']:>10.5f}  "
            f"{d['relative_update_norm']:>9.6f}  "
            f"{_safe_fmt(d['compression_error']):>9}  "
            f"{_safe_fmt(d['retention_ratio']):>7}  "
            f"{_safe_fmt(d['compressed_cosine']):>8}"
        )
    print(f"  [GRAD-DIAG] agg_update_norm={agg_update_norm:.6f}  "
          f"global_update_norm={global_update_norm:.6f}")


# ---------------------------------------------------------------------------
# PART 3 — Routing quality evaluation
# ---------------------------------------------------------------------------

# Accumulator: stores per-round routing-quality rows across the full run.
# main.py appends to this each round; the list is never reset mid-run.
_routing_quality_rows: List[Dict[str, Any]] = []


def record_routing_quality(
    rnd: int,
    results: List[dict],
    grad_diags: Dict[int, Dict[str, float]],
) -> None:
    """
    Append one row per client to the routing-quality accumulator.

    Parameters
    ----------
    rnd         : 1-indexed round number.
    results     : list of client result dicts (post-aggregation so that
                  aggregation_weight is already set).
    grad_diags  : {client_id: gradient_diagnostics_dict} — may be empty.
    """
    for r in results:
        cid = r["client_id"]
        diag = grad_diags.get(cid, {})
        _routing_quality_rows.append({
            "round":            rnd,
            "client_id":        cid,
            "tier":             r.get("tier"),
            "divergence":       r.get("divergence_score"),
            "d_raw":            r.get("d_raw"),
            "local_loss":       r.get("local_loss"),
            "local_train_acc":  r.get("local_train_acc"),
            "update_norm":      diag.get("update_l2_norm"),
            "aggregation_weight": r.get("aggregation_weight"),
            "compressed_norm":  (
                diag.get("retention_ratio") * diag.get("update_l2_norm", 0.0)
                if diag.get("retention_ratio") is not None
                   and not math.isnan(diag.get("retention_ratio", float("nan")))
                else None
            ),
        })


def compute_routing_correlations(
    rows: List[Dict[str, Any]],
    rnd: int,
) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations over the last round's data.

    Returns a dict with keys like ``pearson_div_acc``, ``spearman_div_norm`` …
    """
    # Use only rows from this round
    this_round = [row for row in rows if row["round"] == rnd]
    if len(this_round) < 2:
        return {}

    def _arr(key):
        vals = [row[key] for row in this_round if row.get(key) is not None]
        return np.array(vals, dtype=np.float64)

    div_arr  = _arr("divergence")
    acc_arr  = _arr("local_train_acc")
    loss_arr = _arr("local_loss")
    norm_arr = _arr("update_norm")

    out: Dict[str, float] = {}

    def _pair(name_a, a, name_b, b):
        n = min(len(a), len(b))
        if n < 2:
            return
        a2, b2 = a[:n], b[:n]
        out[f"pearson_{name_a}_{name_b}"]  = _pearson(a2, b2)
        out[f"spearman_{name_a}_{name_b}"] = _spearman(a2, b2)

    _pair("div", div_arr, "acc",  acc_arr)
    _pair("div", div_arr, "loss", loss_arr)
    _pair("div", div_arr, "norm", norm_arr)

    return out


def print_routing_quality(rnd: int, corrs: Dict[str, float]) -> None:
    """Print routing-quality correlation table."""
    if not corrs:
        return
    print(f"\n  [ROUTING-QUALITY] round={rnd}")
    pairs = [
        ("div↔acc",  "pearson_div_acc",  "spearman_div_acc"),
        ("div↔loss", "pearson_div_loss", "spearman_div_loss"),
        ("div↔norm", "pearson_div_norm", "spearman_div_norm"),
    ]
    for label, pk, sk in pairs:
        pv = corrs.get(pk, float("nan"))
        sv = corrs.get(sk, float("nan"))
        print(f"  {label}: pearson={_safe_fmt(pv)}  spearman={_safe_fmt(sv)}")


# ---------------------------------------------------------------------------
# PART 4 — Tier contribution analysis
# ---------------------------------------------------------------------------

def compute_tier_contributions(
    results: List[dict],
    raw_deltas: Dict[int, torch.Tensor],      # {client_id: raw_delta (CPU)}
    compressed_deltas: Dict[int, torch.Tensor], # {client_id: compressed_delta (CPU)}
) -> Dict[str, Any]:
    """
    Compute total update norm separated by tier, before and after compression.

    Returns a dict with keys:
      tier{n}_raw_norm, tier{n}_compressed_norm, tier{n}_count  (n = 1,2,3)
      percent_contribution_{before,after} per tier
    """
    from collections import defaultdict
    raw_norms:  Dict[int, float] = defaultdict(float)
    comp_norms: Dict[int, float] = defaultdict(float)
    counts:     Dict[int, int]   = defaultdict(int)

    for r in results:
        cid  = r.get("client_id")
        tier = r.get("tier", 3)
        if cid in raw_deltas:
            raw_norms[tier]  += float(raw_deltas[cid].norm(p=2).item())
        if cid in compressed_deltas:
            comp_norms[tier] += float(compressed_deltas[cid].norm(p=2).item())
        counts[tier] += 1

    total_raw  = sum(raw_norms.values())  + 1e-12
    total_comp = sum(comp_norms.values()) + 1e-12

    out: Dict[str, Any] = {}
    for t in (1, 2, 3):
        out[f"tier{t}_count"]           = counts[t]
        out[f"tier{t}_raw_norm"]        = raw_norms[t]
        out[f"tier{t}_compressed_norm"] = comp_norms[t]
        out[f"tier{t}_pct_before"]      = 100.0 * raw_norms[t]  / total_raw
        out[f"tier{t}_pct_after"]       = 100.0 * comp_norms[t] / total_comp

    return out


def print_tier_contributions(rnd: int, tc: Dict[str, Any]) -> None:
    """Print tier contribution table."""
    print(f"\n  [TIER-CONTRIB] round={rnd}")
    print(f"  {'Tier':>4}  {'N':>3}  {'raw_norm':>10}  {'%before':>8}  "
          f"{'cmp_norm':>10}  {'%after':>8}")
    for t in (1, 2, 3):
        n   = tc.get(f"tier{t}_count", 0)
        rn  = tc.get(f"tier{t}_raw_norm", 0.0)
        cn  = tc.get(f"tier{t}_compressed_norm", 0.0)
        pb  = tc.get(f"tier{t}_pct_before", 0.0)
        pa  = tc.get(f"tier{t}_pct_after", 0.0)
        print(f"  {t:>4}  {n:>3}  {rn:>10.5f}  {pb:>7.1f}%  "
              f"{cn:>10.5f}  {pa:>7.1f}%")


# ---------------------------------------------------------------------------
# PART 5 — Compression analysis (per-tier averages)
# ---------------------------------------------------------------------------

def compute_compression_analysis(
    results: List[dict],
    raw_deltas: Dict[int, torch.Tensor],
    compressed_deltas: Dict[int, torch.Tensor],
) -> List[Dict[str, Any]]:
    """
    Compute per-tier averages of compression error and retention ratio.

    Returns a list of dicts, one per tier, with keys:
      tier, count, avg_orig_norm, avg_comp_norm, avg_error, avg_retention
    """
    from collections import defaultdict

    data: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for r in results:
        cid  = r.get("client_id")
        tier = r.get("tier", 3)
        if cid not in raw_deltas:
            continue
        raw  = raw_deltas[cid]
        comp = compressed_deltas.get(cid)
        orig_norm = float(raw.norm(p=2).item())
        if comp is not None:
            err       = float((raw - comp).norm(p=2).item())
            comp_norm = float(comp.norm(p=2).item())
            retention = comp_norm / max(orig_norm, 1e-12)
        else:
            err = comp_norm = retention = float("nan")
        data[tier].append({
            "orig_norm":  orig_norm,
            "comp_norm":  comp_norm,
            "error":      err,
            "retention":  retention,
        })

    out = []
    for t in (1, 2, 3):
        rows = data[t]
        if not rows:
            continue
        def _avg(key):
            vals = [row[key] for row in rows
                    if not math.isnan(row[key])]
            return sum(vals) / len(vals) if vals else float("nan")
        out.append({
            "tier":          t,
            "count":         len(rows),
            "avg_orig_norm": _avg("orig_norm"),
            "avg_comp_norm": _avg("comp_norm"),
            "avg_error":     _avg("error"),
            "avg_retention": _avg("retention"),
        })
    return out


def print_compression_analysis(rnd: int, ca: List[Dict[str, Any]]) -> None:
    """Print compression analysis table."""
    print(f"\n  [COMPRESS-ANALYSIS] round={rnd}")
    print(f"  {'Tier':>4}  {'N':>3}  {'orig_norm':>10}  {'comp_norm':>10}  "
          f"{'error':>10}  {'retention':>10}")
    for row in ca:
        t   = row["tier"]
        n   = row["count"]
        on  = row["avg_orig_norm"]
        cn  = row["avg_comp_norm"]
        err = row["avg_error"]
        ret = row["avg_retention"]
        print(f"  {t:>4}  {n:>3}  {on:>10.5f}  {cn:>10.5f}  "
              f"{err:>10.5f}  {ret:>10.5f}")


# ---------------------------------------------------------------------------
# PART 6 — Adaptive tau diagnostics
# ---------------------------------------------------------------------------

def compute_tau_diagnostics(
    raw_d_scores: List[float],
    mu: float,
    sigma: float,
    tau_low: float,
    tau_high: float,
    results: List[dict],
) -> Dict[str, Any]:
    """
    Collect tau-related statistics without modifying any values.

    Returns a dict with keys:
      mu, sigma, tau_low, tau_high, spread, coeff_variation,
      n_tier1, n_tier2, n_tier3,
      hist_p10, hist_p25, hist_p50, hist_p75, hist_p90
    """
    arr = np.array(raw_d_scores, dtype=np.float64)
    spread = float(arr.max() - arr.min()) if len(arr) > 1 else 0.0
    cv = (sigma / max(abs(mu), 1e-12)) if mu != 0.0 else 0.0

    n_tier = {1: 0, 2: 0, 3: 0}
    for r in results:
        t = r.get("tier", 3)
        n_tier[t] = n_tier.get(t, 0) + 1

    percentiles = (
        np.percentile(arr, [10, 25, 50, 75, 90]).tolist()
        if len(arr) >= 2 else [float("nan")] * 5
    )

    return {
        "mu":              mu,
        "sigma":           sigma,
        "tau_low":         tau_low,
        "tau_high":        tau_high,
        "spread":          spread,
        "coeff_variation": cv,
        "n_tier1":         n_tier[1],
        "n_tier2":         n_tier[2],
        "n_tier3":         n_tier[3],
        "hist_p10":        percentiles[0],
        "hist_p25":        percentiles[1],
        "hist_p50":        percentiles[2],
        "hist_p75":        percentiles[3],
        "hist_p90":        percentiles[4],
    }


def print_tau_diagnostics(rnd: int, td: Dict[str, Any]) -> None:
    """Print adaptive tau diagnostic summary."""
    print(f"\n  [TAU-DIAG] round={rnd}")
    print(f"  mu={td['mu']:.8f}  sigma={td['sigma']:.8f}  "
          f"tau_low={td['tau_low']:.8f}  tau_high={td['tau_high']:.8f}")
    print(f"  spread={td['spread']:.8f}  CV={td['coeff_variation']:.4f}")
    print(f"  tiers: T1={td['n_tier1']}  T2={td['n_tier2']}  T3={td['n_tier3']}")
    print(f"  percentiles: "
          f"p10={td['hist_p10']:.8f}  p25={td['hist_p25']:.8f}  "
          f"p50={td['hist_p50']:.8f}  p75={td['hist_p75']:.8f}  "
          f"p90={td['hist_p90']:.8f}")
