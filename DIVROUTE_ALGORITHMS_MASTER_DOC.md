# DivRoute-FL — Master Algorithm & Method Reference

This document is a complete, self-contained technical description of every algorithm,
formula, and mechanism implemented in this repository. It is written so that someone
with no prior context on this project (e.g. a fresh LLM conversation) can understand
the system's design, math, and rationale purely from this document.

## 0. High-level system description

This is a **Federated Learning (FL) research framework** ("DivRoute-FL") built on
PyTorch. It simulates many clients (e.g. 25) training local copies of a shared global
model (SimpleCNN for CIFAR-10, or a modified ResNet-18 for CIFAR-100) on non-IID data
partitions, and a central server that aggregates their updates every communication
round. The central research contribution is **DivRoute**: a divergence-based adaptive
routing mechanism that decides, per client per round, (a) whether the client's update
is worth transmitting at full fidelity, reduced fidelity, or not at all, and (b) how
much weight the update should get during aggregation. The goal is to cut
client→server (and optionally server→client) communication volume while preserving
accuracy, by spending bandwidth on clients whose local model has diverged the most
from the global model (i.e., clients carrying the most "new information").

Around this core idea, the codebase implements a large number of supporting
algorithms: non-IID data partitioning (Dirichlet), local training tricks (MixUp,
label smoothing, EMA, cosine LR schedule, gradient clipping), gradient/weight
compression (Top-k sparsification, layer-wise Top-k, error feedback), server-side
aggregation strategies (sample-weighted averaging, divergence-weighted averaging,
softmax weighting, server momentum), adaptive threshold computation (z-score based
and percentile based, with rolling windows and hysteresis/smoothing), several
published FL baselines re-implemented for comparison (FedAvg, FedProx-like
FedSparse, FedZip, FedNTD-style distillation), and an extensive diagnostics/forensic
instrumentation layer used to audit *why* the "DivRoute" pipeline was hurting
accuracy at larger scale (CIFAR-100 / ResNet-18) — layer starvation audits, gradient
quality diagnostics, tier-contribution analysis, etc.

Directory map:
- `divroute_fl/mechanism.py` — divergence, tiering, adaptive-threshold, weighting math (the algorithmic heart)
- `divroute_fl/compression.py` — Top-k sparsification, layer-wise Top-k, error feedback
- `divroute_fl/client.py` — local training loop (MixUp, NTD, FedSparse prox operator, EMA, cosine LR)
- `divroute_fl/server.py` — aggregation, server momentum, checkpoint-relevant state
- `divroute_fl/model.py` — model factory (SimpleCNN, CIFAR-adapted ResNet-18, GroupNorm/WS-Conv variants)
- `divroute_fl/data.py` — Dirichlet non-IID partitioning, train/val/test transforms
- `divroute_fl/config.py` — the single `Config` dataclass controlling every behavior/ablation flag
- `divroute_fl/main.py` — orchestration loop tying everything together, round-by-round
- `divroute_fl/diagnostics.py` — Part 1–6 scientific instrumentation (non-invasive, diagnostic only)
- `baselines/fedzip_baseline.py`, `baselines/fedzip_actual.py` — FedZip (Top-z + k-means quantization) baseline
- `baselines/fedsparse_baseline.py` — FedSparse (L1 proximal) baseline config factory
- `run_audit.py`, `run_b3_ab_experiment.py`, `run_b3_layerwise_audit.py`, `analyze_audit.py`, `analyze_layerwise_audit.py` — forensic "Bottleneck 3" layer-starvation audit tooling

---

## 1. Divergence metrics — measuring how much a client has drifted

All divergence functions live in `divroute_fl/mechanism.py` (core) and
`divroute_fl/diagnostics.py` (alternative metrics, Part 2).

### 1.1 Weight-space cosine divergence (default / legacy metric)

```
compute_divergence(local_model, global_model):
    local_vec  = flatten(local_model.parameters())
    global_vec = flatten(global_model.parameters())
    cos_sim    = cosine_similarity(local_vec, global_vec, eps=1e-8)
    d = 1 - cos_sim
```
- Range: `d ∈ [0, 2]` in theory (`cos_sim ∈ [-1, 1]`), but in practice models stay in a
  similar region of weight space so `d` is small (near 0).
- `d = 0` → local and global weight vectors point in the exact same direction
  (client made no meaningful directional change).
- `d = 2` → vectors point in opposite directions (never really happens in practice).
- Interpretation: measures *directional* drift of the **full weight vector**
  (not just the update/delta), so it is dominated by whichever parameters happen to
  have the largest magnitude in the flattened vector.
- Used as the default (`divergence_metric = "cosine"` in `Config`).

### 1.2 Alternative divergence metrics (Part 2 diagnostics, `diagnostics.py: compute_divergence_metric`)

Selectable via `Config.divergence_metric ∈ {"cosine", "l2", "relative_l2", "layerwise_cosine"}`:

- **`"l2"`**: `d = ||w_local - w_global||_2` — raw Euclidean distance between local
  and global parameter vectors. Not scale-invariant; larger models / larger updates
  automatically produce larger scores.
- **`"relative_l2"`**: `d = ||w_local - w_global||_2 / ||w_global||_2` — normalizes
  the L2 distance by the magnitude of the current global model, giving a
  scale-invariant relative-drift measure. Falls back to `0.0` if `||w_global||_2 < 1e-12`.
- **`"layerwise_cosine"`**: computes `1 - cos_sim` independently **per named
  parameter tensor** (e.g. each conv layer, each BN layer, each FC layer) and
  averages the per-layer divergence scores. This prevents one huge layer (e.g. the
  final FC layer in a many-class model) from dominating the single global cosine
  score; it treats every layer's directional change as equally important regardless
  of its parameter count.

### 1.3 Decoupled / directional divergence (Phase-5 revision)

```
compute_decoupled_divergence(delta, reference_vector, eps=1e-8):
    delta_hat = delta / (||delta||_2 + eps)
    ref_hat   = reference_vector / (||reference_vector||_2 + eps)
    d = clip(1 - cosine_similarity(delta_hat, ref_hat), 0, 1)
```
- Key idea: instead of comparing the client's **absolute weight vector** to the
  global model (which conflates "how far did I move" with "in what direction did I
  move"), this compares the **direction of the client's update** (`delta = w_local -
  w_global`) to the **direction the server's aggregated update moved in the previous
  round** (`reference_vector = server momentum buffer`). Both vectors are first
  L2-normalized ("hat" vectors) so magnitude is fully removed from the score — it is
  a *pure angular* divergence measure.
- `d = 0` → the client moved in exactly the direction the server is already moving
  (server-aligned, "redundant" update — arguably less informative).
- `d = 1` → the client moved in a direction orthogonal (or worse) to the server's
  trajectory (potentially the most "novel" signal).
- Activated via `Config.use_directional_divergence = True`. On round 0, or whenever
  the server momentum buffer doesn't exist yet, this safely falls back to a neutral
  `d = 0.5` for every client (`main.py`, "Phase-5 directional divergence" block).

### 1.4 Dynamic Hybrid Scoring (Bottleneck-2 fix — combines divergence + loss improvement)

```
compute_dynamic_hybrid_scores(div_scores, loss_scores, current_round, total_rounds):
    tau_d    = max(0.05, std(div_scores))
    tau_I    = max(0.05, std(loss_scores))
    progress = current_round / max(1, total_rounds - 1)
    lambda_t = 0.4 + 0.4 * progress        # rises 0.4 -> 0.8 over training
    score    = lambda_t * softmax(div_scores / tau_d)
             + (1 - lambda_t) * softmax(loss_scores / tau_I)
```
- Motivation: pure directional divergence answers "did this client move in a novel
  direction?" but not "did that movement actually help?" This hybrid score blends
  in **local validation loss improvement** (`global_val_loss - local_val_loss`,
  positive when the client's local model beats the current global model on its own
  held-out validation split) so that clients whose updates *both* look novel *and*
  demonstrably improve loss are favored.
- Each component (`div_scores`, `loss_scores`) is passed through its own **temperature-scaled
  softmax** (`safe_softmax`, numerically stabilized by subtracting the max before
  `exp`), so it becomes a probability-like distribution over the selected cohort
  rather than a raw scalar.
- `lambda_t` is a **linear annealing schedule**: early training (low `progress`)
  weighs directional-divergence more (`lambda_t=0.4` → weight given to div term is 0.4,
  to loss term 0.6) — actually re-reading: `score = lambda_t * div_soft + (1-lambda_t) * loss_soft`,
  so `lambda_t` is literally the divergence-term weight, and it *increases* from 0.4 to
  0.8 over the run: early rounds trust the loss-improvement signal more (harder to get
  a reliable divergence signal early on), later rounds trust directional divergence
  more (as the model converges, loss differences shrink and divergence becomes the
  more discriminating signal).
- Only computed when `use_directional_divergence=True` **and** every client in the
  cohort has a valid `local_val_loss`/`global_val_loss` pair (requires
  `local_val_fraction > 0`); otherwise `main.py` falls back to a min-max normalized
  directional-divergence-only score.

### 1.5 EMA smoothing of divergence scores

```
update_ema(ema_scores, client_id, d_current, beta):
    if client_id not seen before: ema_scores[client_id] = d_current
    else: ema_scores[client_id] = beta * ema_scores[client_id] + (1-beta) * d_current
```
- Standard **exponential moving average** per client, persisted across rounds in a
  dict keyed by `client_id` (so it survives a client not being selected for several
  rounds — it just doesn't update during that gap).
- `beta` (`Config.ema_beta`, default `0.85`) controls smoothing: closer to 1 = more
  historical inertia, closer to 0 = reacts fully to the latest score.
- `Config.routing_score ∈ {"raw", "ema"}` decides whether tier assignment compares
  the instantaneous `d_raw` or the smoothed `d_ema` against the thresholds — but note
  Phase-5-revised code in `main.py` *hardcodes* the aggregation weighting to always
  use `d_ema`, and (per the "Active EMA routing lock" comment) also enforces that
  the **routing/tiering decision** itself always uses `d_ema` regardless of the
  `routing_score` config flag's nominal description — this was a deliberate lock
  applied during the Phase-5 revision to keep tiering and weighting consistent.

---

## 2. Tier assignment (the "routing" in DivRoute)

### 2.1 Basic 3-tier rule (`mechanism.py: assign_tier`)

```
assign_tier(d, tau_low, tau_high):
    if d > tau_high: return 1   # high divergence -> Tier 1 (full-fidelity)
    elif d > tau_low: return 2  # moderate divergence -> Tier 2 (lightweight)
    else: return 3              # low divergence / converged -> Tier 3 (skip)
```

**Important inconsistency to note**: `main.py`'s actual production tier-assignment
loop uses the **opposite polarity** from the docstring above:
```python
if d_for_tier <= config.tau_low:  tier = 1
elif d_for_tier <= config.tau_high: tier = 2
else: tier = 3
```
i.e., in the live training loop, **low divergence → Tier 1** (full fidelity) and
**high divergence → Tier 3**. This is the opposite of the tier semantics described in
the `mechanism.assign_tier` docstring (which is only used for the diagnostic
"routing disagreement" table, not for the live tiering decision). Practically, in the
live loop: Tier 1 = clients whose EMA divergence is at/below `tau_low` (most
"settled"/converged clients — get the *most* bandwidth, top-k ratio = `k_ratio_tier1`,
default 0.20), Tier 2 = clients between `tau_low` and `tau_high` (get
`k_ratio_tier2`, default 0.05), Tier 3 = clients above `tau_high` (excluded from
aggregation and transmission entirely by default, unless
`include_tier3_in_aggregation=True`). This is a materially important asymmetry
between the "intended" design (reward high-divergence/novel clients with more
bandwidth) described in most of the code comments, versus the actual live-loop
behavior (reward low-divergence/stable clients with more bandwidth, treat
high-divergence clients as probably-noisy and starve them). Any downstream analysis
of this codebase should re-verify which polarity is active for a given experiment
by reading the specific `main.py` block in use, since ablation flags can change this.

### 2.2 Progress Guarantee (starvation prevention)

If, after tier assignment, **zero** clients out of the selected cohort ended up in
Tier 1 or Tier 2 (i.e., everyone got skipped), the system force-promotes the **top 2
clients by divergence score** (tie-broken by "hasn't been promoted recently") to
Tier 2, so at least some signal reaches the server every round:
```python
p_count = sum(1 for r in results if r["tier"] in (1, 2))
if p_count == 0 and results:
    ranked = sorted(results, key=lambda r: (r["divergence_score"], -last_promoted_round[cid]), reverse=True)
    promote ranked[:2] to tier 2
```

### 2.3 Selection-weight decay (`update_selection_weights`)

Controls which clients are *sampled* in future rounds (separate from tiering, which
controls what happens to clients *once* they're sampled):
```
for each client result r:
    if r.tier == 3 (converged / natural_tier==3):
        selection_weights[client_id] *= gamma      # decay probability of re-selection
    else:
        selection_weights[client_id] = 1.0          # reset to full priority
```
- `gamma` (`Config.gamma`, default `0.85`) < 1 exponentially decays the sampling
  probability of clients repeatedly found to be "converged" (Tier 3), so the server
  spends its limited `clients_per_round` budget preferentially on clients still
  producing useful signal. Any client that stops being Tier-3 immediately regains
  full selection priority (weight reset to 1.0).
- `select_clients` (`server.py`) draws `clients_per_round` clients **without
  replacement**, with probability proportional to these `selection_weights`
  (normalized to sum to 1), via `numpy.random.Generator.choice(..., p=prob)`.

---

## 3. Adaptive threshold computation (`tau_low`, `tau_high`)

Two independently implemented threshold-computation strategies exist, selected by
`Config.threshold_mode ∈ {"fixed", "adaptive_tau", "percentile", "rolling_percentile"}`.

### 3.1 Z-score / mean-sigma adaptive thresholds (`compute_adaptive_taus`)

```
mu, sigma = mean(scores), std(scores)
sigma = max(sigma, 1e-5)                 # guard against variance collapse
tau_low  = max(0.0, mu - tau_alpha * sigma)
tau_high = mu + tau_beta * sigma
if tau_low >= tau_high:                  # degenerate-distribution fallback
    tau_high = tau_low * 1.5 + 1e-6
```
- `tau_alpha` (default 0.5) and `tau_beta` (default 1.0) control how many standard
  deviations below/above the mean define the tier boundaries — an intentionally
  **asymmetric** band (Tier-1 cutoff is closer to the mean than the Tier-3 cutoff),
  reflecting a design choice that few clients should be excluded outright.
- Fed by a **rolling window** (`Config.tau_window`, default 5 rounds): `main.py`
  keeps a list of the most recent `tau_window` rounds' worth of per-client EMA
  scores (`_tau_score_history`), concatenates them, and computes `mu`/`sigma` over
  that pooled multi-round sample (rather than just the current round's ~15 clients),
  which stabilizes the threshold estimate against per-round sampling noise.
- Additional **exponential smoothing** (`Config.tau_smoothing`, default 0.80) is then
  applied to the *candidate* thresholds computed above, blending them with the
  previous round's live thresholds:
  ```
  new_tau_low  = tau_smoothing * old_tau_low  + (1 - tau_smoothing) * candidate_tau_low
  new_tau_high = tau_smoothing * old_tau_high + (1 - tau_smoothing) * candidate_tau_high
  tau_low, tau_high = min(new_low,new_high), max(new_low,new_high)  # order guard
  ```
  This is a second layer of temporal smoothing on top of the rolling-window
  statistic — i.e., thresholds move slowly (80% weight on history) even when the
  underlying divergence distribution shifts round to round, which prevents tier
  assignments from flip-flopping.

### 3.2 Percentile-based thresholds (`compute_percentile_taus`)

```
p33, p67 = 33.33rd and 66.67th percentile of the score distribution
if p33 == p67: p67 += 1e-6   # tie-break degenerate case
if fewer than 3 scores: return (-inf, +inf)  # fallback — nobody gets tiered out
```
- A **non-parametric** alternative to the z-score method: instead of assuming the
  divergence distribution is roughly Gaussian, this just splits the current cohort
  into thirds by rank. Guarantees each tier gets ≈1/3 of clients regardless of the
  actual score distribution's shape.
- `"rolling_percentile"` mode additionally pools scores across a `deque` of the last
  `rolling_window_size` rounds (default 5) before computing percentiles, analogous
  to the rolling window in §3.1.

### 3.3 Convergence protection / threshold freezing

Before recomputing percentile thresholds each round, `main.py` checks whether the
pooled score distribution has essentially collapsed to a point:
```python
spread = std(all pooled scores)
if spread < convergence_floor (default 1e-4) and a previous tau exists:
    reuse the previous round's tau_low/tau_high verbatim (frozen)
else:
    recompute p33/p67 normally
```
This prevents the percentile thresholds from becoming numerically meaningless (e.g.
`p33 ≈ p67 ≈ 0`) once the whole federation has nearly converged and divergence
scores are all tiny — in that regime, freezing the last "sane" threshold is safer
than recomputing noise-dominated percentiles.

---

## 4. Aggregation weighting

### 4.1 Sample-count weighting (FedAvg baseline component)

Every client's contribution is always scaled, at minimum, by
`num_samples_client / total_samples_selected` — the classic FedAvg weighting, so
clients with more local data get proportionally more influence.

### 4.2 Divergence-based weight multipliers (`compute_divergence_weight`)

Given a per-client divergence score `d`, four modes (selected via
`Config.divergence_weight_mode`) convert it into a multiplicative weight on top of
the sample-count weight:
- **`"inverse"`** (legacy): `w = 1 / (d + 1e-6)` — inversely weights by divergence
  (heavily rewards *low*-divergence/stable clients).
- **`"sqrt"`** (default/recommended): `w = sqrt(d) + 1e-6` — the sign is *inverted*
  relative to `"inverse"`: this rewards **higher**-divergence (more novel) clients,
  but with diminishing returns (square-root compression) so a single very-high-`d`
  outlier doesn't dominate the aggregate. Chosen because it's "more stable than raw
  inverse" per the inline comment.
- **`"exp"`** (legacy): `w = exp(-10 * d)` — exponentially *penalizes* higher
  divergence (opposite direction from `"sqrt"`), decaying very fast (10x rate
  constant), effectively a soft version of "only trust clients close to the global
  model."
- **`"softmax"`**: not handled inline by `compute_divergence_weight`; instead the
  caller uses `compute_softmax_weights` (below) over the *whole selected cohort at
  once* rather than per-client independently.

Note the explicit contradiction flagged in the code comments themselves: `"inverse"`
and `"exp"` are labeled "historically inverted (legacy)" while `"sqrt"` is the
"recommended" mode — i.e., earlier iterations of this project weighted *low*
divergence more, and the design was later flipped so that *high* divergence
(interpreted as "informative update") earns more aggregation weight. Whichever mode
is active must be checked against `Config.divergence_weight_mode` for any specific
experiment; do not assume "higher divergence = higher weight" without checking.

### 4.3 Softmax weighting over the cohort (`compute_softmax_weights`)

```
compute_softmax_weights(d_scores, temperature=50.0):
    neg = -d_scores * temperature
    neg -= max(neg)                     # numerical stability trick
    weights = exp(neg) / sum(exp(neg))  # sums to exactly 1.0, no separate normalization needed
```
- This is a **softmax over negative divergence**, i.e. it converts the *smallest*
  divergence scores into the *largest* weights (again the "reward stability" polarity,
  opposite to `"sqrt"` mode's "reward novelty" polarity). `temperature=50.0` makes
  this fairly sharp/peaked (high temperature in this formula's convention → closer to
  winner-take-all on the lowest-divergence client) since it's multiplied directly into
  the exponent rather than dividing.
- Only activated when `use_divergence_weighting=True` **and**
  `divergence_weight_mode == "softmax"` **and** every client in the aggregation
  cohort has a non-`None` `divergence_score`.
- When active, the softmax weight for each client is then multiplied by that
  client's sample-fraction weight (`num_samples / total_samples`) exactly as in the
  other modes, before final renormalization to sum to 1.

### 4.4 Final aggregation weight computation

Regardless of mode, `server.aggregate()`:
1. Computes an unnormalized weight per client as above.
2. Sums all unnormalized weights (`w_sum`).
3. Normalizes: `weight_i = weight_i / w_sum` so all participating clients' weights
   sum to exactly 1.0.
4. Records `aggregation_weight = 0.0` for every client that was excluded from
   aggregation (e.g. Tier 3 by default), and the computed normalized weight for
   everyone else.

### 4.5 Streaming weighted delta accumulation

The server never averages full state_dicts directly; it works entirely in
**delta space**:
```
old_flat  = flatten(global_model parameters)
for each participating client:
    client_flat = flatten(client's uploaded parameters)
    raw_delta   = client_flat - old_flat
    [optional: server-side gradient clipping if raw_delta's L2 norm exceeds grad_clip_norm]
    [compression: raw_delta -> compressed payload -> reconstruct_delta() back to dense tensor]
    agg_delta += weight_i * compressed_delta
new_flat = old_flat + agg_delta
```
This means **compression happens per-client, before weighted summation** — each
client's delta is independently sparsified/quantized, and the server sums the
(possibly lossy) reconstructions. An `active_mask` boolean tensor is also
accumulated across clients, marking every coordinate that *any* participating client
actually transmitted a nonzero value for — used later to mask the server-momentum
update so momentum is only applied to coordinates that received real updates this
round (see §7.2).

### 4.6 Buffer (BatchNorm running-stat) handling

Non-parameter buffers (BatchNorm `running_mean`, `running_var`,
`num_batches_tracked`) are **not** put through delta/weight/compression logic at
all — they are aggregated separately by a simple **sample-weighted average** of the
raw uploaded buffer values (not deltas):
```
buf_new = sum( (num_samples_i / total_samples) * client_buffer_i  for all clean clients )
```
This average is computed over **all clean (non-NaN) uploaded clients**, independent
of tier or whether they participated in the parameter aggregation — *unless*
`Config.bn_mode == "local_bn"`, in which case buffers are **not** aggregated at all
and the server simply keeps its own existing buffer values (because in `"local_bn"`
mode, BN statistics are treated as purely local/personalized state that should never
be globally shared — see §8.3).

---

## 5. Compression algorithms

All in `divroute_fl/compression.py`.

### 5.1 Global Top-k magnitude sparsification (`compress_delta`)

```
k = max(1, floor(k_ratio * numel))
if k >= numel: send the whole delta densely (float32, no indices) — "full model" path
else:
    indices = argtop_k(|delta|)          # magnitude-based, global across ALL parameters
    values  = delta[indices]
    payload = {indices (int32), values (float16/float32)}
```
- **Byte accounting** is explicit and asymmetric by design:
  - Sparse path (`k_ratio < 1.0`): `bytes = k * 8` (4 bytes int32 index + 4 bytes
    float32 value per retained coordinate), or `k * 6` if `use_fp16_upload=True`
    (4-byte index + 2-byte fp16 value).
  - Dense/full path (`k_ratio >= 1.0`): `bytes = numel * 4` (or `numel * 2` under
    fp16) — no index overhead since every coordinate is sent, avoiding
    double-penalizing the FedAvg baseline (`k=1.0`) against the sparse-format byte
    cost.
- **Global** Top-k means the single largest-magnitude `k` coordinates are picked
  across the *entire flattened model* — layers with naturally larger-magnitude
  gradients (e.g. later/final layers) will structurally dominate the selection,
  which is exactly the failure mode the layer-wise variant below was built to fix
  (see §5.2 and the "Bottleneck 3 layer starvation" audit tooling in §11).

### 5.2 Layer-wise independent Top-k (`compress_layerwise_independent`)

```
for each named layer (name, offset, size) in layer_slices:
    k_layer = max(1, floor(size * k_ratio))    # SAME ratio applied independently per layer
    select top-k_layer by |value| WITHIN that layer's slice only
concatenate all per-layer indices/values into one payload
```
- Key difference from §5.1: the *same fraction* `k_ratio` is retained from **every**
  layer independently, rather than a single global budget being redistributed by
  magnitude. This guarantees small/low-magnitude layers (e.g. early conv layers or
  BN-adjacent parameters) always get *some* representation in the transmitted delta,
  preventing "layer starvation" where an entire layer receives zero selected
  coordinates in a given round.
- Activated via `Config.use_layerwise_topk = True`.
- Includes optional forensic CSV logging (`b3_layerwise_audit.csv`) recording, per
  layer per client per round: parameter count, selected count/percentage, and
  pre-/post-selection mean/max absolute value — used by the Bottleneck-3 audit (§11).

### 5.3 Importance-weighted layer-wise Top-k with Hamilton apportionment (`compress_delta_layerwise`)

A more sophisticated alternative (not the one wired into the default pipeline, but
present and usable) that allocates a **fixed total budget** `k_total` across layers
*proportionally to a supplied importance score per layer*, using the
**largest-remainder (Hamilton) apportionment method** — the same algorithm used for
apportioning parliamentary seats:
```
k_total = max(1, floor(k_ratio * total_numel))
raw_k_l = k_total * importance_l / sum(importances)     # ideal (fractional) share per layer
k_l     = floor(raw_k_l)                                 # integer floor per layer
deficit = k_total - sum(k_l)                              # how many "leftover" slots remain
# award the `deficit` remaining slots to the layers with the largest fractional remainder
sorted_by_remainder = argsort(raw_k_l - k_l, descending)
for layer in sorted_by_remainder[:deficit]: k_l[layer] += 1
```
This guarantees the exact total budget `k_total` is hit (no rounding drift) while
respecting each layer's proportional importance as closely as integer rounding
allows. If all importances are zero (degenerate case), it falls back to a uniform
`k_total // L` per layer plus round-robin distribution of the remainder.

**Layer importance itself** is computed server-side as a per-round-updated EMA of
each layer's **RMS (root-mean-square) magnitude** in the globally aggregated delta:
```python
# server.py, after aggregation, once per round:
layer_rms = ||global_delta[layer_slice]||_2 / sqrt(layer_size)     # per layer
layer_importance = beta * layer_importance_old + (1-beta) * layer_rms   # beta = layerwise_ema_beta (0.9)
```
This is computed **exactly once per round** from the single globally-aggregated
delta (not per-client), specifically to avoid what the code calls the "beta^N decay
bug" — if it were instead updated once per client, the EMA would decay by `beta^N`
extra times per round (N = number of clients), over-smoothing importance estimates
and causing every subsequent round's client budget allocations to be computed
against a stale/over-decayed importance vector.

### 5.4 Error feedback (compression residual accumulation)

```
apply_tiered_compression(...):
    if use_error_feedback:
        buf = error_buffers.get(client_id, 0)
        effective_delta = delta + 0.9 * buf              # inject 90% of last round's residual
        payload = compress(effective_delta, k_ratio)
        residual = effective_delta - reconstruct(payload)  # what got dropped/quantized away
        error_buffers[client_id] = residual                # carried into NEXT round for this client
    else:
        payload = compress(delta, k_ratio)
```
- Classic **error-feedback / memory SGD** compression correction: information
  discarded by Top-k in one round isn't lost — it's remembered in a per-client
  residual buffer and re-injected (scaled by `error_feedback_momentum=0.9`) into the
  *next* round's delta before compressing again, so that eventually every
  coordinate's cumulative signal gets transmitted even if any single round's
  magnitude was too small to make the Top-k cut.
- Buffers are explicitly **cleared to zero** whenever a client is assigned Tier 3
  (and Tier-3 exclusion from aggregation is active), specifically to prevent a stale
  residual from a much earlier active round leaking into a future round's delta
  after a long gap of inactivity.
- **Empirically disabled by default** (`use_error_feedback=False`) — the config
  comments explicitly note it was found to *decrease* accuracy at
  `k_ratio_tier2=0.05` ("EF decreases accuracy... accumulates stale residuals →
  −3.8pp" per `get_recommended_divroute_config`'s docstring).

### 5.5 Warm-up and decay schedules for `k_ratio` (`get_adaptive_k_ratios`)

```
if use_k_warmup and round < k_warmup_rounds (30):
    base_k1, base_k2 = k_warmup_tier1 (0.70), k_warmup_tier2 (0.30)   # generous early budget
else:
    base_k1, base_k2 = k_ratio_tier1 (0.20), k_ratio_tier2 (0.05)     # steady-state budget

if use_adaptive_k:
    decay = 1.0 - 0.5 * (round / (num_rounds - 1))       # linear decay to 50% of base by the end
    k1 = max(0.05, base_k1 * decay)
    k2 = max(0.01, base_k2 * decay)
else:
    k1, k2 = base_k1, base_k2
```
- **Warm-up**: for the first `k_warmup_rounds` (default 30), much larger compression
  budgets (70%/30% instead of 20%/5%) are used, on the theory that early training
  needs richer gradient signal to escape a poor random initialization quickly, and
  compression can be tightened later once the model is in a good basin.
- **Adaptive decay** (`use_adaptive_k`): a linear schedule that *shrinks* the budget
  further over the course of training, floored at 5%/1% respectively.
  **Empirically disabled by default** — config comments say "static ratios
  outperform decay schedule" (validated finding, −3.5pp when enabled).

### 5.6 Tier-3 heartbeat (staleness mitigation) (`tier3_heartbeat_payload`)

Tier-3 clients normally receive **zero** download bytes and thus keep training on
an increasingly stale copy of the global model round after round. Every
`tier3_sync_interval` rounds (default 5), instead of nothing, the server compresses
its **current global delta** (the update it just computed this round, not a
client-specific delta) at the Tier-2 ratio and sends that down as a lightweight
"heartbeat" to any Tier-3 client selected that round, partially resynchronizing
them at a fraction of the cost of a full broadcast. **Empirically found to
contribute 0.00pp over a 20-round horizon** and disabled by default
(`use_tier3_sync=False`).

### 5.7 FP16 transmission

`Config.use_fp16_upload` / `use_fp16_download`: independently toggle whether
uploaded compressed values / downloaded full model weights are cast to `float16`
before "transmission" (and back to `float32` on receipt) — a simple 2x bandwidth
reduction simulated by literal `.half()`/`.float()` casts, at the cost of numerical
precision. Download-side fp16 is simulated in `main.py` by casting the entire
global `state_dict` to half and back before handing it to clients for local
training, so the precision loss genuinely affects what clients train on, not just
what's counted in the byte ledger.

---

## 6. Local client training algorithms (`client.py`)

Each call to `FLClient.train()` reconstructs a fresh local model from the received
global `state_dict`, trains it for a number of local epochs, and returns the
resulting weights plus diagnostics. Techniques stacked into this loop:

### 6.1 Cosine-annealed learning rate schedule (`get_local_lr`)

```
get_local_lr(base_lr, min_lr, round_num, total_rounds):
    t = round_num + 1
    return min_lr + 0.5*(base_lr - min_lr) * (1 + cos(pi * t / total_rounds))
```
- Standard **half-cosine decay** from `base_lr` (round 0) down to `min_lr` (final
  round), computed as a **pure function of the global round number** (not of any
  optimizer's internal step count), so every client selected in the same
  communication round trains with the *exact same* learning rate regardless of how
  many times that client has individually been selected before — this preserves
  fairness across clients with different selection frequencies.
- Because the LR is stateless (recomputed fresh each call), the local SGD optimizer
  is *also* reconstructed from scratch on every single `train()` call — there is no
  optimizer momentum carried between rounds or between clients. This is
  intentional; the code comments explicitly call out that momentum buffers cannot
  "bleed" between clients or across rounds this way.

### 6.2 Local epoch warm-up schedule (`_get_local_epochs`, `main.py`)

```
if round < 5:   epochs = max(2, local_epochs // 2)      # half epochs, floor 2
elif round < 15: epochs = max(3, local_epochs - 1)       # local_epochs minus 1, floor 3
else: epochs = local_epochs                               # full schedule
```
Ramps up local computation over the first 15 rounds (fewer local epochs early, when
the global model is far from converged and local overfitting risk / client drift is
highest, more local epochs later). Disabled (`use_epoch_warmup=False`) for baseline
comparisons, so those always train the fixed `local_epochs` value every round.

### 6.3 SGD optimizer configuration

Standard ResNet-CIFAR training recipe: `torch.optim.SGD` with
`momentum=0.9, weight_decay=5e-4, nesterov=True`, at the cosine-scheduled LR
described above. Explicitly matched to "the standard ResNet-18/CIFAR-100 optimiser
recipe" per code comments (matches conventions from FedProx/SCAFFOLD/FedNova-style
FL research papers).

### 6.4 Gradient clipping

`nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)` applied every
mini-batch, before `optimizer.step()`, to prevent NaN/Inf weight explosions
(particularly relevant for ResNet-18 with plain SGD at a relatively high peak LR).
The clip norm (10.0) intentionally matches the server-side `grad_clip_norm` default,
so client-side and (optional) server-side clipping use a consistent scale.

### 6.5 MixUp data augmentation

```
lam ~ Beta(alpha=0.4, alpha=0.4)          # module-level constant, shared by all clients
lam = max(lam, 1 - lam)                    # enforce lam >= 0.5 (standard convention)
index = random permutation of the batch
mixed_x = lam * images + (1 - lam) * images[index]
y_a, y_b = labels, labels[index]
loss = lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
```
- **MixUp** (Zhang et al.) linearly interpolates pairs of training examples *within
  the same mini-batch* and their labels, training the network on convex
  combinations rather than raw examples — a well-established regularizer that
  improves generalization and robustness to label noise. Enforcing `lam ≥ 0.5`
  avoids an arbitrary symmetry-breaking convention issue (otherwise `(x_a, x_b,
  lam)` and `(x_b, x_a, 1-lam)` describe the same mixed example but could be
  double-counted inconsistently).
- The `alpha=0.4` Beta-distribution parameter is a fixed module-level constant
  (`_MIXUP_ALPHA`), not exposed via `Config` — every client and every run uses the
  identical value.
- Combined with **label smoothing** (`nn.CrossEntropyLoss(label_smoothing=0.1)`):
  the code notes that summing two independently-label-smoothed CE terms
  (`lam*CE(pred,y_a) + (1-lam)*CE(pred,y_b)`) is algebraically equivalent to
  smoothing the linearly-interpolated soft label directly, so the two techniques
  compose correctly without double-smoothing artifacts.

### 6.6 FedNTD — Not-True Distillation (`_ntd_loss`)

Implements the loss from "Preservation of the Global Knowledge by Not-True
Distillation in Federated Learning" (NeurIPS 2022):
```
_ntd_loss(student_logits, teacher_logits, y_a, y_b, tau):
    mask out the TRUE class(es) (y_a and, if MixUp is active, y_b) in BOTH student
    and teacher logits by setting them to -1e9 (forces their softmax prob -> 0)
    return KL( softmax(student_masked/tau) || softmax(teacher_masked/tau) ) * tau^2
```
- The "teacher" is a **frozen snapshot of the global model** (loaded fresh each
  `train()` call, `requires_grad_(False)`, `eval()` mode) — i.e. distillation
  targets are the *incoming* global model's own predictions, not some separately
  trained teacher network.
- By masking out the ground-truth class(es) from both distributions before
  computing the KL divergence, the loss specifically preserves the **relative
  ranking/calibration of the global model's beliefs about the wrong classes**
  (inter-class relationships) without fighting against the primary cross-entropy
  objective on the correct class — this is the core insight of the NTD paper: naive
  full-distribution distillation conflicts with local CE loss, but not-true-class-only
  distillation does not.
- Total loss when `ntd_beta > 0`: `loss = mixup_CE_loss + ntd_beta * ntd_loss`.
  Both the student and teacher see the *same MixUp-augmented input* so the
  distillation target is computed on the correct input distribution.
- Recommended operating point per config comments: `ntd_beta=1.0, ntd_tau=3.0` for
  CIFAR-100 (though `get_recommended_divroute_config` actually sets `ntd_beta=0.1`).

### 6.7 EMA shadow model (per-client, per-round, local only)

```
_update_ema(ema_model, live_model, decay=0.95):
    for every tensor (parameters AND buffers) in the state_dict:
        ema_tensor = decay * ema_tensor + (1 - decay) * live_tensor
```
- A **shadow/teacher-style EMA** of the local model's weights is tracked
  step-by-step (updated after every mini-batch, *after* both the optimizer step and
  the FedSparse proximal step if active) throughout local training, using decay
  `0.95` (a ~lower decay than the typical vision recipe of 0.999, deliberately chosen
  for FL's much shorter local trajectories — a comment notes it's meant to mirror
  the "standard torchvision EMA recipe" scaled down).
- Critically, **this EMA model is scoped entirely to one `train()` call** — it is
  never stored on `self` and never transmitted to the server. The code explicitly
  explains why: transmitting the EMA weights instead of the raw post-SGD weights
  would systematically dampen every aggregated update by roughly `0.64x` (given
  `beta=0.95` and ~48 local SGD steps), since the EMA model always lags behind the
  live trajectory. So EMA is purely a local diagnostic/potential-future-use
  artifact, not part of the actual upload path — the client always uploads
  `local_model.state_dict()` (the raw post-training weights), never the EMA shadow.

### 6.8 FedSparse proximal regularization (Stage 1–3, "Algorithm 2" from the FedSparse paper)

Activated when `fedsparse_lambda > 0`. Implements a three-stage scheme:

**Stage 1 — snapshot.** Before local training begins, clone the incoming global
weights per-parameter (`global_snapshot`) as the frozen "anchor" point `w_t` that
the client's weights will be regularized toward staying close to.

**Stage 2/objective.** The paper's stated objective is
`h_k(w) = F_k(w) + lambda * ||w - w_t||_1` (local loss plus an L1 penalty toward the
anchor). Rather than adding the L1 term directly into the backprop loss (which
would be non-differentiable at zero), this is implemented via a **shifted proximal
(soft-thresholding) operator** applied *after* each SGD step:
```
for every mini-batch, after opt.step():
    lr = current learning rate
    delta = param.data - w_t                       # how far this parameter has moved from anchor
    threshold = lr * lambda_j[param_name]           # per-parameter shrinkage threshold
    delta = sign(delta) * max(|delta| - threshold, 0)   # soft-threshold shrinkage toward zero
    param.data = w_t + delta                         # re-anchor
```
This is the standard proximal-gradient trick for L1 regularization: instead of
adding a subgradient of `|w-w_t|` to the loss gradient, apply soft-thresholding
directly to the post-SGD-step delta from the anchor. Any coordinate whose deviation
from the anchor is smaller than `lr * lambda_j` gets **snapped exactly to zero**
(hard sparsity), which is the mechanism that makes FedSparse's uploads naturally
compressible (many exactly-zero coordinates).

**Stage 3 — Iteratively Reweighted (IRW) per-parameter lambda_j.** Rather than
using one global `lambda` for every parameter, each named parameter gets its own
shrinkage strength `lambda_j`, computed from **that parameter's L2 deviation norm
from the previous round this client participated in** (`AU_j`, stored in a
persistent per-client buffer `_irw_norms`):
```
raw_j = _irw_norms.get(param_name, 0.0)     # from the client's PREVIOUS participation round
if all raw_j are equal (including the all-zero bootstrap case):
    lambda_j = fedsparse_lambda for all j       # uniform (same as Stage 2 alone)
else:
    range = max(raw_j) - min(raw_j)
    lambda_j = fedsparse_lambda * (1 - (raw_j - min(raw_j)) / range)   # inverse-scaled
```
Parameters that moved a lot last round (`raw_j` large) get a *smaller* `lambda_j`
this round (less aggressive shrinkage — "it's clearly still learning, don't
sparsify it as hard"), while parameters that were essentially static last round get
shrunk harder toward the anchor. This IRW logic resolves an ordering issue in the
paper's Algorithm 2 (it calls for initializing `lambda_j` *before* training using
that round's own AU norms, which would require a second forward pass) by instead
using each client's own **previous** participation round's norms — available
instantly with no extra compute. On a client's first-ever participation (bootstrap),
`_irw_norms` is empty and every `lambda_j` collapses to the uniform
`fedsparse_lambda`.

After each local training run, `_irw_norms[param_name]` is refreshed to
`||param_after_training - global_snapshot||_2` for use on the client's *next*
participation round.

**EMA/sparsity interaction fix.** Because the EMA shadow model (§6.7) exponentially
averages values *before* the proximal operator can zero them out on a later step,
naively transmitting an EMA model under FedSparse would leave small non-zero
residuals where the raw model has exact zeros, destroying the intended sparsity
pattern (this matters for communication-savings accounting even though EMA isn't
transmitted). A post-hoc mask-and-restore step forces any EMA parameter to exactly
equal the anchor wherever the corresponding raw (live) parameter equals the anchor.

**Server-side upload sparsification** (`fedsparse_sparsify_upload`, separate flag
from `fedsparse_lambda`): masks the raw delta to only the coordinates where
`|delta| > 1e-12` (i.e., relies on FedSparse's own soft-thresholding to have
produced exact zeros) and transmits those as sparse `(index, value)` pairs at 8
bytes/coordinate — the same encoding convention as DivRoute's own Top-k
compression, so byte counts are directly comparable across methods.

### 6.9 BatchNorm handling modes (`bn_mode`, see `model.py` + `client.py`)

Four modes, `Config.bn_mode ∈ {"default", "local_bn", "groupnorm", "ws_groupnorm"}`:
- **`"default"`**: standard `nn.BatchNorm2d`; running statistics are aggregated
  globally by the server via sample-weighted averaging (§4.6).
- **`"local_bn"`**: still uses `nn.BatchNorm2d`, but each client persists its own BN
  running-statistics buffer (`self._local_bn_stats`) locally across rounds
  (loaded back in at the start of each `train()` call via
  `load_state_dict(..., strict=False)`), and the **server never aggregates or
  overwrites** these buffers globally — they stay purely client-personalized. This
  is a known mitigation for BatchNorm's well-documented instability under non-IID
  FL (client-specific feature statistics get contaminated by a globally-averaged BN
  running mean/var computed across very different local distributions).
- **`"groupnorm"`**: every `nn.BatchNorm2d` in the ResNet is structurally replaced
  with `nn.GroupNorm(32, num_features)` (32 groups — chosen because ResNet-18's
  channel counts of 64/128/256/512 are all evenly divisible by 32). GroupNorm
  computes normalization statistics *within each example* (no cross-example running
  statistics at all), which sidesteps the non-IID BN instability problem entirely
  at the cost of GroupNorm's slightly different inductive bias.
- **`"ws_groupnorm"`**: additionally replaces every `nn.Conv2d` with a custom
  **`WSConv2d`** ("Weight Standardized Convolution") layer:
  ```
  w_hat = (w - mean(w, dims=[in_channels,kh,kw])) / (std(w, dims=[in_channels,kh,kw], unbiased=False) + 1e-5)
  ```
  i.e., each output-channel's convolution kernel weights are re-standardized (zero
  mean, unit variance) *at every forward pass* before the convolution is applied.
  Weight Standardization + GroupNorm is a well-known published combination ("Big
  Transfer" / Qiao et al.) that recovers much of BatchNorm's optimization benefits
  without its batch-statistics dependence, making it a natural fit for FL where
  local batch statistics are unreliable proxies for the global data distribution.

---

## 7. Server-side aggregation extras (`server.py`)

### 7.1 NaN/Inf guard

Before any aggregation logic runs, every client result whose uploaded `state_dict`
contains any NaN or Inf value in any tensor is dropped entirely from the round (with
a printed warning). If *all* clients in a round are dropped this way, aggregation is
skipped and `global_delta` is set to all-zeros for that round (no-op update). This
prevents a single numerically unstable client from corrupting the shared global
model.

### 7.2 Server momentum

```
if use_server_momentum:
    momentum_buf = server_momentum * momentum_buf + agg_delta         # classic heavy-ball momentum
    agg_delta = server_lr * momentum_buf * active_mask                # masked to touched coordinates only
```
- Standard **server-side momentum** (as in e.g. FedAvgM), accumulating the
  aggregated per-round delta into a persistent buffer with decay
  `server_momentum` (default 0.9), then scaling by `server_lr` (default 1.0)
  before applying to the global model.
- Masked by `active_mask` — the boolean tensor accumulated during the per-client
  loop marking every coordinate that received a genuinely nonzero contribution from
  at least one client this round — so momentum is not spuriously applied to
  coordinates nobody actually transmitted (which would otherwise let stale momentum
  values leak into untouched parameters).
- **Empirically disabled by default**: config comments explain `β=0.9` combined with
  `η=1.0` produces an effective step size roughly 10x larger than intended
  (`1/(1-0.9) = 10`), which was found to cause a validated −15.9 percentage-point
  accuracy regression — this is the single largest documented negative finding in
  the config's "Key findings" list.

### 7.3 Server-side gradient clipping (`server_clip_updates`)

Optional: if a single client's raw delta L2 norm exceeds `grad_clip_norm` (default
10.0), it's rescaled down to exactly that norm before compression/aggregation —
mirrors the client-side clip norm as a second line of defense against any one
client's update dominating the aggregate.

### 7.4 Client selection sampling

`select_clients` samples `clients_per_round` clients **without replacement** from
`all_ids`, with probability proportional to the current `selection_weights` vector
(§2.3), using a seeded `numpy.random.Generator` (`np.random.default_rng(seed)`)
stored on the server so sampling is fully reproducible given a fixed seed and RNG
state (also checkpointed/restored across resume, see §9).

---

## 8. Data partitioning (`data.py`)

### 8.1 Dirichlet non-IID partitioning (`get_client_datasets`)

The standard **Dirichlet-based label-skew partitioning** scheme widely used in FL
research (e.g. as in FedAvg/FedProx non-IID benchmark setups):
```
for each class c in {0, ..., num_classes-1}:
    cls_idx = shuffle(all sample indices with label == c)
    proportions ~ Dirichlet(alpha, alpha, ..., alpha)   # num_clients-dim draw, one per class
    proportions /= sum(proportions)                      # renormalize (already sums to ~1)
    counts = round(proportions * len(cls_idx))            # per-client sample count for this class
    split cls_idx into contiguous chunks of size counts[client] and hand out to each client
```
- **`alpha`** (`Config.alpha`, default 0.9) is the concentration parameter of the
  symmetric Dirichlet distribution: **lower alpha → more skewed/heterogeneous**
  partitions (each client ends up dominated by very few classes — approaching
  pathological single-class-per-client as `alpha → 0`), **higher alpha → more
  uniform/IID-like** partitions (approaching an even split as `alpha → ∞`). This is
  drawn *independently per class*, so a client's overall class distribution is the
  concatenation of `num_classes` independent Dirichlet draws, one per class —
  the standard construction from "Bayesian Nonparametric Federated Learning of
  Neural Networks" (Yurochkin et al.) and widely reused in FL benchmarks since.
- **Rounding correction**: converting continuous Dirichlet proportions into integer
  sample counts leaves a rounding remainder. For CIFAR-100 (many classes, small
  per-class counts), a **largest-remainder apportionment** is used (same Hamilton
  method as §5.3) to fairly distribute leftover samples to whichever clients were
  closest to rounding up, rather than always favoring low-index clients. For
  CIFAR-10, a simpler round-robin (`leftover_i % num_clients`) correction is used
  instead — a deliberately simpler path since CIFAR-10's larger per-class counts
  make the choice of remainder-distribution method less consequential.
- A fixed seeded `numpy.random.default_rng(seed)` drives both the per-class shuffle
  and the Dirichlet draws, so the exact same partition is reproduced for any given
  `(dataset_name, num_clients, alpha, seed)` tuple.

### 8.2 Held-out local validation split (`get_client_datasets_with_val`)

Purely diagnostic (never used in training/routing/aggregation math). After
producing the identical Dirichlet partition as §8.1, each client's shard is further
split via a **second, independent RNG** seeded `seed + client_id + 99999` (chosen
specifically to never collide with the primary Dirichlet RNG's seed) into a
`(1-val_fraction)` training subset and a `val_fraction` (default 10%) validation
subset. Validation samples are backed by a **separately loaded** `Dataset` object
that uses the no-augmentation test transform, so validation accuracy/loss is never
computed on RandomCrop/Flip/Erasing-perturbed inputs, while training subsets still
use the augmented transform Dataset. Shards smaller than 10 samples get no
validation split at all (`None`) to avoid degenerate near-empty val sets.

### 8.3 Data augmentation pipeline

Training transform (`_make_train_transform`): `RandomCrop(32, padding=4)` →
`RandomHorizontalFlip()` → `ToTensor()` → `Normalize(dataset-specific mean/std)` →
`RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3))` — a standard modern
CIFAR training recipe (crop+flip+erasing), with dataset-specific channel-wise
normalization constants for CIFAR-10 vs CIFAR-100. Test/eval transform
(`_make_test_transform`) is deterministic: `ToTensor()` → `Normalize()` only, no
augmentation. Because every client's `Subset` shares the same single underlying
`torchvision.datasets.CIFAR10/100` object (differing only by index list), all FL
methods being compared see byte-identical augmentation behavior for the same
sample indices, preserving experimental fairness across method comparisons.

---

## 9. Model architectures (`model.py`)

### 9.1 SimpleCNN (CIFAR-10)

`conv(3→16,3x3) → ReLU → MaxPool(2) → conv(16→32,3x3) → ReLU → MaxPool(2) →
flatten → fc(32*8*8→128) → ReLU → fc(128→num_classes)`. ~200K parameters, a
lightweight network suitable for fast CIFAR-10 experimentation.

### 9.2 CIFAR-adapted ResNet-18 (CIFAR-100)

Standard `torchvision.models.resnet18(weights=None)` with two modifications
standard in FL/CIFAR research (matching FedProx/SCAFFOLD/FedNova conventions):
- Replace the stem's `7x7 stride-2` conv with a `3x3 stride-1` conv, and replace
  the stem's `MaxPool` with `nn.Identity()` — this preserves the 32x32 spatial
  resolution through the stem instead of collapsing it to 8x8 before any residual
  block even runs (the default ImageNet-oriented stem is too aggressive for
  32x32-pixel CIFAR inputs and would otherwise make learning essentially
  impossible).
- Replace the final FC layer with `nn.Linear(in_features, num_classes)` to match
  the target class count (100 for CIFAR-100).
- ~11M parameters.

### 9.3 Normalization/conv variants

See §6.9 for `"groupnorm"` and `"ws_groupnorm"` `bn_mode` options, which
structurally rewrite the ResNet's normalization (and optionally convolution)
layers via a recursive `named_children()` walk-and-replace.

---

## 10. Baseline FL methods re-implemented for comparison

### 10.1 FedAvg (`fedavg_baseline_mode=True`)

Disables every DivRoute feature: `k_ratio_tier1 = k_ratio_tier2 = 1.0` (full dense
updates, no compression), `tau_low = tau_high = -1.0` (every client's divergence
trivially exceeds this so — combined with the corresponding tier-assignment
branch — every client is treated uniformly, no skipping), `gamma=1.0` (no selection
decay, uniform sampling), and turns off divergence weighting, server momentum,
error feedback, adaptive tau, and tier-3 sync. This reproduces vanilla FedAvg:
every selected client trains locally and uploads its full dense delta; the server
averages by sample count only.

### 10.2 Uniform Top-5% baseline (`uniform_top5_mode=True`)

Every selected client is forced to Tier 2 with a neutral divergence score of `0.0`
(no divergence computation, no adaptive tau, no tiering logic runs at all) and
compressed at the fixed `k_ratio_tier2` (default 0.05). This isolates **whether
DivRoute's adaptive routing intelligence outperforms naive uniform compression at
an equivalent communication budget** — i.e., it's a communication-matched control
condition for measuring the specific contribution of the routing/tiering decision,
separate from the contribution of compression itself.

### 10.3 FedSparse baseline

See §6.8 for the client-side L1-proximal training mechanism. The
`get_fedsparse_config` factory in `baselines/fedsparse_baseline.py` wires it up as:
vanilla FedAvg *downloads* (full dense global model every round, no tiering/routing)
combined with FedSparse's L1-proximal *local training* and *sparse upload
encoding* (`fedsparse_sparsify_upload=True`, 8 bytes/retained-nonzero-coordinate,
threshold `1e-4`). The docstring explicitly notes an expected and accepted
property: because each retained coordinate costs 8 bytes (vs FedAvg's dense 4
bytes/coordinate), FedSparse's upload cost will *exceed* FedAvg's whenever the
retained (nonzero) fraction is above 50% — this is intentional, to honestly
evaluate raw L1-thresholding's compressibility rather than layering on additional
encoding tricks (bitmasks, run-length encoding) that the original paper doesn't
specify.

### 10.4 FedZip baseline (two variants)

**Byte-cost formula only** (`baselines/fedzip_baseline.py: fedzip_bytes_for_delta`):
```
z = max(1, floor(z_ratio * numel))                       # top-z retained coordinates
bits_per_index = ceil(log2(max(k_clusters, 2)))            # bits needed to encode a cluster label
bytes = z*4 (float32 values) + k_clusters*4 (codebook) + ceil(z * bits_per_index / 8) (packed indices)
```
This models FedZip's published byte formula: top-z sparsification, then
**vector/scalar quantization** of the retained values into `k_clusters` discrete
levels (so each retained coordinate only needs `log2(k_clusters)` bits to identify
its cluster, plus a small shared codebook of the `k_clusters` centroid values,
rather than a full 32-bit float per coordinate).

**Actual implementation** (`baselines/fedzip_actual.py:
fedzip_compress_delta`): performs the real compression, not just the byte-cost
estimate:
```
1. Top-z sparsification: keep the k=z_ratio*numel largest-magnitude delta coordinates (torch.topk)
2. MiniBatchKMeans(n_clusters=min(k_clusters,k)).fit(retained_values.reshape(-1,1))
3. Replace each retained value with its assigned cluster's centroid (lossy scalar quantization)
4. Scatter the quantized values back into a full-size zero tensor at their original indices
```
Uses **scikit-learn's `MiniBatchKMeans`** (a scalable approximate variant of Lloyd's
k-means algorithm that processes data in small random batches rather than the full
dataset each iteration) to cluster the 1-D retained delta values into `k_clusters`
groups (default 3), and every retained value is snapped to its cluster's centroid —
this is genuine **scalar/vector quantization**, distinct from and complementary to
the sparsification step. `fedzip_actual_mode=True` runs this real quantized
reconstruction through the server's normal weighted-aggregation pipeline (in place
of the standard Top-k `apply_tiered_compression` payload), while still using the
shared `fedzip_bytes_for_delta` formula for the byte-cost ledger. The FedZip config
factory also runs `fedavg_baseline_mode=True` on the download/routing side, so
FedZip in this codebase is specifically an *upload-compression-only* baseline
(comparable in scope to FedSparse's upload-only sparsification, and to DivRoute
itself which is also upload-only by default).

---

## 11. "Bottleneck 3" layer-starvation forensic audit tooling

A family of standalone scripts built to diagnose *why* CIFAR-100/ResNet-18
DivRoute runs were losing accuracy relative to FedAvg — specifically testing the
hypothesis that **global Top-k sparsification structurally starves certain layers**
of any representation in the compressed update (see §5.1 vs §5.2). Not part of the
core training loop; run as separate analysis passes over instrumented CSV logs
produced by `compress_delta`/`compress_layerwise_independent`'s optional forensic
logging blocks (`b3_layer_starvation_audit.csv`, `b3_layerwise_audit.csv`).

- **`run_audit.py` / `run_b3_ab_experiment.py`**: run matched A/B training
  experiments (global Top-k vs. layer-wise independent Top-k) while forensic CSV
  logging is enabled, to produce comparable audit data for both compression
  strategies under identical conditions.
- **`run_b3_layerwise_audit.py`**: focused run specifically auditing the
  layer-wise-independent-Top-k path.
- **`analyze_audit.py` / `analyze_layerwise_audit.py`**: post-hoc analysis scripts
  that read the CSV logs and compute the actual **layer starvation diagnostics**:
  for each layer, what fraction of its parameters were ever selected by Top-k
  across the audited rounds, whether any layer's `selected_pct` collapses to
  near-zero (evidence of starvation), and how each layer's `allocation_ratio`
  (its share of selected coordinates divided by its share of total parameters —
  computed inline during `compress_delta`'s forensic logging as `selected_share /
  parameter_share`) compares to the "fair" ratio of 1.0 (a layer receiving exactly
  its proportional share of the sparsification budget).

The per-coordinate forensic logging embedded directly in `compress_delta` (global
Top-k path) records, per layer per client per round: parameter count, selected
count/percentage, each layer's share of total parameters vs. its share of total
selected coordinates, the resulting `allocation_ratio`, and pre-/post-selection
mean/max absolute magnitude — enough data to reconstruct, after the fact, exactly
how unevenly the global magnitude-based Top-k selection was distributed across the
network's layers, and thereby determine whether layer-wise independent Top-k
(§5.2) is a necessary fix.

---

## 12. Diagnostics / instrumentation layer (`diagnostics.py`, "Parts 1–6")

All gated behind individual `Config` boolean flags, all **read-only / non-invasive**
(never modify weights, gradients, routing decisions, or aggregation math) — the
module docstring explicitly guarantees existing experiments remain byte-identical
when all flags are `False`.

- **Part 1 — Gradient quality diagnostics** (`compute_gradient_diagnostics`): per
  client, computes `update_l2_norm` (`||w_local - w_global||_2`),
  `relative_update_norm` (normalized by `||w_global||_2`), and (when a compressed
  reconstruction is available) `compression_error` (`||Δ_orig - Δ_compressed||_2`),
  `retention_ratio` (`||Δ_compressed||_2 / ||Δ_orig||_2`), and `compressed_cosine`
  (directional similarity between the original and compressed delta).
- **Part 2 — Alternative divergence metrics**: see §1.2.
- **Part 3 — Routing quality evaluation**: accumulates a persistent module-level
  list of per-round, per-client rows (`_routing_quality_rows`), then computes
  **Pearson and Spearman correlations** (`compute_routing_correlations`) between
  divergence score and each of local training accuracy, local loss, and update
  norm, for the current round's cohort — used to empirically validate whether the
  divergence signal is actually predictive of anything meaningful (or, per a
  code comment elsewhere in `main.py`, to check whether it might just be a
  disguised transformation of raw update norm).
- **Part 4 — Tier contribution analysis** (`compute_tier_contributions`): sums raw
  and compressed delta L2 norms **per tier**, and computes each tier's percentage
  contribution to the total aggregated update norm both before and after
  compression — quantifies how much of the model's actual movement each round is
  attributable to Tier 1 vs Tier 2 vs (if included) Tier 3 clients.
- **Part 5 — Compression analysis** (`compute_compression_analysis`): per-tier
  averages of original norm, compressed norm, reconstruction error, and retention
  ratio — a tier-level rollup of Part 1's per-client statistics.
- **Part 6 — Adaptive tau diagnostics** (`compute_tau_diagnostics`): logs `mu`,
  `sigma`, `tau_low`, `tau_high`, the raw score spread (`max - min`), the
  coefficient of variation (`sigma / |mu|`), per-tier client counts, and the
  10th/25th/50th/75th/90th percentiles of the current round's score distribution.

`main.py` additionally implements its own **inline routing-dynamics diagnostics**
(the `_diag_hist` / "[DIAG]" print blocks, separate from the `diagnostics.py`
module) tracking, round over round: raw-vs-EMA score standard deviations,
within-round Spearman rank correlation between raw and EMA divergence rankings
(explicitly re-derived at one point per an in-code "BUG FIX" comment, correcting an
earlier version that accidentally compared *across* rounds instead of *within* a
round — a bug that could spuriously produce ±1.0 correlations from as few as 2
overlapping clients), cross-round tier-retention rates per tier, Pearson/Spearman
correlation between divergence and both update norm and local loss, and a set of
internal consistency assertions (e.g. that the T1/T2/T3 sets are pairwise disjoint
and their union equals the full selected cohort) that print warnings — but
deliberately never raise/crash — if violated.

---

## 13. Checkpoint / resume system (`main.py`)

Every 5 completed rounds, a full checkpoint is atomically written (`torch.save` to a
`.tmp` file, then `os.replace` for atomicity) capturing: the global model's
`state_dict`, the server's selection weights / global delta / RNG bit-generator
state / momentum buffer, every client's FedSparse IRW norm buffer
(`_irw_norms`), the main loop's EMA score dict, progress-guarantee last-promoted
round tracker, compression error-feedback buffers, cumulative communication-volume
counters (both DivRoute's and the FedAvg baseline's, for accurate savings-percentage
reporting on resume), the Phase-5 `_previous_global_delta` and rolling
`_tau_score_history` state, the full logger history, and Python/NumPy/CPU-Torch/CUDA
RNG states (for exact reproducibility across a resume). A separate permanent
milestone checkpoint is additionally saved every 50 rounds (not overwritten by later
rounds). On resume, extensive compatibility checks (`method_name`, `dataset_name`,
`model_name`, `seed`, and several `Config` fields) raise explicit errors on any
mismatch rather than silently continuing with an incompatible checkpoint, and
several RNG-state-restoration branches are wrapped in per-component `try/except`
so a version-related failure to restore (e.g.) CUDA RNG state doesn't block the rest
of the resume.

---

## 14. Summary of key empirically-validated findings (from `Config` docstrings)

These are not algorithms per se, but documented experimental conclusions baked into
the recommended default configuration (`get_recommended_divroute_config`), useful
context for interpreting *why* certain features default to disabled despite being
fully implemented:
- `use_server_momentum=False`: β=0.9 with η=1.0 → effective ~10x step size →
  **−15.9 percentage points** accuracy regression.
- `use_error_feedback=False`: at `k_ratio_tier2=0.05`, error feedback accumulates
  stale residuals → **−3.8pp**.
- `use_adaptive_k=False`: late-round linear ratio decay harms convergence →
  **−3.5pp**.
- `use_tier3_sync=False`: the Tier-3 heartbeat contributes **0.00pp** over a
  20-round horizon (i.e., not worth its bandwidth cost at that horizon).
- Validated recommended operating point at CIFAR-10/α=0.9/20 rounds: **~64.1%
  accuracy** with **~84% bidirectional communication savings** vs. FedAvg.
