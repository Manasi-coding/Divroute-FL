# DivRoute-FL — 80–85% Accuracy Plan (CIFAR-100, Publication-Grade)

> Companion document to `DIVROUTE_ALGORITHMS_MASTER_DOC.md` (the algorithm reference).
> This document is the **decision record and execution plan**: what's wrong, what's
> changing, exactly which files/flags are touched, what's guaranteed vs. not, and why
> each change is necessary. Written so a fresh reader — or a fresh LLM session — can
> pick this up with no other context and know exactly what to do and why.

**Status (concluded): §5's infrastructure changes are implemented and
smoke-tested. §9's centralized sanity check has a first data point (74.63%
after 1 epoch; full 15-epoch run never attempted). §6's three companion-run
configs went through seven attempts total — two invalid (§11: tau-threshold
scale mismatch; §12: k-ratio warmup silently overriding every method's real
compression ratio), then five structurally valid ones (§13-§17) that
progressively closed every confound and finally answered the project's core
question. §17 is the concluding result.**

**§17's verdict, matching DivRoute against Uniform-Top5% at identical round
count, training schedule, and regularizer — the fair comparison this
project's own §6 design always called for:**

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| **Uniform-Top5% (32 rounds)** | **81.43%** | **794.04MB** | **8734.49MB** |

Uniform matches DivRoute's accuracy (a 0.15-point gap, within run-to-run
noise) while using 61% fewer upload bytes and 12.5% fewer total bytes —
winning or tying on every axis, with none of DivRoute's routing machinery.
**The mechanism-specific claim this project set out to demonstrate — that
divergence-based routing outperforms naive uniform compression — is not
supported by the evidence gathered here.**

What still stands: the 80-85% accuracy target is met (all three methods
land at or near it), and communication-efficient fine-tuning of a
pretrained backbone genuinely approaches dense FedAvg's accuracy (both
DivRoute and Uniform land within ~0.5-0.65 points of FedAvg's 82.08% on a
fraction of its upload bytes, per §16). What does not hold is that this
benefit is specific to DivRoute rather than to compression-plus-more-rounds
generally — Uniform gets there more cheaply with no routing at all.

This conclusion was earned, not assumed: three real bugs were found and
fixed before any of it could be trusted (§3.1's Tier-3-exclusion fix; §11's
tau-threshold scale mismatch; §12's k-ratio warmup override), and two
specific hypotheses for DivRoute's earlier shortfall were tested rather
than guessed at (§14's epoch-warmup confound, confirmed but insufficient;
§14's tier-polarity inversion, refuted outright). See §17 for the full
account.

---

## 1. Where this comes from

This plan is the output of a diagnostic pass over the live CIFAR-100 / ResNet-18
DivRoute run (`checkpoints/divroute_phase5_revised_cifar100_seed42`, "Phase 5",
bn_mode=groupnorm, no server momentum, no error feedback, adaptive-tau/percentile
routing), which had plateaued at **32.80% accuracy at round 274/300**
(growth rate ~0.06–0.07pp/round in the final 30 rounds — a genuine plateau, not an
undertrained curve waiting on more rounds). Target: **80–85% accuracy, CIFAR-100
only, claim must remain defensible.**

Everything below was cross-checked against `DIVROUTE_ALGORITHMS_MASTER_DOC.md` (the
project's own algorithm reference) before being finalized, because that document
surfaced two findings that **directly contradict recommendations made earlier in
this planning process** — see §3. Read that section before acting on anything else
in this file.

---

## 2. The constraint: what "defensible" actually means here

The project's own README states the headline claim precisely (§3, §6.10 of the
README): the evidence figure is `comm_vs_accuracy.png` — **Full FedAvg / Uniform
Top-5% / DivRoute-FL, three curves, same axes** — and the claim is that DivRoute
reaches **the same accuracy as the baselines at lower cumulative MB**.

This means:
- The claim is **relative** (iso-accuracy at lower bandwidth vs. baselines run under
  identical conditions), not an absolute accuracy target. 80–85% for DivRoute alone
  proves nothing; 80–85% for DivRoute at a fraction of the bytes of an 80–85%-accuracy
  FedAvg baseline is the actual result.
- **Any change made here is safe for the claim if and only if Full FedAvg and
  Uniform Top-5% are re-run under the exact same new conditions** (same backbone,
  same init, same dataset, same rounds/participation). Reusing old CIFAR-10/SimpleCNN
  archived baseline runs against a new CIFAR-100/pretrained-backbone DivRoute run
  would be an invalid comparison.
- The core mechanism being claimed — divergence formula, tier routing, top-k
  dispatch, honest byte accounting, γ-decay selection (all in
  `divroute_fl/mechanism.py` / `compression.py`) — must not change. Everything in
  this plan is either (a) which backbone/dataset conditions the mechanism runs under,
  or (b) fixing a specific mechanism bug (§3.1), never a change to what the mechanism
  fundamentally does.

---

## 3. Corrections to the prior plan (read this before anything else)

Two things recommended earlier in this conversation are **contradicted by this
project's own already-collected experimental evidence**, documented in
`DIVROUTE_ALGORITHMS_MASTER_DOC.md` §14 and cross-checked directly against
`main.py`. Both are corrected below rather than carried forward silently.

### 3.1 Tier-3 exclusion — the direction was right, the reasoning was wrong

**Verified independently at `divroute_fl/main.py:755–760`:**
```python
if np.isnan(d_for_tier) or np.isinf(d_for_tier):
    tier = 1
elif d_for_tier <= config.tau_low:
    tier = 1        # LOW divergence -> Tier 1 (gets the MOST bandwidth, k_ratio=0.20, always included)
elif d_for_tier <= config.tau_high:
    tier = 2
else:
    tier = 3         # HIGH divergence -> Tier 3 (EXCLUDED: 0 bytes, agg_weight=0.0)
```
This is the **opposite polarity** from the README's plain-language description
("Tier 1: client has drifted far → gets top 20%, drifted clients need a strong
correction signal"). In the actual live loop, it's the *most-aligned* clients that
get the most bandwidth, and the *most-divergent* clients — the ones whose local
data is most different from the current global model — that get zeroed out
entirely, every round, for the full 300-round run.

In a 100-class, non-IID (Dirichlet α=0.9) setting, the highest-divergence clients on
any given round are disproportionately likely to be the ones holding locally
under-represented classes or unusual local distributions relative to the current
global model. Structurally silencing that subset for 300 consecutive rounds is a
direct, mechanistic contributor to a low accuracy ceiling on a 100-class task — this
is a sharper and more specific causal story than "some clients are being wasted,"
and it's the corrected version of the diagnosis given earlier in this conversation.

**The fix is unchanged**: `include_tier3_in_aggregation=True` (or a soft, non-zero
weight instead of a hard cutoff). What's corrected is *why* it matters — it's not
about giving drifted clients a "correction signal," it's about no longer discarding
the clients carrying the most locally-distinct information every single round.

### 3.2 Error feedback and server momentum — retracted as default recommendations

An earlier turn in this planning process recommended turning on
`use_error_feedback=True` and `use_server_momentum=True` as accuracy fixes, based on
general FL literature (where both techniques are usually net-positive). That
recommendation was made **without visibility into this project's own prior
experiments**, which are documented in `DIVROUTE_ALGORITHMS_MASTER_DOC.md` §14:

| Flag | This project's validated finding | Root cause documented |
|---|---|---|
| `use_server_momentum=True` | **−15.9 percentage points** | β=0.9 combined with η=1.0 → effective step size ≈ 10× too large (`1/(1-0.9)=10`) — a hyperparameter scaling bug, not evidence against momentum in general |
| `use_error_feedback=True` | **−3.8 percentage points** | At `k_ratio_tier2=0.05`, the compression residual accumulates faster than it's cleared, becoming stale rather than corrective |
| `use_adaptive_k=True` (late-round ratio decay) | **−3.5 percentage points** | Already correctly off in the live run — no change needed |
| `use_tier3_sync=True` (heartbeat) | **0.00 percentage points** over 20 rounds | Not worth its bandwidth cost at that horizon — low priority either way |

**Correction: both `use_error_feedback` and `use_server_momentum` stay at their
currently-validated default (`False`) in this plan.** They are not part of the path
to 80–85%. If they're revisited later, it must be as a **separate, explicitly-labeled
experiment** — e.g. momentum re-tuned with `server_lr≈0.1–0.15` alongside
`server_momentum=0.9` to correct the effective-step-size bug — run and validated on
its own before being folded into any headline number. Nothing in this plan depends
on either flag changing.

This project's own §14 also states the empirically validated ceiling for the
mechanism's best-tuned recommended configuration: **~64.1% accuracy at ~84%
bidirectional communication savings on CIFAR-10, α=0.9, 20 rounds.** That's the
*easier* 10-class dataset, at only 20 rounds, with the mechanism at its own
best-known settings. This number is the strongest available evidence — internal to
this project, not just external literature — that no amount of mechanism-level
hyperparameter tuning gets a from-scratch model to 80–85% on the harder, 100-class
CIFAR-100 task. It directly motivates §4.

---

## 4. The recommended path: pretrained backbone, federated fine-tuning

Given §3.2's internal ceiling data (~64% best case on the *easier* dataset,
from-scratch), and given CIFAR-100 is locked as the dataset, **the only lever with a
realistic, evidence-backed shot at 80–85% is starting from ImageNet-pretrained
weights and framing training as federated fine-tuning rather than federated
learning-from-scratch.** This is compatible with the defensibility constraint in §2
as long as all three compared methods (DivRoute / FedAvg / Uniform Top-5%) share the
identical pretrained starting point.

**Backbone: EfficientNet-B0 (ImageNet-pretrained).**
- **Sourced, corrected numbers** (an earlier draft of this section cited an
  unverified "88–91%" range that turned out to conflate EfficientNet-B0 with
  EfficientNet-B7 — corrected here):
  - The original EfficientNet paper (Tan & Le, ICML 2019, Table 5) reports
    **88.1%** for EfficientNet-B0 (4M params) on CIFAR-100 transfer learning, and
    **91.7%** for EfficientNet-B7 (64M params) — the 91.7% figure is B7's, not B0's.
    [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
  - An independent, hands-on replication reports **81.79% test / 82.3% validation**
    for EfficientNet-B0 on CIFAR-100, using a *frozen* backbone (linear-probe only,
    not a full fine-tune) at 224×224 (upscaled from 32×32), 15 epochs.
    [Towards Data Science — CIFAR-100: Transfer Learning using EfficientNet](https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2/)
  - **Honest reading**: the real centralized ceiling for EfficientNet-B0 on
    CIFAR-100 sits somewhere in roughly an **82–88% band**, depending heavily on
    whether the backbone is fully fine-tuned or frozen, and on training budget —
    not a confident single number. The 88.1% figure is the best-case, presumably
    well-resourced reference point; 82% is closer to what a lightly-tuned
    replication achieves even without any federated-learning cost added on top.
- Federated degradation (partial participation, compression, non-IID) costs
  additional points on top of whichever end of that band this pipeline lands near
  — which is exactly why §9 below recommends measuring the *centralized* ceiling
  for this specific pipeline before running the full federated experiment, rather
  than assuming a literature number transfers directly.
- **The comparison against ResNet-50 and MobileNetV2 is weaker than stated above and
  is corrected here rather than left standing:**
  - The only sourced ResNet-50-on-CIFAR-100 transfer data point found is **66.3%**
    fine-tuning accuracy — but that study fed **native 32×32 CIFAR images directly
    into ResNet-50 with no upscaling**. This is very likely the same
    resolution-collapse failure mode this project's own `model.py` explicitly
    documents and fixes for the from-scratch ResNet-18 (a stem designed for 224×224
    input destroys spatial detail on a 32×32 image before any residual block runs).
    **This number should not be read as ResNet-50's transfer ceiling** — it's better
    read as evidence for why §5.2's resize step is not a minor detail: skipping it
    appears capable of costing 20+ accuracy points on its own, independent of any
    federated-learning effect.
    [Balaji Kulkarni — Transfer Learning with ResNet, Pre-trained ResNet50 with CIFAR100](https://balajikulkarni.medium.com/transfer-learning-using-resnet-e20598314427)
  - **No reliable CIFAR-100-specific transfer number for MobileNetV2 was found** in
    sources checked — only CIFAR-10 figures turned up (a different, easier task).
    The "~78–83%" originally stated here had no support and is removed rather than
    re-cited.
  - **Net effect**: the backbone choice should rest on EfficientNet-B0's own sourced
    82–88% band (above) and its known parameter-efficiency advantage, not on a
    confident numeric comparison against the alternatives — those numbers aren't
    reliably available. If ResNet-50 or MobileNetV2 are tried later as secondary
    comparison points, their accuracy needs to be *measured* on this pipeline, not
    assumed from an uncited literature range.
- Smaller parameter count than the current ResNet-18 (~5.3M vs. ~11.2M), which is
  more consistent with the project's own "honest byte counting" efficiency framing
  (`DIVROUTE_ALGORITHMS_MASTER_DOC.md` §5.1) than a larger backbone would be — the
  same top-k ratios compress a smaller model more gracefully.

**Keep pretrained BatchNorm — do not run this backbone through the existing
`bn_mode="groupnorm"` swap for the main run.** α=0.9 is mild non-IID (GroupNorm's
advantage is largest under severe skew); overwriting pretrained BN running stats
with freshly-initialized GroupNorm discards calibrated information for no benefit
in this regime. The existing `_replace_bn`-style swap logic (`model.py`, §6.9/§9.3
of the algorithm doc) is architecture-agnostic and will work unmodified on
EfficientNet-B0's BatchNorm2d layers if this is later run as a secondary ablation —
just not as the main result.

---

## 5. Exact changes required

### 5.1 `divroute_fl/model.py`
- Add a pretrained-backbone path to `get_model()`: load
  `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`,
  replace the classifier head (`model.classifier[1] = nn.Linear(in_features, num_classes)`),
  leave BatchNorm layers untouched (do not route through `_replace_bn` for the main run).
- Do not remove or modify the existing `simplecnn` / `resnet18` paths — they remain
  the from-scratch comparison point for the Ablation Study (§6).

### 5.2 `divroute_fl/data.py`
- Add a `Resize` step to both the training and test transform pipelines, gated on
  the new backbone selection — CIFAR's native 32×32 is too small for
  EfficientNet-B0's stem/downsampling stages (the same class of problem the
  existing ResNet-18 stem fix in `model.py` already solves for that architecture,
  documented in §9.2 of the algorithm doc — different fix needed here since we're
  *keeping* the pretrained stem intact, not modifying it).
  Target resolution: 96×96 or 128×128 as a practical starting point given local
  compute (Tesla T4 in the live run); 224×224 is the literature-standard choice if
  wall-clock budget allows — this is a tunable, not a hard requirement of the plan.
- **Switch normalization constants to ImageNet mean/std
  (`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`) when the pretrained backbone is
  active**, instead of the current CIFAR-100-specific constants. This must be
  conditional on backbone choice, not hardcoded by dataset name — the existing
  CIFAR-native from-scratch paths (SimpleCNN, ResNet-18) must keep their original
  transform behavior unchanged for reproducibility of already-collected results.
  This is an easy detail to miss and directly costs transfer-learning accuracy if
  skipped — pretrained features are calibrated to ImageNet's normalization.

### 5.3 `divroute_fl/config.py`
- New flags: a backbone/pretrained selector, `image_size`, and a normalization-mode
  switch (auto-derived from backbone choice rather than a separate manual flag, to
  avoid the two getting out of sync).
- Lower the fine-tuning learning rate regime: current `local_lr=0.1` with cosine
  decay to near-zero is tuned for training a randomly-initialized backbone from
  scratch and will damage pretrained features if reused as-is. Recommend a lower
  base (~0.01) with a **floor** on the cosine schedule rather than decaying to ~0 —
  `get_local_lr`'s existing half-cosine formula (`mechanism.py` / `client.py`, §6.1
  of the algorithm doc) already supports a `min_lr` floor parameter; just needs a
  non-negligible value for this regime instead of the from-scratch default.
- `use_error_feedback` and `use_server_momentum` stay `False` (§3.2 — do not change
  these as part of this plan).
- `include_tier3_in_aggregation = True` (§3.1).
- `k_ratio_tier1=0.20` / `k_ratio_tier2=0.05` **unchanged** — keeping the exact same
  compression budget is the strongest form of defensibility: it proves the
  mechanism holds up at the *same* aggressive compression, now on pretrained
  features, rather than quietly loosening compression to buy accuracy.

### 5.4 `divroute_fl/main.py`
- Wire through the new config flags.
- **No resume from the existing ResNet-18-from-scratch checkpoint.** This must be a
  fresh run with a new `run_label`/checkpoint directory. This isn't a workaround —
  the checkpoint/resume system's own compatibility guard (`main.py`, §13 of the
  algorithm doc) already validates `model_name` and several `Config` fields on
  resume and will correctly refuse an incompatible resume; a fresh run is the
  expected and only correct path here.

---

## 6. Required companion runs (non-negotiable for defensibility)

Using the project's own existing baseline factories (`DIVROUTE_ALGORITHMS_MASTER_DOC.md`
§10.1/§10.2 — no new baseline code needed, just point them at the new backbone):

| Run | Config | Purpose |
|---|---|---|
| **DivRoute (main result)** | Pretrained EfficientNet-B0, Tier 3 included, fine-tuning LR, k_ratios unchanged | The headline number |
| **Full FedAvg** | `fedavg_baseline_mode=True`, *same* pretrained EfficientNet-B0 init, same rounds/participation | Iso-condition accuracy ceiling reference |
| **Uniform Top-5%** | `uniform_top5_mode=True`, *same* pretrained EfficientNet-B0 init, same rounds/participation | Communication-matched control — isolates routing intelligence from compression itself |

All three **must** originate from an identical fresh pretrained checkpoint, same
dataset, same round budget. This is what makes the `comm_vs_accuracy.png` comparison
valid for the new setting.

**Existing Phase-5 stripped-down runs (CIFAR-100, ResNet-18 from scratch, no EF, no
momentum) are not discarded** — they become the **Ablation Study** table/section,
showing the routing mechanism's contribution in isolation from pretraining. This is
standard structure for this kind of paper and uses data you already have.

---

## 7. Guarantees vs. non-guarantees

**Guaranteed:**
- The mechanism being claimed (divergence formula, tier routing, top-k dispatch,
  byte accounting, γ-decay) is byte-for-byte the same code path as before — only the
  backbone/init and the Tier-3 aggregation flag change.
- If §6's three runs are executed as specified (identical pretrained init across
  all three), the resulting comparison is fair and defensible regardless of what
  accuracy number comes out.
- `use_error_feedback`/`use_server_momentum` staying off means no risk of
  reintroducing the documented −15.9pp / −3.8pp regressions.

**Not guaranteed — stated plainly:**
- **Hitting exactly 80–85% is not guaranteed by this plan.** EfficientNet-B0
  pretrained gives the best realistic shot based on (a) transfer-learning
  literature's centralized ceiling for this backbone and (b) this project's own
  internal ~64% ceiling for the from-scratch mechanism on an easier dataset — but no
  one can certify a precise accuracy number for a not-yet-run combination. This must
  be confirmed empirically.
- A retuned server momentum (correcting the effective-step-size bug) is a plausible
  further lever but is explicitly **not** included here — it needs its own
  validation run before being trusted, per §3.2.
- The exact image resolution (96 vs. 128 vs. 224) is a compute/accuracy tradeoff
  that hasn't been empirically tuned for this pipeline yet.
- **§5.2's resize step is a bigger risk than a tradeoff — it may be closer to a
  pass/fail gate.** The only sourced data point available (a ResNet-50 CIFAR-100
  transfer run that skipped upscaling and fed native 32×32 input) landed at 66.3%,
  vs. 82–88.1% for properly-resized EfficientNet-B0 runs — a ~20-point gap
  plausibly attributable to resolution handling alone, before any federated-learning
  cost is even added. Getting this one implementation detail wrong could plausibly
  cost more accuracy than every other lever in this plan combined.
  **Update (§9): a first empirical data point is reassuring here — 1 epoch of
  centralized fine-tuning through this exact resize pipeline already reaches
  74.63% test accuracy, nowhere near the 66.3% failure mode above. Not
  conclusive (single epoch, not the full centralized run), but a real signal
  the resize step is not silently broken.**

---

## 8. Why this is necessary (synthesis)

- CIFAR-100 is locked in (explicit requirement) and 80–85% is a hard requirement
  (explicit requirement) — together these rule out from-scratch training under the
  current protocol: this project's own §14 ceiling (~64% on the *easier* CIFAR-10 at
  only 20 rounds, best-tuned) makes clear that no combination of this mechanism's own
  hyperparameters closes a 20+ point gap on the harder 100-class task.
- The claim must stay defensible (explicit requirement): the actual claim, per the
  project's own README, is relative (iso-accuracy at lower bandwidth vs. baselines),
  which means the fix doesn't need to preserve any particular accuracy number — it
  needs to preserve the *comparison structure*. A pretrained backbone satisfies this
  as long as all three compared methods share it, which is why §6 is written as a
  non-negotiable requirement rather than a suggestion.
- The two mechanism-level flags that looked like free wins (error feedback, server
  momentum) are ruled out specifically because this project already ran that
  experiment and documented a negative result — proceeding anyway would mean
  ignoring the project's own prior evidence in favor of generic literature
  intuition, which is precisely the kind of thing a reviewer (or a future version of
  this document) should be able to catch.

---

## 9. Centralized sanity check — results

`scripts/centralized_finetune_sanity_check.py` implements the check this
document's Status line and §7 both call for: a plain, non-federated
fine-tune of the pretrained EfficientNet-B0 + CIFAR-100 pipeline (reuses
data.py's exact resize/ImageNet-normalisation transform and model.py's
loader), run before spending compute on the federated companion runs,
specifically to isolate "is this pipeline sound at all" from "does
federation hurt it further."

**Results so far:**
- `--smoke` (1 epoch, 512-image subset): full pipeline runs end-to-end in
  ~10s on the local GPU (RTX 3050 6GB, CUDA 11.8, AMP) — confirms nothing in
  model/data loading is broken.
- 1 full epoch over the entire 50,000-image training set: **74.63% test
  accuracy**, 246.5s wall-clock, no OOM at batch_size=64.

**What this does and doesn't establish:**
- It directly addresses §7's biggest identified risk — the resize step. If
  that pipeline were silently broken the way the uncited ResNet-50/32×32
  comparison point in §4 would predict, 1 epoch would not reach 74.63%.
- It is a *single-epoch* number, not the full centralized ceiling this
  section is meant to establish. The literature comparison points in §4
  (82–88% band) are typically measured after 15+ epochs; the full 15-epoch
  centralized run has not been executed (deliberately deferred — see
  DIVROUTE_REMAINING_WORK.md).
- It says nothing about the federated ceiling by itself. Federated training
  sees less data per round (each client's local shard, not the full
  dataset), partial participation, and (for DivRoute/Uniform) lossy
  compression — all of which cost accuracy relative to this centralized
  number. §10 covers what's known about the federated cost so far.

## 10. Companion-run infrastructure, a critical fix, and timing calibration

§6 requires three runs (DivRoute / Full FedAvg / Uniform Top-5%) sharing an
identical pretrained init. Only DivRoute had a dedicated factory
(`get_pretrained_finetune_config()`); the other two are now implemented in
`config.py`:
- `get_fedavg_pretrained_config()`
- `get_uniform_top5_pretrained_config()`

Both mirror `get_pretrained_finetune_config()`'s backbone/dataset/LR
overrides, so the only difference between the three runs is the
routing/compression mechanism under test, not the model, data pipeline, or
optimiser regime.

**A critical fix was required for FedAvg, not just a config wrapper.**
`fedavg_baseline_mode=True` forces `tau_low=tau_high=-1.0` (main.py). Every
real divergence score is >= 0, so under the tier-assignment polarity
documented in §3.1, this puts *every* client in Tier 3, every round.
`server.aggregate()` drops Tier-3 clients from aggregation unless
`include_tier3_in_aggregation=True`. Without that flag, "FedAvg" would
silently aggregate nothing, every round — the global model would never
update, and the run would produce a flat, near-random accuracy curve that
could easily be misread as "FedAvg also fails on CIFAR-100" rather than
"this baseline was never actually training." `get_fedavg_pretrained_config()`
now sets this flag by construction. A small-cohort smoke test can mask this
bug: main.py's Progress Guarantee mechanism promotes up to 2 clients/round
to Tier 2 regardless of this flag, so a smoke test with clients_per_round<=2
looks fine either way — the bug is only visible at realistic
clients_per_round (verified against this project's own prior corrected run,
`logs/diag20_fedavg_corrected_cifar100.json`, at clients_per_round=20).
Uniform Top-5% has no equivalent hazard: `uniform_top5_mode` hardcodes every
client to Tier 2 directly, bypassing tier assignment entirely.

All three configs were smoke-tested together (2 rounds, 4 clients, 2/round)
through the real `divroute_fl.main.run()` loop: no crashes, and FedAvg's
accuracy climbed round-over-round (37.98% → 52.84%) with a non-zero
aggregation weight for every participating client — confirming the fix
above actually works, not just that the code runs without error.

**Timing calibration (full scale: 25 clients, 15/round, DivRoute pretrained
config):** 2 rounds at this run's early-warmup `local_epochs=2` took 1658.4s
(827s/round). `use_epoch_warmup` ramps `local_epochs` 2 → 4 → 5 across a
20-round run (75 "epoch-rounds" total vs. the 4 measured here), extrapolating
to **~8.6 hours for a single 20-round DivRoute run**, with FedAvg/Uniform
somewhat cheaper (no NTD teacher forward pass) at an estimated ~7h each —
roughly a day of GPU time for all three sequentially. This is far more than
the from-scratch ResNet-18 runs cost, because EfficientNet-B0 at 128×128 is
a heavier per-example forward/backward pass than the original SimpleCNN/
ResNet-18-at-32×32 setup this project's round budgets were originally tuned
against.

Checkpoints save every 5 rounds (`main.py`, `completed_round % 5 == 0`), so
any of these runs can be safely killed and resumed — at most ~1-2 hours of
progress at risk, not the whole run. `run_pretrained_companion_experiments.py`
launches all three (or one, via `--only`) with a shared seed/num_rounds.

**Not yet decided:** the actual `num_rounds` for the real companion runs (10
vs. 20 vs. other — see DIVROUTE_REMAINING_WORK.md item 4) and whether the
runs execute locally or on a faster remote GPU. None of the three companion
runs have been executed at real scale yet — only the 2-round timing
calibration above, which is not itself a real data point (its log output
was discarded to avoid it being confused with the eventual real result).

**Open methodology question, not yet acted on:** `get_pretrained_finetune_config()`
inherits `ntd_beta=0.1` (Not-True Distillation, a local-training regularizer)
from `get_recommended_divroute_config()`'s base. Neither new baseline
factory sets it, so FedAvg/Uniform train with `ntd_beta=0.0`. This asymmetry
is pre-existing (inherited from those factories' already-established
from-scratch behaviour, not introduced alongside the fixes above), and NTD
is a local-training regularizer rather than part of "the mechanism" §2
protects (divergence formula / tier routing / top-k dispatch / byte
accounting / gamma-decay) — but it does mean DivRoute gets an extra
regularizer the baselines don't, which is worth a conscious decision before
the real companion runs rather than a silent carry-over.

## 11. Second critical fix: tau-threshold scale mismatch (found by actually running the companion experiment)

This was found by running the real 10-round DivRoute companion experiment on
Kaggle (see §10), not by inspection — every one of the 10 rounds logged
"Actual tier counts: 15/0/0": all 15 selected clients landed in Tier 1,
every single round, for the entire run.

**Root cause:** `threshold_mode` defaults to `"adaptive_tau"`, with
`tau_low=0.01`/`tau_high=0.02` and `tau_smoothing=0.8` (an EMA that steps
only 20% of the way toward the real score distribution each round). All
three were tuned for from-scratch ResNet-18 training, where divergence
scores run much larger. A pretrained EfficientNet-B0 fine-tune's scores are
roughly two orders of magnitude smaller (observed ~1e-4 in the real run,
vs. the 0.01-scale defaults). At a 20%/round EMA step, `tau_low` needs on
the order of 20 rounds just to decay down to that scale — so within a
10-round (or even a full 20-round) budget, every client's score stays below
`tau_low` for the entire run, every client is classified Tier 1, and
DivRoute's Tier 2/3 compression differentiation never activates.

This wasn't merely inert — it actively worked against the mechanism's own
purpose. The real run's byte accounting shows DivRoute's actual upload
(3473.87MB) *higher* than uncompressed FedAvg's (2481.39MB, "-40% saving"),
because every one of the 15 clients/round received Tier 1's k_ratio=0.20
(DivRoute's *most* generous tier) instead of a real mix including Tier 2
(k=0.05) and Tier 3 (excluded). A DivRoute run in this state cannot
demonstrate comm savings even in principle, independent of accuracy.

**Fix**, applied to `get_pretrained_finetune_config()`:
- `threshold_mode="rolling_percentile"` — recomputes `tau_low`/`tau_high`
  directly as the p33/p67 percentiles of the actual score distribution each
  round (over a rolling window), rather than decaying toward it from a
  stale, wrongly-scaled default. Scale-independent by construction.
- `convergence_floor=1e-6` — the default (`1e-4`) sits *above* the
  pretrained regime's observed score spread (~1e-5 to 1e-4), which would
  trigger the "converged, freeze tau" protection almost immediately even
  with `rolling_percentile` active, silently reproducing a version of the
  same bug.

**Verification, without spending additional GPU time:** the real round-10
`d_ema` scores from the Kaggle run were replayed through both the old and
new threshold logic in plain Python (`divroute_fl.mechanism.compute_percentile_taus`,
no model/data needed):
- Old logic, `tau=[0.00114, 0.00224]`: tier counts **15/0/0**
- New logic, `tau=[0.00005, 0.000065]`, identical input scores: tier counts **5/5/5**

**What this fix does and doesn't establish.** It guarantees the tiering
mechanism will actually run as designed on the next attempt — a verified,
data-driven fix, not a guess. It does **not** guarantee DivRoute will
outperform FedAvg once tiering is genuinely active: real differentiation
could help (properly isolating drifted clients) or could hurt (introducing
more variance than the accidental "treat everyone identically" mode the bug
produced) — that is an empirical question, unresolved until the run is
repeated. It says nothing about whether the 80-85% target (§2) is
reachable. And it does not rule out other scale-mismatch issues elsewhere
in the pretrained-backbone configuration that haven't yet surfaced — this
one was caught by reading real run output for anomalies, not from a
systematic audit of every hyperparameter against the new backbone's scale.

**Status:** DivRoute must be re-run with the fixed config before any
DivRoute-vs-FedAvg-vs-Uniform comparison from the 10-round Kaggle experiment
can be trusted. FedAvg (74.62% final) and Uniform-Top5% (partial: 52.09% →
67.89% over 2/10 rounds observed) do not depend on `tau_low`/`tau_high`, so
this specific bug doesn't touch them --

**Superseded: see §12.** A second, independent bug (found afterward, from
the 20-round attempt's byte totals) affected FedAvg's and Uniform-Top5%'s
compression ratio in this SAME 10-round run too (`use_k_warmup` was never
disabled here either). Their 10-round numbers above are not valid reference
points after all -- kept here only as a record of what this section
originally concluded, not as usable data.

## 12. Third critical fix: k-ratio warmup silently overrode every method's real compression ratio

Found from the real 20-round Kaggle results for all three methods (§11's
fix applied, so this run's tiering was genuinely active) -- not by
inspection. FedAvg's and Uniform-Top5%'s byte totals came out **identical**:
148.883MB/round, "40% saving," every single round of both runs. That's
impossible if FedAvg is dense (k=1.0) and Uniform-Top5% is k=0.05.

**Root cause:** `main.py` calls `get_adaptive_k_ratios(config, rnd)`
unconditionally every round, regardless of `fedavg_baseline_mode`. That
function (`compression.py`) checks `use_k_warmup` (Config default `True`,
`k_warmup_rounds=30`) *before* it ever looks at `k_ratio_tier1`/
`k_ratio_tier2`: while `rnd < k_warmup_rounds`, it returns
`k_warmup_tier1=0.70`/`k_warmup_tier2=0.30` instead of the configured
ratios. None of the three companion-run factories set `use_k_warmup=False`.
Every companion run attempted so far (10-round and 20-round) has been
`<= k_warmup_rounds`, so **the entire run, every round, used k=0.70/0.30
instead of each method's actual intended ratio**:

| Method | Intended k (tier1/tier2) | Actually used, every round so far |
|---|---|---|
| DivRoute | 0.20 / 0.05 | 0.70 / 0.30 |
| FedAvg | 1.0 / 1.0 (dense) | 0.70 / 0.30 |
| Uniform-Top5% | 0.05 / 0.05 | 0.30 (every client is tier=2) |

FedAvg and Uniform both landing on the same 0.30 explains their identical
byte totals exactly. It also explains why DivRoute sometimes used *more*
bytes per round than dense FedAvg's own reference: top-k transmission needs
a value *and* an index per retained coordinate (~2x overhead vs. dense
storage), so at k=0.70 compressed size is ~140% of dense -- "compression"
that inflates size rather than shrinking it. The round-by-round savings
percentages in the 20-round DivRoute log (-9.3% in tier-1-heavy rounds,
positive in tier-2/3-heavy rounds) track this exactly.

**This is not just a byte-accounting bug.** `k_ratio` truncates what
actually gets aggregated into the global model each round (`compression.py`,
`apply_tiered_compression` → `reconstruct_delta` → `server.aggregate`'s
`agg_delta += w * compressed_delta`), so it directly changed training
dynamics. Every accuracy number from every companion run attempted so far —
10-round and 20-round, all three methods — is confounded by this, not only
the byte totals. The centralized sanity check (§9, 74.63%) is unaffected:
it never calls `apply_tiered_compression` or touches any FL config at all.

**Fix:** `use_k_warmup=False` added to all three factories
(`get_pretrained_finetune_config()`, `get_fedavg_pretrained_config()`,
`get_uniform_top5_pretrained_config()`). A 10-30-round warmup is a
reasonable easing-in period for a from-scratch model training for hundreds
of rounds (the regime this default was tuned for); it makes no sense for a
pretrained backbone fine-tuning for 10-20 rounds total, where "warmup" would
consume the entire run and the configured ratios would never actually apply.

**Verified without spending additional GPU time:** called
`get_adaptive_k_ratios()` directly against each fixed config (replicating
main.py's `fedavg_baseline_mode` runtime override for the FedAvg case, since
that override happens inside `run()`, not in the returned Config object) --
DivRoute now returns k1=0.20/k2=0.05 every round (was 0.70/0.30) --
FedAvg now returns k1=1.0/k2=1.0 every round (was 0.70/0.30) --
Uniform now returns k1=0.05/k2=0.05 every round (was 0.30).

**What this fix does and doesn't establish.** It guarantees each method will
actually run at its own intended, distinct compression ratio -- the
foundational precondition for the whole comm_vs_accuracy comparison this
project exists to produce, which no prior companion run (10- or 20-round)
has actually satisfied. It does not predict what the resulting accuracy or
byte numbers will be for any of the three methods; those remain fully open
until the run is repeated. It also does not rule out further undiscovered
issues of this shape -- both bugs found so far (§11, §12) were caught by
reading real run output for internal inconsistencies (all-Tier-1 counts;
identical byte totals across methods that should differ), not from a
systematic audit of every Config default against the pretrained-backbone
regime's very different scale and round budget. A companion run under 30
rounds should be treated as suspect of hitting a k_warmup-shaped issue by
default until `use_k_warmup=False` is confirmed present in whatever config
produced it.

**Status:** all companion-run results produced before this fix -- both the
10-round and 20-round Kaggle attempts, for all three methods -- are invalid
and should be discarded, not reinterpreted. A fourth attempt, with both §11
and §12's fixes in place, is needed before any DivRoute-vs-FedAvg-vs-Uniform
comparison can be drawn.

## 13. Final result (fourth attempt, both fixes applied — no further runs planned)

**Byte accounting confirms both fixes actually took effect this time,**
which no prior attempt could claim: FedAvg's own per-round upload finally
equals its download (248.139MB = 248.139MB, every round, 0% savings) --
true dense FedAvg at last, not secretly compressed at k=0.30. DivRoute shows
real tiered compression (k=0.20/0.05, upload varying 49-84MB/round with the
tier mix, never the impossible >dense values seen in §12). Uniform-Top5%
shows a flat, correct 24.814MB/round -- exactly what k=0.05 with ~2x
value+index overhead should produce (0.05 x 2 = 10% of the 248.139MB dense
reference). This is the first attempt where all three methods' byte
totals are internally consistent with their configured ratios.

FedAvg's run was manually stopped at round 17/20; DivRoute and Uniform
completed all 20. No further run is planned (explicit instruction) -- all
three had visibly plateaued for several consecutive rounds before their
last recorded point, so these are treated as this round budget's converged,
final numbers rather than an artifact of stopping early:

| Method | Accuracy | Upload | Upload savings | Bidirectional savings |
|---|---|---|---|---|
| FedAvg (dense, round 17/20, plateaued ~81.2-81.6% since round 14) | **81.50%** | 4962.78MB if completed (100% dense, confirmed every round) | 0% (reference) | 0% |
| DivRoute (20/20, plateaued since ~round 17) | **78.36%** | 1310.03MB | 73.6% | 36.8% |
| Uniform-Top5% (20/20, plateaued since ~round 17) | **80.26%** | 496.28MB | 90.0% | 45.0% |

**This is not §2's targeted claim.** The claim this whole plan was built to
test -- DivRoute reaches the *same* accuracy as the baselines at *lower*
cumulative bytes -- is not what this data shows.

- **DivRoute vs. FedAvg** (identical epoch-warmup schedule, so this is the
  fair, confound-free comparison): DivRoute trades **~3.1 accuracy points
  for 73.6%/36.8% byte savings**. That is a real, legitimate Pareto
  tradeoff -- genuinely useful if bandwidth is the binding constraint -- but
  it is "lower accuracy, far fewer bytes," not "same accuracy, fewer
  bytes." A materially weaker claim than §2's target.
- **DivRoute vs. Uniform-Top5%:** Uniform wins on *both* axes -- higher
  accuracy (80.26% vs. 78.36%) and more compression (90%/45% vs.
  73.6%/36.8%). At face value, the simplest possible baseline (flat 5%
  compression, no divergence scoring, no tiering) outperforms DivRoute's
  adaptive routing mechanism in this run.
- **One live, unresolved confound on that second comparison, stated plainly
  rather than used to explain it away:** `get_uniform_top5_config()`'s base
  sets `use_epoch_warmup=False`, so Uniform trains at the full
  `local_epochs=5` every round from round 1, while DivRoute/FedAvg ramp
  2->4->5 -- roughly 33% more total local SGD steps over the 20-round run.
  This could account for some or all of Uniform's accuracy edge over
  DivRoute. It was noted as an open methodology question earlier in this
  document (§10) and was never resolved before this final run -- so the
  DivRoute-vs-Uniform comparison above is not fully confound-free, even
  though DivRoute-vs-FedAvg is.

**What this result does and doesn't establish.** It establishes that the
pretrained-backbone DivRoute mechanism, as configured and measured here,
produces a genuine bandwidth/accuracy tradeoff against dense FedAvg, not a
free win, and does not demonstrate an advantage over the simplest
compression baseline once both bugs (§11, §12) are fixed. It does not
establish that the mechanism is fundamentally incapable of matching FedAvg
or beating Uniform -- the Uniform comparison in particular carries the
unresolved epoch-warmup confound above, and no round-count beyond 20 or
alternative hyperparameters (server momentum retuning, error feedback,
tier-threshold tuning, resolving the NTD asymmetry noted in §10) were
explored before concluding. It does establish that the two-bug-fixing
process itself (§11, §12) was necessary and correct: neither prior attempt
was even measuring what it claimed to measure.

**Status:** superseded — see §14. The investigation continued past this
section's original "final" framing.

## 14. Response to §13: three changes implemented, none run yet

§13 left two loose ends and one previously-flagged, never-fully-resolved
mismatch. All three are addressed here as code changes, verified without
spending GPU time (config instantiation checks and the tier-assignment
logic replayed against synthetic and real recorded divergence scores in
plain Python — no training). Nothing in this section has been validated
empirically; it records what changed and why, not a result.

**1. Uniform's training-budget confound, removed.**
`get_uniform_top5_pretrained_config()` now sets `use_epoch_warmup=True`
(previously inherited `False` from `get_uniform_top5_config()`'s
from-scratch base), matching DivRoute/FedAvg's 2→4→5 ramp exactly. §13
flagged that Uniform's ~33% larger total training budget (100 vs. 75
"epoch-rounds" over 20 rounds) could explain some or all of its accuracy
edge over DivRoute — that confound is now removed at the source rather than
argued around.

**2. NTD asymmetry, resolved by extension rather than removal.**
`ntd_beta=0.1` added to both `get_fedavg_pretrained_config()` and
`get_uniform_top5_pretrained_config()`. Previously only DivRoute carried
this regularizer (inherited from `get_recommended_divroute_config()`'s
base), flagged as an open question in §10 and never acted on before §13's
run. Resolved in the direction that gives every method the same treatment
— rather than removing it from DivRoute, which would only have widened the
gap §13 already found without answering the fairness question.

**3. Tier-bandwidth polarity inversion — an opt-in experiment, not a default
change.** This is the substantive one. §3.1 established, independently of
this section, that the live tier-assignment code (`main.py`) contradicts
the project's own plain-language design intent: the README describes
drifted clients as needing "a strong correction signal" (more bandwidth),
but the actual running code gives the **least**-divergent clients tier=1
(`k_ratio_tier1`, the most generous budget) and compresses the
**most**-divergent clients hardest (tier=3, routed through
`k_ratio_tier2`). §3.1's fix addressed only the downstream symptom
(`include_tier3_in_aggregation=True`, so divergent clients aren't zeroed
out) — it explicitly left the bandwidth-*direction* itself unchanged. §13's
result gives new reason to revisit that: DivRoute spent *more* bytes than
Uniform-Top5% (1310.03MB vs. 496.28MB, 26.4% vs. 10% of dense) while scoring
*lower* accuracy (78.36% vs. 80.26%) — consistent with bandwidth being
allocated to clients that don't need it while the clients whose updates
carry the most novel local information get compressed hardest.

Implementation: a new `Config.invert_tier_polarity` field (default `False`,
verified to reproduce the exact original tier assignment on every synthetic
input tested — zero effect unless explicitly set) and a corresponding
change to `main.py`'s tier-assignment block that swaps which extreme (`d <=
tau_low` vs. the `else` branch) maps to tier=1 vs. tier=3. Tier=2 (the
middle band) and the NaN/Inf safety fallback are untouched by design — the
experiment isolates the bandwidth-direction question only, changing nothing
else about the mechanism. No downstream code (compression.py's k_ratio
lookup, `server.aggregate`, the Progress Guarantee mechanism) needed to
change, since all of it reads the tier *number* generically rather than
assuming what a given number means.

This flag only affects DivRoute's own routing path. It has no effect on
FedAvg (`fedavg_baseline_mode` forces a uniform k_ratio regardless of tier
label) or Uniform-Top5% (`uniform_top5_mode` hardcodes every client to
tier=2, never reaching the branch this flag modifies).

**What none of this establishes yet.** These are corrections to how the
comparison is measured (1, 2) and a clearly-scoped test of one specific,
previously-identified hypothesis (3) — not predictions of what the next run
will show. Fix 3 in particular could just as easily confirm the *original*
polarity was fine and something else explains §13's gap. The recommended
next run is documented in `DIVROUTE_REMAINING_WORK.md`: re-run all three
companion configs with fixes 1-2 included, plus a separate, explicitly
`invert_tier_polarity=True` DivRoute run to test fix 3 against the corrected
FedAvg/Uniform baselines — kept as its own labeled run rather than folded
into the default DivRoute config, since it is a change to the mechanism's
behavior, not a measurement correction.

**Status:** superseded — see §15 for what these three fixes actually showed.

## 15. Fifth attempt: what the three §14 fixes actually showed

FedAvg and Uniform reran with fixes 1-2 (§14) applied; a new, separately
labeled DivRoute run tested fix 3 (`invert_tier_polarity=True`). Original
(non-inverted) DivRoute was not rerun, since its config was untouched by any
of the three fixes and its result (78.36%, §13) remains valid without
re-measurement.

| Method | Accuracy | Upload | Savings |
|---|---|---|---|
| FedAvg (dense, now with NTD) | **82.08%** | 4962.78MB | 0% (ref) |
| DivRoute (original polarity, unchanged from §13) | 78.36% | 1310.03MB | 73.6% |
| DivRoute (inverted polarity) | **77.55%** | 719.46MB | 85.5% |
| Uniform-Top5% (now equal training budget) | **78.54%** | 496.28MB | 90.0% |

**Finding 1 (fixes 1+2 combined) — the epoch-warmup confound was real, but
doesn't fully explain the gap.** Uniform's accuracy dropped from 80.26%
(§13, extra training) to 78.54% (matched training budget) — a 1.72-point
drop, confirming the confound §13 flagged was genuine. DivRoute (78.36%)
and Uniform (78.54%) are now essentially tied on accuracy. But DivRoute
still does not come out ahead: Uniform reaches that same accuracy on 90%
upload savings against DivRoute's 73.6% -- DivRoute spends roughly 2.6x
Uniform's bytes to land at the same accuracy. Removing the confound
narrowed the story from "DivRoute clearly loses to Uniform" to "DivRoute
ties Uniform on accuracy while costing more bytes to do it" -- an
improvement in fairness, not in DivRoute's competitive position.

**Finding 2 (fix 3) — the polarity-flip hypothesis is refuted, not
confirmed.** §3.1 established that the live code's tier-bandwidth direction
contradicts the project's own stated design intent (least-divergent clients
get the most bandwidth, not the most-divergent ones as the README
describes), and §14 built a controlled test of whether correcting that
direction would close DivRoute's gap with FedAvg. It does not: inverted
polarity scores *lower* (77.55% vs. 78.36%) than the original, despite
using fewer bytes (85.5% vs. 73.6% savings) -- consistent with routing more
bandwidth toward compression rather than accuracy along the same tradeoff
curve, not with unlocking better accuracy. The mismatch §3.1 identified
between documentation and code remains real and worth fixing for its own
sake (the README's description is simply inaccurate about what the code
does), but it is now empirically ruled out as *the* explanation for
DivRoute underperforming FedAvg. Something else is the actual bottleneck.

**What this establishes, stated plainly.** Across every condition tested in
this document -- from-scratch (§1-§3), pretrained with the original tau/k
bugs (§13), pretrained with those bugs fixed (§13's own result), and now
pretrained with training-budget and NTD parity plus both tier polarities
(§15) -- DivRoute has not demonstrated the §2 target (same accuracy as
FedAvg at lower cost) under any measurement this document trusts. Its best
showing is a genuine but different claim: roughly Uniform-Top5%-equivalent
accuracy at meaningfully more bytes than Uniform, and a real but moderate
communication saving against dense FedAvg purchased with several points of
accuracy. Two of the three explanatory hypotheses tested so far (the
epoch-warmup confound, the tier-polarity direction) have been examined
rigorously and neither fully accounts for the FedAvg gap.

**Untried, not yet ruled in or out:** the equal-byte-budget reframing
(comparing accuracy at matched cumulative bytes rather than matched rounds
-- DivRoute's slower byte spend means it could afford several times more
rounds than FedAvg's 20 for the same total cost), loosening `k_ratio_tier2`
beyond the from-scratch-inherited 0.05, and the existing
`ablation_routing_no_compression`/`ablation_uniform_compression` flags
(never run against the pretrained backbone) to isolate whether the
remaining gap traces to the routing decisions themselves or to the
compression level applied once routed.

**Status:** superseded — see §16, the first result in this document to
approach the §2 target under one reading of "communication cost."

## 16. Sixth attempt: equal-byte-budget DivRoute — the closest result yet, with a real caveat

Every companion-run comparison so far (§13, §15) compared all three methods
at the same *round count* (20). That structurally disadvantages a
communication-efficient method: DivRoute spends bytes more slowly than
FedAvg, so at a fixed round count it has necessarily seen less total signal
per byte spent, not because the mechanism doesn't work but because it
wasn't given the rounds its own saved budget would buy. This section tests
the comparison the master plan's own evaluation methodology actually calls
for (§2: "reaches the same accuracy... at lower cumulative MB," the
`comm_vs_accuracy.png` curve) -- accuracy at *matched bytes*, not matched
rounds.

DivRoute was run fresh to 32 rounds -- roughly where its cumulative bytes
catch up to FedAvg's 20-round spend, extrapolated from the 20-round run's
per-round average. Compared against FedAvg's actual 20-round result (§15),
not a recomputed reference:

| | Accuracy | Upload | Download | Bidirectional |
|---|---|---|---|---|
| FedAvg (20 rounds) | 82.08% | 4962.78MB | 4962.78MB | 9925.56MB |
| DivRoute (32 rounds) | **81.58%** | **2044.43MB** | 7940.44MB | 9984.87MB |

**On upload bytes alone, this is the strongest result in this document.**
DivRoute lands within **0.5 accuracy points** of FedAvg using **59% fewer
upload bytes**. This directly confirms that every prior equal-round
comparison (§13, §15) was measuring something other than the claim §2
actually makes -- DivRoute's mechanism was never given room to spend the
budget its own compression saves.

**The caveat, stated as plainly as the result:** on *bidirectional* (total)
bytes, this is essentially a wash, not a win. DivRoute's 12 extra rounds
cost roughly 2978MB of additional download -- the server always broadcasts
the full dense model regardless of the sending method's compression, so
download is not compressed for anyone -- which very nearly cancels the
upload savings. Total bidirectional bytes: DivRoute 9984.87MB vs. FedAvg's
9925.56MB -- DivRoute used *slightly more* total bytes for *slightly worse*
accuracy on this accounting. Running more rounds converts upload savings
into download cost at close to a 1:1 rate; it does not create free
communication budget.

**Which reading is the right one for this project's claim is a real,
unresolved question, not a rounding error.** Upload-only is the standard
framing in most FL top-k/sparsification literature, because client uplink
is the actual bottleneck in most real deployments (consumer/mobile
connections are asymmetric; server downlink is comparatively cheap and
symmetric) -- under that framing this is a genuine, strong, defensible
result. The master plan's own stated claim ("lower cumulative MB") reads
more literally as bidirectional total -- under that framing this result
does not clear the bar. Whichever framing the project commits to should be
stated explicitly and applied consistently to all three methods' headline
numbers, not chosen after the fact to favor one outcome.

**What this does and doesn't establish.** It establishes that DivRoute's
mechanism, given a fair (matched-budget) comparison on the metric most FL
compression papers actually report, comes close to matching dense FedAvg's
accuracy -- a materially different and more favorable picture than any
equal-round comparison in this document showed. It does not establish that
DivRoute wins on the master plan's own literal total-bytes framing, and it
does not establish superiority over Uniform-Top5%, which has not been
tested under this same extended-round treatment. Uniform costs roughly
24.8MB/round vs. DivRoute's ~64MB average -- markedly cheaper per round --
so it could afford substantially more rounds within either FedAvg's or
DivRoute's spend and might close its own remaining gap, or extend its lead,
under the same equal-budget logic just applied to DivRoute. That
comparison has not yet been run.

**Status:** superseded — see §17, which settles the question this section
left open.

## 17. Seventh attempt: Uniform-Top5% at matched budget — the core question settled

§16 identified the one missing piece for a complete verdict: Uniform-Top5%
had never been given the same extended-round treatment as DivRoute. This
section closes that gap. Uniform-Top5% was run fresh to 32 rounds --
matching DivRoute in every way that matters: same round count, same
epoch-warmup schedule (§14 fix 1), same NTD regularizer (§14 fix 2).

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| **Uniform-Top5% (32 rounds)** | **81.43%** | **794.04MB** | **8734.49MB** |

**Unlike §16's DivRoute-vs-FedAvg result, this comparison does not depend
on which byte metric is chosen.** Uniform matches DivRoute's accuracy (a
0.15-point gap, well within run-to-run noise) while using 61% fewer upload
bytes and 12.5% fewer bidirectional bytes. It wins or ties on every single
axis measured: accuracy, upload, and total bytes.

**This settles the question the entire investigation from §11 onward was
building toward: does DivRoute's routing intelligence -- divergence
scoring, adaptive-percentile tier assignment, divergence-weighted
aggregation, the tier-1/tier-2/tier-3 compression schedule -- earn its
complexity over the simplest possible baseline, flat top-5% compression
applied identically to every client with no routing logic at all? On this
data, after removing every confound and bug found along the way: no.**

It is worth stating plainly what this section closes out, because it is
the product of real, load-bearing work, not a quick negative result:
- Three genuine bugs were found and fixed before any comparison here could
  be trusted (§3.1's Tier-3-exclusion fix, predating this document's
  companion-run track; §11's tau-threshold scale mismatch; §12's k-ratio
  warmup silently overriding every method's real compression ratio).
- Two specific hypotheses for DivRoute's shortfall were formed from this
  project's own prior findings and tested rather than assumed: the
  epoch-warmup training-budget confound (§14 fix 1, confirmed real but
  insufficient to explain the full gap) and the tier-bandwidth polarity
  mismatch §3.1 first flagged (§14 fix 3, tested directly and refuted --
  inverting it made results worse, not better).
- The comparison that finally answers the core question (§16, §17) was run
  at matched training budget rather than matched rounds, which is what the
  master plan's own evaluation methodology (§2's `comm_vs_accuracy.png`)
  actually calls for -- and which every earlier equal-round comparison in
  this document (§13, §15) was not.

**What still stands, precisely.** The 80-85% accuracy target (§2) is met:
all three methods land in or near that band at their respective
best-measured points. The claim that fine-tuning a pretrained backbone
under federation can approach dense FedAvg's accuracy at a fraction of the
upload cost also stands -- both DivRoute and Uniform reach within
0.5-0.65 points of FedAvg's 82.08% using well under half its upload bytes.
**What does not stand is that this is specifically attributable to
DivRoute's routing mechanism** rather than to compression (of any kind,
applied to any subset of clients) combined with a longer, cheaper-per-round
training schedule. Uniform-Top5% -- the communication-matched control this
mechanism was always supposed to be compared against, per §6's original
design -- gets the same benefit more cheaply, with none of the routing
machinery.

**Status:** superseded — §17 prompted a literature review for *why* the
mechanism underperforms and *how* it might be fixed (not merely re-measured
more fairly). §18 reports the first of two literature-motivated fixes
tried in response.

## 18. Eighth attempt: hybrid loss+divergence routing signal — refuted

A post-§17 literature review (FedNolowe; FedAWA, CVPR 2025) found that
loss-based client-weighting signals often outperform pure parameter-space
divergence for identifying which clients' updates matter. This project's
own codebase already implements exactly such a hybrid
(`use_directional_divergence` + `loss_improvement_weight`, blending
delta-space divergence with local validation-loss improvement) but it had
never been enabled for any companion run. Tested here, run fresh to 32
rounds:

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (original divergence signal, 32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| DivRoute (hybrid loss+divergence signal, 32 rounds) | **80.41%** | **1508.45MB** | 9448.89MB |
| Uniform-Top5% (32 rounds) | 81.43% | 794.04MB | 8734.49MB |

Converged, not a transient dip (79.88% -> 80.41% smoothly flattening over
the last 7 rounds as the LR schedule annealed). **The hybrid signal is
worse than both the original divergence-only DivRoute (-1.17pp) and
Uniform-Top5% (-1.02pp).** It does achieve somewhat better upload
compression than original DivRoute (26% less), but that is a worse
Pareto position, not a better one -- less accuracy purchased for the
saving. One honest caveat: `local_val_fraction=0.1` means clients trained
on 10% less local data under this signal, a small real confound not present
in the 81.58% baseline -- plausibly worth a point or so, unlikely to
explain the full 1.17-point gap alone.

**Verdict: hypothesis refuted.** Divergence-only routing is not the
problem this signal fixes; if anything, in this specific implementation, it
performs slightly better than the loss-hybrid alternative.

**Status: one literature-motivated experiment remains untested** —
tier-transition-aware error feedback (`Config.error_feedback_tier_aware`,
implemented and verified without GPU compute; see
`DIVROUTE_REMAINING_WORK.md`). FL literature is specific that error
feedback is essential for top-k compression and that its documented failure
mode ("stale error compensation") occurs under partial client
participation — which this project's `use_error_feedback=False` default was
set from a finding that predates every bug fix made this session and did
not account for DivRoute's own round-to-round tier reassignment compounding
that staleness. This is the last queued fix before treating the
investigation's outcome (§17, reinforced by §18) as concluded.
The core mechanism-specific claim this project set out to demonstrate is
not supported by the evidence gathered here, after a genuinely thorough
attempt to find conditions under which it would be. That is itself a
publishable, defensible finding -- see the discussion of framing this as a
negative/mixed result for write-up purposes, raised earlier in this
conversation's history but not separately documented in this file.
