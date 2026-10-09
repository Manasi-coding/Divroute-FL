# DivRoute-FL — 80–85% Accuracy Plan (CIFAR-100, Publication-Grade)

> Companion document to `DIVROUTE_ALGORITHMS_MASTER_DOC.md` (the algorithm reference).
> This document is the **decision record and execution plan**: what's wrong, what's
> changing, exactly which files/flags are touched, what's guaranteed vs. not, and why
> each change is necessary. Written so a fresh reader — or a fresh LLM session — can
> pick this up with no other context and know exactly what to do and why.

**Status (concluded): §5's infrastructure changes are implemented and
smoke-tested. §9's centralized sanity check has a first data point (74.63%
after 1 epoch; full 15-epoch run never attempted). §6's three companion-run
configs went through fourteen attempts total — two invalid (§11:
tau-threshold scale mismatch; §12: k-ratio warmup silently overriding
every method's real compression ratio), then twelve structurally valid
ones (§13-§24). §17's negative verdict — DivRoute's routing doesn't beat
naive uniform compression — held firmly at `alpha=0.9` (near-IID) through
§20. §21-§24 then tested real heterogeneity (`alpha=0.1`) with a full
3-seed-vs-3-seed matched comparison — the condition DivRoute's own design
premise requires to have any signal to route on — and found a clean
statistical tie on accuracy, with DivRoute still paying a consistent
2.4-2.6x upload premium. §24 is the concluding result.**

**§24's verdict — the first fully seed-matched (3 vs. 3) comparison in the
investigation, real heterogeneity, plain error feedback:**

| | Final-round acc (mean, n=3) | Trailing-5-round mean (n=3) | Upload |
|---|---|---|---|
| DivRoute | 78.35% (±1.07pp) | 77.91% (±0.59pp) | 1989.84MB |
| **Uniform-Top5%** | 78.47% (±0.55pp) | 78.20% (±0.56pp) | **794.04MB** |

The gap between methods (0.12-0.29pp) is smaller than either method's own
seed-to-seed spread (0.55-1.07pp) — a clean statistical tie, not a win for
either side. DivRoute costs a consistent 2.4-2.6x Uniform's upload bytes
for that tie, on every one of six seeded runs across both methods.

**The mechanism-specific claim this project set out to demonstrate —
divergence-based routing outperforms naive uniform compression — is not
supported anywhere across twelve structurally valid attempts, four
distinct literature-motivated hypotheses, and two heterogeneity regimes
(near-IID and real non-IID, the latter seed-matched 3-vs-3).** At
alpha=0.9, DivRoute loses outright (§17, §20). At alpha=0.1 — where its
design premise should matter most — it ties Uniform on accuracy while
still costing 2.5x the bytes. There is no condition tested in this
investigation where the routing mechanism earns its complexity.

**What still stands, and is the project's real, defensible contribution:**
error-feedback-corrected top-k compression enables pretrained-backbone
federated fine-tuning to reach (and per §20, slightly exceed) dense
FedAvg's accuracy at a large communication saving — regardless of which
client the bandwidth is routed to. That is a genuine, publishable finding.
It is a claim about error-feedback-corrected compression in general, not
about DivRoute's divergence-based routing specifically. See §17-§24 for
the full, seed-matched account of how this conclusion was reached.

---

**§17-§20 verdict (still the reference point at alpha=0.9, unchanged):**

**§20's verdict, the two ablations that settle what §19 left open:**

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute + tier-aware EF (§19) | 82.99% | 1974.95MB | 9915.39MB |
| DivRoute + plain EF (§20) | **83.32%** | 2103.98MB | 10044.43MB |
| **Uniform-Top5% + plain EF (§20)** | **83.24%** | **794.04MB** | **8734.49MB** |

Plain error feedback beats the tier-aware version for DivRoute
(+0.33pp, at *more* bytes not fewer — the tier-change reset was throwing
away useful signal, not protecting against staleness). And Uniform-Top5%
+ plain EF ties DivRoute's accuracy (0.08pp gap — noise) using **62% less
upload and 13% less total bytes**. Error feedback is a general fix for
biased top-k compression, not something DivRoute's routing unlocks — it
helps flat uniform compression at least as much, exactly as the
non-DivRoute-specific FL literature on EF would predict.

**The mechanism-specific claim this project set out to demonstrate — that
divergence-based routing outperforms naive uniform compression — is not
supported by the evidence gathered here, across ten attempts and four
distinct, literature-motivated hypotheses tested for why it should be.**

What still stands: the 80-85% accuracy target is met and exceeded (all
methods now land at 81-83%+), and communication-efficient fine-tuning of a
pretrained backbone, with error feedback added, now slightly *exceeds*
dense FedAvg's accuracy (82.08%) at a fraction of its upload bytes. What
does not hold is that this benefit is specific to DivRoute rather than to
error-feedback-corrected top-k compression generally — Uniform gets there
more cheaply with no routing at all.

This conclusion was earned, not assumed: three real bugs were found and
fixed before any of it could be trusted (§3.1's Tier-3-exclusion fix;
§11's tau-threshold scale mismatch; §12's k-ratio warmup override), and
four specific hypotheses for DivRoute's shortfall were tested rather than
guessed at (§14's epoch-warmup confound, confirmed but insufficient;
§14's tier-polarity inversion, refuted; §18's hybrid loss+divergence
signal, refuted; §19's tier-aware error feedback, initially promising,
refuted by §20's ablations). See §17-§20 for the full account.

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

## 19. Ninth attempt: tier-transition-aware error feedback — the first positive result

The last queued fix from §18. FL literature is unanimous that error
feedback (accumulating the residual top-k drops, adding it back next round)
is close to essential for top-k compression, but also documents a specific
failure mode under partial participation and non-stationary routing: a
client's residual buffer, computed under one k_ratio, becomes stale or
actively harmful if blended into a delta compressed at a *different*
k_ratio next time that client participates. DivRoute's own tier
reassignment (a client can be Tier 1 one round, Tier 3 the next) triggers
exactly this every time a client's tier changes. `error_feedback_tier_aware`
resets a client's residual buffer to zero on any tier change instead of
blending a residual computed under the wrong compression level. Run fresh
to 32 rounds, `use_error_feedback=True, error_feedback_tier_aware=True`,
everything else identical to the original §17 DivRoute config:

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (original, no EF, 32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| Uniform-Top5% (32 rounds) | 81.43% | 794.04MB | 8734.49MB |
| FedAvg (dense, 20 rounds) | 82.08% | 4962.78MB | 9925.56MB |
| **DivRoute + tier-aware error feedback (32 rounds)** | **82.99%** | **1974.95MB** | **9915.39MB** |

Confirmed converged, not a transient spike: rounds 24-32 read 82.20 ->
82.55 -> 82.49 -> 82.75 -> 82.71 -> 82.78 -> **83.00** -> 82.76 -> 82.99, a
tight plateau in the 82.5-83.0% band for the last 9 rounds, not a one-round
peak.

**This is the first result in the entire investigation where DivRoute wins
outright, on every axis that matters, without a caveat that undoes it:**

- **Beats FedAvg on accuracy (+0.91pp) while using 60% less upload and a
  statistically-tied bidirectional total** (9915.39MB vs. 9925.56MB, a
  0.1% difference — noise). This is the actual headline claim §2 was
  chasing from the start: same-or-better accuracy than dense FedAvg, for a
  fraction of the upload cost, with no bidirectional penalty this time
  (contrast §16, where the equal-byte-budget DivRoute run matched FedAvg's
  accuracy but came out bidirectionally *worse*, not better).
- **Strictly dominates the original DivRoute mechanism** (§17): higher
  accuracy (+1.41pp) *and* fewer upload bytes (-3.4%) *and* fewer
  bidirectional bytes (-0.7%). Tier-aware EF is a pure improvement over
  vanilla DivRoute with the same routing logic — not a different tradeoff
  point, a better one on every axis.
- **Beats Uniform-Top5% on accuracy (+1.56pp) at a real but modest
  bidirectional cost** (+13.5%, 9915.39MB vs. 8734.49MB). This is the one
  place a tradeoff still exists: Uniform remains the cheapest option in
  absolute bytes, but DivRoute+tier-aware-EF is the first version of the
  mechanism to buy meaningfully higher accuracy for that extra cost, rather
  than losing on both axes (§17) or losing on accuracy for a worse tradeoff
  (§18).

**Why this fixes what §17 could not:** §17 established that routing alone
(divergence scoring + tiering + adaptive thresholds) doesn't outperform
flat compression — the routing *signal* wasn't the missing piece. What was
missing was handling the interaction between routing and error feedback:
DivRoute's own tier churn was corrupting the one mechanism (EF) that makes
biased top-k compression competitive with dense training in the first
place. Once EF's residual buffer is made routing-aware (reset on tier
change instead of blended across incompatible k_ratios), the routing
signal's information is no longer being paid for with corrupted gradient
history, and the underlying tiering mechanism turns out to have been sound
all along — its value was being masked by an implementation gap, not
absent.

**Caveats, stated plainly:**
- This is one run, not a repeated/averaged result — no seed variance
  estimate exists yet for any config in this investigation, so a ~1pp gap
  should be treated as a real signal but not a guaranteed-reproducible
  constant until reproduced.
- Not yet tested: whether plain (non-tier-aware) `use_error_feedback=True`
  alone — without the tier-aware reset — would recover most of this gain.
  If it would, the "tier-aware" refinement specifically (rather than error
  feedback in general) contributes less than this result alone suggests.
  This is the natural next ablation before treating tier-aware EF as the
  load-bearing fix.
- Not yet tested: whether Uniform-Top5% also improves under plain error
  feedback (Uniform has no tiers, so `error_feedback_tier_aware` is a
  no-op for it, but ordinary EF still applies). If Uniform gains similarly
  from EF alone, part of this result is "error feedback helps top-k
  compression in general" rather than something specific to DivRoute's
  routing.

**Status: this reopens the project's core claim.** §17's negative verdict
("routing doesn't beat naive compression") stands for *routing without
correctly-integrated error feedback*. It does not stand for DivRoute as a
whole once EF is fixed to work correctly with tiering. The two ablations
above should run before this is written up as the final result — they
determine whether the win is really about tier-aware routing+EF
specifically, or about error feedback more generally with DivRoute
incidentally along for the ride.

## 20. Tenth attempt: the two §19 ablations — tier-aware EF refuted, §17's verdict reinstated

Both queued ablations run fresh to 32 rounds, identical setup to §19
otherwise. Both confirmed converged, not spikes: DivRoute+plain-EF
plateaus 82.7% -> 83.3% over rounds 24-32 (final 83.32%); Uniform+plain-EF
plateaus 82.4% -> 83.3% over the same window (final 83.24%).

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute + tier-aware EF (§19) | 82.99% | 1974.95MB | 9915.39MB |
| **DivRoute + plain EF (no tier-aware reset)** | **83.32%** | 2103.98MB | 10044.43MB |
| **Uniform-Top5% + plain EF** | **83.24%** | **794.04MB** | **8734.49MB** |

**Ablation 1 result: tier-aware EF is refuted, not confirmed.** Plain
error feedback — blending the residual every round regardless of tier
change, exactly the "naive" version the literature review flagged as
prone to stale-error problems under tier churn — beats the tier-aware
version outright (83.32% vs. 82.99%, +0.33pp) while saving *fewer* bytes,
not more (2103.98MB vs. 1974.95MB upload). The "reset residual on tier
change" hypothesis from §18/§19's literature review does not hold up here:
resetting the buffer throws away real signal the model needed, rather than
protecting it from staleness. §19's positive result was real, but its
explanation was wrong — the gain came from adding error feedback at all,
not from making it tier-aware.

**Ablation 2 result: Uniform gains just as much from plain EF, at a
fraction of the bytes.** Uniform-Top5% + plain EF (83.24%) is
statistically tied with DivRoute + plain EF (83.32%, a 0.08pp gap — noise)
while using **62% less upload and 13% less bidirectional total**
(794.04MB vs. 2103.98MB upload; 8734.49MB vs. 10044.43MB bidirectional).
Error feedback is not something DivRoute's routing unlocks — it is a
general fix for biased top-k compression that helps flat uniform
compression at least as much, exactly as the general (non-DivRoute-specific)
FL literature on error feedback would predict.

**Status: §17's negative verdict is reinstated, on stronger evidence than
before.** The sequence across §17-§20 now reads: routing alone doesn't
beat naive compression (§17); a loss-based hybrid signal doesn't fix it
(§18); tier-aware error feedback looked like it fixed it (§19) but the two
follow-up ablations show the gain was from plain error feedback, which
helps Uniform-Top5% just as much for a fraction of the bytes (§20). Ten
attempts, three real infrastructure bugs fixed, and four distinct
hypotheses for why DivRoute's routing should matter — tested, not assumed
— have not produced a case where the routing mechanism specifically earns
its complexity over flat compression. What does hold, unchanged since
§17: the 80-85% accuracy target is met, and communication-efficient
fine-tuning of a pretrained backbone reaches (now slightly exceeds, with
EF) dense FedAvg's accuracy — that benefit is not specific to DivRoute.

**What would still move this**, if pursued: a seed-variance study (no
result in this investigation has been repeated across seeds, so every
delta discussed here, including this one, is a single-run signal); and the
routing-vs-compression-level ablation flagged repeatedly since §14
(`ablation_routing_no_compression` / `ablation_uniform_compression`,
never run on the pretrained backbone) — the one diagnostic in the codebase
that isolates routing from compression level directly rather than by
comparing separately-configured runs.

## 21. Eleventh attempt: real heterogeneity (alpha=0.1) — the gap closes, but the result is ambiguous, not a win

Every attempt through §20 held `alpha=0.9` (near-IID) fixed and only tuned
routing/EF mechanics. Since DivRoute's premise requires clients to differ
enough that bandwidth allocation matters, this attempt changed the one
variable never touched: partitioned data at `alpha=0.1` (standard
strongly-non-IID FL benchmark setting), everything else identical to
§20's best config (plain error feedback, no tier-aware reset), DivRoute vs.
Uniform-Top5% head to head, both fresh to 32 rounds.

| | Final-round accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (alpha=0.1, plain EF) | **79.58%** | 2019.61MB | 9960.06MB |
| Uniform-Top5% (alpha=0.1, plain EF) | 79.11% | **794.04MB** | **8734.49MB** |

Taken at face value: DivRoute wins on accuracy (+0.47pp) for the first
time in this investigation's history against Uniform, at real
heterogeneity. **But this reading doesn't survive closer inspection.**

**Rounds 15-23 (pre-noise) were smooth and monotonic for both methods, and
Uniform was slightly ahead**, not DivRoute: round 23 read DivRoute 77.26%
vs. Uniform 77.55%. **Rounds 24-32 then became noisy for both methods**
(unlike every alpha=0.9 run in this document, which plateaued smoothly) —
DivRoute oscillated 76.91%-79.58% and Uniform oscillated 77.16%-79.36%
round to round, a ~2.5pp swing for each, roughly 5x the round-to-round
noise seen at alpha=0.9. Averaging the last 5 rounds instead of reading
the final round alone **flips the ranking**: DivRoute's trailing-5 mean is
78.56%, Uniform's is 78.84% — Uniform ahead by 0.28pp on the more
noise-robust reading, the opposite of what the final-round-only comparison
shows.

**Verdict: inconclusive, not a win.** Whether DivRoute or Uniform is
"ahead" at alpha=0.1 depends entirely on which round is read as the
endpoint — a coin-flip-sized effect sitting inside round-to-round noise
5x larger than anything seen at alpha=0.9, from a single seed. What the
alpha hypothesis from §20 (routing needs real heterogeneity to have
anything to route on) did predict correctly: **the gap genuinely closes**
at alpha=0.1 — DivRoute went from decisively behind Uniform on both axes
at alpha=0.9 (§17, §20) to statistically tied at alpha=0.1. That's a real,
directionally-correct finding. It just isn't yet strong enough evidence to
call DivRoute the winner, and even if the final-round reading is taken at
face value, DivRoute's edge would be bought at **2.5x Uniform's upload
bytes** (2019.61MB vs. 794.04MB) — an expensive tradeoff for a gap this
size even before accounting for the noise.

**What this requires before it can be called either way: seed repeats.**
This is the first result in the investigation where the single-seed
methodology used throughout is actually load-bearing for the conclusion —
every earlier alpha=0.9 comparison had a large enough, smooth enough gap
that one seed was sufficient; this one does not. 2-3 seeds each for
DivRoute and Uniform at `alpha=0.1` (same plain-EF config) are needed
before treating either "DivRoute wins under heterogeneity" or "still no
real difference" as established.

## 22. Twelfth attempt: DivRoute seed repeats at alpha=0.1 — the seed-42 result looks like a lucky draw

Two more DivRoute seeds run at `alpha=0.1` (seeds 43, 44), same plain-EF
config as §21's seed 42. Uniform has not yet been repeated — this section
only resolves DivRoute's own seed variance; the DivRoute-vs-Uniform
question still needs Uniform's seeds too (see status below).

| Seed | Final-round acc | Trailing-5-round mean | Upload | Bidirectional |
|---|---|---|---|---|
| 42 (§21) | 79.58% | 78.56% | 2019.61MB | 9960.06MB |
| 43 | 77.70% | 77.40% | 2069.24MB | 10009.69MB |
| 44 | 77.77% | 77.77% | 1880.66MB | 9821.10MB |
| **Mean (n=3)** | **78.35%** | **77.91%** | 1989.84MB | 9930.28MB |
| **Std (n=3, sample)** | ±1.07pp | ±0.59pp | — | — |

All three converged (rounds 28-32 checked for each; no transient spikes).

**Seed 42 was the high outlier, not the typical case.** Two of three seeds
(43, 44) landed 1.8-1.9 points below seed 42's final-round number, and
below it on the trailing-5-round mean too. DivRoute's own seed-to-seed
spread at alpha=0.1 (±0.59-1.07pp) is now comparable to or larger than any
of the DivRoute-vs-Uniform gaps discussed in §21 — which retroactively
explains why §21's single-seed comparison flipped depending on
methodology: it wasn't measuring a real effect, it was measuring where one
particular seed happened to land inside a noisy distribution.

**Comparing the 3-seed DivRoute mean against Uniform's single alpha=0.1
data point (§21: 79.11% final / 78.84% trailing-5, not yet itself
repeated) now favors Uniform, not DivRoute:** DivRoute's mean (78.35%
final / 77.91% trailing-5) sits 0.76-0.93pp *below* Uniform's one
observation, and Uniform's number is at or above the top of DivRoute's own
3-seed range. This comparison is still asymmetric (3 DivRoute seeds vs. 1
Uniform seed) and not yet conclusive on its own, but it moves the needle
in the opposite direction from §21's face-value reading, not the same one.
DivRoute's upload cost also stayed consistently ~2.4-2.6x Uniform's across
all three seeds (1880.66-2069.24MB vs. 794.04MB) — the byte disadvantage
did not shrink alongside the accuracy question.

**Status: not yet resolved, but trending against the "alpha=0.1 rescues
DivRoute" hypothesis.** Uniform-Top5% needs the same two seed repeats
(43, 44) at alpha=0.1 before a real seed-matched comparison exists. Until
then, treat DivRoute's alpha=0.1 mean (78.35% final-round, 77.91%
trailing-5) as the best current estimate of its true performance at real
heterogeneity — noticeably lower than the single number §21 highlighted,
and no longer clearly ahead of Uniform's (still single-seed) reference
point.

## 23. Thirteenth attempt: Uniform seed 43 at alpha=0.1 — both methods show comparable noise, gap shrinks to statistical tie

One of Uniform's two queued seed repeats (seed 43; seed 44 not yet run).
Converged (rounds 28-32 plateau 77.4%->78.7%->78.1%, no spike).

| Seed | Final-round acc | Trailing-5-round mean | Upload |
|---|---|---|---|
| 42 (§21) | 79.11% | 78.84% | 794.04MB |
| 43 (this) | 78.14% | 77.96% | 794.04MB |
| **Mean (n=2)** | **78.63%** | **78.40%** | 794.04MB |

vs. DivRoute's 3-seed mean from §22: **78.35% final-round / 77.91%
trailing-5**.

**Uniform shows its own real seed-to-seed spread too** (0.97pp final-round,
0.88pp trailing-5, between its two seeds so far) — comparable in size to
DivRoute's 3-seed spread (§22: ±1.07pp / ±0.59pp). With that context, the
current between-method gap (Uniform's mean ahead by 0.28-0.49pp) is now
smaller than either method's own within-method seed noise. **This reads as
a statistical tie, not a Uniform win or a DivRoute win** — a genuine
change from §22's reading, where DivRoute's mean looked clearly behind a
single Uniform data point; with Uniform's own noise now visible, that
apparent gap mostly evaporates.

**What is not ambiguous: DivRoute's byte cost.** Every DivRoute seed at
alpha=0.1 has used 1880-2069MB upload; every Uniform seed has used exactly
794.04MB (Uniform's compression ratio is fixed and deterministic, so this
number does not vary by seed). DivRoute costs a consistent ~2.4-2.6x
Uniform's upload for accuracy that, on current evidence, is statistically
indistinguishable from Uniform's — not better, and this is the clearest
finding across §21-§23: even where DivRoute's routing is *not* worse, it
does not buy anything for the extra bytes it spends.

**Status: one seed short of a clean, fully seed-matched verdict.** Uniform
seed 44 (same config, `seed=44`) is the last piece — once it lands, both
methods will have exactly 3 seeds each at alpha=0.1 and the comparison can
be read with proper means and spreads on both sides rather than one side
still partially single-seed.

## 24. Fourteenth attempt: Uniform seed 44 — full 3-vs-3 verdict at real heterogeneity, and it's a clean tie

Uniform's third and final queued seed at alpha=0.1. Converged (rounds
28-32 plateau 77.16%->78.28%->78.16%, no spike).

| Seed | Final-round acc | Trailing-5-round mean | Upload |
|---|---|---|---|
| 42 | 79.11% | 78.84% | 794.04MB |
| 43 | 78.14% | 77.96% | 794.04MB |
| 44 (this) | 78.16% | 77.80% | 794.04MB |
| **Uniform, mean (n=3)** | **78.47% (±0.55pp)** | **78.20% (±0.56pp)** | **794.04MB** |
| **DivRoute, mean (n=3, §22)** | 78.35% (±1.07pp) | 77.91% (±0.59pp) | 1989.84MB |

**This is the first fully seed-matched (3 seeds vs. 3 seeds) comparison in
the entire investigation, and it is a clean statistical tie.** The gap
between the two methods' means (Uniform ahead by 0.12pp final-round,
0.29pp trailing-5) is smaller than either method's own seed-to-seed
standard deviation (0.55-1.07pp) — well inside noise, in both directions,
by both accuracy readings. Neither "DivRoute wins under heterogeneity" nor
"Uniform still wins outright" is supported by this data; what's supported
is that **the two methods perform indistinguishably on accuracy at real
non-IID splits**, a genuine change from the clean, decisive Uniform wins
at alpha=0.9 (§17, §20).

**What is unambiguous, across all six runs on both sides: bytes.**
DivRoute used 1880.66-2069.24MB upload on every seed; Uniform used exactly
794.04MB on every seed (deterministic — no tiering, so no per-seed
variation is possible). DivRoute pays a consistent, real **2.4-2.6x
upload premium for zero measurable accuracy benefit** at alpha=0.1. This
is now a stronger, cleaner, and more damaging result for DivRoute's core
claim than §17's original alpha=0.9 finding: at near-IID, DivRoute lost
outright; at real heterogeneity — the condition its own design premise
requires to have any signal to route on — it does not lose, but it also
does not win, and it still costs 2.5x the bytes to tie.

**Status: this closes the alpha question.** Across both heterogeneity
regimes tested (alpha=0.9: §17/§20, alpha=0.1: §21-§24, 3 seeds each at
the harder setting), there is no condition found in eleven structurally
valid attempts where DivRoute's divergence-based routing earns its
complexity over flat uniform top-k compression. The honest, evidence-based
statement of what this project has actually demonstrated: **error-feedback-
corrected top-k compression enables pretrained-backbone federated
fine-tuning to reach dense-FedAvg-competitive (in fact slightly exceeding,
per §20) accuracy at a large communication saving — and this holds
regardless of which client gets the bandwidth, not because of DivRoute's
specific routing intelligence.** That routing intelligence, exhaustively
tested across four distinct hypotheses (original signal, tier-polarity
inversion, hybrid loss+divergence signal, tier-aware error feedback) and
two heterogeneity regimes, has not once demonstrated a measurable
advantage.

**What would still move this, if pursued further (not required, at
diminishing returns given the above):** the routing-vs-compression-level
ablation flagged since §14 (`ablation_routing_no_compression` /
`ablation_uniform_compression`), run at alpha=0.1, would give one final,
maximally direct confirmation by isolating routing from compression level
within a single run rather than across separately-configured ones — but
given how clean and consistent the tie is across 6 independent runs here,
it is unlikely to overturn this conclusion, only to reconfirm it more
directly. At this point, writing up the negative/mixed result — which
this investigation has earned through unusually thorough, adversarial
self-testing (three real bugs fixed, four hypotheses tested, two
heterogeneity regimes, seed-matched comparisons) — is a more productive
use of remaining effort than further experiments chasing a positive result
the evidence does not support.

## Post-§24 literature check and strategic pivot

A literature search after §24 confirmed the "error-feedback-corrected
top-k compression reaches near-dense accuracy" claim is well-established
(EFSkip/AAAI 2025, FedSparQ, EF-Feddr all do this in FL settings already)
— real, but not novel on its own. More importantly, **federated LoRA/adapter
fine-tuning already achieves far larger communication savings** for the
same overall goal (95-99%+ reduction vs. this project's 60-90% upload
savings; Fed-SB reports up to 230x). And **HeteRo-Select (2025)**, which
uses client informativeness (not raw bandwidth) to drive participation and
compression allocation, reportedly *does* outperform uniform baselines —
the opposite of what DivRoute found here, in a closely related setting.
Full discussion of both findings was given directly to the user; not
duplicated here.

Decision: pursue two follow-ups in sequence rather than continuing to tune
DivRoute in its current setup (alpha=0.9/0.1 Dirichlet on a pretrained
EfficientNet-B0 backbone), which has been tested exhaustively (§13-§24).

1. **Harder heterogeneity (this section, infrastructure now built).** Every
   heterogeneity level tested so far (alpha=0.9, alpha=0.1) is Dirichlet —
   even at alpha=0.1, clients still see 38-55 of 100 classes each (verified
   via `_dirichlet_partition_indices()`, synthetic-label check below), a
   skewed-but-broad mixture, not the "each client mostly sees a couple of
   classes" pathological non-IID setting most divergence/importance-routing
   literature (including HeteRo-Select-style methods) is usually validated
   against. Added `Config.partition_mode` ("dirichlet" default / "pathological")
   and `Config.shards_per_client` (default 2) plus `data.py`'s
   `_shard_partition_indices()`: sorts all training indices by label, cuts
   into `num_clients * shards_per_client` contiguous shards, deals
   `shards_per_client` shards to each client at random (McMahan et al.,
   2017-style pathological split). Verified without GPU on synthetic
   CIFAR-100-shaped labels (100 classes x 500 samples): `shards_per_client=2`
   gives exactly 4 distinct classes per client (vs. 38-55 for Dirichlet(0.1))
   — confirmed materially harder heterogeneity, not a marginal change.
   Dirichlet path confirmed byte-for-byte unchanged (same RNG call sequence,
   refactored into `_dirichlet_partition_indices()` without altering logic);
   both paths confirmed deterministic per-seed, seed-sensitive, and cover
   every index exactly once with no duplicates. `main.py` wired through
   (`get_client_datasets`/`get_client_datasets_with_val` calls,
   `logger_meta`, and a new `[train] partition: ...` startup log line for
   traceability) and `Config` factory kwargs confirmed to accept both new
   fields (`get_pretrained_finetune_config(partition_mode=...)` etc.).
2. **LoRA/adapter update compression (not yet started).** Apply the
   existing EF+top-k+tiering machinery to LoRA update tensors instead of
   full-model deltas — addresses the LoRA-competitiveness gap directly
   (rather than competing with it) and reopens the routing question in a
   lower-dimensional update space that may retain more per-client signal
   than full-model deltas from a shared pretrained init. Requires new
   `model.py`/`client.py` work (LoRA injection, adapter-only local
   training); not started this session.

Not yet run: the actual pathological-partitioning DivRoute-vs-Uniform
comparison. See `DIVROUTE_REMAINING_WORK.md` for the ready-to-run commands.

## 25. FedAvg at alpha=0.1 — closing the missing accuracy baseline

Real dense FedAvg run at `alpha=0.1`, 20 rounds, seed 42 (shard sizes
min=1167/max=3407/mean=2000 confirm identical partitioning to the
DivRoute/Uniform alpha=0.1 seed-42 runs from §21-§24). This was the one
gap flagged earlier: every prior "without sacrificing accuracy" claim at
real heterogeneity rested on comparing compressed methods against each
other, never against a real dense baseline at that same heterogeneity
level. Converged (rounds 16-20: 76.33% -> 76.69% -> 76.79% -> 77.42% ->
77.49%, still inching up slightly but flattening — consistent with every
other FedAvg curve in this project, not a spike).

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| **FedAvg (dense, alpha=0.1, 20 rounds, real)** | 77.49% | 4962.78MB | 9925.56MB |
| DivRoute, mean (n=3, alpha=0.1, §22) | **78.35%** | 1989.84MB | 9930.28MB |
| Uniform-Top5%, mean (n=3, alpha=0.1, §24) | **78.47%** | **794.04MB** | **8734.49MB** |

**This closes the gap.** At real heterogeneity, exactly as at alpha=0.9
(§20), both compressed methods don't just avoid sacrificing accuracy
relative to dense FedAvg — they slightly *exceed* it: DivRoute +0.86pp at
59.9% less upload, Uniform +0.98pp at 84.0% less upload. The
"error-feedback-corrected compression reaches near-dense (here, slightly
above-dense) accuracy at large communication savings" claim now holds with
a real baseline in both heterogeneity regimes tested in this project, not
just the easier near-IID one. (Caveat: this is one FedAvg seed against the
3-seed DivRoute/Uniform means — a real, though minor, asymmetry; the
comparison is conservative in FedAvg's favor if anything, since FedAvg
here is a single point rather than an averaged one.)

This does not change §24's routing verdict — DivRoute and Uniform are
still statistically tied with each other at alpha=0.1, and Uniform still
does it more cheaply. What it does establish is that the *general*
(non-DivRoute-specific) claim — the one worth publishing per the earlier
literature discussion — is now fully validated, not just plausible, at
both heterogeneity levels tested.

## 26. Fifteenth attempt: pathological non-IID (2 shards/client) — promising, but neither method has converged

First run using the new `partition_mode="pathological"` infrastructure
(§ "Post-§24 literature check and strategic pivot"). `[train] partition:
pathological (2 shards/client)` and uniform shard sizes (min=max=2000)
confirm the infrastructure applied correctly. DivRoute and Uniform, both
plain-EF, 32 rounds, single seed (42).

| | Final-round acc | Upload |
|---|---|---|
| DivRoute | 44.84% | 2069.24MB |
| Uniform-Top5% | 37.10% | **794.04MB** |

**Absolute accuracy collapsed relative to every Dirichlet run in this
project** (37-45% here vs. 77-83% at alpha=0.1/0.9) — expected, not a bug:
2 classes/client causes severe client drift under plain local SGD, a
well-documented FedAvg failure mode under pathological non-IID that
typically needs many more communication rounds (or drift-correction
methods like FedProx/SCAFFOLD, neither implemented here) to stabilize.

**Critical caveat, unlike every prior result in this document: neither
method has converged.** Checked rounds 15-32 (not just the usual last
5-9): DivRoute climbs 32.27% -> 44.84% (+12.6pp) and Uniform climbs
30.49% -> 37.10% (+6.6pp) over that window, with no flattening. Every
number above should be read as "accuracy after 32 rounds," not "achievable
accuracy" — extrapolating a verdict from an unconverged curve would repeat
exactly the mistake §22 caught for alpha=0.1 seed 42, just one step
earlier in the pipeline (there the problem was single-seed noise on a
converged curve; here it's reading a curve before it's had the chance to
converge at all).

**What is nonetheless worth noting:** DivRoute leads Uniform in every one
of the last 9 rounds (24-32), not just the final one — average margin
+2.44pp, ranging +0.12pp to +7.74pp. That is a qualitatively different,
more consistent pattern than §21's alpha=0.1 result, where seed 42's
apparent DivRoute lead flipped sign under a trailing-round-average and
then failed to replicate across seeds (§22). Consistent, directional
leads across 9 noisy-but-unconverged rounds is a more promising early
signal than anything seen in the Dirichlet regime — but it is not yet
evidence of anything, because an unconverged curve's ranking can still
change completely once both methods actually plateau.

**Status: extend rounds before seed-repeating.** The right next
diagnostic here is *not* immediately seed-repeating at 32 rounds (that
would just produce several more unconverged, hard-to-interpret curves) —
it's extending this exact run to more rounds until both methods actually
plateau, the same convergence bar applied to every other result in this
document. Only once a real plateau is found should seed repeats follow,
mirroring the alpha=0.1 methodology (§21-§24) at whatever round count
that turns out to be.

## FedProx: a targeted fix for the instability §26 found

A literature review of how existing systems combat exactly what §26 showed
(unstable, slow-converging curves under severe heterogeneity — textbook
"client drift") turned up the standard, well-established fix:
**FedProx** (Li et al., 2020) adds a proximal term
`(mu/2) * ||w_local - w_global||^2` to each client's local loss, pulling
every local step back toward the global model it started from. Considered
alongside **SCAFFOLD** (control-variate-based drift correction — more
powerful, corrects bias rather than damping it, but requires persistent
per-client state transmitted every round, adding real communication
overhead that cuts against this project's efficiency framing) and
**FedNova** (aggregation-weight normalization for heterogeneous local step
counts — relevant given shard sizes vary, but addresses a narrower slice
of the problem). FedProx was implemented first: zero extra communication,
directly targets the observed instability, and the codebase already had
an almost-identical pattern to build on (`fedsparse_lambda`'s frozen
global-weight snapshot in `client.py`, previously used for a post-hoc
proximal *thresholding* step, not a loss term — FedProx is implemented as
its own independent mechanism, not a reuse of that one, since the two are
structurally different operations that happen to share the "snapshot the
global weights" step).

**Implemented:** `Config.use_fedprox` (default `False`) / `Config.fedprox_mu`
(default `0.01`, literature range 0.001-1.0 by heterogeneity severity) in
`config.py`; `client.py`'s `train()` gained a `fedprox_mu` parameter, a
frozen global-weight snapshot (independent of the existing FedSparse one),
and the proximal term added to the loss right after the NTD block, mirroring
that block's exact style; `main.py` wired through the call site and added
`fedprox: ...` to the `[train]` startup log line for traceability.

**Verified without GPU:** config defaults and factory-kwarg overrides
checked (`use_fedprox=False` by default, a true no-op — both the snapshot
block and the loss-addition block are gated on `fedprox_mu > 0.0`, which
never holds unless a caller explicitly opts in). The proximal term's value
and gradient were checked against the analytical formula on a toy tensor
(`d/dw[(mu/2)*sum((w-w_g)^2)] = mu*(w-w_g)`) — both matched exactly.

**Not yet run.** The most informative first test is FedProx *at the same
32-round pathological setting just analyzed* — not immediately combined
with a round extension — because FedProx's actual promise is faster/more
stable convergence in the *same* round budget, not just eventual
convergence given enough rounds. If it visibly stabilizes and flattens the
curves within 32 rounds, that both confirms the client-drift diagnosis and
may remove the need to extend rounds at all. If curves are still unstable
even with FedProx, that points toward SCAFFOLD (or simply more rounds)
instead. See `DIVROUTE_REMAINING_WORK.md` for the ready-to-run commands.

## 27. Sixteenth attempt: FedProx (mu=0.01) at the pathological setting — hypothesis not confirmed

Both DivRoute and Uniform, pathological (2 shards/client), 32 rounds,
`use_fedprox=True, fedprox_mu=0.01` (confirmed active via the `fedprox:
True (mu=0.01)` startup log line in both runs). Compared directly against
§26's non-FedProx curves at the same setting.

| | Final acc, no FedProx (§26) | Final acc, FedProx mu=0.01 | Delta |
|---|---|---|---|
| DivRoute | 44.84% | 40.57% | -4.27pp |
| Uniform-Top5% | 37.10% | 36.71% | -0.39pp |

**No stabilization.** Rounds 15-32 still climb steadily for both methods
(no flattening) with round-to-round noise of the same magnitude as
without FedProx — the client-drift-correction hypothesis is not confirmed
by this data. Final accuracy got slightly worse for Uniform and
noticeably worse for DivRoute, not better.

**More importantly, FedProx erased the one interesting pattern §26
found.** Without FedProx, DivRoute led Uniform in all 9 of the last 9
rounds (avg +2.44pp). With FedProx, DivRoute leads in only 4 of 9 — the
ranking flips round to round now, much closer to the coin-flip pattern
seen throughout the alpha=0.1 Dirichlet results (avg gap shrinks to
+0.77pp). Whatever produced §26's consistent lead, this specific
intervention removed it rather than sharpening it — the opposite of the
hoped-for "cleaner signal" effect.

**Caveat before ruling FedProx out entirely:** `mu=0.01` is the weak end
of the literature's cited 0.001-1.0 range; it's possible a stronger mu
would show a real stabilizing effect this one didn't. But given this
investigation's track record — every "maybe a stronger/different version
of this will work" follow-up has come up short so far (§15 polarity
inversion, §18 hybrid signal, §20's tier-aware-EF ablations, and now
this) — further mu-tuning is a low-expected-value use of compute relative
to the two paths already agreed: extend rounds (without FedProx, since it
didn't help) to see if §26's promising signal survives actual convergence,
or move on to the LoRA pivot, the second half of the plan regardless of
how the heterogeneity track resolves.

**Recommendation: extend rounds without FedProx next**, not more FedProx
tuning. §26's 9/9-consistent-lead pattern is still the most promising
signal in this entire investigation and deserves to be checked at
convergence before being abandoned — but that check should isolate one
variable at a time, and FedProx has now been shown not to help at this
strength.

## 28. Seventeenth attempt: DivRoute at 64 rounds (pathological, no FedProx) — it converges, ~3pp below §26's number

DivRoute only (`fedprox: False` confirmed in the startup line); the
Uniform 64-round run was not in this log, so **no DivRoute-vs-Uniform
conclusion is possible yet.** Pathological (2 shards/client), plain EF,
seed 42. Upload 4203.00MB (73.5% saving), bidirectional 20083.89MB
(36.8% saving); tier mix stayed healthy to the end (e.g. 9/3/3 at round
50, 11/1/3 at round 62 — no collapse).

**Trajectory: converged.** Climbs 2.5% (round 1) to ~39% by round 23, then
sits on a flat plateau for the remaining ~40 rounds:

| Window | Mean acc | Notes |
|---|---|---|
| Rounds 33-44 | 41.16% | |
| Rounds 45-64 | 41.06% | round-to-round std ~0.94pp |
| Peak single round | 43.52% (round 43) | |
| Final (round 64) | 39.97% | |

The plateau is flat from about round 30 (window means 41.16% vs. 41.06%
are indistinguishable), so ~41% is DivRoute's converged level under this
setting — the convergence question §26 left open is answered for DivRoute.

**§26's 44.84% overstated the converged level.** The 32-round run's final
accuracy (44.84%; trailing-5 mean 43.59%) sits 2.5-3.8pp above the 64-round
plateau. The two runs are not round-for-round comparable — the cosine LR
schedule is a function of `num_rounds`, so a 64-round run follows a
different learning-rate path from round 1 — and this data cannot separate
"favorable trajectory" from "short-schedule annealing effect." (Annealing
alone is not an obvious explanation: the 64-round run also reaches the LR
floor by its end and shows no late bump.) What it does show is the same
lesson as §22: a single run's final number, read before the curve is
stress-tested, overstated DivRoute's level.

**Data-integrity note:** rounds 26-27 are missing from the pasted console
log (round 25 is followed directly by round 28). The learning-rate values
on either side match the cosine schedule exactly (0.007016 at round 25,
0.006378 at round 28) and accuracy continues smoothly (38.46% to 38.73%),
so training appears unbroken and this is most likely a live-output gap
(browser disconnect) rather than a restart. It sits in the pre-plateau
phase and does not affect the plateau estimate; the run's JSON log should
contain all rounds if verification is wanted.

**What this changes for the open question.** §26's interesting signal was
DivRoute leading Uniform in 9/9 of the last rounds at 32 rounds. Two facts
now bear on it: DivRoute's converged level is ~41%, and Uniform's noisy
32-round trailing-5 mean was already 40.4% while still rising. If
Uniform's 64-round plateau lands near 40-41%, §26's lead was a transient
(DivRoute happening to sit in a favorable excursion at round 32), and the
verdict is another tie; if it lands near or below ~38%, DivRoute has a real
~3pp ceiling advantage at pathological heterogeneity and seed repeats
become the priority. Judgment, not a result: this shifts the balance
modestly toward the tie outcome (roughly 20-25% odds of a real advantage,
down from my earlier 30-35%).

**Status: the Uniform 64-round run (same config) is the one missing
number.** Also still open: no dense FedAvg baseline exists at pathological
heterogeneity, so "without sacrificing accuracy" is unchecked at this
setting (it was closed for alpha=0.9 in §20 and alpha=0.1 in §25).

## 29. Uniform at 64 rounds — partial log, a real resume, and a resume bug found and fixed

**What the pasted log contains.** A stitched Kaggle notebook: rounds 1-31
of the original run, then two notebook cells (an `ls` of the checkpoint
folder and the resume command), then a resumed segment covering rounds
56-59. Rounds 32-55 are not in the paste, and it ends at round 59 with no
`[done]` line (mid-run or stopped again), so **there is no final Uniform
64-round number yet.** The `ls` shows `latest.pt` timestamped 15:58 and
`round_050.pt` at 15:11 — i.e. the original process had carried on well past
round 31 (checkpoint at round 55) while the *console display* stopped at 31;
the "interruption at 31" was most likely the live output freezing, the same
kind of gap seen in §28's DivRoute log (rounds 26-27). The run's JSON log is
flushed every round and restored from the checkpoint on resume, so it should
hold rounds 1-59+ in full.

**Resume works.** `Resuming from checkpoint ... Loaded checkpoint ...
Resuming from round 55/64`, then round 56 with `local_lr=0.001343`, which
matches the cosine schedule at round 56 (0.001342). Model, server state,
error-feedback buffers, cumulative byte counters and logger history all
restored without error.

**Bug found in the resume path, now fixed.** Both torch RNG restores failed
(`RNG state must be a torch.ByteTensor`) and were silently skipped. Cause:
`torch.load(..., map_location=device)` puts the saved RNG-state tensors on
the GPU, `torch.set_rng_state` only accepts CPU tensors, and the old
`isinstance(..., torch.ByteTensor)` guard did not catch a CUDA tensor.
Reproduced locally with the identical message; fixed in `main.py` by
moving the states to CPU (`.cpu().to(torch.uint8)`) before restoring, and
confirmed that a restored state reproduces the same random numbers.
Consequence for the run in flight: client selection is unaffected (the
python/numpy/server RNGs restored with no warning), but torch's stream
(MixUp draws, batch order) restarted from the process-start seed instead of
continuing — statistically valid training, not bit-identical to an
uninterrupted run. Worth a sentence in any write-up that reports this run
(it was resumed from round 55). The fix only matters for future resumes.

**Provisional comparison (same 64-round schedule, so the LR paths match).**
Only what overlaps in the two logs; not a verdict.

| Window | DivRoute (§28) | Uniform | Note |
|---|---|---|---|
| Rounds 21-25, 28-31 (9 rounds) | 38.82% | 38.54% | tie |
| Rounds 56-59 (4 rounds) | 42.08% | 43.40% | Uniform +1.3pp, noise is ~±1pp/round |

The two curves track each other closely through the overlap and nothing here
shows a persistent DivRoute lead — consistent with the tie outcome and
against §26's 9/9 lead being a real ceiling advantage at the 32-round
schedule. Caveat: 13 comparable rounds, gaps of 24 rounds missing, single
seed. **Need Uniform's full plateau (rounds 45-64 mean) before calling it.**

## 30. Decision: routing/heterogeneity track closed by deduction; LoRA novelty premise corrected

**Deduction from the partial Uniform 64-round log (no rerun).** Uniform's
late rounds (56-59: 42.72/41.62/45.51/43.73; 29-31: 43.40/42.59/39.94) sit at
or above DivRoute's converged level (41.06%, §28). A real ~3pp DivRoute
ceiling advantage would need Uniform to plateau near 38%; four consecutive
late rounds >=41.6% (noise ~1pp at low LR) rule that out. So §26's
32-round lead was schedule-specific — it also disappeared under FedProx
(§27) and does not appear on the matched 64-round schedule. Routing has now
failed to beat uniform top-k at all three heterogeneity levels: alpha=0.9
(loses), alpha=0.1 (ties, 3 seeds each), pathological (ties or trails,
single seed, Uniform's final plateau not observed), at a consistent
2.4-2.6x upload premium. Caveat that stays: single seed at pathological, no
dense FedAvg baseline there.

**Correction to an earlier statement.** I said compressing LoRA updates with
EF+top-k+tiering looked like unclaimed territory (with a caveat that I had
not checked closely). Reading the closest abstracts shows it is crowded:
FedSRD (importance-aware sparsified LoRA uploads, up to 90% communication
reduction, LLMs, WWW 2026), EcoLoRA (adaptive LoRA sparsification +
round-robin segments, up to 79% communication-time reduction, LLMs), and a
two-stage wireless federated-LoRA paper with sparsified updates evaluated on
CIFAR-100 with ViT-Base (arXiv 2505.00333). Abstracts only; whether each uses
error feedback is not stated. Consequence: LoRA-update compression cannot be
the paper's novel positive method. Its only defensible role is a scoped
generality test of the negative finding (does routing help in low-rank
update space?) and/or an extreme-savings regime, and it is not required for
the study the current evidence supports.

**Supported claim set.** (C1) EF-corrected top-k matches or exceeds dense
FedAvg accuracy at 60-90% less upload (alpha=0.9, 0.1; unchecked at
pathological) — real, not novel. (C2) Divergence routing gives no accuracy
benefit over uniform top-k at any tested heterogeneity and costs 2.4-2.6x
the upload — the strong, well-controlled negative result. (C3) Method:
three measurement bugs found/fixed, plus single-seed traps (alpha=0.1 seed
42 outlier; 32-round final overstating the plateau by ~3pp). (C4) Refuted
mechanism attempts: polarity inversion, hybrid loss+divergence signal,
tier-aware EF, FedProx(0.01).

**Next step: consolidate and write.** Build the claims-vs-evidence map and
results tables from these sections; then choose only the gap-closing runs a
reviewer would demand, in priority order: (1) a second backbone/dataset for
generality, (2) dense FedAvg at pathological (only if pathological numbers
are reported), (3) seeds at pathological (same condition), (4) optional
LoRA generality cell.


## 31. FEMNIST from scratch — implementation, verified facts, and an important correction

**Why this setting.** Every result in §13-§30 fine-tunes a shared ImageNet-pretrained
EfficientNet-B0 on CIFAR-100 with *simulated* partitions. The routing negative result
(C2) is therefore exposed to two objections: the heterogeneity is synthetic, and a
shared pretrained init may starve divergence-based routing of signal (a hypothesis of
mine, not a tested fact). FEMNIST trained from scratch tests C2 in the regime DivRoute
was designed for, on a real per-writer partition. It also serves the generality demand
(a second dataset and a second, non-pretrained regime).

**Correction to my earlier recommendation.** I described FEMNIST's natural partition as
strong heterogeneity and "the setting most favorable to routing". Measured, that is
overstated (see below): FEMNIST's *label* skew is only slightly above what IID sampling
noise alone produces. Its heterogeneity is mostly handwriting-style (feature) shift.
The from-scratch regime is still a valid and informative test; "strongly non-IID" is not
a claim this dataset supports.

**Verified dataset facts** (read from the Hub dataset `flwrlabs/femnist`, not assumed):
one `train` split only (814,277 examples; ~200 MB), **no official test split**; 3,597
distinct `writer_id`s; samples/writer min 18, median 178, mean 226.4, max 583; 62 classes
(0-9, A-Z, a-z), class frequencies from 2,214 to 44,706; 28x28 8-bit grayscale PNG, white
background (pixel mean 0.9636, std 0.1594 on 5,000 random samples).

**Protocol (my own, documented in `femnist_data.py`; not borrowed from any paper).**
Eligible writers have >= 100 samples (drops 39 of 3,597). 200 eligible writers are held
out with a fixed seed (independent of the run seed) as the global test set: 45,971
samples, identical for every method and seed, and never seen in training, so accuracy
measures generalisation to unseen writers. Training clients are `num_clients` writers
drawn from the remaining pool (3,358) with the run seed; one writer = one client.
**Consequence: absolute accuracies are not comparable to numbers reported under other
FEMNIST protocols (e.g. per-user sample splits).**

**Measured heterogeneity** (`scripts/femnist_sanity_check.py`, 100 writers, seed 42):
clients hold 106-416 samples (median 175, mean 216.1; 21,605 total) and cover 10-62
classes (median 57, mean 54.6 of 62). Mean total-variation distance of a client's label
distribution from the pooled one: **0.249 for FEMNIST writers vs 0.197 for an IID split
with identical client sizes** (the sampling-noise floor). Yardsticks from the same
function on synthetic CIFAR-100-shaped labels (25 clients, 2,000 samples each): Dirichlet(0.9)
0.378, Dirichlet(0.1) 0.730, pathological 0.960. Caveat: TV also depends on client size
and class count, so the yardsticks have a different noise floor; the like-for-like
comparison is 0.249 vs 0.197.

**Centralised reference** (same script; from-scratch CNN, pooled 100 training writers,
lr 0.03, batch 32, 20 epochs, no MixUp/NTD): held-out-writer accuracy reaches 82.0% by
epoch 3 and then fluctuates between 81.4% and 82.8% (epochs 5-20; final 81.9%). This is
the ceiling to read federated results against, with the caveat that federated local
training additionally applies MixUp and NTD.

**What was implemented.** `divroute_fl/femnist_data.py` (loader/protocol);
`FemnistCNN` in `model.py` (conv5x5-32, pool, conv5x5-64, pool, fc-512, fc-62; no
BatchNorm; **1,690,046 parameters**, counted); dispatch and guards in `data.py`
(`partition_mode='natural'` is accepted only for FEMNIST, and Dirichlet/pathological only
for CIFAR, so a run's label can never disagree with its data); `client.py` loader branch
extended so FEMNIST keeps `drop_last=False` like CIFAR-100 (the old `num_classes == 100`
test would have silently dropped samples from these small shards); `main.py` no longer
hard-codes 10-vs-100 classes; three factories `get_femnist_{fedavg,uniform,divroute}_config()`
mirroring the pretrained ones (same optimisation, `use_k_warmup=False`, same `ntd_beta`,
same epoch-warmup, `rolling_percentile` thresholds for DivRoute, plain error feedback for
both compressed methods). Aggregation weighting was checked in `server.py`: all three
methods weight by client sample count (DivRoute additionally multiplies its divergence
weight), so there is no hidden weighting confound across methods.
Loading speed: cold start (open dataset + build 100 clients + build the test set) went
from ~155 s to ~33 s. The whole gain is reading the writer/label columns straight from
the Arrow table (~97 s -> ~5 s, values identical to the list-based read). Decoding all
writers in one batched call was tried and measured to be no faster than per-writer
decoding (6.3 s vs 5.7 s, identical pixels), so the simple per-writer loop is kept.
Client pixels/labels re-verified against an independent per-image decode on 3 clients.

**Verification performed (all local, real data).** Same seed -> identical clients;
different seed -> different writers; 0 overlap between test writers and the training
pool. Regression: partition guards behave as intended, CIFAR-10 keeps `drop_last=True`,
CIFAR-100 unchanged, existing pretrained factories unchanged. 6-round smoke test of all
three methods: dense FedAvg upload = 101.403 MB/round, exactly
1,690,046 x 4 B x 15 clients; Uniform = 10.140 MB/round (exactly 10% of dense, as in the
CIFAR runs); DivRoute tiers active and varied (5/5/5 in round 1 -> 20.280 MB, matching
5x2.704 + 10x0.676 MB computed by hand); round-5 checkpoint written. (Round-6 accuracies of
47-52% come from a 6-round schedule and say nothing about method ranking.) Divergence
scores are ~2e-4 to 8e-3, i.e. below the 0.01/0.02 default thresholds, confirming that
scale-independent thresholds were the right choice again.

**Learning-rate calibration (dense FedAvg only, so no compressed method is favoured).**
Candidates {0.01, 0.03, 0.1}, cosine floor = lr/10, 60 rounds, seed 42, plateau = mean of
last 10 rounds: 0.01 -> 77.77% (round-to-round std 0.19), **0.03 -> 79.42% (0.31)**,
0.1 -> 79.10% (0.37). 0.03 is best and interior to the grid, but its margin over 0.1
(0.32pp) is about one round-to-round std, from one seed and three candidates - a coarse
choice, not a finely tuned one. 60 rounds is **not converged** at lr 0.03 (10-round window
means 51.9, 72.7, 75.9, 77.0, 78.5, 79.4), so the study's round count is set separately
below.

**Round count (150), from a full dense-FedAvg run at the chosen lr.** FedAvg, lr 0.03
(cosine floor 0.003), seed 42, 150 rounds. Ten-round window means, computed by hand
from the run's per-round accuracies: 51.86, 72.68, 75.84, 76.61, 78.27, 78.72, 79.21,
79.76, 80.02, 80.52, 80.44, 80.93, 81.00, 81.35, **81.44%**. The last two windows differ
by 0.09pp (round-to-round std of the last window ~0.28pp), so the curve has plateaued at
about 81.4% - close to the ~82% centralised reference (81.4-82.8%, which trains on the same
100 writers but without MixUp/NTD). Dense upload for the run is 150 x 101.403 = 15,210 MB.
The three FEMNIST factories therefore default to `num_rounds=150`. **This plateau is
verified for dense FedAvg only.** Error-feedback top-k may converge more slowly, so each
method's own last two 10-round windows must be checked before any comparison; if they
differ by more than ~0.5pp, the run needs more rounds.

**How to run (implemented, not executed by me).** `scripts/run_femnist_experiments.py`
runs the 3 methods x seeds 42/43/44 (default) at 150 rounds from the repo root. It skips
runs whose JSON already holds the full round count, auto-resumes interrupted runs, names
runs `femnist_<method>_r<rounds>_s<seed>`, deletes each finished run's checkpoint folder
(error-feedback checkpoints are ~375 MB after 6 rounds - measured - and grow towards
~690 MB with 100 clients, so keeping nine runs plus milestone copies could approach
Kaggle's disk limit), and prints the exact `summarize_runs.py` command for the results.
Time: two 60-round dense-FedAvg runs took 775 s and 911 s wall-clock on the RTX 3050
laptop I used (the first also paid the one-off dataset load), i.e. about 13-15 s per
round, so roughly 32-38 min per 150-round run and ~5-6 h for the nine-run grid at that
rate. Kaggle speed and the compressed methods' overhead were not measured.

**What is verified vs not.** Verified locally on real data: the loader and protocol
(determinism, disjoint writers, pixel-exact against an independent decode), the model
size, the three configs' byte accounting and tier activation in a 6-round smoke test,
regression of the CIFAR paths, the lr calibration, and the 150-round FedAvg plateau.
**Not verified:** `run_femnist_experiments.py` itself was written but never executed (its
skip, cleanup and summary-print logic are unexercised; the calls it makes were); nothing has
been run on Kaggle (internet must be on for the ~200 MB Hub download, and I did not check
whether `datasets` is preinstalled there - `pip install datasets` if `import datasets`
fails); and no compressed method has been run beyond 6 rounds.

**Disclose in any write-up.** (1) The protocol (>=100 samples/writer, 200 fixed held-out
test writers, training writers sampled per seed) is this project's own, so absolute
accuracies are not comparable with other FEMNIST reports. (2) Label skew is mild (TV 0.249
vs IID floor 0.197); the heterogeneity is mostly handwriting style. (3) lr was tuned on
FedAvg only, on one seed and three values. (4) MixUp is hard-coded on in `client.py` for
every method; it is identical across methods but not standard for character images.
(5) Normalisation constants come from a 5,000-image sample. (6) The 150-round plateau
was checked for FedAvg only. (7) 100 clients / 15 per round is a documented choice, not a
tuned one.


## 32. FEMNIST result: Uniform beats DivRoute by a real margin, at half the bytes

All 9 runs (fedavg/uniform/divroute x seeds 42/43/44, 150 rounds) completed on Kaggle.
`fedavg_r150_s42` was skipped (its log already existed from an earlier session on that
Kaggle account) and its numbers are not in this paste; the core routing comparison below
does not need FedAvg. Read from the pasted console log (not the raw JSON) via grep, so
transcription risk exists but was cross-checked against the printed `[summary]` lines
where both were present -- they agree everywhere checked.

**Plateau check first** (last 20 rounds each, split into two 10-round windows,
per section 31's stated bar of "well under 0.5pp" between them):

| Run | w131-140 | w141-150 | diff | last-20 mean | last-20 std |
|---|---|---|---|---|---|
| uniform s42 | 81.19 | 81.34 | +0.15 | 81.27 | 0.26 |
| divroute s42 | 80.43 | 80.68 | +0.25 | 80.55 | 0.55 |
| fedavg s43 | 81.61 | 81.77 | +0.16 | 81.69 | 0.29 |
| uniform s43 | 81.64 | 81.79 | +0.15 | 81.71 | 0.32 |
| divroute s43 | 81.02 | 81.12 | +0.10 | 81.07 | 0.28 |
| fedavg s44 | 81.57 | 81.63 | +0.06 | 81.60 | 0.26 |
| uniform s44 | 81.49 | 81.58 | +0.08 | 81.53 | 0.26 |
| **divroute s44** | 80.59 | 79.97 | **-0.62** | 80.28 | 0.69 |

Seven of eight runs clear the bar. **divroute s44 does not** (-0.62pp, its own noisiest
run at std 0.69) -- driven by a single-round dip to 78.13% at round 144 with no NaN/Inf
warning nearby, i.e. ordinary client-sampling noise rather than an error, but the window
mean is real and not excluded below. Kept in, flagged, not discarded.

**DivRoute vs. Uniform, 3 seeds each, plateau = mean of the last 20 rounds
(the comparison this whole FEMNIST addition exists to make):**

| | Seed 42 | Seed 43 | Seed 44 | Mean | Across-seed std | Upload |
|---|---|---|---|---|---|---|
| DivRoute | 80.55% | 81.07% | 80.28% | **80.63%** | 0.40 | 3190.8MB |
| Uniform-Top5% | 81.27% | 81.71% | 81.53% | **81.50%** | 0.22 | **1521.0MB** |

Gap (Uniform - DivRoute) = **+0.87pp**, against a max across-seed std of 0.40pp -- **larger
than noise, the first time in this entire investigation that a DivRoute-vs-Uniform gap has
exceeded that bar in either direction with 3 seeds.** Not driven by the flagged seed
either: dropping divroute-s44 entirely, the remaining two seeds alone already show the
same direction and a similar size (Uniform ahead by 0.72pp at seed 42, 0.64pp at seed
43). DivRoute also costs 2.10x Uniform's upload (3190.8MB vs. 1521.0MB) for this loss,
not a tradeoff.

**This is the strongest evidence against the routing claim (C2) obtained anywhere in the
project.** It was measured in the one setting explicitly chosen to give the mechanism its
best chance: real per-writer partitioning (not simulated) and from-scratch training (not
fine-tuning from a shared pretrained init, the confound hypothesised in section 31 as a
reason CIFAR-100 pretrained fine-tuning might be suppressing a real routing signal). With
that confound removed, DivRoute does not tie here as it did on CIFAR-100 -- it loses by a
margin larger than the seed noise. Combined with alpha=0.9 (loses), alpha=0.1 (ties, 3
seeds), pathological (ties/trails, 1 seed): four heterogeneity/regime combinations tested,
and divergence-based routing has not once beaten uniform top-k compression by a real
margin, while losing outright in the two settings farthest from near-IID pretrained
fine-tuning (pathological simulated, and now FEMNIST real-partition from-scratch).

**Caveats that still apply.** Section 31's disclosure list in full (protocol is this
project's own, mild label skew, lr tuned coarsely on FedAvg only, MixUp on for a
character-image task, one flagged non-plateaued run). Numbers here are from a console
paste, not the JSON logs; re-verifying with `scripts/summarize_runs.py` against the
actual JSON files once downloaded would remove that risk and is a cheap check worth
doing before this goes in a final table.

**Not yet done:** no dense FedAvg baseline for seed 42 (skipped run); no check of whether
FedProx or a different lr changes this (not motivated by anything observed here -- the
runs converged and the gap is real, so there is no instability to fix this time).


## 33. Evidence-strength audit across both datasets, and the two FEMNIST loose ends

**Tooling fix, verified.** `scripts/summarize_runs.py`'s compare step now reports every
pair of `--group`s, not just the first two -- needed because section 32's finding is a
three-way comparison (FedAvg/Uniform/DivRoute), not two. Checked against the real FEMNIST
numbers (reconstructed from the section-32 console log) before use: reproduces gap -0.87pp
DivRoute-vs-Uniform (larger than noise), surfaces DivRoute-vs-FedAvg at -1.01pp (also
larger than noise, provisional on 2 FedAvg seeds), and Uniform-vs-FedAvg at -0.14pp
(within noise, a tie) -- all three consistent with the numbers already in section 32,
computed independently by the script rather than by hand this time.

**The two loose ends require a Kaggle command, not something I can do from here** --
I have no access to the Kaggle session or its files, only the console text pasted into
this conversation. One command closes both:
```
!python scripts/run_femnist_experiments.py --only fedavg --seeds 42
!python scripts/summarize_runs.py --last 20 \
    --group DivRoute logs/femnist_divroute_r150_s42.json logs/femnist_divroute_r150_s43.json logs/femnist_divroute_r150_s44.json \
    --group Uniform  logs/femnist_uniform_r150_s42.json logs/femnist_uniform_r150_s43.json logs/femnist_uniform_r150_s44.json \
    --group FedAvg   logs/femnist_fedavg_r150_s42.json logs/femnist_fedavg_r150_s43.json logs/femnist_fedavg_r150_s44.json
```
The first line either finds `femnist_fedavg_r150_s42.json` already in that session's
`logs/` (prints `[skip]`, instant) or runs it fresh (~12 min, matching the other FedAvg
seeds' wall-clock). The second line re-derives every section-32 number from the actual
JSON files rather than a console-log grep, and adds the FedAvg comparison section 32
could not make (seed 42 was missing). Paste the output back; if any of its numbers
disagree with section 32's hand-extracted ones, the JSON version is authoritative.

**Evidence-strength tiers, both datasets, current state.** Built so a paper draft can be
checked against it sentence by sentence rather than trusting prose written weeks apart:

| Setting | Method comparison | Seeds | Convergence check | Tier |
|---|---|---|---|---|
| CIFAR-100 pretrained, alpha=0.9 | DivRoute vs Uniform vs FedAvg | 1 each | passed (plateau) | B - single seed, but the gaps are large relative to any noise observed elsewhere (SS17/S20) |
| CIFAR-100 pretrained, alpha=0.1 | DivRoute vs Uniform | 3 each | "passed" only at the 32-round budget; a 64-round Uniform run is still climbing (S35) | **A for equal-budget ranking at 32 rounds; NOT a converged-level claim** - seed-matched, gap smaller than spread, a tie at that budget (SS21-24, amended S35) |
| CIFAR-100 pretrained, alpha=0.1 | Uniform/DivRoute vs FedAvg | 3 vs 1, and S25 compared 32-round compressed runs to a 20-round FedAvg | **S25's "compressed beats FedAvg" is a budget artifact, see S36**; matched 64-round seed-42 pair: Uniform trails FedAvg by ~0.5pp, neither converged | C - single seed, unconverged; S25's claim withdrawn as an equal-rounds statement (S36) |
| CIFAR-100 pretrained, pathological | DivRoute vs Uniform | 1 each | DivRoute passed; Uniform's plateau not directly observed, only deduced from partial late rounds | **C - weakest link.** No dense FedAvg baseline at all here (SS26-30) |
| FEMNIST from scratch | DivRoute vs Uniform | 3 each | 7/8 runs passed; divroute-s44 missed the bar by 0.62pp (flagged, kept in) | **A** - seed-matched, gap larger than spread, a real loss for DivRoute (S32) |
| FEMNIST from scratch | FedAvg vs Uniform vs DivRoute | 3 vs 3 vs 3 | passed (seed 42 added S34) | **A** - seed-matched; FedAvg ties Uniform, DivRoute loses to both (S34) |

**What this means for how the paper should state things.** The alpha=0.1 CIFAR-100 tie
and the FEMNIST loss are both Tier A -- seed-matched, and can be stated as plainly as the
evidence lets you (a real tie; a real loss with upload cost on top). The pathological
CIFAR-100 result (Tier C) should not be stated as confirmed in the same register --
"single seed, Uniform's plateau not directly observed" needs to stay attached to it, not
be dropped for cleaner prose. If pathological CIFAR-100 numbers go in a table at all
without further runs, they belong in a clearly-marked supplementary/lower-confidence row,
not alongside the Tier A results. Section 21's original caveat about single-seed noise
overstating a lead (later confirmed by section 22) is exactly the failure mode Tier C is
still exposed to.


## 34. FedAvg seed 42 closes the loop, and a Kaggle-persistence gap found in the process

`fedavg_r150_s42` was run (not skipped this time -- see below) and converged cleanly:
last two 10-round windows 81.34% -> 81.42% (+0.09pp), plateau mean **81.38%**, std 0.24.

**The follow-up `summarize_runs.py` command failed**: `FileNotFoundError` on
`logs/femnist_divroute_r150_s42.json`. Cause, confirmed by the run printing `RUNNING`
rather than `[skip]` for fedavg-s42 this time: this was a different Kaggle session than
the one behind section 32, and `/kaggle/working` does not persist between sessions unless
a version's output was explicitly saved. The other 8 runs' JSON files are not lost, they
are simply not present in *this* session -- and were never re-run, so section 32's numbers
stand. Practical fix for future runs: download `logs/` (or click "Save Version" so Kaggle
attaches the output) before a session ends, rather than assuming it carries over.

**Full seed-matched three-way comparison, computed by hand from the console numbers
(section 32's for Uniform/DivRoute, this run's for the missing FedAvg seed) since the
JSON files needed for `summarize_runs.py` are not available in this session:**

| | Seed 42 | Seed 43 | Seed 44 | Mean | Across-seed std |
|---|---|---|---|---|---|
| FedAvg (dense) | 81.38% | 81.69% | 81.60% | **81.56%** | 0.16 |
| Uniform-Top5% | 81.27% | 81.71% | 81.53% | 81.50% | 0.22 |
| DivRoute | 80.55% | 81.07% | 80.28% | 80.63% | 0.40 |

FedAvg vs. Uniform: gap +0.05pp vs. spread 0.22pp -> **within noise, a clean tie.**
FedAvg vs. DivRoute: gap +0.92pp vs. spread 0.40pp -> **larger than noise.**
Uniform vs. DivRoute: gap +0.87pp vs. spread 0.40pp -> larger than noise (unchanged
from section 32, now confirmed independent of the FedAvg question).

**This promotes the FedAvg comparison from Tier B (provisional, 2 seeds) to Tier A
(seed-matched, 3 vs. 3 vs. 3) in the section 33 table.** Uniform's "communication savings
without sacrificing accuracy" claim now holds on FEMNIST with the same confidence as
everywhere else in the project. DivRoute is confirmed, not just suggested, to cost real
accuracy relative to dense FedAvg here -- it is the one method of the three that loses
against doing nothing compressed at all, and it does so on a fully seed-matched basis.

**Remaining, now smaller, gap: these numbers are console-log-derived (cross-checked
against every printed `[summary]` line encountered so far, all consistent), not
re-verified against the raw JSON.** Re-deriving them would cost a fresh ~5-6h run of the
8 missing logs (or recovering the previous session's saved output, if any) for a check
that has not caught a single discrepancy anywhere in this project when console and JSON
have both been available. Given that track record and the cost, this is not worth blocking
on -- log the caveat in any write-up (see section 31/32's disclosure list) rather than
re-running for it, unless the previous session's output turns out to be free to recover.


## 35. A 64-round Uniform run at alpha=0.1 shows the 32-round alpha=0.1 results were budget-limited

Source: `uniform_dirichlet_a01_ef_64r.json` (uploaded without comment). Verified from the file
itself: CIFAR-100, pretrained EfficientNet-B0, Dirichlet alpha=0.1, seed 42, Uniform + plain error
feedback, 64 rounds, 15 clients/round. Byte accounting is exact for Uniform (24.814 MB upload
every round, all 15 clients in tier 2, total 1588.1 MB). **This is not the pathological run I
asked for** (that command was `uniform_pathological_ef_64r`), and its metadata has no
`partition_mode` field, so it was produced by code older than that logging change -- I cannot tell
from the file which command made it, and the pathological results are still pending.

**Not converged at 64 rounds.** 10-round window means: 53.73, 74.86, 77.45, 78.40, 79.17, 79.82,
80.38 (last window = rounds 61-64 only: 4 rounds). Last two full windows (r45-54 vs r55-64):
79.39 -> 80.21, diff +0.82pp, above this project's own 0.5pp plateau bar. Plateau(last 20) =
79.80% (std 0.51), final 80.66%.

**Consequence for section 24 (alpha=0.1, 32 rounds).** Same seed 42, same method: 32-round
schedule gave 79.11% final / 78.84% trailing-5; the 64-round schedule gives 80.66% final / 79.80%
plateau, about +1pp. The cosine learning-rate schedule depends on `num_rounds`, so this is not a
like-for-like comparison, but it does show the 32-round numbers were limited by the round budget,
and that the "plateau" I read off rounds 28-32 then was at least partly the LR annealing to its
floor rather than convergence. What this does NOT undermine: section 24's comparison gave both
methods the same 32-round budget, so DivRoute-vs-Uniform at that budget is still like-for-like.
What it DOES limit: any statement that those are converged accuracies. Unknown: whether DivRoute
gains the same ~1pp from 64 rounds, so whether the tie persists at a larger budget is untested.
The same caution applies to the alpha=0.9 32-round results (not re-checked at longer budgets).
FEMNIST (150 rounds, windows differ 0.06-0.25pp for 7 of 8 checked runs) is not affected.

**Disclosure for the write-up:** state the alpha=0.1 and alpha=0.9 CIFAR-100 comparisons as
"at a fixed 32-round budget", not as converged accuracies. Cheapest way to close it: DivRoute
alpha=0.1 seed 42 at 64 rounds with the same config (pairs with this Uniform run).


## 36. Matched-budget FedAvg at alpha=0.1 (64 rounds): section 25's "compressed beats FedAvg" is withdrawn

Source: `fedavg_dirichlet_a01_64r.txt` (console log, no accompanying text). Verified from the log:
CIFAR-100, dense FedAvg, Dirichlet alpha=0.1, 64 rounds, 15/25 clients per round, upload
15,880.89 MB = 64 x 248.139 MB (dense, exact). The log prints no seed, but shard sizes (min 1167,
max 3407, mean 2000) are identical to the seed-42 alpha=0.1 partition used before, so I treat it as
seed 42; unconfirmed. It has no `partition_mode` line, i.e. produced by older code, like the Uniform
file in section 35. This is the matched partner to section 35's Uniform run.

| 64 rounds, alpha=0.1, seed 42 (probable) | Final | Last-20 mean (std) | r45-54 -> r55-64 | Upload |
|---|---|---|---|---|
| FedAvg (dense) | 80.88% | **80.28%** (0.50) | 79.90 -> 80.67 (+0.77) | 15,880.9 MB |
| Uniform + EF | 80.66% | 79.80% (0.51) | 79.39 -> 80.21 (+0.82) | 1,588.1 MB |

Uniform trails FedAvg by 0.48pp on the last-20 plateau and 0.22pp on the final round; the per-round
difference over the last 20 rounds averages -0.48pp (std 0.19; rounds are autocorrelated, so this
std understates uncertainty). 10-window means: FedAvg 57.66, 76.15, 78.28, 78.96, 79.71, 80.38,
80.63 (last window = 4 rounds); Uniform 53.73, 74.86, 77.45, 78.40, 79.17, 79.82, 80.38. **Neither
run has converged** (both +0.8pp between the last two windows). Single seed.

**What this corrects.** Section 25 reported that DivRoute (78.35%) and Uniform (78.47%) *exceed*
dense FedAvg (77.49%) at alpha=0.1. That compared compressed runs at 32 rounds with a FedAvg run at
20 rounds, so extra rounds, not compression, produced the "gain". At matched rounds the sign flips:
dense FedAvg is slightly ahead. The same mismatch underlies section 20's alpha=0.9 statement
(FedAvg 20 rounds 82.08% vs compressed 32 rounds ~83.3%); I have no matched-round FedAvg at
alpha=0.9, so that claim is likewise unsupported as an equal-rounds statement. Earlier messages of
mine saying compressed methods "slightly exceed dense FedAvg" on CIFAR-100 should be read as
withdrawn in that form.

**What still holds, stated precisely.** (a) Equal rounds (alpha=0.1, 64): Uniform + EF is within
~0.5pp of dense FedAvg at 10% of the upload (single seed, unconverged). (b) Equal *total
(bidirectional) bytes*: download is dense every round, so 32 compressed rounds cost about what 20
dense rounds cost (section 16's framing); on that axis the compressed runs did score higher. These
are different claims and the paper must say which one it makes. The FEMNIST FedAvg-vs-Uniform tie
(150 rounds, 3 seeds each, converged) is a matched-round result and is unaffected.

**Open:** DivRoute at alpha=0.1, 64 rounds, same seed, would complete this matched trio; without it
the 64-round routing comparison is unavailable.


## 37. FEMNIST ablation results, tight-budget test, code audit, and the revised conclusion

**Equivalence check.** After the code cleanup (section 36's follow-up audit: forensic CSV writers
removed, unused code removed, EMA update skipped unless FedSparse, tier rule unified, new
`random_tier_assignment` flag), FedAvg/Uniform/DivRoute seed-42 reproduce the earlier Kaggle final
accuracies exactly (0.8164 / 0.8160 / 0.8044) on the same GPU type, from the JSON logs. Seed 42
only; training is cudnn-deterministic.

**Audit finding.** DivRoute differs from Uniform in more than routing: per-layer vs global top-k,
divergence-weighted aggregation (sample share x sqrt(d); higher divergence -> more weight, while
bandwidth goes to LOWER divergence), selection-weight decay (gamma=0.85) for tier-3 clients, and
rolling-percentile thresholds on an EMA of divergence. So "DivRoute < Uniform" cannot be attributed
to routing without ablation.

**Ablation at the 5% scale (FEMNIST, 150 rounds, 3 seeds, plateau = last-20-round mean).**
FedAvg 81.56 (0.16), Uniform 81.50 (0.22), Uniform k=0.105 ("matched", ~DivRoute's bytes) 81.56
(0.12), `pure` (tiered k only: global top-k, no divergence weighting, gamma=1) 81.50 (0.19),
`purerandom` (same, tier labels shuffled) 81.56 (0.16), DivRoute 80.63 (0.40), DivRoute with
global top-k 81.02 (0.34), without selection decay 80.98 (0.10), without divergence weighting
79.84 (0.52). Removing all three extras recovers ~0.9pp (to 81.50); divergence tiers equal random
tiers equal flat Uniform at matched bytes. Components interact (removing divergence weighting
alone hurts), so no single extra is blamed. Two leave-one-out runs miss the 0.5pp window bar by
~0.01-0.04pp (noweight s42/s43) and one is at -0.67 (nolayer s44).

**Why a second operating point.** At 5% Uniform already equals dense FedAvg, so no allocation
scheme can help there. Uniform seed-42 plateau by budget: k=5% 81.27, 2% 81.08, 1% 80.54, 0.5%
80.05 (FedAvg 81.38): a 1.3pp loss at 0.5%.

**Tight budget (0.5% scale; ratios x0.1; 3 seeds; all runs pass the window bar).** Per-seed
plateaus (s42/s43/s44): Uniform 0.5%: 80.05/80.85/80.47 (80.46, 152MB); `pure`: 80.37/81.11/80.67
(80.72, ~311MB); `purerandom`: 80.54/81.19/80.92 (80.88, ~311MB); `matched` (flat Uniform at
1.05%, 319MB): 80.60/81.41/81.00 (81.00). Paired by seed: matched-minus-Uniform-0.5% +0.55 (sd
0.02, 3/3); pure-minus-purerandom -0.17 (3/3 negative); pure-minus-matched -0.29 (3/3 negative);
purerandom-minus-matched -0.12 (3/3 negative). Same-seed runs should share client sequences
(gamma=1, flat sampling) - inferred from the code, not verified. With 3 seeds, 3-of-3 same sign has
sign-test p=0.25: consistent, not significant. Reading: extra bytes help (+0.55pp); redistributing
bytes by divergence does not, and is slightly worse than random or flat allocation.

**Revised conclusion (replaces earlier "routing fails" wording).** Divergence-guided bandwidth
allocation gave no benefit over random allocation or flat Uniform at matched bytes, at two
operating points (5%: ties; 0.5%: consistently slightly worse), on FEMNIST from scratch. The full
DivRoute system's deficit at 5% comes from its non-routing components (per-layer top-k,
divergence-weighted aggregation, selection decay), not from the routing signal. Limits: one
dataset for the ablations; CIFAR-100 comparisons remain fixed-budget and partly single-seed;
effects at the tight budget are 0.1-0.3pp with 3 seeds; lr tuned on FedAvg only; own FEMNIST
protocol; MixUp on for all methods.

**Literature check on extensions (searches, not exhaustive; nothing here is a result).**
FourierFT (Gao et al., ICML 2024, PMLR 235:14884-14901, arXiv 2405.03003) fine-tunes a pretrained
model by learning a small set of spectral coefficients of the weight change, recovered by inverse
DFT; official code lives in HF PEFT. It is a PEFT method; I found no federated use of it. One search
for DCT/FFT gradient compression in federated learning surfaced none; adjacent: compressed
sensing / random-projection compression. Federated PEFT with frozen CLIP/ViT backbones (PromptFL,
FedCLIP, FedDAT, pFedMMA) is established; one FedPEFT submission reports 328MB -> 0.68MB per client
for ViT-Base (as reported there, unverified by me). None of this changes the present evidence.
Ollama is an LLM serving tool and has no role in this FL image-classification study. I have not
profiled where wall-clock time goes; no claim about speed-ups from any transform is made.


## 38. Literature verification pass (searches + abstract reads; correction to earlier claims)

**Correction.** Earlier sections said HeteRo-Select "reportedly outperforms uniform baselines". The
abstract (read directly, arXiv 2508.06692) says only: a per-client informativeness score jointly
governs client selection, compression ratio and aggregation weight, with bandwidth as a hard
ceiling; informativeness is not defined in the abstract; datasets include CIFAR-10 (logistic
regression to ResNet-18); headline 1.78x speedup and 18.2% traffic reduction on CIFAR-10 under the
FedCG protocol, measured against bandwidth-driven allocation. No uniform or random-allocation
baseline is mentioned. So that paper does not contradict the FEMNIST result, and the claim is
withdrawn. The full paper was not read; its theory result (score-proportional budget beats uniform
top-k error under convexity-type assumptions) is a theorem, not an empirical control.

**Related prior art found (abstracts only).** LBAT (arXiv 2609.39646) allocates rank/bits across
LAYERS under strict uplink budgets and claims to beat uniform rank/quantisation/fixed compression
in extreme-budget regimes, on tabular tasks, without numbers in the abstract. Biased client
selection (Power-of-Choice, Cho et al.) reports gains over random selection; other studies find
random sampling competitive depending on dataset. I found no paper reporting a matched-byte
random-allocation control for per-client compression ratios; this is "not found in these searches",
not proof none exists.

**Fourier / spectral / VLM.** FourierFT (ICML 2024, arXiv 2405.03003) is a PEFT method (spectral
coefficients of the weight change); no federated FourierFT and no DCT/FFT gradient compression
paper turned up. Federated PEFT with frozen CLIP/ViT (PromptFL, FedCLIP, FedDAT, pFedMMA) is
established and addresses a different problem. A 2025 study reports top-k's sorting cost as its
compute drawback (secondhand); where our wall-clock goes has NOT been profiled.

**Positioning.** The contribution to claim: a matched-byte random-tier control, plus component
ablation, showing divergence-guided per-client allocation has no benefit on FEMNIST at two
operating points. Limits unchanged (section 37).


## 39. Recent-literature positioning (2025-2026; abstracts/snippets unless stated) and claim-strength assessment

**Read in full text (PDF extracted locally):** Wan et al., "Enhancing Communication Compression via
Discrepancy-aware Calibration for FL", ICLR 2026. It changes WHICH coordinates (or singular
components) a client keeps, ranking them by output discrepancy measured on a small local
calibration set instead of by magnitude; every client keeps the same budget; classic error feedback
is used; baselines are Top-k and ATOMO vs their re-ranked variants at the same communication
budget; CIFAR-10/100 and Fashion-MNIST, ViT-tiny/small/base and CNNs, three independent trials; no
FEMNIST. Headline: 18.9% relative accuracy gain at compression ratio 0.1 for CIFAR-10 with
ViT-tiny, against "the baseline". It is a within-client selection rule, a different axis from
between-client budget allocation, so it does not contradict this study.

**Seen in search results only (not verified beyond the snippet):** FedSparQ (arXiv 2511.05591;
adaptive threshold sparsification + half precision + error feedback; ~90% fewer bytes than FedAvg);
SA-PEF (arXiv 2601.20738; step-ahead partial error feedback; results not read); an error-feedback
theory paper (arXiv 2605.31594; content not read); SparsyFed (arXiv 2504.05153; per-layer/per-round
adaptive sparsity, CIFAR-100 LDA(0.1) at 95% sparsity, claims gains over Top-K, ZeroFL, FLASH); a
2026 edge-IoT paper (4-bit quantisation + top-1% + error feedback, ~98.87% cumulative communication
reduction, 70.30% +/- 0.40 over three seeds); EcoLoRA (federated LoRA; top-k comparable at low
compression, degrading as compression increases, per the authors). Common ingredients in this
literature: sparsification + quantisation + error feedback; layer/threshold/round adaptivity;
calibration-aware selection; PEFT. No paper found reporting a matched-byte random-allocation
control for per-client budgets (not found, not proven absent).

**How this study differs.** It tests WHO gets how many bytes (between-client allocation by a
divergence score) with a random-allocation control at equal bytes, a component ablation of the
full DivRoute system, paired same-seed comparisons, and two operating points. Published adaptive
methods report gains for their own scheme, mostly without that control.

**Claim strength (honest).** Strong: matched-byte random control; component ablation; 3 seeds with
paired same-sign differences; converged runs; exact-reproduction check. Weak: decisive test is on
one dataset (FEMNIST); CIFAR-100 evidence is fixed-budget and confounded by DivRoute's extra
components; effects are 0.1-0.3pp at n=3 (sign test p=0.25); scope is "this divergence-based
scheme as formulated", not "allocation in general"; no published competing allocation rule
(HeteRo-Select / score-proportional budgets) was implemented; lr tuned on FedAvg only; own FEMNIST
protocol. Suitable as a workshop / TMLR-style negative-result paper; not a top-venue claim as is.
Hardening, in order of value: (1) a score-proportional-budget arm (tests the principle, not one
implementation); (2) second dataset at the tight budget; (3) signal bake-off; (4) third budget
point at 1%.


## 40. Weak-point audit with verified evidence (HeteRo-Select full text read; statistics recomputed)

**HeteRo-Select, full text (PDF extracted locally) - corrects section 38.** Its "informativeness" is a
min-max-normalised composite S_k(t) = V'_k + lambda_D*D_k + lambda_F*F'_k + lambda_St*St'_k, where V'_k is
the normalised LOCAL LOSS of the global model (plus diversity, fairness, staleness terms), not a
weight-space divergence. Per-client ratio: theta_tk = clip(min((S_k/S_bar_t)*theta_t, theta_cap_k),
theta_min, 1) with a cosine-scheduled round budget theta_t (avg 0.20, floor 0.08). The system also uses
FedProx local training, server momentum, adaptive error feedback and a curvature-weighted coordinate
criterion. Evidence: Table I baselines are copied from the FedCG paper (not re-run); HeteRo-Select is
mean +/- std over 3 seeds on CIFAR-10/100; ablations (Table II) are single-seed (42). In that ablation
(CIFAR-10): uniform compression allocation 72.22% final / 2,124 MB vs adaptive (primary) 73.03% / 2,010 MB,
i.e. +0.81pp single-seed; in the same table "pure magnitude top-k" (73.24) and "uniform LR" (74.08) are
at or above the primary configuration. Their Theorem IV.2 (score-proportional allocation lowers
aggregate top-k error) assumes convex, decreasing error and that more informative clients gain more
from extra budget. So there IS a published single-seed empirical gain for score-proportional
allocation, with a different (loss-based) score and a confounded system; our divergence-score test
does not test their rule.

**Statistics, tight budget (3 seeds, recomputed).** Paired t-test p: pure-vs-matched 0.0105,
pure-vs-purerandom 0.077, purerandom-vs-matched 0.140, matched-vs-Uniform0.5% 0.0003. Sign/Wilcoxon
p=0.25 for every 3-of-3 pairing (the minimum possible at n=3). A two-sided sign test needs 6 seeds in
agreement for p<0.05 (n=5: 0.0625). Power at the observed effect size (d=1.96, estimated from the
same data, so optimistic): n=3 0.46, n=4 0.74, n=5 0.90, n=6 0.97.

**Datasets (verified).** FEMNIST label skew is mild (mean TV 0.249 vs 0.197 for an IID split);
Dirichlet(0.1) on CIFAR-100-shaped labels gives TV 0.730. The flwrlabs HF organisation lists 23
datasets; the 10 visible did not include CelebA, Shakespeare or Sentiment140, so their availability is
unverified. CIFAR wall-clock is unmeasured recently (one old T4 figure: 164 min per 20 DivRoute rounds).

**Order of work.** (1) three more seeds (45-47) at the 0.5% scale for pure/purerandom/matched; (2) a
score-proportional-allocation arm using the published rule (loss-based score first, composite after
reading its Eq. 3-5), compared against random and matched-uniform at equal bytes, alongside our
divergence and update-norm scores; (3) a second dataset in a strong-label-skew regime, after a timing
calibration.


## 41. Cross-check of an external (Gemini Deep Research) audit against primary sources

An LLM-generated literature audit was checked claim by claim against primary sources. Rule: nothing
from it enters the paper unless it appears below as VERIFIED with its source. Nothing in the code was changed.

**HeteRo-Select Table II, re-extracted from the PDF (arXiv 2508.06692v3).** CIFAR-10, seed 42:
Compression: Uniform 72.22 peak / 72.22 final / R@70% 86 / 3,009 s / 2,124 MB; Adaptive (primary)
73.03 / 73.03 / 83 / 2,913 s / 2,010 MB (+0.81pp, single seed). Other rows: w/o V 72.71; w/o D 72.25
(final 71.10); w/o F,St 72.08; pure magnitude top-k 73.24; static beta=0.90 72.13; uniform LR 74.08
(best); uniform aggregation 72.28. This CONFIRMS section 40 and CONTRADICTS the external audit's
Table II (72.18 vs 70.41, +1.77pp, other rows and MB/s columns), which is wrong. The "uniform
compression" ablation is defined as theta_k = theta_t (Sec. III); the text does not say the selection
score or aggregation weights are removed in that arm, so they were presumably kept (inferred, not stated).
Table I: baseline rows are "cited directly from Table I of [26]" (FedAvg, OptRate, FlexCom, AdaSample,
FedCG), HeteRo-Select rows simulated locally; main results 3 seeds (42-44); ablations and cross-scale
runs single seed (42). Stated venue target: IEEE ICDM (Nov 2026); arXiv v1 8 Aug 2025, v3 18 Aug 2026;
acceptance not confirmed.

**Dataset statistics, verified from primary PDFs.** LEAF (arXiv 1812.01097, Table 1), devices / samples:
FEMNIST 3,550 / 805,263; Sent140 660,120 / 1,600,498; Shakespeare 1,129 / 4,226,158; CelebA 9,343 /
200,288; Reddit 1,660,820 / 56,587,343. (Our Hugging Face FEMNIST copy has 3,597 writers / 814,277
samples, so it differs from LEAF's release.) Hsu et al. 2020 (arXiv 2003.08082): iNaturalist-User-120k =
9,275 clients, 1,203 classes, 120,300 examples; Landmarks-User-160k = 164,172 images, 2,028 landmarks,
1,262 users, test split 19,526 images. FLamby Fed-Camelyon16: 2 clients (two hospitals), 399 slides
(search-summary source, secondary). The external audit's figures for Shakespeare (660), Reddit,
iNaturalist-User (1,203 clients / 565K images) and Fed-Camelyon16 (5 clients) were WRONG.

**Venues.** Thomsen, Taylor, Dieuleveut, "A Tight Theory of Error Feedback Algorithms in Distributed
Optimization" (arXiv 2605.31594): ICML 2026 poster page exists; the abstract covers only tight EF/EF21
convergence analyses, with nothing about budget allocation across clients. SA-PEF (arXiv 2601.20738):
arXiv listing says TMLR 2026; seed/dataset details not read.

**Unresolved.** Wan et al. (ICLR 2026) dataset list: section 39 (from the PDF) says CIFAR-10/100 and
Fashion-MNIST; the external audit says Tiny-ImageNet; the arXiv-less GitHub README lists neither.
Re-read the PDF before citing either.

**Not usable (unsourced in the external audit):** the EF error-bound formula and "three conditions"
for when non-uniform allocation helps; attribution of an "allocation cannot help under EF" result to
Thomsen et al.; "empirical reliability" claims about weight-divergence scores; "reviewers require >=5
seeds"; the "precedents" (Charles et al. 2021, Li et al. 2020, Reddi et al. 2021 are about cohort size,
FedAvg convergence theory and adaptive server optimisers respectively, not allocation null results);
"Claim 2/5 VERIFIED" (absence cannot be shown by search terms; Claim 5 only restates our own data).

**Verified or re-derived:** 1 - cos(w0 + D, w0) ~ (1/2)|D_perp|^2/|w0|^2 * (1 - 2 D_par/|w0|) for small D
(derivation re-checked; leading term proportional to the squared orthogonal update). t_crit(df=2,
two-sided 0.05) = 4.303; sign-test minimum p at n=3 = 0.25.

**Reviewer objections worth acting on (ideas, not facts):** (1) error-feedback-off control, to test
whether EF masks allocation effects; (2) continuous score-proportional allocation instead of three tiers;
(3) instantaneous vs EMA divergence; (4) 5-6 seeds so a sign test can reach p<0.05.

**Error-feedback control (added after §41).** FEMNIST uniform/divroute factories set `use_error_feedback=True`
(`config.py` lines 732, 743); the library default is False. `scripts/run_femnist_experiments.py --no-ef` now
turns EF off for any compressed variant (label tag `_noef`). Planned control, not yet run:
`--variant pure purerandom matched uniform --kscale 0.1 --no-ef --seeds 42 43 44`.
README.md and DIVROUTE_ALGORITHMS_MASTER_DOC.md now carry an OUTDATED banner (old tier polarity, "Tier 3 = skip");
DIVROUTE_PROJECT_OVERVIEW.md already states the live polarity and the Tier-3 fix.


## 42. Six-seed result at the 0.5% budget and the error-feedback-off control

Plateau = mean of last 20 rounds, FEMNIST, `--kscale 0.1`, bytes matched (upload ~310-319 MB; Uniform 152 MB).
Seeds 45-47 JSONs read this session; seeds 42-44 EF-on values from earlier sessions (4 of 9 re-verified from JSON:
pure s42/s44, purerandom s43, uniform s42; the other five not re-read). Seeds 45-47 may have run on a different
GPU type than 42-44 (not recorded) - a possible, unmeasured source of offset.

**EF on, 6 seeds (paired by seed).**
pure 80.60 +/-0.32; purerandom 80.81 +/-0.28; matched 80.95 +/-0.30 (mean +/- sd).
pure-purerandom -0.22pp (95% CI [-0.30,-0.13], 0/6 seeds positive, paired t p=0.001, sign p=0.031);
pure-matched -0.35pp ([-0.44,-0.26], 0/6, t p<0.001, sign p=0.031);
purerandom-matched -0.13pp ([-0.20,-0.06], 0/6, t p=0.006, sign p=0.031). Same ordering in every seed:
matched > purerandom > pure.

**EF off, 3 seeds (42-44).** pure 79.94, purerandom 80.33, matched 80.49, uniform(0.5%) 79.80.
pure-purerandom -0.40pp (0/3, t p=0.049); pure-matched -0.55pp (0/3, t p=0.066); purerandom-matched -0.16pp
(0/3, t p=0.119); matched-uniform +0.69pp (3/3, t p=0.033). Sign p=0.25 for every 3/3 (minimum at n=3).
EF effect (on-off, seeds 42-44): pure +0.78, purerandom +0.55, matched +0.51pp, each 3/3.

**Reading.** (1) Divergence tiers do not help: they are slightly worse than random tiers and than flat allocation
at equal bytes, with and without EF, and the 6-seed EF-on ordering is consistent. (2) Error feedback does not hide
an allocation benefit: the ordering is unchanged with EF off. (3) Flat allocation beats Uniform-at-half-the-bytes
by ~0.5-0.7pp, i.e. most of what tiers "gain" over Uniform is just spending ~2x the bytes. (4) pure < purerandom
means the divergence score is not merely uninformative but, in this setup, worse than random; why is untested
(candidates: low-divergence clients are the ones with least to add; per-client k variance itself costs, matched >
purerandom). Not tested: inverted polarity with the `pure` components (existing `invert` variant keeps the full
DivRoute components).
Caveats: effects are 0.1-0.4pp on one dataset/model; t-tests assume near-normal differences at n=3-6; no
correction for the several comparisons made; EF-off covers only 3 seeds.


## 43. Inverted polarity (`pureinvert`) and Uniform at 0.5%, six seeds

`pureinvert` = `pure` (tiered k only) with tier polarity swapped: the 20% budget goes to the MOST divergent third.
Same `--kscale 0.1`, 150 rounds, seeds 42-47, all JSONs read; bytes ~ pure (~310 MB). Uniform seeds 45-47 read from
JSON (80.32/80.57/80.43); seed 42 re-verified (80.05); seeds 43-44 (80.85/80.47) from earlier-session numbers.
Plateau (last 20 rounds, mean, sd over 6 seeds): pureinvert 80.93 (sd 0.24; per seed 80.72/81.20/80.97/80.62/
80.88/81.21), matched 80.95, purerandom 80.82, pure 80.60, uniform 0.5% 80.45.
Paired differences (6 seeds): pureinvert-pure +0.34pp (CI [+0.16,+0.51], 6/6, t p=0.004, sign p=0.031);
pureinvert-purerandom +0.12 ([+0.01,+0.23], 6/6, t p=0.040); pureinvert-matched -0.01 ([-0.15,+0.13], 2/6, t p=0.86);
pureinvert-uniform +0.49 (6/6, t p=0.002); matched-uniform +0.50 (6/6, t p<0.001); pure-uniform +0.15 (5/6, t p=0.089).
Reading: routing direction matters (more bandwidth to the least-divergent clients costs ~0.3pp vs the opposite
direction), but neither direction beats flat allocation at equal bytes: inverted ties matched, and the benefit over
Uniform is explained by spending ~2x the bytes. Caveats: GPU type of seeds 45-47 not recorded; per-seed values are
paired by seed index only; one dataset/model; the mechanism for the direction effect is untested.
