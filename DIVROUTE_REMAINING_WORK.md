# Remaining Work — DivRoute-FL Accuracy Plan

Checklist only. Full rationale: `DIVROUTE_ACCURACY_MASTER_PLAN.md`.

## RESOLVED: full 3-vs-3-vs-3 FEMNIST comparison, all seed-matched (Tier A)
`fedavg_r150_s42` ran and converged (last two 10-round windows +0.09pp, plateau
81.38%). The planned JSON re-verification command failed instead:
`logs/femnist_divroute_r150_s42.json` was not found. Cause: this was a *different*
Kaggle session than the one behind the section-32 results, and `/kaggle/working`
does not persist across sessions unless a version's output was saved -- the other 8
logs are not lost, just absent from this session. **Lesson for future runs:
download `logs/` (or "Save Version") before a session ends.** Computed the full
comparison by hand instead (console numbers, cross-checked against every printed
`[summary]` line so far -- all consistent):

| | Seed 42 | Seed 43 | Seed 44 | Mean | Across-seed std |
|---|---|---|---|---|---|
| FedAvg (dense) | 81.38% | 81.69% | 81.60% | **81.56%** | 0.16 |
| Uniform-Top5% | 81.27% | 81.71% | 81.53% | 81.50% | 0.22 |
| DivRoute | 80.55% | 81.07% | 80.28% | 80.63% | 0.40 |

FedAvg vs. Uniform: **tie** (gap +0.05pp, spread 0.22pp). FedAvg vs. DivRoute and
Uniform vs. DivRoute: both **larger than noise** (+0.92pp and +0.87pp vs. 0.40pp) --
DivRoute loses to both other methods, confirmed, not just suggested. This is now
fully seed-matched (Tier A in section 33's table, promoted from Tier B) and does not
need the raw JSON to be trusted further: re-deriving it would cost ~5-6h for a check
that has not caught a discrepancy anywhere in this project when both were available.
Not worth blocking on unless the previous session's Kaggle output turns out to be
free to recover (worth a quick check of Kaggle's Version/Output history).

On FEMNIST, DivRoute is the only one of the three methods that costs real accuracy
relative to doing nothing compressed at all -- Uniform ties dense FedAvg at 90% less
upload; DivRoute trails both at 2.1x Uniform's upload. Full account:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` section 34 (section 32 for the original DivRoute
vs. Uniform finding).

## Evidence-strength tiers across both datasets (section 33)
Full table in the master plan. Summary: the CIFAR-100 alpha=0.1 tie and the FEMNIST
loss are both Tier A (seed-matched, gap vs. spread computed) and can be stated
plainly. CIFAR-100 pathological (non-IID, 2 shards/client) is Tier C -- single seed,
Uniform's plateau only deduced from partial late rounds, no dense FedAvg baseline at
all -- and must keep that caveat attached if used in any table, not be cleaned up for
prose. This is the one setting left where the evidence is meaningfully weaker than
everywhere else in the project.

## Superseded (kept for history): earlier FEMNIST tracking note
Purpose: test the routing claim (C2) where DivRoute was designed to work (no shared
pretrained init) on a real per-writer partition, and add a second dataset. Caveat
first: FEMNIST's label skew is only mild (mean TV 0.249 vs 0.197 for an IID split);
the heterogeneity is mostly handwriting style. Full facts, protocol, verification and
the disclosure list: `DIVROUTE_ACCURACY_MASTER_PLAN.md` section 31.

What to upload to Kaggle: the `divroute_fl/` folder (now includes `femnist_data.py`)
and the `scripts/` folder (`run_femnist_experiments.py`, `summarize_runs.py`), with the
same setup cell as before. Internet must be ON (the loader downloads ~200 MB from the
Hugging Face Hub). I did not check whether `datasets` is preinstalled on Kaggle; if
`import datasets` fails, run `pip install datasets` first.

Run, from the project folder (defaults: seeds 42 43 44, 150 rounds):
```
python scripts/run_femnist_experiments.py
```
- Interrupted or session died: run the exact same command again. Finished runs are
  skipped, an interrupted run resumes from its checkpoint.
- Single run: `python scripts/run_femnist_experiments.py --only divroute --seeds 42`
- Expect roughly 32-38 min per run at ~13-15 s/round (measured on my laptop for dense
  FedAvg; Kaggle and the compressed methods not measured), so ~5-6 h for all nine.
- Logs land in `logs/femnist_<method>_r150_s<seed>.json`. When it finishes it prints the
  exact `summarize_runs.py` command (DivRoute first, so the tie/not-tie verdict compares
  DivRoute vs Uniform).

Before comparing anything: check each method's plateau. The last two 10-round window
means should differ by well under 0.5pp (dense FedAvg: 81.35 -> 81.44%). If a compressed
method is still climbing, re-run it with a larger `--num-rounds` (new label, no
collision). Reference points: centralised ~82%; dense FedAvg plateau ~81.4% (seed 42).

Not yet done / not verified: `run_femnist_experiments.py` was never executed (written
and reviewed only); nothing has run on Kaggle; no compressed method has run past 6
rounds; no dense FedAvg baseline other than seed 42.

## RESULT: FedAvg at alpha=0.1 — the missing accuracy baseline, now filled
Real dense FedAvg, alpha=0.1, 20 rounds, seed 42: **77.49%**, 4962.78MB
upload. Converged (rounds 16-20 flatten 76.33%->77.49%). Both compressed
methods' 3-seed means beat it: DivRoute 78.35% (+0.86pp, -59.9% upload),
Uniform 78.47% (+0.98pp, -84.0% upload). Confirms "without sacrificing
accuracy" holds at real heterogeneity too, not just alpha=0.9 — same
pattern as §20. Full analysis: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §25.
Doesn't change §24's routing verdict (DivRoute/Uniform still tied with
each other); closes the general (non-routing) claim's evidence gap.

## RESULT of the fifteenth attempt — pathological non-IID (2 shards/client): promising, but not converged
First real run under the new pathological partitioning. `[train]
partition: pathological (2 shards/client)` and uniform shard sizes
(2000/2000/2000) confirm it applied correctly.

| | Final-round acc (32 rounds, single seed) | Upload |
|---|---|---|
| DivRoute | 44.84% | 2069.24MB |
| Uniform-Top5% | 37.10% | **794.04MB** |

Absolute accuracy collapsed vs. every Dirichlet run (37-45% here vs.
77-83% before) — expected under 2-classes/client client drift, not a bug.
**Neither method has converged**: checked rounds 15-32, DivRoute climbs
32.27%->44.84% and Uniform climbs 30.49%->37.10% with no flattening.
Promising signal worth flagging: DivRoute leads Uniform in every one of
the last 9 rounds (avg +2.44pp), a more consistent pattern than alpha=0.1
ever showed — but not yet evidence of anything, since an unconverged
curve's ranking can still flip before it plateaus. Full analysis:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §26.

## RESULT of the sixteenth attempt — FedProx (mu=0.01): hypothesis not confirmed
`fedprox: True (mu=0.01)` confirmed active in both logs. Compared directly
against §26's non-FedProx curves at the identical (pathological, 32-round)
setting.

| | No FedProx (§26) | FedProx mu=0.01 | Delta |
|---|---|---|---|
| DivRoute | 44.84% | 40.57% | -4.27pp |
| Uniform-Top5% | 37.10% | 36.71% | -0.39pp |

**No stabilization** — rounds 15-32 still climb steadily with no
flattening, same noise magnitude as without FedProx. Final accuracy got
worse, not better, for both methods (DivRoute notably so). **Worse: the
one interesting pattern in §26 disappeared** — DivRoute led Uniform in
all 9 of the last 9 rounds without FedProx (avg +2.44pp); with FedProx
that drops to 4 of 9, a coin-flip pattern (avg +0.77pp), much closer to
the noisy alpha=0.1 Dirichlet results. Full analysis:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §27.

Caveat: `mu=0.01` is the weak end of the literature's 0.001-1.0 range, so
this doesn't rule out FedProx at a higher mu. But given this
investigation's track record — every "maybe a stronger version will work"
follow-up has come up short so far — further mu-tuning is lower priority
than the two paths already agreed on.

## RESULT of the seventeenth attempt — DivRoute at 64 rounds: converged, ~41%, below §26's number
DivRoute only (Uniform's 64-round run is not in the log yet). `fedprox:
False` confirmed. Plateaus from about round 30: rounds 33-44 mean 41.16%,
rounds 45-64 mean 41.06% (round-to-round std ~0.94pp); peak 43.52% (round
43), final 39.97%. Upload 4203.00MB (73.5% saving); tier mix stayed
healthy. So ~41% is DivRoute's converged level here, and §26's 44.84%
(trailing-5 43.59%) overstated it by 2.5-3.8pp — same lesson as §22
(single-run final numbers mislead). Rounds 26-27 are missing from the
pasted console log only (LR on either side matches the cosine schedule
exactly; most likely a Kaggle live-output gap). **No DivRoute-vs-Uniform
conclusion is possible until Uniform's 64-round run exists.** Full
analysis: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §28.

## DECISION (post-§29): the routing/heterogeneity track is closed — no more runs on it
Deduced from the partial Uniform log, no rerun needed. Uniform's late rounds
(56-59: 42.72/41.62/45.51/43.73; 29-31: 43.40/42.59/39.94) sit at or above
DivRoute's converged 41.06%. For DivRoute to hold a real ~3pp ceiling edge,
Uniform would have to plateau near 38% — four consecutive late rounds all
>=41.6% make that essentially impossible (>5 sigma at ~1pp noise). So §26's
32-round lead was schedule-specific (it also vanished under FedProx), and
routing has now failed to beat uniform compression at all three heterogeneity
levels (alpha=0.9 loses; alpha=0.1 ties, 3 seeds; pathological ties or trails,
single seed), while costing 2.4-2.6x the upload. My odds of a real DivRoute
advantage at pathological: <10% (was 20-25%). Also corrected: the LoRA pivot
was NOT unclaimed territory — see item 4. **Next step: consolidate and write
(claims-vs-evidence map, results tables), then pick 1-2 gap-closing runs.**
Full reasoning: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §30.

**Tooling added:** `scripts/summarize_runs.py` reads the run JSON logs directly
(plateau mean over the last N rounds, across-seed mean/spread, upload and
bidirectional MB, and a gap-vs-spread tie/not-tie verdict between two method
groups). Validated against the hand-computed §28 plateau (41.06% +/- 0.94) and
the §24 3-vs-3 comparison. Use it instead of console pastes for every number in
the write-up, e.g.:
```
python scripts/summarize_runs.py --last 20 \
  --group DivRoute logs/d42.json logs/d43.json logs/d44.json \
  --group Uniform  logs/u42.json logs/u43.json logs/u44.json
```

## (superseded, now optional) STATUS: Uniform 64-round run was in flight (paste ended at round 59)
The Uniform 64-round run (checkpoint at round 55, resumed, log ends at round
59, no `[done]`) has no final number yet. Provisional overlap with DivRoute's
64-round run (same LR schedule): rounds 21-31 tie (38.82% vs 38.54%), rounds
56-59 Uniform +1.3pp — nothing resembling a persistent DivRoute lead, but 13
comparable rounds is not a verdict. Resume itself works; a torch-RNG restore
bug (RNG tensors loaded onto the GPU, `set_rng_state` needs CPU) was found in
the log, reproduced locally, and fixed in `main.py` (client selection was
never affected; only torch's MixUp/batch-order stream restarted). Details:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §29.

1. **Let the run reach round 64** (if the notebook stopped at 59, re-run the
   same command *without* `fresh=True` — it resumes from the latest
   checkpoint). Then read the whole trajectory from the JSON (console rounds
   32-55 are missing from the paste; the JSON has them; its `round` field is
   0-indexed, so console round N = JSON round N-1):
   ```bash
   python -c "
   import json, statistics as st
   h = json.load(open('logs/pretrained_companion_uniform_pathological_ef_64r.json'))['history']
   acc = {e['round']+1: e['test_accuracy'] for e in h}
   print('rounds logged:', min(acc), '-', max(acc), '| count:', len(acc))
   print(*[f'{r}:{acc[r]:.4f}' for r in sorted(acc)])
   w = [acc[r] for r in range(45, 65) if r in acc]
   print('mean rounds 45-64: %.4f (n=%d) std %.4f' % (st.mean(w), len(w), st.stdev(w)))
   "
   ```
   Compare against DivRoute's rounds 45-64 mean of 41.06% (§28). Read both
   by plateau means, not final rounds. Near 40-41%: tie, §26's lead was a
   transient. Near or below ~38%: real ~3pp DivRoute advantage, seed
   repeats become the priority.
2. **Dense FedAvg at pathological heterogeneity** — no baseline exists
   there, so "without sacrificing accuracy" is unchecked at this setting.
   Needed before any pathological-setting numbers go in a write-up.
3. **Once both plateaus are known: seed repeats**, same as §21-§24's
   methodology, before trusting any DivRoute-vs-Uniform gap under
   pathological non-IID.
4. **(Novelty premise corrected — see §30.) LoRA/adapter update compression — not started, second half of the
   pivot, independent of the above.** Apply EF+top-k+tiering to LoRA
   updates instead of full-model deltas. Needs new `model.py`/`client.py`
   work (LoRA injection, adapter-only local training) before any runs are
   possible. Can proceed in parallel with or after the above, per the
   user's "both in sequence" decision.

## Earlier NEXT STEP (post-§24 — investigation concluded; write-up, not more runs; superseded by pivot above)
§24 completed the full 3-vs-3 seed-matched comparison at alpha=0.1:
DivRoute 78.35% (±1.07pp) vs. Uniform 78.47% (±0.55pp) final-round —
**a clean statistical tie**, with DivRoute costing a consistent 2.4-2.6x
Uniform's upload bytes (1989.84MB avg vs. exactly 794.04MB every seed) for
it. Combined with alpha=0.9's decisive Uniform wins (§17, §20), there is
no heterogeneity regime tested where DivRoute's routing earns its
complexity. **The core mechanism-specific claim is not supported.** What
does hold: error-feedback-corrected top-k compression reaches (and per
§20, slightly exceeds) dense FedAvg's accuracy at large communication
savings — a real result, just not one specific to DivRoute's routing.

Nothing further is required to reach a defensible conclusion. If pursuing
more anyway, in priority order:

1. **Write up the negative/mixed result.** This is the actual next step —
   not another experiment. The investigation earned this conclusion
   through unusually thorough self-testing (three real infrastructure bugs
   found and fixed; four distinct hypotheses for why routing should matter,
   each tested rather than assumed; two heterogeneity regimes; seed-matched
   comparisons at the harder one). That rigor is itself worth stating
   plainly in a write-up, not just the final numbers.
2. **Optional, diminishing-returns confirmation:** the routing-vs-compression-
   level ablation flagged since §14 (`ablation_routing_no_compression` /
   `ablation_uniform_compression`), run at alpha=0.1, would isolate routing
   from compression level within a single run. Given how clean the 6-run
   tie already is, this is expected to reconfirm, not overturn, §24 — worth
   doing only if the write-up specifically needs that more direct
   isolation.
3. **Seed-variance check at alpha=0.9** — still technically unconfirmed
   (§13-§20 are single-seed), but those gaps were large and smooth, unlike
   alpha=0.1's noisy regime — low priority.
2. **Routing-vs-compression-level ablation at alpha=0.1** — flagged
   repeatedly since §14, never run at all: `ablation_routing_no_compression`
   (keep routing, force dense) and `ablation_uniform_compression` (keep
   DivRoute's compression schedule, disable routing) isolate routing from
   compression level directly. Worth running once the seed-matched
   comparison above is in.
3. **Seed-variance check at alpha=0.9 too** — every §13-§20 result is also
   single-seed; lower priority than #1 since those gaps were large and
   smooth (not noisy the way alpha=0.1 is), but still technically
   unconfirmed.

## Earlier NEXT STEP (post-§19 — two ablations, now run, see RESULT below)
§19 found DivRoute + tier-transition-aware error feedback hit 82.99%
accuracy, beating FedAvg, dominating original DivRoute, and beating
Uniform-Top5%. Two ablations were queued to check why — both have now run;
see the RESULT section below for what they showed (tier-aware EF refuted,
§17's verdict reinstated).

## Earlier NEXT STEP (post-§17 literature review — both now run, see below)
§17 settled that DivRoute doesn't beat Uniform-Top5% at matched conditions.
A literature review turned up a specific, well-documented explanation and
one already-built-but-unused alternative, both implemented/ready now:

1. **Hybrid loss+divergence routing signal** — zero new code, just unused
   config flags (`use_directional_divergence`, `loss_improvement_weight`,
   `local_val_fraction`). Literature (FedNolowe, FedAWA/CVPR2025) suggests
   loss-based signals often beat pure parameter-divergence for identifying
   which clients' updates matter.
   ```
   get_pretrained_finetune_config(use_directional_divergence=True, local_val_fraction=0.1,
       directional_div_weight=0.5, loss_improvement_weight=0.5,
       run_label='divroute_hybrid_signal', num_rounds=32, fresh=True)
   ```
   Caveat: `local_val_fraction=0.1` means clients train on 90% not 100% of
   local data — a small, real confound vs. the 81.58% baseline.

2. **Tier-transition-aware error feedback** — new code, implemented and
   verified (no GPU): `Config.error_feedback_tier_aware` (default `False`,
   confirmed byte-identical to existing EF behavior when off). FL literature
   is unanimous that error feedback is essential for top-k to work well, but
   also documents a specific failure mode ("stale error compensation") under
   partial client participation -- which this project's own `use_error_feedback=False`
   choice was based on a finding (-3.8pp) that predates every bug fix made
   this session and didn't account for DivRoute's own round-to-round tier
   reassignment compounding that staleness. When `error_feedback_tier_aware=True`,
   a client's residual buffer resets to zero whenever its tier changes,
   instead of blending a residual computed under a different k_ratio.
   ```
   get_pretrained_finetune_config(use_error_feedback=True, error_feedback_tier_aware=True,
       run_label='divroute_tier_aware_ef', num_rounds=32, fresh=True)
   ```

Both have now run — see RESULT sections below (§18 hybrid: refuted; §19
tier-aware EF: the project's first positive result, see NEXT STEP above
for the two follow-up ablations before calling it final).

## RESULT of the fourteenth attempt — Uniform seed 44: full 3-vs-3 verdict, clean tie — investigation concluded
Converged (rounds 28-32 plateau, no spike).

| | Final-round acc (mean, n=3) | Trailing-5-round mean (n=3) | Upload |
|---|---|---|---|
| DivRoute | 78.35% (±1.07pp) | 77.91% (±0.59pp) | 1989.84MB |
| **Uniform-Top5%** | 78.47% (±0.55pp) | 78.20% (±0.56pp) | **794.04MB** |

First fully seed-matched (3 vs. 3) comparison in the investigation. Gap
(0.12-0.29pp) is smaller than either method's own seed spread — **a clean
statistical tie**, DivRoute still costing 2.4-2.6x Uniform's bytes for it.
Combined with alpha=0.9's decisive Uniform wins, no heterogeneity regime
tested supports the routing mechanism earning its complexity. Full
analysis: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §24.

**This concludes the accuracy investigation.** See NEXT STEP above:
write-up is the actual next step now, not more runs.

## RESULT of the thirteenth attempt — Uniform seed 43 at alpha=0.1: gap shrinks to a statistical tie
Converged (rounds 28-32 plateau, no spike).

| Seed | Final-round acc | Trailing-5-round mean | Upload |
|---|---|---|---|
| 42 (§21) | 79.11% | 78.84% | 794.04MB |
| 43 (this) | 78.14% | 77.96% | 794.04MB |
| **Mean (n=2)** | **78.63%** | **78.40%** | 794.04MB |

vs. DivRoute's 3-seed mean (§22): 78.35% final / 77.91% trailing-5.
Uniform's own seed spread (0.97pp) is comparable to DivRoute's (1.07pp) —
so the remaining gap between the two methods' means (0.28-0.49pp) is now
inside the noise either method shows on its own. **Reads as a tie, not a
win for either method.** DivRoute's byte cost (~2.4-2.6x Uniform's) is the
one unambiguous finding: no accuracy edge to show for it so far. Full
analysis: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §23.

**Next step: Uniform seed 44** — see NEXT STEP above. Last data point for
a clean, fully seed-matched (3 vs. 3) verdict.

## RESULT of the twelfth attempt — DivRoute seed repeats at alpha=0.1: seed 42 was a lucky draw
Two more DivRoute seeds (43, 44), alpha=0.1, plain EF, same config as §21.
Both converged (checked rounds 28-32, no spikes).

| Seed | Final-round acc | Trailing-5-round mean | Upload |
|---|---|---|---|
| 42 (§21) | 79.58% | 78.56% | 2019.61MB |
| 43 | 77.70% | 77.40% | 2069.24MB |
| 44 | 77.77% | 77.77% | 1880.66MB |
| **Mean (n=3)** | **78.35%** | **77.91%** | 1989.84MB |

**Seed 42 was the high outlier.** DivRoute's 3-seed mean (78.35% final /
77.91% trailing-5) is now *below* Uniform's single alpha=0.1 data point
from §21 (79.11% / 78.84%) — the opposite of what §21's face-value reading
suggested. Not yet final: Uniform hasn't been re-seeded, so this is 3
DivRoute seeds vs. 1 Uniform seed. Full analysis:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §22.

**Next step: Uniform seed repeats (43, 44) at alpha=0.1** — see NEXT STEP
above. This is the one piece left before the alpha=0.1 comparison has a
real, seed-matched answer.

## RESULT of the eleventh attempt — alpha=0.1 (real heterogeneity): ambiguous, seed repeats needed
Both run fresh to 32 rounds, plain EF, everything else identical to §20's
best config except `alpha=0.1` instead of `0.9`.

| | Final-round acc | Trailing-5-round mean | Upload | Bidirectional |
|---|---|---|---|---|
| DivRoute (alpha=0.1, plain EF) | 79.58% | 78.56% | 2019.61MB | 9960.06MB |
| Uniform-Top5% (alpha=0.1, plain EF) | 79.11% | **78.84%** | **794.04MB** | **8734.49MB** |

Rounds 24-32 were noisy for both methods (~2.5pp round-to-round swings,
~5x the noise seen at alpha=0.9) — final-round reading favors DivRoute,
trailing-5-round mean favors Uniform. **The alpha hypothesis was
directionally right (the DivRoute/Uniform gap genuinely closes at real
heterogeneity, from a clear Uniform win at alpha=0.9 to a statistical tie
here) but this single-seed result can't yet say which method actually
wins.** DivRoute's best-case reading also costs 2.5x Uniform's upload
bytes. Full analysis: `DIVROUTE_ACCURACY_MASTER_PLAN.md` §21.

**Next step is seed repeats at alpha=0.1, not another hyperparameter
variation** — see NEXT STEP above.

## RESULT of the tenth attempt — the two §19 ablations: tier-aware EF refuted
Both run fresh to 32 rounds. Both confirmed converged: DivRoute+plain-EF
plateaus 82.7% -> 83.3% over rounds 24-32 (final 83.32%); Uniform+plain-EF
plateaus 82.4% -> 83.3% over the same window (final 83.24%).

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute + tier-aware EF (§19) | 82.99% | 1974.95MB | 9915.39MB |
| **DivRoute + plain EF** | **83.32%** | 2103.98MB | 10044.43MB |
| **Uniform-Top5% + plain EF** | **83.24%** | **794.04MB** | **8734.49MB** |

**Ablation 1: tier-aware EF is refuted.** Plain EF (no tier-change reset)
beats the tier-aware version for DivRoute (+0.33pp) while using *more*
bytes, not fewer — the reset was discarding useful signal, not protecting
against staleness. §19's win was real but its explanation was wrong.

**Ablation 2: Uniform gains just as much, far more cheaply.** Uniform +
plain EF (83.24%) ties DivRoute + plain EF (83.32%, 0.08pp — noise) using
62% less upload and 13% less total bytes. Error feedback is a general fix
for biased top-k compression, not something DivRoute's routing unlocks.

**§17's negative verdict is reinstated.** Full account:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §20.

## RESULT of the ninth attempt — tier-transition-aware error feedback: superseded by §20, kept for history
Run fresh to 32 rounds. Confirmed converged, not a spike: rounds 24-32 read
82.20 -> 82.55 -> 82.49 -> 82.75 -> 82.71 -> 82.78 -> 83.00 -> 82.76 ->
82.99, a tight plateau for the last 9 rounds.

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (original, no EF, 32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| Uniform-Top5% (32 rounds) | 81.43% | 794.04MB | 8734.49MB |
| FedAvg (dense, 20 rounds) | 82.08% | 4962.78MB | 9925.56MB |
| **DivRoute + tier-aware error feedback (32 rounds)** | **82.99%** | **1974.95MB** | **9915.39MB** |

**First result in the investigation with no undoing caveat.** Beats FedAvg
on accuracy (+0.91pp) with 60% less upload and a tied bidirectional total
(9915.39 vs 9925.56MB — noise). Strictly dominates original DivRoute
(higher accuracy, fewer upload bytes, fewer bidirectional bytes — not a
different tradeoff point, a better one). Beats Uniform-Top5% on accuracy
(+1.56pp) at a real but modest +13.5% bidirectional cost. Full analysis:
`DIVROUTE_ACCURACY_MASTER_PLAN.md` §19.

**Not yet final** — see NEXT STEP above for the two ablations (plain EF
without tier-aware reset; Uniform+EF) needed to confirm the gain is
specifically about DivRoute's tiering interacting with EF, not just "error
feedback helps top-k compression" in general.

## RESULT of the eighth attempt — hybrid loss+divergence signal: refuted
Run to 32 rounds, converged (79.88% -> 80.41% smoothly flattening over the
last 7 rounds as LR annealed -- a stable final number, not a transient dip).

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (original divergence signal, 32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| DivRoute (hybrid loss+divergence signal, 32 rounds) | **80.41%** | **1508.45MB** | 9448.89MB |
| Uniform-Top5% (32 rounds) | 81.43% | 794.04MB | 8734.49MB |

**Hypothesis #1 refuted.** The hybrid signal is worse than both the
original DivRoute (-1.17pp) and Uniform (-1.02pp). It does compress more
than original DivRoute (26% less upload) but that's a worse tradeoff, not a
better one -- less accuracy for the saving. (Caveat: `local_val_fraction=0.1`
means clients trained on 10% less local data here, a small real confound --
unlikely to explain the full gap alone, but worth naming honestly.)

One literature-motivated experiment remains: tier-transition-aware error
feedback (item 2 above), not yet run.

## RESULT of the fourth attempt (superseded as "final" — see NEXT STEP below)
The fourth attempt (both critical fixes #2 and #3 applied) is the first
structurally valid one: FedAvg's own byte accounting finally matches true
dense (248.139MB up = down, every round, 0% savings — confirms `use_k_warmup`
is really off this time), DivRoute shows real 0.20/0.05 tiering, Uniform
shows a flat, correct 5% (24.814MB/round, matching k=0.05 exactly).
FedAvg's run was manually stopped at round 17/20 (no rerun planned per
explicit instruction); DivRoute and Uniform completed all 20. All three had
visibly plateaued for several rounds before their last recorded point, so
these are treated as converged, final numbers for this round budget:

| Method | Accuracy | Upload | Upload savings | Bidir savings |
|---|---|---|---|---|
| FedAvg (dense, round 17/20, plateaued ~81.2-81.6% since round 14) | **81.50%** | 4962.78MB if completed (100% dense every round, confirmed) | 0% (reference) | 0% |
| DivRoute (20/20, plateaued since ~round 17) | **78.36%** | 1310.03MB | 73.6% | 36.8% |
| Uniform-Top5% (20/20, plateaued since ~round 17) | **80.26%** | 496.28MB | 90.0% | 45.0% |

**Conclusion — not the headline claim.** DivRoute vs. FedAvg (same
epoch-warmup schedule, the fair comparison): DivRoute trades ~3.1 points of
accuracy for 73.6%/36.8% byte savings — a real Pareto tradeoff, but "lower
accuracy, way fewer bytes," not "same accuracy, fewer bytes." DivRoute vs.
Uniform-Top5%: Uniform wins on both axes (higher accuracy AND more
compression) in this run. One live confound on that specific comparison:
Uniform runs with `use_epoch_warmup=False` (5 local epochs every round from
round 1) vs. DivRoute/FedAvg's 2->4->5 ramp -- ~33% more total local
training over the run -- which could explain some or all of Uniform's edge,
but is unconfirmed with this data. Taken at face value, this run does not
show DivRoute's routing intelligence outperforming flat uniform compression.

## RESULT of the fifth attempt (fixes 1-3 tested, see below)
FedAvg and Uniform reran with fixes 1-2; a new DivRoute run tested fix 3
(`invert_tier_polarity=True`). Original (non-inverted) DivRoute was not
rerun — its config didn't change, so its 78.36% result still stands.

| Method | Accuracy | Upload | Savings |
|---|---|---|---|
| FedAvg (dense, now with NTD) | **82.08%** | 4962.78MB | 0% (ref) |
| DivRoute (original polarity, unchanged) | 78.36% | 1310.03MB | 73.6% |
| DivRoute (inverted polarity) | **77.55%** | 719.46MB | 85.5% |
| Uniform-Top5% (now equal training budget) | **78.54%** | 496.28MB | 90.0% |

**Finding 1 — the epoch-warmup confound was real.** Uniform's accuracy
dropped 80.26% -> 78.54% once matched to DivRoute/FedAvg's training
schedule, confirming its earlier edge was partly just more training. But
DivRoute doesn't come out ahead even so — DivRoute (78.36%) and Uniform
(78.54%) are now essentially tied on accuracy, with Uniform still reaching
that on 90% savings vs. DivRoute's 73.6%. The confound explained *some* of
the gap, not all of it: DivRoute still doesn't out-Pareto Uniform.

**Finding 2 — the polarity-flip hypothesis is refuted, not confirmed.**
Inverting which clients get the generous tier made accuracy *worse*
(78.36% -> 77.55%), not better, despite saving more bytes (85.5% vs.
73.6%). §3.1's flagged mismatch between the README's stated intent and the
live code's actual bandwidth direction is real as a documentation/design
inconsistency, but it is **not** the explanation for DivRoute underperforming
FedAvg — the original polarity is mildly better, not backwards in the way
that would fix the gap.

**Status: across both polarities tested, DivRoute has not beaten
Uniform-Top5% on the accuracy/bytes tradeoff, and remains ~3.7-4.5pp behind
dense FedAvg.** Still untried (see below): the equal-byte-budget reframing,
loosening `k_ratio_tier2`, and the existing routing-vs-compression-level
ablation flags.

## RESULT of the sixth attempt — equal-byte-budget DivRoute (fresh 32-round run)
DivRoute run fresh (not resumed) to 32 rounds -- roughly the round count
where its cumulative bytes catch up to FedAvg's 20-round spend. Compare
against FedAvg's actual 20-round result (5th attempt), not a recomputed
reference:

| | Accuracy | Upload | Download | Bidirectional |
|---|---|---|---|---|
| FedAvg (20 rounds) | 82.08% | 4962.78MB | 4962.78MB | 9925.56MB |
| DivRoute (32 rounds) | **81.58%** | **2044.43MB** | 7940.44MB | 9984.87MB |

**Depends entirely on which "bytes" the claim is about.**
- **Upload only:** DivRoute lands within **0.5 accuracy points** of FedAvg
  using **59% fewer upload bytes**. This is the closest this investigation
  has come to the §2 target, and confirms the equal-*rounds* comparisons
  used everywhere before this were structurally unfair to a
  communication-efficient method -- DivRoute needed more rounds to spend an
  equivalent budget, and wasn't given them.
- **Bidirectional (total) bytes:** essentially a wash. DivRoute's 12 extra
  rounds cost ~2978MB of extra (uncompressed) download -- download is dense
  for every method regardless of compression, since the server always
  broadcasts the full model -- which very nearly cancels the upload
  savings. DivRoute used *slightly more* total bytes (9984.87 vs.
  9925.56MB) for *slightly worse* accuracy on this accounting.

Which framing is right depends on what "communication cost" means for the
claim: upload-only is the standard framing in most FL compression
literature (client uplink is the real-world bottleneck, not server
downlink), and by that measure this is a genuine, strong result. Bidirectional
total is the more literal reading of the master plan's "cumulative MB"
language, and by that measure this run does not clear the bar.

**Still missing for a complete comparison:** Uniform-Top5% has not been
tested at extended rounds. It costs even less per round (~24.8MB vs.
DivRoute's ~64MB average), so it could afford far more rounds within the
same budget -- unknown whether it closes its own small gap or extends its
lead under the same treatment DivRoute just got.

## RESULT of the seventh attempt — Uniform at 32 rounds: the core question settled
Uniform-Top5% run fresh to 32 rounds, matching DivRoute in every way that
matters (same round count, same epoch-warmup schedule, same NTD):

| | Accuracy | Upload | Bidirectional |
|---|---|---|---|
| DivRoute (32 rounds) | 81.58% | 2044.43MB | 9984.87MB |
| **Uniform-Top5% (32 rounds)** | **81.43%** | **794.04MB** | **8734.49MB** |

**Unlike the upload-vs-bidirectional ambiguity in the DivRoute/FedAvg
comparison, this one doesn't depend on which byte metric is chosen.**
Uniform matches DivRoute's accuracy (0.15pp gap -- noise) while using 61%
fewer upload bytes and 12.5% fewer total bytes. It wins or ties on every
axis: accuracy, upload, bidirectional.

**This settles the question this whole investigation was building toward:
does DivRoute's routing intelligence (divergence scoring, tiering, adaptive
percentile thresholds, divergence-weighted aggregation) earn its complexity
over the simplest possible baseline (flat 5% compression, no routing at
all)? On this data: no.** Six attempts, three real bugs found and fixed
(§ tau-threshold scale mismatch, k-ratio warmup override, plus the earlier
Tier-3-exclusion fix), two hypotheses tested and ruled out
(epoch-warmup confound, tier-bandwidth polarity) -- and once every
confound is removed, the mechanism-specific claim does not hold. This is a
clean, well-earned negative result on the core thesis, not an ambiguous one.

**What still stands, for the record:** the 80-85% accuracy target is met by
all three methods now. DivRoute and Uniform both land within ~0.5-0.65
points of FedAvg's accuracy using a fraction of its upload bytes -- the
"communication-efficient fine-tuning of a pretrained backbone reaches
FedAvg-competitive accuracy" claim holds. What does not hold is that
*routing specifically* (as opposed to *any* compression, applied uniformly)
is what buys that efficiency.

## Earlier NEXT STEP (fixes 1-3, now tested — kept for reference)
Three changes were implemented, verified without GPU compute (config
instantiation + tier-assignment logic replayed against synthetic/real
scores in plain Python), then run — see RESULT above for what they showed.

1. **Uniform's training-budget confound removed.** `get_uniform_top5_pretrained_config()`
   now sets `use_epoch_warmup=True` (was `False`), matching DivRoute/FedAvg's
   2->4->5 schedule exactly. Uniform no longer gets ~33% more total local
   training than the other two — the DivRoute-vs-Uniform comparison from the
   fourth attempt's result is no longer confound-free without this.
2. **NTD asymmetry resolved.** `ntd_beta=0.1` added to both
   `get_fedavg_pretrained_config()` and `get_uniform_top5_pretrained_config()`
   (previously only DivRoute had it, inherited from
   `get_recommended_divroute_config()`). All three methods now train with
   the same local-training regularizer.
3. **Tier-bandwidth polarity inversion, as an opt-in experiment.** New
   `Config.invert_tier_polarity` flag (default `False`, zero effect on
   existing behaviour) plus the corresponding `main.py` tier-assignment
   change. Tests whether the live code's bandwidth direction — least-divergent
   clients get tier=1 (most bandwidth), most-divergent get tier=3 (most
   compressed) — is backwards relative to the project's own stated design
   intent (see master plan §3.1, which flagged this mismatch but never
   fixed the allocation direction, only the aggregation-exclusion symptom).
   Run via `get_pretrained_finetune_config(invert_tier_polarity=True, run_label=...)`.
   Only meaningful for DivRoute — FedAvg's tier assignment is moot
   (`fedavg_baseline_mode` forces uniform k regardless of tier) and Uniform
   never reaches this code path (`uniform_top5_mode` hardcodes tier=2).

**Recommended next run, when ready:** re-run all three with fixes 1-2 baked
in (drop-in — same commands as before), plus a fourth, separate DivRoute
run with `invert_tier_polarity=True` to test fix 3 as its own experiment
against the corrected FedAvg/Uniform baselines. Also still open: the
equal-byte-budget reframing (extend DivRoute's round count to match
FedAvg's total byte spend, ~76 rounds by the fourth attempt's per-round
average, rather than comparing at equal round count) and the two smaller
levers (loosening `k_ratio_tier2`, running the existing
`ablation_routing_no_compression`/`ablation_uniform_compression` flags on
the pretrained backbone to isolate routing from compression level) — none
of these require code changes, only launch parameters.

## Missing

| # | Item | Status |
|---|---|---|
| 1 | Centralized sanity-check script | **Done** — see below. Unaffected by fixes #2/#3 (doesn't use the FL config/compression path at all). |
| 2 | FedAvg + pretrained backbone config | **Done** — `get_fedavg_pretrained_config()` in `config.py`. `include_tier3_in_aggregation=True` is *required*, not optional — see the function's docstring and "Critical fix #1" below. |
| 3 | Uniform-Top5% + pretrained backbone config | **Done** — `get_uniform_top5_pretrained_config()` in `config.py`. |
| 4 | Round count for the 3 companion runs (§6) | Still genuinely open — no byte-valid run has completed yet at any round count, so there's no trustworthy convergence trend to read. 20 is the current working default (see NEXT STEP), to be revisited once a byte-valid run actually lands. |
| 5 | Image resolution (128, in `data.py`) | Untuned, but empirically validated as workable via the (unaffected) centralized sanity check: 74.63% test acc after 1 epoch at 128px, no OOM at batch_size=64 on a 6GB GPU |
| 6 | Execution environment (local CUDA vs. remote) | **Resolved, remote is much faster.** Local: ~8.6h extrapolated for 20 rounds. Kaggle T4: a (byte-invalid, but timing-representative) 20-round DivRoute run took 163.67 min — roughly 3x faster than local. Kaggle is the recommended execution environment. |

### Critical fix folded into item 2 (read before touching `get_fedavg_pretrained_config()`)
`fedavg_baseline_mode=True` alone is **not** a working FedAvg baseline under
the current tier-assignment code. It forces `tau_low=tau_high=-1.0`
(main.py), and every real divergence score is >= 0, so *every* client is
assigned Tier 3, every round. `server.aggregate()` drops Tier-3 clients
unless `include_tier3_in_aggregation=True` (server.py: `if r["tier"] != 3 or
include_tier3_in_aggregation`). Without that flag, a "FedAvg" run would
silently aggregate nothing and the global model would never update — the
`clients_per_round<=2` case is a trap here, because main.py's Progress
Guarantee mechanism promotes up to 2 clients/round to Tier 2 regardless of
this flag, which can mask the bug in a small smoke test (confirmed while
validating this config: a 2-clients-per-round smoke test looked fine either
way; the flag only becomes visibly load-bearing once clients_per_round > 2).
Verified independently against a real prior run at realistic scale
(`logs/diag20_fedavg_corrected_cifar100.json`, `clients_per_round=20`): every
client shows a non-zero `aggregation_weight` despite `tier=3`, specifically
because that run set this flag (`run_diag20_fedavg_corrected.py`).
`get_fedavg_pretrained_config()` now bakes this in so it can't be dropped by
accident.

### Critical fix #2: DivRoute's tau thresholds never reached the pretrained regime's scale
Found by actually running the real 10-round Kaggle companion experiment, not
by inspection. **Every one of the 10 rounds logged "Actual tier counts:
15/0/0"** — all 15 selected clients landed in Tier 1, every round, for the
entire run. Root cause: `threshold_mode` defaults to `"adaptive_tau"` with
`tau_low`/`tau_high` starting at `0.01`/`0.02` and `tau_smoothing=0.8` (20%
EMA step toward the real distribution per round) — all tuned for from-scratch
ResNet-18 divergence scores. A pretrained-backbone fine-tune's scores run
~100x smaller (observed ~1e-4, vs. the 0.01-scale defaults); at 20%/round
decay, `tau_low` needs ~20 rounds just to reach that scale, so inside a
10-round (or even 20-round) budget every client's score stays below
`tau_low` the whole time -> tier 1 for everyone -> Tier 2/3 compression
never activates. This wasn't just inert: DivRoute's actual upload came out
*higher* than uncompressed FedAvg's (3473.87MB vs. 2481.39MB, "-40%
saving") because every client got Tier 1's k_ratio=0.20 instead of a real
mix including Tier 2 (k=0.05) — the opposite of the comm-savings the whole
mechanism exists to produce.

**Fix (applied to `get_pretrained_finetune_config()`):**
`threshold_mode="rolling_percentile"` (recomputes tau_low/tau_high directly
as the p33/p67 of the actual score distribution every round — scale-independent,
no dependency on the from-scratch defaults) + `convergence_floor=1e-6` (the
1e-4 default sits *above* the pretrained regime's observed score spread,
which would freeze tau updates almost immediately even with
rolling_percentile active).

**Verified without spending more GPU time**, by replaying the real round-10
`d_ema` scores from the Kaggle log through both the old and new
threshold logic in plain Python:
- Old (`adaptive_tau`, tau=[0.00114, 0.00224]): tier counts **15/0/0**
- New (`rolling_percentile`, tau=[0.00005, 0.000065] on that same data):
  tier counts **5/5/5**

**What this fix does and doesn't establish:** it guarantees the tiering
mechanism will actually activate on the next run — a real, verified,
data-driven fix, not a guess. It does **not** guarantee DivRoute will beat
FedAvg once tiering is active (genuinely active tiering could help or hurt
relative to the accidental "treat everyone the same" mode the bug produced —
unknown until re-run), and it says nothing about the 80-85% target. It also
doesn't rule out other scale-mismatch issues elsewhere in the pretrained
config that haven't surfaced yet — this one was caught by reading real run
output, not from a complete audit.

### Critical fix #3: use_k_warmup silently overrode every method's real compression ratio
Found by reading the real 20-round Kaggle results for all 3 methods (not by
inspection) — FedAvg's and Uniform-Top5%'s byte numbers came out
**identical** (148.883MB/round, "40% saving", every single round of both),
which is impossible if FedAvg is dense (k=1.0) and Uniform is k=0.05.

**Root cause:** `main.py` calls `get_adaptive_k_ratios(config, rnd)`
unconditionally every round (regardless of `fedavg_baseline_mode`).
That function (`compression.py`) checks `use_k_warmup` (Config default
`True`, `k_warmup_rounds=30`) *before* ever looking at
`k_ratio_tier1`/`k_ratio_tier2`: while `rnd < k_warmup_rounds`, it returns
`k_warmup_tier1=0.70`/`k_warmup_tier2=0.30` instead. None of the 3 companion
factories ever set `use_k_warmup=False`. Since every companion run so far
has been <=20 rounds (< the 30-round warmup window), **every round of every
run used k=0.70/0.30 instead of each method's actual configured ratio**:

| Method | Intended k (tier1/tier2) | Actually used (all runs so far) |
|---|---|---|
| DivRoute | 0.20 / 0.05 | 0.70 / 0.30 |
| FedAvg | 1.0 / 1.0 (dense) | 0.70 / 0.30 |
| Uniform-Top5% | 0.05 / 0.05 | 0.30 (always tier 2) |

FedAvg and Uniform landing on the identical 0.30 explains their identical
byte numbers exactly. It also explains why DivRoute sometimes used *more*
bytes than dense FedAvg's reference: top-k storage needs a value **and** an
index per retained coordinate (~2x overhead), so at k=0.70 compressed size
is ~140% of dense -- compression that inflates size instead of shrinking it.

**This isn't just a byte-accounting bug.** `k_ratio` truncates what actually
gets aggregated into the global model each round, so it changed training
dynamics directly -- every accuracy number from every companion run so far
(10-round and 20-round, all 3 methods) is confounded by this, not just the
byte totals.

**Fix:** `use_k_warmup=False` added to all three factories
(`get_pretrained_finetune_config()`, `get_fedavg_pretrained_config()`,
`get_uniform_top5_pretrained_config()`). A 10-30-round warmup is reasonable
for a from-scratch model training hundreds of rounds; it makes no sense for
a pretrained backbone fine-tuning for 10-20 rounds total, where "warmup"
would consume the entire run.

**Verified without spending GPU time:** called `get_adaptive_k_ratios()`
directly against each fixed config (replicating main.py's `fedavg_baseline_mode`
runtime override for the FedAvg case, since that override happens inside
`run()`, not in the config object itself) --
DivRoute: k1=0.20/k2=0.05 every round (was 0.70/0.30) --
FedAvg: k1=1.0/k2=1.0 every round (was 0.70/0.30) --
Uniform: k1=0.05/k2=0.05 every round (was 0.30).

**Status: every companion-run result so far (10-round and 20-round, all 3
methods) is invalid and must be discarded, not reinterpreted.** All three
need a fourth re-run with this fix. The centralized sanity check
(`scripts/centralized_finetune_sanity_check.py`, 74.63%) is unaffected -- it
doesn't use `apply_tiered_compression` or any FL config at all.

## Done (implemented + smoke-tested)
`model.py` (pretrained loader, BN handling, groupnorm fix), `data.py` (resize/
ImageNet-norm pipeline), `config.py` (`local_lr_min`, `get_pretrained_finetune_config`),
`main.py` wiring. See master plan doc's status line.

`config.py`'s `get_fedavg_pretrained_config()` and
`get_uniform_top5_pretrained_config()` — companion-baseline factories for
§6, mirroring `get_pretrained_finetune_config()`'s backbone/dataset/LR
overrides. Smoke-tested together (2 rounds, 4 clients, 2/round) by running
all three configs through the real `divroute_fl.main.run()` loop: no
crashes, FedAvg accuracy climbs round-over-round (37.98% -> 52.84%) with
non-zero aggregation weight for every participating client (confirming the
Tier-3 fix above actually works, not just that it doesn't crash).

`run_pretrained_companion_experiments.py` — launches all 3 (or just one, via
`--only`) back to back with identical seed/num_rounds so the comparison
stays valid per §6; writes `logs/pretrained_companion_<name>.json` +
`.walltime.txt` per run. Used for the full-scale timing calibration above
(`--only divroute --num-rounds 2 --fresh`); that calibration's own log
output was discarded (it's a 2-round partial result, not a real data point,
and shares the real run's filename so it would otherwise look like one).

Note (not yet acted on): `get_pretrained_finetune_config()` inherits
`ntd_beta=0.1` from `get_recommended_divroute_config()`'s base, but
`get_fedavg_pretrained_config()` / `get_uniform_top5_pretrained_config()` do
not set it (defaults to `0.0`) — this asymmetry is pre-existing (inherited
from those factories' established from-scratch behaviour, not introduced by
this session's changes) and NTD is a local-training regularizer, not part of
"the mechanism" §2 protects, but it does mean DivRoute trains with an extra
regularizer the baselines don't get. Worth a conscious decision before the
real runs, not a silent carry-over.

`scripts/centralized_finetune_sanity_check.py` — plain (non-federated)
EfficientNet-B0 fine-tune on the full CIFAR-100 train set, reusing
`data.py`/`model.py`'s exact pretrained-backbone pipeline (same resize,
ImageNet normalisation, and label-smoothing/SGD+momentum/grad-clipping
recipe as `client.py`, so the ceiling it measures is comparable to what the
federated runs will use). Validated two ways:
  - `--smoke` (1 epoch, 512-image subset): confirms the full pipeline runs
    end-to-end in ~10s.
  - 1 full epoch over all 50,000 training images (`logs/_timing_probe_cifar100.json`):
    **74.63% test accuracy**, 246.5s/epoch, no OOM. This is a single-epoch
    number, not the full 15-epoch result, but it already lands well inside
    the plan's sourced 82-88% literature band's trajectory and confirms the
    resize/normalisation step (the plan's biggest identified risk, §7) is
    working correctly rather than silently collapsing accuracy.
