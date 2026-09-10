# DivRoute-FL — Project Overview

A plain-language summary of the whole project: what it is, how it's built, what's been proven, what's still open, and what research says about fixing it. For the full technical decision log, see `DIVROUTE_ACCURACY_MASTER_PLAN.md`. For a live checklist, see `DIVROUTE_REMAINING_WORK.md`.

---

## 1. What is this project, in plain terms?

**Federated learning (FL)** is a way to train one shared AI model across many devices (phones, hospitals, etc.) without those devices ever sending their raw data anywhere. Instead, each device trains the model a little on its own data, then sends back just the *changes* it made. A central server combines everyone's changes into an updated shared model, and the cycle repeats.

The problem: sending those "changes" back and forth, every round, for every device, uses a lot of bandwidth. Most research in this space is about **compressing** those changes without losing much accuracy.

**DivRoute-FL's idea:** not every device's changes are equally useful. Some devices end up with a model that looks a lot like the current shared model ("aligned"); others end up quite different ("divergent"). DivRoute measures this difference and gives *more bandwidth* (less compression) to some clients and *less bandwidth* (more compression) to others, instead of treating everyone the same. The hope: same accuracy, less total data sent.

**The question this project set out to answer:** does that smart routing actually work better than just compressing everyone by the same amount?

---

## 2. Architecture

### 2.1 The training loop, step by step

| Step | What happens |
|---|---|
| 1. Setup | Server starts with a model already pretrained on ImageNet (EfficientNet-B0) |
| 2. Select clients | Each round, 15 of 25 simulated clients are picked to participate |
| 3. Send model | Server sends the full, uncompressed current model to those 15 clients (download is never compressed — see §2.3) |
| 4. Local training | Each client trains the model a bit further on its own local data (2 to 5 passes, ramping up over the run) |
| 5. Measure divergence | Server compares each client's trained model to the model it started from, and scores how different it is |
| 6. Assign tiers | Clients are sorted into 3 groups based on that score (see §2.2) |
| 7. Compress & upload | Each client sends back only part of its update — how much depends on its tier |
| 8. Aggregate | Server combines all the compressed updates into a new shared model, weighted by each client's data size |
| 9. Repeat | Steps 2–8 repeat for a set number of rounds |

### 2.2 The routing mechanism (DivRoute's core idea)

| Tier | Who's in it | How much of their update gets sent |
|---|---|---|
| Tier 1 | Least-divergent third of clients (most "aligned" with the current model) | 20% |
| Tier 2 | Middle third | 5% |
| Tier 3 | Most-divergent third | 5% (originally designed to send 0%, i.e. be skipped entirely — this turned out to be a bug, see §4.2) |

**Important, non-obvious detail:** the tier boundaries (which "divergence score" counts as low/medium/high) are *not* fixed — they're recalculated every round from that round's actual scores, so roughly a third of clients always land in each tier.

### 2.3 The two things being compared against

| Method | What it does | Purpose |
|---|---|---|
| **FedAvg** | No compression at all — every client sends its full update | The "gold standard" ceiling — most accurate, most expensive |
| **Uniform-Top5%** | Every client sends exactly 5% of its update, no divergence scoring, no tiers | The simplest possible compression — a control to test whether DivRoute's "smartness" is worth anything |

### 2.4 Model, data, and training setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 (100 classes, 50,000 training images) |
| Model | EfficientNet-B0, pretrained on ImageNet, fine-tuned via FL |
| Image size | Resized to 128×128 (CIFAR images are natively only 32×32) |
| Data split across clients | Non-IID — each client gets a different, skewed mix of classes |
| Total simulated clients | 25 |
| Clients per round | 15 |
| Local training extras | Label smoothing, MixUp, and a regularizer called NTD (Not-True Distillation) |
| Where it runs | Started on a local laptop GPU, moved to Kaggle's free GPU for speed |

### 2.5 Codebase map

| File | What it's responsible for |
|---|---|
| `divroute_fl/main.py` | The main training loop — runs every round, does routing, calls aggregation |
| `divroute_fl/model.py` | Loads the model (pretrained EfficientNet-B0 or from-scratch options) |
| `divroute_fl/data.py` | Loads CIFAR-100, splits it across clients, handles resizing/normalization |
| `divroute_fl/client.py` | What one client does during local training |
| `divroute_fl/server.py` | Combines client updates into the new shared model |
| `divroute_fl/compression.py` | The actual compression logic (which % of an update to keep) |
| `divroute_fl/config.py` | All the settings/knobs for every experiment, as named presets |
| `scripts/centralized_finetune_sanity_check.py` | A non-federated sanity check — proves the model/data pipeline works at all, before testing it federated |
| `run_pretrained_companion_experiments.py` | Runs DivRoute / FedAvg / Uniform-Top5% back to back for a fair comparison |

---

## 3. What's been achieved

### 3.1 Bugs found and fixed

Three real, measurement-breaking bugs were found and fixed during this project — each one made earlier results untrustworthy until it was caught.

| # | Bug | What was actually happening | Fix |
|---|---|---|---|
| 1 | Tier-3 exclusion | The most-divergent third of clients were being fully skipped (0% sent) every single round, permanently discarding their data's contribution | Include them, but heavily compressed instead of skipped entirely |
| 2 | Threshold scale mismatch | The routing thresholds were tuned for the old (from-scratch) setup and were ~100x too large for the new pretrained setup — so every client ended up in the same tier, and routing never actually activated | Recalculate thresholds fresh from real data every round instead of using fixed old values |
| 3 | Compression "warm-up" override | A setting meant to ease very long (300+ round) training runs into compression gradually was silently overriding every method's real compression settings, for the *entire* short run — invalidating every method's byte counts *and* accuracy | Turn the warm-up off for these short runs |

### 3.2 Fairness fixes

| Issue | Fix |
|---|---|
| Uniform-Top5% was training more per round than DivRoute/FedAvg (an unfair head start) | Gave all three methods the same training schedule |
| Only DivRoute had an extra training regularizer (NTD) that the other two didn't | Gave it to all three methods equally |

### 3.3 Key results (the numbers that matter)

The fairest comparison so far — all three methods given a similar amount of total communication budget to work with:

| Method | Final accuracy | Data sent (upload) |
|---|---|---|
| FedAvg (no compression) | 82.08% | 4,962.78 MB |
| **DivRoute (our method)** | **81.58%** | **2,044.43 MB** |
| Uniform-Top5% (simplest baseline) | 81.43% | 794.04 MB |

**What this shows:**

| Question | Answer |
|---|---|
| Did we hit the original accuracy target (80–85%)? | **Yes** — all three methods land in or near that range |
| Does federated fine-tuning of a pretrained model get close to full-cost accuracy using much less data? | **Yes** — both DivRoute and Uniform get within ~0.5–0.7 points of FedAvg using a fraction of the bytes |
| Does DivRoute's *smart routing* beat *dumb uniform compression*? | **No** — Uniform matches DivRoute's accuracy (a 0.15-point gap, basically noise) while sending **61% fewer** bytes |

That last row is the central open problem of the project.

---

## 4. The main remaining problem

**In one sentence:** the smart part of DivRoute — deciding which clients deserve more bandwidth — doesn't actually help, compared to just giving everyone the same small amount of bandwidth.

This was tested as fairly as possible: same number of training rounds, same training schedule, same regularizer, same model, same data. Under those conditions, the simplest possible method (compress everyone equally) does just as well, for less cost.

---

## 5. What research says, and what we've already tried

A literature review was done specifically to find out *why* this might be happening, and what other researchers have done about similar problems.

### 5.1 Leading explanations from research

| Possible reason | What research says |
|---|---|
| Missing "error feedback" | A well-known technique where, if part of an update gets dropped during compression, the server remembers what was dropped and adds it back in next time. Research says this is *essential* for this kind of compression to work well — without it, accuracy suffers in a well-documented way. This project currently has it turned off. |
| Error feedback breaks under DivRoute's own design | Research also shows error feedback specifically struggles when (a) only some clients participate each round, and (b) a client's compression amount keeps changing round to round. DivRoute does *both* — a client can be in Tier 1 one round and Tier 3 the next — so even if error feedback were turned on as-is, it likely wouldn't help without adjustment. |
| Wrong signal for measuring "how different" a client is | The current method measures difference using raw model weights. Research suggests measuring based on each client's *training loss* (how well it's doing) is often a better predictor of which clients matter. |

### 5.2 Fixes already tried, and what happened

| Fix tried | Idea (from research) | Result |
|---|---|---|
| Loss-based routing signal instead of pure weight-difference | Might identify more useful clients | **Made things worse** — 80.41% accuracy, below both DivRoute's original 81.58% and Uniform's 81.43% |
| Tier-aware error feedback | Fix error feedback so it doesn't break when a client's tier changes round to round | **Built, not tested yet** — this is the next and last research-backed fix queued |

---

## 6. What's remaining — a straight checklist

| # | Item | Status |
|---|---|---|
| 1 | Try tier-aware error feedback | Built and verified (no GPU needed to check the logic), not yet run |
| 2 | If that doesn't help either | Consider the negative result itself as the finding — it's still a real, publishable contribution (see §7) |
| 3 | Decide which "communication cost" the project's claim is about | Upload-only (the standard framing in most research papers, since a device's *upload* speed is usually the real bottleneck) makes DivRoute's numbers look good; total (upload + download) makes it closer to a wash. This needs to be picked and stated clearly, not left ambiguous |
| 4 | Run one more diagnostic (already planned, not yet done) | A test that turns off routing but keeps compression, and a test that turns off compression but keeps routing — to see which part is actually responsible for the current results |

---

## 7. If/when this gets written up for publication

| Venue type | Fit | Why |
|---|---|---|
| FL-specific workshops (NeurIPS, ICML) | **Best first target** | Lower barrier, active community, values careful methodology even without a clean "win" |
| MLSys (full conference) | **Best topical fit** | Its whole focus is exactly this — is a method actually more efficient in practice, measured honestly |
| NeurIPS / ICML / ICLR main track | Possible, but a stretch right now | Usually wants either a clear win or a much bigger set of experiments (more datasets, more models) than this project currently has |
| arXiv | Do this regardless of venue | Free, makes the work citable immediately, doesn't block submitting elsewhere later |

**Why this is still a good paper even with a "negative" core result:** the project found and fixed three real bugs that were making the numbers wrong, ran the comparison as fairly as anyone reasonably could, tested two separate research-backed fixes, and reported exactly what did and didn't work. That kind of rigor and honesty is specifically what current research surveys say the field wants more of — not another paper that quietly picks whichever framing makes the numbers look best.
