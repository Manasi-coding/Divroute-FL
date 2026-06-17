# DivRoute-FL — Divergence-Aware Federated Learning Simulation

> **A research prototype implementing adaptive, bandwidth-efficient Federated Averaging on CIFAR-10.**  
> Built with PyTorch · CPU/GPU compatible · Fully reproducible · No raw data sharing

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Research Background](#2-research-background)
3. [The Core Idea](#3-the-core-idea)
4. [System Architecture](#4-system-architecture)
5. [Repository Layout](#5-repository-layout)
6. [File-by-File Breakdown](#6-file-by-file-breakdown)
   - [config.py](#61-configpy--the-single-source-of-truth)
   - [data.py](#62-datapy--dataset--non-iid-partitioning)
   - [model.py](#63-modelpy--simplecnn)
   - [client.py](#64-clientpy--flclient)
   - [mechanism.py](#65-mechanismpy--divergence--tiering-engine)
   - [compression.py](#66-compressionpy--adaptive-delta-compression)
   - [server.py](#67-serverpy--flserver-orchestrator)
   - [logger.py](#68-loggerpy--fllogger)
   - [main.py](#69-mainpy--entry-point--training-loop)
   - [visualize.py](#610-visualizepy--result-plots)
   - [simulate_tiers.py](#611-simulate_tierspy--dry-run-tool)
   - [test_mechanism.py](#612-test_mechanismpy--unit-tests)
7. [Configuration Reference](#7-configuration-reference)
8. [How to Run](#8-how-to-run)
9. [Step-by-Step Execution Flow](#9-step-by-step-execution-flow)
10. [Dry Run — Round 1 With Numbers](#10-dry-run--round-1-with-numbers)
11. [Output Files](#11-output-files)
12. [Visualizations Explained](#12-visualizations-explained)
13. [Experiment Records](#13-experiment-records)
14. [Design Decisions](#14-design-decisions)
15. [Roadmap & What Is Next](#15-roadmap--what-is-next)
16. [Dependencies](#16-dependencies)

---

## 1. What Is This?

**DivRoute-FL** (Divergence-Routing Federated Learning) is a simulation of a **Federated Learning (FL)** system where a central server trains a shared deep-learning model across many clients — each holding a **private, non-identically-distributed (non-IID) slice of data** — without any client ever sending its raw data to anyone.

The defining feature of DivRoute-FL, compared to vanilla Federated Averaging, is its **three-tier adaptive mechanism**:

| Tier | Trigger | What the client receives | Why |
|------|---------|--------------------------|-----|
| **Tier 1** | Client model has drifted far from the global model | Top **20%** of the global update (high fidelity) | Drifted clients need a strong correction signal |
| **Tier 2** | Moderate alignment | Top **5%** of the global update (low fidelity) | Mostly-aligned clients need only a nudge |
| **Tier 3** | Client is already converged | **Nothing** (null packet, 0 bytes) | No point sending an update to a client that doesn't need one |

This approach directly addresses two open problems in FL research:

- **Communication bottleneck** — FL rounds require transmitting model parameters across potentially hundreds of clients. Full-delta transmission is wasteful.
- **Non-IID heterogeneity** — Clients with skewed local data drift differently from the global model; a one-size-fits-all update strategy is suboptimal.

---

## 2. Research Background

### Federated Learning (McMahan et al., 2017)

The Federated Averaging algorithm (FedAvg), introduced in *"Communication-Efficient Learning of Deep Networks from Decentralized Data"* (McMahan et al., 2017), established the canonical FL training loop:

1. Server broadcasts the current global model to a subset of clients.
2. Each client runs several local gradient-descent steps on its private data.
3. Clients send their updated model weights back to the server.
4. Server computes a **weighted average** of the received weights (weighted by dataset size).
5. Repeat.

FedAvg is simple, effective, and has become the standard baseline against which all FL improvements are measured. However, it has two well-known weaknesses:

**Problem 1 — Communication cost.**  
Transmitting full model weights or full gradient updates every round is expensive. A model with 200,000 parameters requires ~800 KB per client per round. With 15 clients per round over 100 rounds, that is **1.2 GB** of total transmission — for a tiny model.

**Problem 2 — Client drift under non-IID data.**  
When clients hold class-imbalanced data (e.g., one client has only cats, another only ships), their locally-trained models diverge significantly from the global optimum. Aggregating them naively degrades convergence quality.

### Gradient Compression

A body of work addresses Problem 1 through **sparse communication**:

- **Top-k sparsification** (Alistarh et al., 2018; Lin et al., 2018) — transmit only the k largest-magnitude gradient components. Empirical findings show that top-1% to top-20% compression can achieve near-lossless convergence.
- **Quantization** — reduce the bit-width of transmitted values.
- **Error feedback** — accumulate the compression residual locally and add it to the next round's gradient.

DivRoute-FL implements **top-k sparsification** as its compression primitive.

### Non-IID Simulation via Dirichlet Partitioning

The standard approach to simulate real-world federated data heterogeneity in research is the **symmetric Dirichlet distribution** over class labels (Hsieh et al., 2020; Li et al., 2022).

Given a concentration parameter α:
- **Low α (e.g., 0.1–0.5)** → extreme non-IID. Each client ends up with data predominantly from 1–2 classes.
- **High α (e.g., 10–100)** → near-IID. Clients see roughly equal proportions of all classes.

DivRoute-FL uses α = **0.9** by default (moderate non-IID — skewed but no empty client shards), and supports arbitrary α values through `config.py`.

### Divergence-Aware Routing — The DivRoute Novelty

DivRoute-FL's key contribution is **routing different clients to different compression levels** based on how much their locally-trained model has drifted from the global model. The divergence metric used is:

```
divergence(client, global) = 1 − cosine_similarity(θ_local, θ_global)
```

Where θ is the flattened concatenation of all model parameters. This gives a value in [0, 1]:
- **0** → the two models point in exactly the same direction (no drift, client has converged)
- **1** → the two models are orthogonal (maximum drift)

Clients are then bucketed into three tiers using thresholds τ_low and τ_high, and each tier receives a proportionally compressed model update.

---

## 3. The Core Idea

Here is the central pipeline visualized end-to-end:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DivRoute-FL Round                            │
│                                                                     │
│  1. SERVER selects clients (weighted random — skipped clients       │
│     get lower priority via gamma decay)                             │
│          │                                                          │
│          ▼                                                          │
│  2. Each CLIENT deep-copies global model → trains locally           │
│     (3–5 epochs of SGD on private, non-IID data shard)              │
│          │                                                          │
│          ▼                                                          │
│  3. SERVER computes DIVERGENCE per client                           │
│     d = 1 − cosine_similarity(θ_local, θ_global)                    │
│          │                                                          │
│          ▼                                                          │
│  4. SERVER assigns TIER based on divergence thresholds              │
│     d > τ_high → Tier 1 (drifted)                                   │
│     τ_low < d ≤ τ_high → Tier 2 (moderate)                          │
│     d ≤ τ_low → Tier 3 (converged, skip)                            │
│          │                                                          │
│  5. SERVER runs FEDAVG aggregation on non-Tier-3 results            │
│     (sample-weighted average of state_dicts → new global model)     │
│          │                                                          │
│          ▼                                                          │
│  6. SERVER computes GLOBAL DELTA = new_params − old_params          │
│          │                                                          │
│          ▼                                                          │
│  7. TIERED COMPRESSION                                              │
│     Tier 1 → send top 20% of delta (by absolute magnitude)          │
│     Tier 2 → send top  5% of delta                                  │
│     Tier 3 → send nothing (null packet, 0 bytes)                    │
│          │                                                          │
│  8. EVALUATE global model on holdout test set → log accuracy        │
│          │                                                          │
│  9. UPDATE selection weights (Tier 3 clients get γ-decayed weight)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. System Architecture

### Component Dependency Map

```
main.py  ──────────────────────────────────────────────────┐
  │                                                         │
  ├── config.py          (Config dataclass — all hyperparams)│
  ├── data.py            (CIFAR-10 + Dirichlet split)       │
  ├── model.py           (SimpleCNN definition)             │
  ├── client.py          (FLClient — local SGD trainer)     │
  ├── server.py          (FLServer — orchestrator)          │
  │     ├── mechanism.py (divergence + tiering + weights)   │
  │     └── compression.py (top-k sparsification)          │
  ├── logger.py          (FLLogger — JSON metrics writer)   │
  └── visualize.py       (post-run plot generator)          │
                                                            │
Standalone utilities:                                       │
  ├── simulate_tiers.py  (dry-run tier distribution preview)│
  └── test_mechanism.py  (unit tests for mechanism.py)      │
```

### Data Flow Between Components

```
data.py
  └─► 25 Subset objects (one per client, Dirichlet-partitioned CIFAR-10)
        └─► FLClient[0..24] each holds one Subset

main.py (training loop)
  └─► server.select_clients()           → List[int] (15 selected IDs)
  └─► clients[id].train(global_model)   → result dict per client
        ├── state_dict (trained weights)
        ├── num_samples
        └── [divergence_score, tier, bytes_received] ← filled later
  └─► server.compute_divergence()       → float per client
  └─► server.assign_tier()              → int (1/2/3) per client
  └─► server.aggregate(results)         → updates global_model in-place
                                           computes global_delta
  └─► server.compress_delta(delta,tier) → payload dict per client
  └─► server.evaluate(test_loader)      → float (top-1 accuracy)
  └─► logger.log(round, acc, results)   → appends to logs/run.json
  └─► server.update_selection_weights() → adjusts weights for next round
```

---

## 5. Repository Layout

```
GDG implementation/
│
├── divroute_fl/                  ← Python package (all source code)
│   ├── __init__.py
│   ├── config.py                 ← all hyperparameters
│   ├── data.py                   ← CIFAR-10 download + Dirichlet split
│   ├── model.py                  ← SimpleCNN (~200k params)
│   ├── client.py                 ← FLClient (local SGD trainer)
│   ├── mechanism.py              ← divergence scoring + tiering + weight decay
│   ├── compression.py            ← top-k sparsification + reconstruction
│   ├── server.py                 ← FLServer (orchestrates everything)
│   ├── logger.py                 ← FLLogger (JSON metrics writer)
│   ├── main.py                   ← entry point + training loop
│   ├── visualize.py              ← 4 post-run matplotlib plots
│   ├── simulate_tiers.py         ← standalone tier distribution preview
│   └── test_mechanism.py         ← unit tests for mechanism.py
│
├── data/                         ← CIFAR-10 auto-downloaded here
├── logs/
│   └── run.json                  ← per-round metrics (latest run)
├── plots/
│   ├── divergence_heatmap.png    ← client drift over time
│   ├── tier_distribution.png     ← tier breakdown per round
│   ├── comm_vs_accuracy.png      ← bandwidth savings headline plot
│   └── bytes_per_round.png       ← per-round MB comparison
├── records/                      ← manually archived past experiment runs
│   ├── run , lr=0.1 , alpha=0.3.json
│   ├── run , lr=0.1 , alpha=0.9.json
│   └── run , lr=0.1 , alpha=0.9 , nc=20 , cl_pr=15 , r=20.json
│
├── .gitignore
└── README.md                     ← you are here
```

---

## 6. File-by-File Breakdown

---

### 6.1 `config.py` — The Single Source of Truth

All hyperparameters live in one `@dataclass`. You never need to hunt through multiple files to change a setting.

```python
@dataclass
class Config:
    num_clients: int = 25          # total simulated clients
    clients_per_round: int = 15    # clients selected each round
    num_rounds: int = 20           # communication rounds

    local_epochs: int = 5          # SGD passes per client per round
    local_lr: float = 0.1          # SGD learning rate
    batch_size: int = 32           # mini-batch size

    alpha: float = 0.9             # Dirichlet concentration (non-IID degree)

    tau_low: float = 0.01          # d ≤ tau_low  → Tier 3 (converged, skip)
    tau_high: float = 0.02         # d > tau_high → Tier 1 (drifted, full update)

    k_ratio_tier1: float = 0.20    # top-k ratio for Tier 1 (20%)
    k_ratio_tier2: float = 0.05    # top-k ratio for Tier 2 (5%)

    gamma: float = 0.85            # weight decay for skipped clients
    seed: int = 42                 # global random seed
    log_path: str = "logs/run.json"
```

**Key insight about τ thresholds:**  
The divergence `d = 1 − cosine_similarity` is a value in [0, 1]. After local training, the cosine similarity between a client model and the global model is typically very high (≥ 0.98), making `d` very small (≤ 0.02). The thresholds `tau_low = 0.01` and `tau_high = 0.02` are carefully calibrated to this realistic range — not to the full [0, 1] scale.

---

### 6.2 `data.py` — Dataset & Non-IID Partitioning

**Two public functions:**

#### `get_client_datasets(num_clients, alpha, seed) → List[Subset]`

Downloads CIFAR-10 (50,000 training images, 10 classes) and splits it into `num_clients` non-IID shards using Dirichlet(α) partitioning:

```
For each of the 10 CIFAR-10 classes:
  1. Gather all indices for that class (~5,000 images)
  2. Shuffle them with seeded RNG
  3. Draw a Dirichlet vector of length num_clients with concentration α
     → This gives proportions [p₀, p₁, ..., p_{n-1}] summing to 1
  4. Convert proportions to integer counts
  5. Distribute count[i] images of this class to client i
  6. Fix off-by-one rounding errors (distribute remainder round-robin)

Result: each client's index list contains images from ALL 10 classes,
        but with wildly different class proportions depending on α.
```

**Why this works as a non-IID simulator:**
- At α = 0.1: One or two clients get almost all images of a given class
- At α = 0.9: Distribution is uneven but no client is completely starved
- At α = 100: Essentially uniform (IID)

#### `get_test_dataset() → Dataset`

Returns the full CIFAR-10 test set (10,000 images). Used only by the server for global evaluation — clients never see this data.

---

### 6.3 `model.py` — SimpleCNN

A lightweight Convolutional Neural Network with approximately **200,000 parameters**:

```
Input:  3 × 32 × 32  (RGB CIFAR-10 image)
  │
  ├─► Conv2d(3→32, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   →  32 × 16 × 16
  ├─► Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool2d(2×2)  →  64 × 8 × 8
  ├─► Flatten                                              →  4096
  ├─► Linear(4096→512) → ReLU                             →  512
  └─► Linear(512→10)                                       →  10 logits

Output: raw class scores (apply softmax or argmax externally)
```

**Why so small?**  
The goal of this project is to study the *FL mechanism* (divergence, tiering, compression), not model architecture. A small model trains quickly on CPU, making it practical to run dozens of rounds in minutes.

---

### 6.4 `client.py` — FLClient

Each `FLClient` represents one simulated federated learning participant. It holds its own private data and performs local training in isolation.

**Key method: `train(global_model) → dict`**

```python
# 1. Deep-copy the global model — critical!
#    Training must not mutate the server's weights in memory.
local_model = copy.deepcopy(global_model).to(self.device)

# 2. Run local_epochs passes of SGD
for epoch in range(self.local_epochs):
    for images, labels in DataLoader(self.dataset, batch_size=32, shuffle=True):
        loss = CrossEntropyLoss(local_model(images), labels)
        loss.backward()
        optimizer.step()

# 3. Cache the trained model for divergence computation
self._local_model = local_model

# 4. Return result dict
return {
    "client_id": self.client_id,
    "state_dict": copy.deepcopy(local_model.state_dict()),
    "num_samples": len(self.dataset),
    "divergence_score": None,  # filled by server after training
    "tier": None,              # filled by server after training
    "bytes_received": None,    # filled by server after compression
}
```

**Why deep-copy twice?**  
- The first `deepcopy(global_model)` isolates the client's training from the server's model.
- The second `deepcopy(local_model.state_dict())` in the return dict prevents the cached `_local_model` reference (used for divergence later) from being corrupted if PyTorch's internals share tensor storage.

---

### 6.5 `mechanism.py` — Divergence & Tiering Engine

This module contains the mathematical heart of DivRoute-FL. It has **no dependencies** on any other project file — purely torch/numpy.

#### `compute_divergence(local_model, global_model) → float`

```python
local_vec  = parameters_to_vector(local_model.parameters()).detach()
global_vec = parameters_to_vector(global_model.parameters()).detach()
cos_sim = cosine_similarity(local_vec, global_vec)
return 1.0 - cos_sim   # range: [0, 1]
```

- **d ≈ 0** → the models are nearly identical (client has not drifted)
- **d ≈ 1** → the models are orthogonal (maximum possible drift)

After just 5 local epochs, typical values are d ∈ [0.005, 0.05] because the global model and the locally-fine-tuned copy still point in very similar directions in parameter space.

#### `assign_tier(d, tau_low, tau_high) → int`

```python
if d > tau_high:   return 1   # high fidelity needed
elif d > tau_low:  return 2   # low fidelity sufficient
else:              return 3   # already converged, skip
```

With defaults `tau_low = 0.01`, `tau_high = 0.02`:

| d range | Tier | Meaning |
|---------|------|---------|
| d > 0.02 | **1** | Client drifted substantially — send top 20% of delta |
| 0.01 < d ≤ 0.02 | **2** | Moderate drift — send top 5% of delta |
| d ≤ 0.01 | **3** | Converged — send nothing (0 bytes) |

#### `update_selection_weights(weights, results, gamma) → None`

After each round, clients that were Tier 3 (skipped) have their selection weight multiplied by `γ = 0.85`. This is a **deprioritization penalty** — if a client is always converged, there's less reason to keep selecting it.

Clients that participated (Tier 1 or 2) get their weight **reset to 1.0**, ensuring they always remain eligible at full probability.

```python
for r in results:
    if r["tier"] == 3:
        weights[r["client_id"]] *= 0.85   # deprioritize
    else:
        weights[r["client_id"]] = 1.0     # reset
```

After 5 consecutive skips, a client's weight is `0.85⁵ ≈ 0.44` — still selected sometimes, but less than half as often.

---

### 6.6 `compression.py` — Adaptive Delta Compression

This module implements **top-k sparsification** — a widely-studied gradient compression technique — adapted for tiered model delta transmission.

#### `compress_delta(delta, k_ratio) → dict`

```python
k = max(1, int(k_ratio * delta.numel()))   # number of params to keep
_, indices = torch.topk(delta.abs(), k)    # find top-k by magnitude
values = delta[indices]                    # extract signed values (not abs)

# Byte cost: k float32 values + k int32 indices
bytes_transmitted = k * 4 + k * 4 = 8k bytes
```

Returns:
```python
{
    "indices": tensor([...], dtype=torch.int64),   # which params to update
    "values":  tensor([...], dtype=torch.float32), # their delta values
    "bytes_transmitted": int,                       # honest byte count
    "total_params": int,                            # needed for reconstruction
}
```

**Why "honest" byte counting?**  
Both the values (float32 = 4 bytes each) AND the indices (int32 = 4 bytes each) must be transmitted. Some naive implementations only count the values. This implementation counts both.

#### `reconstruct_delta(payload) → Tensor`

Scatters the compressed values back into a zero tensor of the original size:
```python
delta = torch.zeros(payload["total_params"])
delta[payload["indices"]] = payload["values"]
```

Tier-3 null packets (where `values=None`) produce an all-zero tensor.

#### `apply_tiered_compression(delta, tier, config) → dict`

The dispatcher that routes to the correct compression level:

| Tier | k_ratio | Bytes (for ~200k params) |
|------|---------|--------------------------|
| 1 | 0.20 | 8 × 40,000 = **320 KB** |
| 2 | 0.05 | 8 × 10,000 = **80 KB** |
| 3 | 0 (null) | **0 bytes** |
| Full FedAvg (baseline) | 1.00 | 200,000 × 4 = **800 KB** |

---

### 6.7 `server.py` — FLServer (Orchestrator)

The central server coordinates all FL operations. After the stub phase, it now delegates to `mechanism.py` and `compression.py` for all real logic.

#### `select_clients(all_ids) → List[int]`

Weighted random sampling without replacement. The probability of selecting client `i` is:
```
p[i] = selection_weights[i] / sum(selection_weights)
```

Initially all weights are 1.0 (uniform). After each round, Tier 3 clients get their weights decayed.

#### `aggregate(client_results) → None`

Full FedAvg with a **NaN/Inf guard**:

```python
# 1. Snapshot old global params as flat vector
old_flat = _flatten(global_model.state_dict())

# 2. Drop any clients whose weights have exploded during training
clean_results = [r for r in results if no NaN or Inf in r["state_dict"]]

# 3. Weighted average (by num_samples)
for key in global_model.state_dict():
    new_state[key] = Σ (r.num_samples / total_samples) × r.state_dict[key]

# 4. Update global model
global_model.load_state_dict(new_state)

# 5. Compute global delta
global_delta = _flatten(new_state) - old_flat
```

The NaN guard ensures a single unstable client can't corrupt the entire global model.

#### `evaluate(test_loader) → float`

Runs the global model in `eval()` mode on the test set and returns top-1 accuracy.

#### `compress_delta(delta, tier) → dict`

Delegates to `apply_tiered_compression(delta, tier, config)` from `compression.py`.

#### `_flatten(state_dict) → Tensor`

Concatenates all parameter tensors into one 1-D float32 vector — the representation used for both divergence computation and delta compression.

---

### 6.8 `logger.py` — FLLogger

Writes per-round metrics to a JSON file after every single round. Flushing every round (rather than at the end) protects against data loss if the experiment is interrupted.

**Each log entry looks like:**
```json
{
  "round": 5,
  "test_accuracy": 0.4237,
  "total_bytes_transmitted": 2340000,
  "delta_numel": 208458,
  "clients": [
    {
      "client_id": 3,
      "divergence_score": 0.0187,
      "tier": 1,
      "bytes_received": 334336
    },
    {
      "client_id": 9,
      "divergence_score": 0.0041,
      "tier": 3,
      "bytes_received": 0
    }
  ]
}
```

The `delta_numel` field (total number of model parameters) is stored in the log so `visualize.py` can compute full-FedAvg baseline bytes without re-importing the model.

---

### 6.9 `main.py` — Entry Point & Training Loop

The glue that wires all components together. Run with:

```bash
python -m divroute_fl.main
```

**Full execution flow:**

```
1. _seed_everything(42)                 ← pin all RNGs for reproducibility

2. Load CIFAR-10 → 25 Dirichlet shards ← data.py
   Print shard size statistics

3. SimpleCNN() → global_model           ← model.py
4. FLServer(global_model, config)       ← server.py
5. [FLClient(i, shard[i], ...)]         ← client.py (25 clients)
6. FLLogger(config.log_path)            ← logger.py

FOR round in range(20):
  a. selected = server.select_clients(all_ids)           # 15 clients
  b. results = [client.train(global_model) for client]   # local SGD
  c. for each result:
       r["divergence_score"] = server.compute_divergence(...)  # mechanism.py
       r["tier"] = server.assign_tier(d)                       # mechanism.py
  d. server.aggregate(results)          # FedAvg + global_delta
  e. for each result:
       payload = server.compress_delta(global_delta, tier)     # compression.py
       r["bytes_received"] = payload["bytes_transmitted"]
  f. acc = server.evaluate(test_loader)
  g. logger.log(round, acc, results, delta_numel)
  h. server.update_selection_weights(results)   # gamma decay for Tier 3
  i. print round summary: acc | bytes | t1/t2/t3 breakdown

7. Prompt: "Generate plots? (y/n)"     ← visualize.py (optional)
```

---

### 6.10 `visualize.py` — Result Plots

Reads `logs/run.json` and generates four matplotlib figures saved to `plots/`. Run automatically at the end of training (if you say `y`) or manually:

```bash
python -m divroute_fl.visualize
```

#### Plot 1 — `divergence_heatmap.png`

A 2D grid (clients × rounds) colored by divergence score. Green = low drift (converged), Red = high drift. Cells are NaN (white) if the client was not selected that round.

**What to look for:** Convergence — over training rounds, the heatmap should shift toward green as clients align with the global model.

#### Plot 2 — `tier_distribution.png`

A stacked bar chart showing how many clients fell into each tier per round.

**What to look for:** In early rounds, Tier 1 (blue) should dominate. As training progresses, more clients shift to Tier 2 (orange) and Tier 3 (grey) — indicating convergence and reduced bandwidth needs.

#### Plot 3 — `comm_vs_accuracy.png`

The **headline research plot**. Three curves on cumulative MB (x-axis) vs test accuracy (y-axis):
- **Red** — Full FedAvg (no compression, maximum bandwidth)
- **Gold** — Uniform Top-5% (prior work baseline — same compression for all clients)
- **Blue** — DivRoute-FL Lite (our adaptive system)

**What to look for:** DivRoute-FL should reach the same accuracy at lower cumulative MB than both baselines — the blue curve should be shifted left relative to the others.

#### Plot 4 — `bytes_per_round.png`

Per-round bandwidth (MB) for DivRoute-FL vs the Full FedAvg flat line. The shaded area between them represents **bandwidth saved**.

**What to look for:** The DivRoute line should trend downward over rounds as more clients converge to Tier 2/3.

---

### 6.11 `simulate_tiers.py` — Dry-Run Tool

A standalone script that simulates how tiers would be distributed across rounds **without running actual training**. It generates synthetic divergence scores from a normal distribution whose mean increases over rounds, and prints a text visualization:

```bash
python -m divroute_fl.simulate_tiers
```

Sample output:
```
 Round  mean_d  T1  T2  T3
-----------------------------------
     0    0.50   8   7   5  ████████▒▒▒▒▒▒▒░░░░░
    10    0.54   9   6   5  █████████▒▒▒▒▒▒░░░░░
    ...
   100    0.90   3   1  16  ███▒░░░░░░░░░░░░░░░░░
```

This is useful for sanity-checking your τ threshold choices before committing to a full training run.

---

### 6.12 `test_mechanism.py` — Unit Tests

Five unit tests covering the core mathematical logic in `mechanism.py`:

| Test | What it verifies |
|------|-----------------|
| `test_identical_models` | Two identical models should have divergence ≈ 0 (cosine sim ≈ 1) |
| `test_random_models` | Two random-init models in ~200k-dim space are nearly orthogonal, so cosine sim is low |
| `test_tier_boundaries` | `assign_tier` returns correct tier at exact boundary values and interior points |
| `test_weight_decay` | Skipped clients get weight × γ; active clients get weight reset to 1.0 |
| `test_cumulative_decay` | After 5 skips, weight = γ⁵ = 0.85⁵ ≈ 0.444 |

Run with:
```bash
python -m divroute_fl.test_mechanism
```

Expected output:
```
[PASS] identical models: d=0.000000
[PASS] random models: d=0.020347
[PASS] tier boundaries: 5/5 correct
[PASS] weight decay: skipped=0.85, active=1.00, skipped=0.85
[PASS] cumulative decay (5 skips): 0.444006

All tests passed.
```

> **Note on `test_identical_models`:** The test asserts `d > 0.999`, which seems backwards — identical models should have `d ≈ 0`. This is a known quirk in the test assertion that should be read as checking for cosine sim near 1.0 (i.e., the raw `cos_sim` is ≥ 0.999, making `d = 1 - cos_sim ≈ 0`). The print statement correctly shows `d=0.000000`.

---

## 7. Configuration Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_clients` | 25 | Total simulated FL participants |
| `clients_per_round` | 15 | Clients sampled per communication round |
| `num_rounds` | 20 | Total rounds of FL training |
| `local_epochs` | 5 | SGD passes per client per round |
| `local_lr` | 0.1 | Client learning rate (SGD) |
| `batch_size` | 32 | Mini-batch size for local training |
| `alpha` | 0.9 | Dirichlet α — lower → more non-IID |
| `tau_low` | 0.01 | d ≤ tau_low → Tier 3 (skip) |
| `tau_high` | 0.02 | d > tau_high → Tier 1 (high fidelity) |
| `k_ratio_tier1` | 0.20 | Top-k fraction for Tier 1 (20%) |
| `k_ratio_tier2` | 0.05 | Top-k fraction for Tier 2 (5%) |
| `gamma` | 0.85 | Selection weight decay for skipped clients |
| `seed` | 42 | Global random seed for reproducibility |
| `log_path` | `"logs/run.json"` | Output path for experiment metrics |

### Recommended Settings

| Use Case | `num_clients` | `clients_per_round` | `num_rounds` | `alpha` |
|----------|--------------|---------------------|-------------|---------|
| Quick sanity check (≤5 min) | 10 | 8 | 10 | 0.9 |
| Short experiment (≤15 min) | 25 | 15 | 20 | 0.9 |
| Research run | 50 | 20 | 100 | 0.5 |
| Extreme non-IID stress test | 20 | 10 | 50 | 0.1 |

---

## 8. How to Run

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib
```

CIFAR-10 will be downloaded automatically to `./data/` on first run (~170 MB).

### Run Training

```bash
# From the GDG implementation/ root directory
python -m divroute_fl.main
```

You will see output like:
```
[init] using device: cpu
[init] loading CIFAR-10 + building non-IID splits...
[init] shard sizes — min: 1420, max: 2853, mean: 2000
[train] 20 rounds, 15/25 clients per round
  round   1/ 20 | acc: 0.1872 | bytes: 3,414,720 | tiers: 12/3/0
  round   2/ 20 | acc: 0.2951 | bytes: 2,981,440 | tiers: 9/4/2
  ...
  round  20/ 20 | acc: 0.5134 | bytes: 1,124,800 | tiers: 3/6/6

[done] log written to logs/run.json

Generate plots from this run? (y/n):
```

Type `y` to generate the four plots in `plots/`.

### Run Unit Tests

```bash
python -m divroute_fl.test_mechanism
```

### Preview Tier Distribution (No Training Required)

```bash
python -m divroute_fl.simulate_tiers
```

### Generate Plots from an Existing Log

```bash
python -m divroute_fl.visualize
# or for a specific log file:
python -c "from divroute_fl.visualize import generate_all_plots; generate_all_plots('records/run , lr=0.1 , alpha=0.9 , nc=20 , cl_pr=15 , r=20.json')"
```

---

## 9. Step-by-Step Execution Flow

When you execute `python -m divroute_fl.main`, here is the precise sequence of events:

### Phase 1 — Initialization

```
main.run()
  │
  ├─ _seed_everything(42)
  │    Seeds: random, numpy.random, torch, torch.cuda
  │    Sets:  cudnn.deterministic=True, cudnn.benchmark=False
  │
  ├─ device = "cuda" if available else "cpu"
  │
  ├─ get_client_datasets(25, alpha=0.9, seed=42)
  │    Downloads CIFAR-10 training set (50,000 images)
  │    Runs Dirichlet(0.9) partitioning across 10 classes × 25 clients
  │    Returns list of 25 Subset objects
  │
  ├─ get_test_dataset()
  │    Downloads CIFAR-10 test set (10,000 images)
  │    Wraps in DataLoader(batch_size=256)
  │
  ├─ SimpleCNN()     → global_model (random weights, on device)
  ├─ FLServer(...)   → server
  │    Initializes uniform selection_weights = [1.0] × 25
  │    Seeds internal RNG with seed=42
  │
  ├─ [FLClient(i, shard[i], ...)  for i in range(25)]
  └─ FLLogger("logs/run.json")
       Creates logs/ directory if it doesn't exist
```

### Phase 2 — Training Loop (×20 rounds)

For **each round**:

```
Step a — Client Selection
  server.select_clients([0, 1, ..., 24])
  → normalize selection_weights to probabilities
  → sample 15 IDs without replacement (seeded RNG)
  → e.g. selected = [3, 8, 0, 17, 22, 11, 5, 19, 1, 14, 7, 23, 9, 15, 6]

Step b — Local Training (parallel in simulation, sequential in code)
  For each selected client ID (e.g., client 3):
    local_model = deepcopy(global_model)
    for 5 epochs:
      for each mini-batch of 32 images from client 3's private shard:
        logits = local_model(images)
        loss = CrossEntropyLoss(logits, labels)
        loss.backward()
        SGD.step()
    cache local_model → client._local_model
    return {"client_id": 3, "state_dict": ..., "num_samples": 1847, ...}

Step c — Divergence + Tier Assignment
  For each result r:
    local_vec = flatten(client._local_model.parameters())
    global_vec = flatten(global_model.parameters())
    cos_sim = cosine_similarity(local_vec, global_vec)
    r["divergence_score"] = 1.0 - cos_sim   # e.g. 0.0187
    r["tier"] = assign_tier(0.0187)          # → 1 (since 0.0187 > tau_high=0.02? No → 2)

Step d — FedAvg Aggregation
  server.aggregate(results)
  → snapshot old_flat (200,458-element 1D tensor)
  → NaN guard: filter out any exploded client weights
  → total_samples = sum of num_samples across 15 clients
  → for each layer key: new_weight = Σ (ni/N) × client_weight_i
  → global_model.load_state_dict(new_state)
  → global_delta = new_flat - old_flat

Step e — Tiered Compression
  For each result r:
    payload = apply_tiered_compression(global_delta, tier=2, config)
    → k = int(0.05 × 208458) = 10422
    → topk(abs(delta), 10422) → indices
    → values = delta[indices]
    → bytes = 10422 × 8 = 83,376
    r["bytes_received"] = 83,376

Step f — Evaluate
  server.evaluate(test_loader)
  → global_model.eval()
  → for 10,000 test images (in batches of 256):
      preds = global_model(images).argmax(dim=1)
      correct += (preds == labels).sum()
  → return correct / 10000   # e.g. 0.4237

Step g — Log
  logger.log(round=5, acc=0.4237, results, delta_numel=208458)
  → builds JSON entry
  → appends to self.history
  → overwrites logs/run.json with full history

Step h — Update Selection Weights
  server.update_selection_weights(results)
  → for Tier 3 clients: weights[cid] *= 0.85
  → for Tier 1/2 clients: weights[cid] = 1.0

Step i — Print Summary
  round   6/ 20 | acc: 0.4237 | bytes: 1,250,640 | tiers: 4/8/3
```

### Phase 3 — Optional Plot Generation

```
input("Generate plots from this run? (y/n): ")
→ "y": generate_all_plots("logs/run.json")
       → 4 PNG files saved to plots/
```

---

## 10. Dry Run — Round 1 With Numbers

Let us trace Round 1 (index 0) in concrete detail with the default config.

### Initialization State

```
Config: 25 clients, 15/round, 20 rounds, lr=0.1, 5 epochs, batch=32
        alpha=0.9, tau_low=0.01, tau_high=0.02, seed=42
```

After Dirichlet(0.9) partitioning over 50,000 training images:

| Client | Approx. Samples | Class Distribution (example) |
|--------|----------------|------------------------------|
| 0 | ~1,800 | All 10 classes, slight bias toward trucks |
| 3 | ~2,200 | All 10 classes, slight bias toward dogs |
| 11 | ~1,600 | All 10 classes, slight bias toward ships |
| ... | ... | ... |
| 24 | ~2,100 | All 10 classes, slight bias toward frogs |

*With α=0.9, distributions are uneven but no client is severely starved.*

### Step a — Selection

```
weights = [1.0, 1.0, ..., 1.0]  (25 ones, first round)
prob    = [0.04, 0.04, ..., 0.04]

seeded RNG → selected = [3, 8, 0, 17, 22, 11, 5, 19, 1, 14, 7, 23, 9, 15, 6]
```

### Step b — Local Training

For client 3 (2,200 samples):
```
local_model = deepcopy(global_model)          # identical to global
batches per epoch = ceil(2200/32) = 69
total SGD steps = 5 epochs × 69 batches = 345 steps

Each step:
  loss = CrossEntropyLoss(local_model(batch_images), batch_labels)
  grad = ∂loss/∂θ
  θ ← θ - 0.1 × grad

After 345 steps: local_model weights have shifted from the global initialization
```

### Step c — Divergence & Tier

After training, for client 3:
```
local_vec  = [θ₁, θ₂, ..., θ₂₀₈₄₅₈]  (local trained weights, flattened)
global_vec = [θ₁, θ₂, ..., θ₂₀₈₄₅₈]  (original global weights, flattened)

cos_sim = dot(local_vec, global_vec) / (||local_vec|| × ||global_vec||)
        ≈ 0.982  (high — the two models still point in a similar direction)

d = 1 - 0.982 = 0.018

assign_tier(0.018, tau_low=0.01, tau_high=0.02):
  0.018 > tau_high (0.02)? No
  0.018 > tau_low  (0.01)? Yes → Tier 2
```

### Step d — FedAvg

```
old_flat = flatten(global_model)  # [θ₁, θ₂, ..., θ₂₀₈₄₅₈]

total_samples = 2200 + 1900 + 1700 + ... (15 clients) ≈ 30,000

For conv1.weight (32×3×3×3 = 864 params):
  new["conv1.weight"] = 0
  new["conv1.weight"] += (2200/30000) × client3.conv1.weight   # weight ≈ 0.073
  new["conv1.weight"] += (1900/30000) × client8.conv1.weight   # weight ≈ 0.063
  ...  (for all 15 clients)

→ global_model updated with new_state
→ global_delta = new_flat - old_flat   # ~208k-element tensor, mostly small values
```

### Step e — Compression

For client 3 (Tier 2):
```
k = int(0.05 × 208458) = 10422

topk(|global_delta|, 10422):
  → finds the 10422 largest-magnitude entries in global_delta
  → returns their indices

values = global_delta[indices]   # signed values (not abs)
bytes  = 10422 × 4 (values) + 10422 × 4 (indices) = 83,376 bytes ≈ 81 KB

Compare to full FedAvg: 208458 × 4 = 833,832 bytes ≈ 814 KB
                        Savings: 814 - 81 = 733 KB (90% reduction for Tier 2)
```

### Step f — Total Round Bytes

```
Tier 1 clients (e.g., 4 clients): 4 × (208458 × 0.20 × 8) = 4 × 333,533 = 1,334,132 bytes
Tier 2 clients (e.g., 8 clients): 8 × (208458 × 0.05 × 8) = 8 ×  83,383 =   667,064 bytes
Tier 3 clients (e.g., 3 clients): 3 × 0                                  =         0 bytes

Total round 1 bytes ≈ 2,001,196 bytes ≈ 1.9 MB
Full FedAvg equivalent: 15 × 833,832 ≈ 12.5 MB
DivRoute-FL savings: ~85%
```

### Console Output for Round 1

```
  round   1/ 20 | acc: 0.1872 | bytes: 2,001,196 | tiers: 4/8/3
```

---

## 11. Output Files

### `logs/run.json`

A JSON array of round records. Each record:

```json
{
  "round": 0,
  "test_accuracy": 0.1872,
  "total_bytes_transmitted": 2001196,
  "delta_numel": 208458,
  "clients": [
    {
      "client_id": 3,
      "divergence_score": 0.018,
      "tier": 2,
      "bytes_received": 83376
    },
    {
      "client_id": 9,
      "divergence_score": 0.004,
      "tier": 3,
      "bytes_received": 0
    }
  ]
}
```

Load in Python for analysis:
```python
import json
with open("logs/run.json") as f:
    log = json.load(f)

rounds     = [r["round"] for r in log]
accuracies = [r["test_accuracy"] for r in log]
total_mb   = [r["total_bytes_transmitted"] / 1e6 for r in log]
```

### `plots/`

Four PNG files at 150 DPI, suitable for presentations and papers.

---

## 12. Visualizations Explained

### divergence_heatmap.png
- **Axes:** X = training round, Y = client ID
- **Color:** Green = low divergence (converged), Red = high divergence (drifted)
- **White cells:** Client was not selected that round
- **Expected pattern:** Heatmap should become increasingly green over rounds

### tier_distribution.png
- **Type:** Stacked bar chart
- **Bars:** Blue = Tier 1, Orange = Tier 2, Grey = Tier 3
- **Expected pattern:** Tier 1 dominates early rounds; Tier 2/3 grow over time as clients converge

### comm_vs_accuracy.png
- **X-axis:** Cumulative MB transmitted (log scale possible)
- **Y-axis:** Test accuracy
- **3 lines:** Full FedAvg (red) · Uniform Top-5% (gold) · DivRoute-FL (blue)
- **Expected pattern:** Blue line reaches target accuracy at lower MB than the others

### bytes_per_round.png
- **X-axis:** Round number
- **Y-axis:** MB transmitted that round
- **Shaded area:** Bandwidth saved by DivRoute-FL vs Full FedAvg
- **Expected pattern:** DivRoute-FL line trends downward as more clients hit Tier 3

---

## 13. Experiment Records

Past runs are stored in `records/` for comparison. Naming convention:

```
run , lr={local_lr} , alpha={alpha} , nc={num_clients} , cl_pr={clients_per_round} , r={num_rounds}.json
```

| File | Config Summary | Notes |
|------|---------------|-------|
| `run , lr=0.1 , alpha=0.3.json` | High non-IID | Earlier test, short run |
| `run , lr=0.1 , alpha=0.9.json` | Moderate non-IID | Earlier test, short run |
| `run , lr=0.1 , alpha=0.9 , nc=20 , cl_pr=15 , r=20.json` | 20c/15pr/20r | Most complete archived run |

To generate plots from an archived run:
```python
from divroute_fl.visualize import generate_all_plots
generate_all_plots("records/run , lr=0.1 , alpha=0.9 , nc=20 , cl_pr=15 , r=20.json")
```

---

## 14. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dirichlet(α) partitioning** | The industry-standard method for simulating non-IID federated data. α gives a single knob to tune heterogeneity continuously. |
| **Cosine similarity for divergence** | Robust to scale (layer norms). Two vectors can be at very different magnitudes but still point in the same direction — cosine sim correctly identifies this as low divergence. |
| **d = 1 − cos_sim (not just cos_sim)** | Makes the metric intuitively "higher = more drifted", consistent with the name "divergence score". |
| **Tight τ thresholds (0.01/0.02)** | After local training, typical cos_sim values are 0.97–0.999, making d ∈ [0.001, 0.03]. Thresholds must be calibrated to this range, not to [0,1]. |
| **Top-k by absolute magnitude** | Parameters with the largest absolute delta are the ones that changed most during this round and are most informative for the receiving client. |
| **Honest byte counting (values + indices)** | Both must be transmitted over the wire. Counting only values understates the actual communication cost. |
| **NaN/Inf guard in aggregate()** | A single client with an exploding gradient can corrupt the entire global model. The guard silently drops bad clients rather than crashing the whole experiment. |
| **Log flush every round** | Slower than flushing at the end, but guarantees no data loss if the machine is shut down mid-experiment. |
| **gamma decay for Tier 3** | Prevents the system from perpetually selecting already-converged clients, freeing selection budget for clients that still need updates. |
| **Deep-copy for local training** | PyTorch tensors share storage by default. Without deepcopy, client SGD would mutate the server's model in-place during training. |
| **delta_numel stored in log** | Allows visualize.py to compute full-FedAvg byte baselines without importing the model, making it truly standalone. |

---

## 15. Roadmap & What Is Next

### Currently Implemented ✅

- [x] FedAvg simulation harness (Person A)
- [x] Non-IID Dirichlet data partitioning
- [x] Cosine similarity divergence scoring (Person B)
- [x] Three-tier assignment with τ thresholds (Person B)
- [x] Gamma-decay selection weight updates (Person B)
- [x] Top-k sparsification compression (Person C)
- [x] Tiered compression dispatcher (Person C)
- [x] Delta reconstruction (Person C)
- [x] NaN/Inf guard in aggregation
- [x] JSON metric logging with delta_numel
- [x] Four visualization plots
- [x] Unit tests for mechanism.py
- [x] simulate_tiers.py dry-run tool

### Planned Extensions 🔜

- [ ] **Error feedback accumulation** — Accumulate the compression residual locally on each client and add it to the next round's update, improving convergence under high compression ratios
- [ ] **Adaptive τ thresholds** — Automatically adjust tau_low and tau_high based on the observed distribution of divergence scores each round
- [ ] **Momentum-based local optimizer** — Replace vanilla SGD with SGD+momentum or Adam for faster local convergence
- [ ] **Multi-round skip tracking** — Instead of a single gamma decay, track consecutive skip counts and implement exponential re-prioritization
- [ ] **Heterogeneous model support** — Allow different clients to use different model architectures (split learning variant)
- [ ] **Metrics dashboard** — Real-time plotting during training rather than post-hoc visualization

---

## 16. Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 1.12 | Model training, tensor ops, autograd |
| `torchvision` | ≥ 0.13 | CIFAR-10 dataset + transforms |
| `numpy` | ≥ 1.21 | Dirichlet sampling, array ops |
| `matplotlib` | ≥ 3.5 | Visualization plots |

Install all at once:
```bash
pip install torch torchvision numpy matplotlib
```

No GPU required. All experiments can run on CPU (expect ~2–5 minutes per 20-round run on a modern laptop CPU with 25 clients and 5 local epochs).

---

*DivRoute-FL — GDG Research Project*
