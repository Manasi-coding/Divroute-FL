# DivRoute-FL

A PyTorch research simulator for client-dependent compression in federated learning. It compares divergence-based upload allocation with dense FedAvg, uniform compression, random allocation, reversed routing, and size-based allocation.

## Setup

Use Python 3.10 or newer with a compatible PyTorch/torchvision installation. The experiments also use NumPy, SciPy, Matplotlib, scikit-learn, and Hugging Face `datasets` for FEMNIST. Dataset downloads and pretrained backbones require internet access on their first use.

```bash
python -m pip install torch torchvision numpy scipy matplotlib scikit-learn datasets
```

For reproducible experiments, retain the exact installed versions, GPU details, configuration, source revision, logs, and checkpoints. Run commands from the repository root.

## Experiments

FEMNIST uses one writer per client and a fixed set of held-out test writers. CIFAR-10 uses Dirichlet partitions. Dataset and model defaults are defined by the factories in `divroute_fl/config.py`; the generic `Config()` defaults differ from the experiment presets.

```bash
# Main FEMNIST comparison
python scripts/run_femnist_experiments.py --variant fedavg uniform divroute --seeds 42 43 44

# Equal-budget allocation controls at the tighter compression setting
python scripts/run_femnist_experiments.py --variant pure purerandom pureinvert matched uniform --kscale 0.1 --num-rounds 150 --seeds 42 43 44

# DAGC-inspired allocation and controls
python scripts/run_femnist_experiments.py --variant dagc dagcrev dagcrandom --kscale 0.1 --num-rounds 150 --seeds 42 43 44

# CIFAR-10 comparison
python scripts/run_femnist_experiments.py --dataset cifar10 --variant dagc dagcrev dagcrandom --kscale 0.1 --num-rounds 150 --local-epochs 5 --lr 0.03 --seeds 42 43 44

# Pretrained CIFAR-100 companion experiments
python run_pretrained_companion_experiments.py --help
```

`--kscale` multiplies retention ratios. With `--kscale 0.1`, Uniform retains 0.5%, the tiered variants retain 2%/0.5%, and `matched` retains 1.05%. Byte matching is approximate; compare the recorded counts. DAGC variants implement a proportional allocation rule adapted to partial client participation, not the complete published algorithm.

Finished runs are skipped when their complete JSON logs are present. Interrupted runs resume from available checkpoints. Checkpoints are written every five rounds, with additional milestones every 50 rounds. The runner removes a finished run's checkpoint directory unless `--keep-checkpoints` is supplied. Preserve both logs and checkpoints between notebook sessions.

## Training and communication

Each selected client starts from the broadcast global state and trains locally. The server computes routing scores and assigns tiers before compressing and aggregating client updates.

- The normal polarity assigns Tier 1 to the lowest scores, Tier 2 to the middle scores, and Tier 3 to the highest scores. Inverted routing swaps the extreme tiers.
- Tier 1 uses `k_ratio_tier1`. Tier 2 and included Tier 3 clients use `k_ratio_tier2`. The FEMNIST and CIFAR-10 study presets include Tier 3 in aggregation; the generic configuration can exclude it.
- Error feedback, threshold adaptation, weighting, selection decay, and momentum are controlled independently by configuration.
- Sparse uploads count values and int32 indices. Dense uploads have no index overhead. Downloads are full-model broadcasts, optionally simulated in FP16.
- This is an in-process simulator: communication counts describe simulated payloads, not measured network traffic.

## Code layout

| Module | Responsibility |
|---|---|
| `main.py` | Experiment setup, round orchestration, scoring, and routing |
| `client.py` | Local training, regularizers, and local evaluation |
| `server.py` | Client selection, compressed aggregation, and global evaluation |
| `compression.py` | Global/layerwise Top-k, reconstruction, and error feedback |
| `mechanism.py` | Divergence metrics, tiers, thresholds, and allocation rules |
| `checkpointing.py` | Checkpoint persistence and state restoration |
| `reporting.py`, `diagnostics.py` | Console reports and experiment diagnostics |
| `logger.py`, `visualize.py` | JSON logs and plots |
| `data.py`, `femnist_data.py`, `model.py` | Datasets, partitions, and architectures |
| `config.py` | Configuration fields and experiment presets |

## Results and checks

```bash
python scripts/summarize_runs.py --last 20 logs/*.json
python scripts/paired_stats.py
python divroute_fl/test_mechanism.py
python -m unittest discover -s tests -v
```

The summary script reads raw JSON logs. The paired-statistics script reads `results/plateaus.csv`; new runs must be incorporated into that table before they appear in its comparisons. Its tests include paired t-intervals, sign-flip permutation tests, and Holm correction. Summary-script gap-versus-spread labels are descriptive, not significance tests.

The regression tests use small CPU datasets and models without downloading data. Full research runs and CUDA validation remain separate from these tests.

Historical experiment decisions and results are recorded in `DIVROUTE_ACCURACY_MASTER_PLAN.md`. Earlier overview and algorithm documents describe superseded experiments as well as the current study; use the implementation and preset configurations to resolve discrepancies.
