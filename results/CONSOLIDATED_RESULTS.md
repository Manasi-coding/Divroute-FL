# Consolidated experiment results — 2026-11-11

All 28 planned runs are complete: 26 original JSON logs and two console reconstructions.
The master table contains 118 distinct dataset/setting/variant/seed rows. Duplicate downloads and interrupted logs are excluded from the completed batch.

## Accuracy

Plateau = mean accuracy over rounds 131–150. Values are percentage points; spread is the sample standard deviation across seeds.

### FEMNIST main comparison

| Method | Seeds | Mean ± SD (%) |
|---|---:|---:|
| pure | 8 | 80.7212 ± 0.4193 |
| purerandom | 8 | 80.9328 ± 0.3974 |
| pureinvert | 8 | 81.0624 ± 0.3843 |
| matched | 8 | 81.0362 ± 0.3742 |
| uniform | 8 | 80.5769 ± 0.4491 |

### FEMNIST DAGC-inspired comparison

| Method | Seeds | Mean ± SD (%) |
|---|---:|---:|
| dagc | 3 | 81.0823 ± 0.3414 |
| dagcrev | 3 | 80.9562 ± 0.3700 |
| dagcrandom | 3 | 80.9553 ± 0.4176 |
| matched | 3 | 81.0033 ± 0.4050 |

### CIFAR-10 DAGC-inspired comparison

| Method | Seeds | Mean ± SD (%) |
|---|---:|---:|
| dagc | 3 | 55.1915 ± 0.5243 |
| dagcrev | 3 | 55.8795 ± 0.7147 |
| dagcrandom | 3 | 54.9167 ± 1.4541 |
| matched | 3 | 55.6567 ± 0.6014 |

## Paired comparisons

The existing six-test main families are retained. Each dataset has a separate five-test DAGC-inspired family. All inference is exploratory; confidence intervals below are unadjusted paired t-intervals. Permutation p-values are Holm-adjusted within each stated family, not across the entire study.

### FEMNIST 0.5%, EF on (eight seeds)

| A − B | n | Difference (pp) | 95% t CI | Holm t p | Holm permutation p |
|---|---:|---:|---|---:|---:|
| pure − purerandom | 8 | -0.2116 | [-0.2703, -0.1528] | 0.000267 | 0.046875 |
| pure − matched | 8 | -0.3150 | [-0.3937, -0.2363] | 0.000185 | 0.046875 |
| purerandom − matched | 8 | -0.1034 | [-0.1675, -0.0394] | 0.013135 | 0.046875 |
| pureinvert − pure | 8 | +0.3412 | [+0.2229, +0.4594] | 0.000744 | 0.046875 |
| pureinvert − matched | 8 | +0.0262 | [-0.0872, +0.1396] | 0.601882 | 0.585938 |
| matched − uniform | 8 | +0.4593 | [+0.3344, +0.5842] | 0.000267 | 0.046875 |

### FEMNIST 0.5%, DAGC-inspired allocation (seeds 42-44)

| A − B | n | Difference (pp) | 95% t CI | Holm t p | Holm permutation p |
|---|---:|---:|---|---:|---:|
| dagc − matched | 3 | +0.0790 | [-0.0920, +0.2500] | 0.434226 | 1.000000 |
| dagc − dagcrandom | 3 | +0.1270 | [-0.0663, +0.3203] | 0.422816 | 1.000000 |
| dagc − dagcrev | 3 | +0.1262 | [+0.0306, +0.2217] | 0.148022 | 1.000000 |
| dagcrandom − matched | 3 | -0.0480 | [-0.1680, +0.0720] | 0.434226 | 1.000000 |
| dagcrev − matched | 3 | -0.0472 | [-0.1341, +0.0398] | 0.434226 | 1.000000 |

### CIFAR10 0.5%, DAGC-inspired allocation (seeds 42-44)

| A − B | n | Difference (pp) | 95% t CI | Holm t p | Holm permutation p |
|---|---:|---:|---|---:|---:|
| dagc − matched | 3 | -0.4652 | [-1.3801, +0.4497] | 0.801025 | 1.000000 |
| dagc − dagcrandom | 3 | +0.2748 | [-2.1281, +2.6778] | 0.801025 | 1.000000 |
| dagc − dagcrev | 3 | -0.6880 | [-2.1904, +0.8144] | 0.801025 | 1.000000 |
| dagcrandom − matched | 3 | -0.7400 | [-3.0038, +1.5238] | 0.801025 | 1.000000 |
| dagcrev − matched | 3 | +0.2228 | [-0.3720, +0.8177] | 0.801025 | 1.000000 |

## Provenance and limits

- Original routing is worse than random tiers and flat allocation in all eight FEMNIST seeds. Inverted routing has no established advantage over flat allocation; failure to detect a difference is not an equivalence proof.
- Neither dataset establishes a DAGC-inspired advantage over flat allocation after correction. These are adapted allocation rules, not a complete reproduction of published DAGC.
- The two reconstructed CIFAR-10 logs preserve all 150 round accuracies and all 2,250 client summaries each. Exact byte counts and unprinted diagnostics remain null; rounded communication summaries are labelled as such. These files cannot resume training.
- Earlier comparison rows retain their existing provenance. Eight earlier-session values and three older console-derived values remain in the master table; the other historical JSON-designated rows have not all been reverified here.
- CIFAR-10 seed 43 remains unconverged by the existing 0.5 pp last-two-window criterion (DAGC +2.148, reversed +1.449, randomized +3.896 pp). A fixed-horizon result must not be described as a converged plateau for these runs.
- Matching is approximate for discrete-tier controls. DAGC-family upload budgets match closely, but this does not establish equal total computation or measured network performance.
- Preserve raw sources and this manifest. The results directory is currently ignored by Git; local consolidation is not an external backup.

## Reproduce

```bash
python scripts/consolidate_completed_runs.py --input-dir PATH_TO_DOWNLOADS
python scripts/paired_stats.py
```
