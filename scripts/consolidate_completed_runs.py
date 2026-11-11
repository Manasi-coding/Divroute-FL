"""Consolidate the audited November experiment batch and regenerate statistics."""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

import paired_stats as ps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def identity(filename):
    name = re.sub(r' \(\d+\)(?=\.json$)', '', filename)
    match = re.fullmatch(r'(femnist|cifar10)_(.+?)(?:_lr0\.03_ep5)?_ks0\.1_r150_s(\d+)\.json', name)
    if not match:
        raise ValueError(filename)
    dataset, variant, seed = match.groups()
    return (dataset, '0.5%', variant, int(seed)), name


def validate(raw):
    history = raw['history']
    if [r['round'] for r in history] != list(range(150)):
        raise ValueError('Expected all 150 rounds in order')
    for r in history:
        if not 0 <= r['test_accuracy'] <= 1:
            raise ValueError('Invalid accuracy')
        clients = r['clients']
        if len(clients) != 15 or len({c['client_id'] for c in clients}) != 15:
            raise ValueError('Missing or repeated client rows')
        if raw['metadata'].get('source_type') != 'console_reconstruction':
            for direction in ['upload', 'download']:
                if r[f'total_{direction}_bytes'] != sum(c[f'{direction}_bytes'] or 0 for c in clients):
                    raise ValueError('Inconsistent byte accounting')
    return [100 * r['test_accuracy'] for r in history]


def paired_comparisons(data):
    families = []
    for dataset, setting, pairs, label in ps.FAMILIES:
        rows = []
        for a, b in pairs:
            seeds = sorted({k[3] for k in data if k[:3] == (dataset, setting, a)} &
                           {k[3] for k in data if k[:3] == (dataset, setting, b)})
            if len(seeds) < 3:
                continue
            diff = np.array([data[(dataset, setting, a, s)][0] - data[(dataset, setting, b, s)][0] for s in seeds])
            ci = stats.t.interval(.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
            rows.append(dict(a=a, b=b, seeds=seeds, difference_pp=float(diff.mean()),
                             ci95_pp=[float(x) for x in ci], positive_seeds=int((diff > 0).sum()),
                             t_p=float(stats.ttest_1samp(diff, 0).pvalue), permutation_p=ps.perm_p(diff),
                             non_original_json_values=sum(not data[(dataset, setting, v, s)][1].startswith('JSON')
                                                          for v in [a, b] for s in seeds)))
        for r, t, p in zip(rows, ps.holm([r['t_p'] for r in rows]), ps.holm([r['permutation_p'] for r in rows])):
            r.update(t_holm=float(t), permutation_holm=float(p))
        families.append(dict(label=label, dataset=dataset, setting=setting, comparisons=rows))
    return families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    sources = []
    for name in ['new_run_audit_2026-11-09.json', 'new_run_audit_2026-11-10.json']:
        audit = json.loads((RESULTS / name).read_text())
        sources.extend(r for r in (audit['runs'] if isinstance(audit, dict) else audit) if r['complete'])
    selected = {}
    logdir = RESULTS / 'run_logs'
    logdir.mkdir(exist_ok=True)
    for entry in sources:
        key, canonical = identity(entry['file'])
        source = args.input_dir / entry['file']
        if not source.exists():
            source = logdir / canonical
        raw = json.loads(source.read_text(encoding='utf-8'))
        validate(raw)
        if raw['metadata']['dataset'] != key[0] or raw['metadata']['seed'] != key[3]:
            raise ValueError('Filename and metadata disagree')
        if key in selected and selected[key][0] != raw:
            raise ValueError(f'Conflicting complete histories: {key}')
        selected[key] = (raw, source, logdir / canonical)
    for source in sorted((RESULTS / 'reconstructed_logs').glob('*.json')):
        key, _ = identity(source.name)
        raw = json.loads(source.read_text()); validate(raw)
        if key in selected:
            raise ValueError(f'Original and reconstructed logs conflict: {key}')
        selected[key] = (raw, source, source)
    expected = {(d, '0.5%', v, s) for d in ['femnist', 'cifar10']
                for v in ['dagc', 'dagcrev', 'dagcrandom'] for s in [42, 43, 44]}
    expected |= {('femnist', '0.5%', v, s) for v in ['pure', 'purerandom', 'matched', 'pureinvert', 'uniform'] for s in [48, 49]}
    if set(selected) != expected:
        raise ValueError(f'Batch mismatch: missing={expected-set(selected)}, extra={set(selected)-expected}')
    table = RESULTS / 'plateaus.csv'
    archive = RESULTS / 'archive'
    archive.mkdir(exist_ok=True)
    backup = archive / 'plateaus_before_2026-11-11.csv'
    if not backup.exists():
        shutil.copy2(table, backup)
    with table.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; old_rows = list(reader)
    rows = {}
    for r in old_rows:
        key = (r['dataset'], r['setting'], r['variant'], int(r['seed']))
        if key in rows:
            raise ValueError(f'Duplicate CSV row: {key}')
        rows[key] = r
    manifest = []
    for key, (raw, source, dest) in sorted(selected.items()):
        if source.resolve() != dest.resolve():
            if dest.exists() and json.loads(dest.read_text()) != raw:
                raise ValueError(f'Refusing to overwrite: {dest}')
            shutil.copy2(source, dest)
        acc = validate(raw)
        reconstructed = raw['metadata'].get('source_type') == 'console_reconstruction'
        origin = 'console reconstruction' if reconstructed else 'JSON'
        relative = dest.relative_to(ROOT).as_posix()
        rows[key] = dict(dataset=key[0], setting=key[1], variant=key[2], seed=str(key[3]),
                         plateau_last20=f'{statistics.mean(acc[-20:]):.6f}', rounds='150',
                         last_window_drift=f'{statistics.mean(acc[-10:])-statistics.mean(acc[-20:-10]):.6f}',
                         source=origin, file=relative,
                         note='Recovered accuracy from complete console; exact bytes unavailable.' if reconstructed else 'Validated original JSON; duplicate downloads excluded.')
        manifest.append(dict(dataset=key[0], variant=key[2], seed=key[3], file=relative,
                             source=origin, sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    with table.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows.values())
    (RESULTS / 'completed_batch_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    console = RESULTS / 'source_console'; console.mkdir(exist_ok=True)
    for name in ['11th_1.txt', '11th_2.txt']:
        source = args.input_dir / name
        if source.exists() and source.resolve() != (console / name).resolve():
            shutil.copy2(source, console / name)
    data = ps.load()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for ds, setting, pairs, label in ps.FAMILIES:
            ps.family(data, ds, setting, pairs, label)
    (RESULTS / 'paired_statistics.txt').write_text(output.getvalue(), encoding='utf-8')
    families = paired_comparisons(data)
    (RESULTS / 'paired_statistics.json').write_text(json.dumps(families, indent=2, allow_nan=False)+'\n')
    report = ['# Consolidated experiment results — 2026-11-11', '',
              'All 28 planned runs are complete: 26 original JSON logs and two console reconstructions.',
              'The master table contains '+str(len(rows))+' distinct dataset/setting/variant/seed rows. Duplicate downloads and interrupted logs are excluded from the completed batch.', '',
              '## Accuracy', '', 'Plateau = mean accuracy over rounds 131–150. Values are percentage points; spread is the sample standard deviation across seeds.', '']
    for label, dataset, variants, seeds in [
        ('FEMNIST main comparison','femnist',['pure','purerandom','pureinvert','matched','uniform'],list(range(42,50))),
        ('FEMNIST DAGC-inspired comparison','femnist',['dagc','dagcrev','dagcrandom','matched'],[42,43,44]),
        ('CIFAR-10 DAGC-inspired comparison','cifar10',['dagc','dagcrev','dagcrandom','matched'],[42,43,44])]:
        report += ['### '+label, '', '| Method | Seeds | Mean ± SD (%) |', '|---|---:|---:|']
        for v in variants:
            vals = [data[(dataset,'0.5%',v,s)][0] for s in seeds]
            report.append(f'| {v} | {len(vals)} | {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} |')
        report.append('')
    report += ['## Paired comparisons', '', 'The existing six-test main families are retained. Each dataset has a separate five-test DAGC-inspired family. All inference is exploratory; confidence intervals below are unadjusted paired t-intervals. Permutation p-values are Holm-adjusted within each stated family, not across the entire study.', '']
    for family in [families[0], families[-2], families[-1]]:
        report += ['### '+family['label'], '', '| A − B | n | Difference (pp) | 95% t CI | Holm t p | Holm permutation p |', '|---|---:|---:|---|---:|---:|']
        for r in family['comparisons']:
            lo,hi=r['ci95_pp']
            report.append(f"| {r['a']} − {r['b']} | {len(r['seeds'])} | {r['difference_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r['t_holm']:.6f} | {r['permutation_holm']:.6f} |")
        report.append('')
    report += ['## Provenance and limits', '',
        '- Original routing is worse than random tiers and flat allocation in all eight FEMNIST seeds. Inverted routing has no established advantage over flat allocation; failure to detect a difference is not an equivalence proof.',
        '- Neither dataset establishes a DAGC-inspired advantage over flat allocation after correction. These are adapted allocation rules, not a complete reproduction of published DAGC.',
        '- The two reconstructed CIFAR-10 logs preserve all 150 round accuracies and all 2,250 client summaries each. Exact byte counts and unprinted diagnostics remain null; rounded communication summaries are labelled as such. These files cannot resume training.',
        '- Earlier comparison rows retain their existing provenance. Eight earlier-session values and three older console-derived values remain in the master table; the other historical JSON-designated rows have not all been reverified here.',
        '- CIFAR-10 seed 43 remains unconverged by the existing 0.5 pp last-two-window criterion (DAGC +2.148, reversed +1.449, randomized +3.896 pp). A fixed-horizon result must not be described as a converged plateau for these runs.',
        '- Matching is approximate for discrete-tier controls. DAGC-family upload budgets match closely, but this does not establish equal total computation or measured network performance.',
        '- Preserve raw sources and this manifest. The results directory is currently ignored by Git; local consolidation is not an external backup.', '',
        '## Reproduce', '', '```bash', 'python scripts/consolidate_completed_runs.py --input-dir PATH_TO_DOWNLOADS', 'python scripts/paired_stats.py', '```', '']
    (RESULTS / 'CONSOLIDATED_RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'Consolidated {len(selected)} complete runs; master table: {len(rows)} rows.')
    print(output.getvalue())


if __name__ == '__main__':
    main()
