"""Recover analysis logs from complete console output without inventing missing data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROUND = re.compile(
    r'^\s*round\s+(\d+)/(\d+)\s*\| acc:\s*([\d.]+)\s*\| tiers:\s*(\d+)/(\d+)/(\d+)'
    r'\s*\| up:\s*([\d.]+)MB down:\s*([\d.]+)MB'
    r'\s*\(bidir save ([\d.-]+)%, cum ([\d.-]+)%\)'
    r'\s*\| tau:\s*\[([\d.eE+-]+),\s*([\d.eE+-]+)\]'
    r'\s*\| d:\s*\[([\d.eE+-]+),\s*([\d.eE+-]+)\]'
    r'\s*\| epochs:\s*(\d+)\s*\| local_lr=([\d.eE+-]+)'
)
CLIENT = re.compile(
    r'^\s*Client\s+(\d+)\s*\| Local Train Acc:\s*([\d.]+)'
    r'\s*\| Local Val Acc:\s*(N/A|[\d.]+)'
    r'\s*\| Agg Weight:\s*([\d.]+)\s*\| Tier:\s*(\d+)'
)


def reconstruct(path):
    path = Path(path)
    payload = path.read_bytes()
    text = payload.decode('utf-8-sig')
    run_ids = re.findall(r'^RUNNING:\s*(\S+)', text, re.M)
    if len(run_ids) != 1:
        raise ValueError('Expected exactly one run per console file')
    run_id = run_ids[0]
    name = re.fullmatch(r'(cifar10|femnist)_(.+)_r(\d+)_s(\d+)', run_id)
    if not name:
        raise ValueError(f'Unrecognized run identifier: {run_id}')
    dataset, _, target, seed = name.groups()
    target = int(target)
    setup = re.search(r'\[train\] (\d+) rounds, (\d+)/(\d+) clients/round', text)
    if not setup or int(setup[1]) != target:
        raise ValueError('Missing or conflicting training configuration')
    selected, population = int(setup[2]), int(setup[3])
    history = []
    for line in text.splitlines():
        match = ROUND.match(line)
        if match:
            values = match.groups()
            number, total = map(int, values[:2])
            if total != target:
                raise ValueError('Inconsistent target round count')
            history.append({
                'round': number - 1, 'test_accuracy': float(values[2]),
                'total_bytes_transmitted': None, 'total_upload_bytes': None,
                'total_download_bytes': None, 'delta_numel': None,
                'clients': [], 'upload_mb': float(values[6]),
                'download_mb': float(values[7]),
                'communication_savings': float(values[8]),
                'console_metrics': {
                    'tier_counts': [int(v) for v in values[3:6]],
                    'cumulative_saving_pct': float(values[9]),
                    'tau_low': float(values[10]), 'tau_high': float(values[11]),
                    'divergence_min': float(values[12]), 'divergence_max': float(values[13]),
                    'local_epochs': int(values[14]), 'local_lr': float(values[15]),
                },
            })
            continue
        match = CLIENT.match(line)
        if match:
            if not history:
                raise ValueError('Client metrics precede the first round')
            cid, acc, val, weight, tier = match.groups()
            history[-1]['clients'].append({
                'client_id': int(cid), 'divergence_score': None, 'tier': int(tier),
                'upload_bytes': None, 'download_bytes': None, 'bytes_received': None,
                'aggregation_weight': float(weight), 'local_train_acc': float(acc),
                'local_val_acc': None if val == 'N/A' else float(val), 'grad_diag': None,
            })
    if [r['round'] for r in history] != list(range(target)):
        raise ValueError('Rounds are missing, duplicated, or out of order')
    for r in history:
        clients = r['clients']
        if len(clients) != selected or len({c['client_id'] for c in clients}) != selected:
            raise ValueError(f'Incomplete client rows in round {r["round"] + 1}')
        counts = [sum(c['tier'] == tier for c in clients) for tier in (1, 2, 3)]
        if counts != r['console_metrics']['tier_counts']:
            raise ValueError('Client tiers disagree with the round summary')
    if f'[done] log written to logs/{run_id}.json' not in text:
        raise ValueError('Missing successful completion marker')
    summary = {}
    for field, pattern in {
        'upload_mb': r'\[summary\] upload \(C->S\)\s*:\s*([\d.]+) MB',
        'download_mb': r'\[summary\] download \(S->C\):\s*([\d.]+) MB',
        'bidirectional_mb': r'\[summary\] bidirectional\s*:\s*([\d.]+) MB',
        'final_accuracy': r'\[summary\] final acc\s*:\s*([\d.]+)',
        'walltime_minutes': r'wall-clock:\s*([\d.]+) min',
    }.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f'Missing final summary field: {field}')
        summary[field] = float(match[1])
    if summary['final_accuracy'] != history[-1]['test_accuracy']:
        raise ValueError('Final accuracy disagrees with round history')
    partition = re.search(r'partition: (\w+) \(alpha=([\d.]+)\)', text)
    return {
        'metadata': {
            'dataset': dataset, 'model': None, 'seed': int(seed),
            'num_clients': population, 'clients_per_round': selected,
            'local_epochs': max(r['console_metrics']['local_epochs'] for r in history),
            'partition_mode': partition[1] if partition else None,
            'alpha': float(partition[2]) if partition else None,
            'run_label': run_id, 'source_type': 'console_reconstruction',
        },
        'provenance': {
            'source_file': path.name, 'source_sha256': hashlib.sha256(payload).hexdigest(),
            'original_logger_json': False,
            'limitations': [
                'Not a byte-for-byte recovery of the original JSON and not a training checkpoint.',
                'Accuracy uses the same four-decimal fraction precision printed by the logger.',
                'Client metrics are rounded console values; full-precision weights are unavailable.',
                'Exact byte counts, per-client divergence scores, and unprinted diagnostics are null.',
                'Per-round MB values have three decimals; final summary MB values have two decimals.',
                'Model identity and unprinted configuration fields are not inferred from defaults.',
            ],
        },
        'console_summary': summary, 'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results/reconstructed_logs'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        raw = reconstruct(path)
        output = args.output_dir / (raw['metadata']['run_label'] + '.json')
        serialized = json.dumps(raw, indent=2, allow_nan=False) + '\n'
        if output.exists() and output.read_text(encoding='utf-8') != serialized:
            raise FileExistsError(f'Refusing to overwrite different content: {output}')
        output.write_text(serialized, encoding='utf-8')
        print(f'{output}: {len(raw["history"])} complete rounds')


if __name__ == '__main__':
    main()
