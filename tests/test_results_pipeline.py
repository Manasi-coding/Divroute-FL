"""Console reconstruction preserves reported precision and rejects incomplete runs."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.reconstruct_console_logs import reconstruct
from scripts.summarize_runs import load_run


def console_text():
    label = 'cifar10_dagcrev_lr0.03_ep5_ks0.1_r2_s44'
    lines = [f'RUNNING: {label}', '[train] 2 rounds, 2/25 clients/round',
             '[train] partition: dirichlet (alpha=0.1)']
    for number, accuracy in [(1, .50), (2, .55)]:
        lines += [
            f'  round {number}/2 | acc: {accuracy:.4f} | tiers: 1/1/0 | up: 0.338MB down: 16.119MB (bidir save 49.0%, cum 49.0%) | tau: [0.001, 0.002] | d: [0.0005, 0.003] | epochs: 5 | local_lr=0.003000',
            '    Client 0 | Local Train Acc: 0.9500 | Local Val Acc: N/A | Agg Weight: 0.4000 | Tier: 1',
            '    Client 1 | Local Train Acc: 0.9600 | Local Val Acc: N/A | Agg Weight: 0.6000 | Tier: 2',
        ]
    lines += [f'[done] log written to logs/{label}.json', '[summary] final acc : 0.5500',
              '[summary] upload (C->S) : 0.68 MB', '[summary] download (S->C): 32.24 MB',
              '[summary] bidirectional : 32.91 MB', f'[{label}] wall-clock: 1.0 min']
    return '\n'.join(lines)


class ReconstructionTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'console.txt'
            source.write_text(text, encoding='utf-8')
            return reconstruct(source)

    def test_recovery_preserves_known_and_unknown_fields(self):
        raw = self.parse(console_text())
        self.assertEqual([r['round'] for r in raw['history']], [0, 1])
        self.assertEqual([r['test_accuracy'] for r in raw['history']], [.5, .55])
        self.assertFalse(raw['provenance']['original_logger_json'])
        self.assertIsNone(raw['metadata']['model'])
        for row in raw['history']:
            self.assertIsNone(row['total_upload_bytes'])
            self.assertIsNone(row['clients'][0]['divergence_score'])
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'reconstructed.json'
            file.write_text(json.dumps(raw), encoding='utf-8')
            summary = load_run(file)
        self.assertEqual(summary['up'], .68)
        self.assertEqual(summary['down'], 32.24)
        self.assertTrue(summary['reconstructed'])

    def test_missing_round_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(console_text().replace('round 2/2', 'round 3/2'))

    def test_missing_client_is_rejected(self):
        lines = console_text().splitlines()
        index = next(i for i, line in enumerate(lines) if 'Client 1' in line)
        del lines[index]
        with self.assertRaises(ValueError):
            self.parse('\n'.join(lines))

    def test_conflicting_final_accuracy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(console_text().replace('final acc : 0.5500', 'final acc : 0.5600'))


if __name__ == '__main__':
    unittest.main()
