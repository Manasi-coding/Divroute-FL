"""Small CPU regressions; no dataset or pretrained-weight downloads."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from divroute_fl import client, main
from divroute_fl.compression import (
    apply_tiered_compression,
    compress_delta,
    reconstruct_delta,
)
from divroute_fl.config import Config
from divroute_fl.mechanism import dagc_k_ratios


def small_model(name, num_classes, bn_mode="default"):
    return nn.Sequential(
        nn.Flatten(), nn.Linear(48, 12), nn.ReLU(), nn.Linear(12, num_classes)
    )


class CompressionTests(unittest.TestCase):
    def test_dense_and_sparse_payloads(self):
        delta = torch.tensor([1.0, -4.0, 2.0, -3.0])
        for fp16, dense_bytes, sparse_bytes in [(False, 16, 16), (True, 8, 12)]:
            with self.subTest(fp16=fp16):
                dense = compress_delta(delta, 1.0, fp16)
                sparse = compress_delta(delta, 0.5, fp16)
                self.assertEqual(dense["bytes_transmitted"], dense_bytes)
                self.assertEqual(sparse["bytes_transmitted"], sparse_bytes)
                torch.testing.assert_close(
                    reconstruct_delta(dense), delta, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    reconstruct_delta(sparse),
                    torch.tensor([0.0, -4.0, 0.0, -3.0]),
                    rtol=0,
                    atol=0,
                )

    def test_feedback_reuses_buffer_and_resets_on_tier_change(self):
        cfg = Config(
            use_error_feedback=True,
            error_feedback_tier_aware=True,
            include_tier3_in_aggregation=True,
            k_ratio_tier1=0.5,
            k_ratio_tier2=0.25,
        )
        delta = torch.tensor([1.0, -4.0, 2.0, -3.0])
        buffers, tiers = {0: torch.ones_like(delta)}, {0: 1}
        with patch(
            "divroute_fl.compression.torch.zeros_like", wraps=torch.zeros_like
        ) as zeros:
            payload = apply_tiered_compression(
                delta, 1, cfg, buffers, 0, error_buffer_tiers=tiers
            )
            self.assertEqual(zeros.call_count, 0)
        expected = delta + cfg.error_feedback_momentum * torch.ones_like(delta)
        torch.testing.assert_close(
            buffers[0], expected - reconstruct_delta(payload), rtol=0, atol=0
        )
        payload = apply_tiered_compression(
            delta, 3, cfg, buffers, 0, error_buffer_tiers=tiers
        )
        torch.testing.assert_close(
            buffers[0], delta - reconstruct_delta(payload), rtol=0, atol=0
        )
        self.assertEqual(tiers[0], 3)
        cfg.include_tier3_in_aggregation = False
        payload = apply_tiered_compression(
            delta, 3, cfg, buffers, 0, error_buffer_tiers=tiers
        )
        self.assertEqual(payload["bytes_transmitted"], 0)
        self.assertTrue(torch.equal(buffers[0], torch.zeros_like(delta)))

    def test_dagc_budget_and_direction(self):
        for sizes in ([1, 2, 8], [16, 600, 6144]):
            for mean in (0.0105, 0.5, 1.0):
                for direction in (1, -1):
                    ratios = dagc_k_ratios(sizes, mean, direction=direction)
                    self.assertAlmostEqual(sum(ratios), len(sizes) * mean, places=10)
                    self.assertTrue(all(0 <= k <= 1 for k in ratios))
                    self.assertEqual(ratios, sorted(ratios, reverse=direction < 0))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(991)
        cls.data = [
            TensorDataset(
                torch.randn(n, 3, 4, 4, generator=g),
                torch.randint(0, 10, (n,), generator=g),
            )
            for n in (8, 10, 12, 14)
        ]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_training(self, directory, options, rounds=2, resume=False):
        previous_dir = Path.cwd()
        captured = []
        real_server = main.FLServer

        class TrackingServer(real_server):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        cfg = Config(
            num_clients=4,
            clients_per_round=3,
            num_rounds=rounds,
            local_epochs=1,
            batch_size=4,
            local_lr=0.03,
            seed=42,
            model_name="simplecnn",
            dataset_name="cifar10",
            log_path="logs/run.json",
            skip_plot_prompt=True,
            fresh=not resume,
            include_tier3_in_aggregation=True,
            use_k_warmup=False,
            **options,
        )
        try:
            os.chdir(directory)
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(torch.cuda, "is_available", return_value=False)
                )
                stack.enter_context(patch.object(main, "get_model", small_model))
                stack.enter_context(patch.object(client, "get_model", small_model))
                stack.enter_context(
                    patch.object(main, "get_client_datasets", return_value=self.data)
                )
                stack.enter_context(
                    patch.object(main, "get_test_dataset", return_value=self.data[0])
                )
                stack.enter_context(patch.object(main, "FLServer", TrackingServer))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main.run(cfg)
            return (
                json.loads(Path(cfg.log_path).read_text()),
                copy.deepcopy(captured[-1].global_model.state_dict()),
                torch.get_rng_state(),
            )
        finally:
            os.chdir(previous_dir)

    def assert_same_run(self, first, second):
        self.assertEqual(first[0], second[0])
        for key in first[1]:
            torch.testing.assert_close(first[1][key], second[1][key], rtol=0, atol=0)
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_deterministic_training_variants(self):
        variants = [
            dict(fedavg_baseline_mode=True),
            dict(uniform_top5_mode=True),
            dict(use_error_feedback=True),
            dict(random_tier_assignment=True),
            dict(invert_tier_polarity=True),
            dict(dagc_mean_k=0.0105),
            dict(dagc_mean_k=0.0105, dagc_direction=-1),
            dict(dagc_mean_k=0.0105, dagc_shuffle=True),
            dict(use_fp16_upload=True, use_fp16_download=True, use_layerwise_topk=True),
            dict(ntd_beta=0.1, use_fedprox=True),
            dict(fedsparse_lambda=0.01, fedsparse_sparsify_upload=True),
        ]
        for options in variants:
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as a,
                tempfile.TemporaryDirectory() as b,
            ):
                self.assert_same_run(
                    self.run_training(a, options), self.run_training(b, options)
                )

    def test_uniform_checkpoint_resume_matches_full_run(self):
        options = dict(uniform_top5_mode=True, use_error_feedback=True)
        with tempfile.TemporaryDirectory() as directory:
            complete = self.run_training(directory, options, rounds=6)
            resumed = self.run_training(directory, options, rounds=6, resume=True)
            self.assert_same_run(complete, resumed)


if __name__ == "__main__":
    unittest.main()
