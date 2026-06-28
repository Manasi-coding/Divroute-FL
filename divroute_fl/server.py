from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .mechanism import (assign_tier, update_selection_weights,
                         compute_divergence_weight, compute_softmax_weights)
from .compression import apply_tiered_compression, reconstruct_delta


class FLServer:
    def __init__(self, global_model: nn.Module, config: Config, device: torch.device):
        self.global_model = global_model.to(device)
        self.config = config
        self.device = device
        self.selection_weights = np.ones(config.num_clients, dtype=np.float64)
        self.global_delta: torch.Tensor | None = None
        self._rng = np.random.default_rng(config.seed)
        self._momentum_buf: torch.Tensor | None = None

    def select_clients(self, all_ids: List[int]) -> List[int]:
        weights = self.selection_weights[all_ids].copy()
        prob = weights / weights.sum()
        return self._rng.choice(all_ids, size=self.config.clients_per_round,
                                 replace=False, p=prob).tolist()

    def aggregate(self, client_results: List[dict], error_buffers: dict) -> None:
        old_flat = self._flatten_params(self.global_model.state_dict())

        # NaN guard
        clean = [r for r in client_results
                 if not any(torch.isnan(v).any() or torch.isinf(v).any()
                             for v in r["state_dict"].values())]
        dropped = len(client_results) - len(clean)
        if dropped > 0:
            print(f"  [warn] dropped {dropped} client(s) with NaN/Inf updates")
        if not clean:
            print("  [warn] all clients NaN/Inf — skipping aggregation")
            self.global_delta = torch.zeros_like(old_flat)
            return

        # per-client deltas: clip -> compress -> reconstruct
        # Tier 3 clients are excluded from client_flats — contributing zero delta
        # would dilute aggregation weights without providing any gradient signal
        client_flats = []
        for r in clean:
            cf = self._flatten_params(r["state_dict"])
            raw_delta = cf - old_flat

            norm = torch.norm(raw_delta)
            if norm > self.config.grad_clip_norm:
                raw_delta = raw_delta * (self.config.grad_clip_norm / norm)

            # compress client's upload delta; sets bytes and clears buffer for Tier 3
            payload = apply_tiered_compression(
                raw_delta, r["tier"], self.config, error_buffers, r["client_id"])
            r["bytes_received"] = payload["bytes_transmitted"]
            # Upload: client always sends its full local model back (float32 state_dict).
            # This is independent of which tier/compression the server used for download.
            r["upload_bytes"] = r["num_samples"] and sum(
                p.numel() * 4 for p in self.global_model.parameters()
            )

            if payload["values"] is None:      # Tier 3 — skip aggregation
                continue

            compressed_delta = reconstruct_delta(payload).to(self.device)
            client_flats.append((r, compressed_delta))

        # all selected clients may be Tier 3 in late training
        if not client_flats:
            print("  [warn] all clients Tier 3 — skipping aggregation")
            self.global_delta = torch.zeros_like(old_flat)
            return

        # compute aggregation weights over Tier 1/2 clients only
        total_samples = sum(r["num_samples"] for r, _ in client_flats)

        use_softmax = (
            self.config.use_divergence_weighting
            and self.config.divergence_weight_mode == "softmax"
            and all(r.get("divergence_score") is not None for r, _ in client_flats)
        )

        if use_softmax:
            d_scores = [r["divergence_score"] for r, _ in client_flats]
            div_weights = compute_softmax_weights(d_scores)
            weights = [dw * (r["num_samples"] / total_samples)
                       for dw, (r, _) in zip(div_weights, client_flats)]
        else:
            weights = []
            for r, _ in client_flats:
                sample_w = r["num_samples"] / total_samples
                if self.config.use_divergence_weighting and r.get("divergence_score") is not None:
                    div_w = compute_divergence_weight(
                        r["divergence_score"], self.config.divergence_weight_mode)
                    weights.append(sample_w * div_w)
                else:
                    weights.append(sample_w)

        # normalise
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        agg_delta = sum(w * d for w, (_, d) in zip(weights, client_flats))

        # server momentum
        if self.config.use_server_momentum:
            if self._momentum_buf is None:
                self._momentum_buf = torch.zeros_like(agg_delta)
            self._momentum_buf = (self.config.server_momentum * self._momentum_buf
                                   + agg_delta)
            agg_delta = self.config.server_lr * self._momentum_buf

        new_flat = old_flat + agg_delta
        
        # Reconstruct full state_dict: parameters get the clipped/compressed update,
        # buffers get a simple sample-weighted average.
        param_names = {n for n, _ in self.global_model.named_parameters()}
        new_sd = OrderedDict()
        
        offset = 0
        for k, v in self.global_model.state_dict().items():
            if k in param_names:
                numel = v.numel()
                new_sd[k] = new_flat[offset:offset + numel].reshape(v.shape).to(v.dtype)
                offset += numel
            else:
                # BN Buffers
                if client_flats:
                    buf_sum = sum((r["num_samples"] / total_samples) * r["state_dict"][k].to(self.device) 
                                  for r, _ in client_flats)
                    new_sd[k] = buf_sum
                else:
                    new_sd[k] = v
                    
        self.global_model.load_state_dict(new_sd)
        self.global_delta = agg_delta

    def evaluate(self, test_loader: DataLoader) -> float:
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.global_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def assign_tier(self, d: float) -> int:
        return assign_tier(d, self.config.tau_low, self.config.tau_high)

    def update_selection_weights(self, results: list) -> None:
        update_selection_weights(self.selection_weights, results, self.config.gamma)

    def tier3_heartbeat_payload(self) -> dict:
        """
        Phase 1.2 (Tier-3 staleness fix, Option A — minimal sync heartbeat).

        Tier-3 clients normally receive 0 bytes and train on an increasingly
        stale global model. Every `tier3_sync_interval` rounds, instead of
        sending nothing, the server sends the current global delta compressed
        at the Tier-2 ratio. This keeps Tier-3 clients roughly in sync with
        the global model while preserving most of the communication savings
        (the heartbeat costs the same as a Tier-2 update, not a full delta).

        Returns a compression payload (same shape as `apply_tiered_compression`'s
        Tier-2 output) describing what the client receives. The caller records
        `bytes_transmitted` against that client's `bytes_received` for this
        round.

        Note: this heartbeat compresses the *current global delta* (the update
        the server just produced this round), not a client-specific upload
        delta — it represents what the server pushes DOWN to a stale Tier-3
        client to refresh its local copy of the global model. It does not use
        or modify any client's upload error buffer.
        """
        if self.global_delta is None:
            return {"indices": None, "values": None, "bytes_transmitted": 0,
                    "total_params": 0}

        payload = apply_tiered_compression(
            self.global_delta, tier=2, config=self.config,
            error_buffers=None, client_id=None)
        return payload

    def _flatten(self, state_dict: OrderedDict) -> torch.Tensor:
        return torch.cat([v.flatten().float().to(self.device) for v in state_dict.values()])

    def _flatten_params(self, state_dict: OrderedDict) -> torch.Tensor:
        param_names = {n for n, _ in self.global_model.named_parameters()}
        return torch.cat([v.flatten().float().to(self.device) 
                          for k, v in state_dict.items() if k in param_names])

    def _load_flat(self, flat: torch.Tensor) -> None:
        sd = self.global_model.state_dict()
        offset = 0
        new_sd = OrderedDict()
        for k, v in sd.items():
            numel = v.numel()
            new_sd[k] = flat[offset:offset + numel].reshape(v.shape).to(v.dtype)
            offset += numel
        self.global_model.load_state_dict(new_sd)