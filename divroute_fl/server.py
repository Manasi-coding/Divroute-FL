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
import divroute_fl.client as _fl_client  # read DEBUG_BN live at call-time


class FLServer:
    def __init__(self, global_model: nn.Module, config: Config, device: torch.device):
        self.global_model = global_model.to(device)
        self.config = config
        self.device = device
        self.selection_weights = np.ones(config.num_clients, dtype=np.float64)
        self.global_delta: torch.Tensor | None = None
        self._rng = np.random.default_rng(config.seed)
        self._momentum_buf: torch.Tensor | None = None
        self._layer_slices = None
        self._layer_importance = None

    def _get_layer_slices(self) -> list:
        if self._layer_slices is not None:
            return self._layer_slices
        param_names = {n for n, _ in self.global_model.named_parameters()}
        slices = []
        offset = 0
        for k, v in self.global_model.state_dict().items():
            if k in param_names:
                numel = v.numel()
                slices.append((k, offset, numel))
                offset += numel
        self._layer_slices = slices
        return slices

    def select_clients(self, all_ids: List[int]) -> List[int]:
        weights = self.selection_weights[all_ids].copy()
        prob = weights / weights.sum()
        return self._rng.choice(all_ids, size=self.config.clients_per_round,
                                 replace=False, p=prob).tolist()

    # ── DEBUG: rounds at which client-0 delta statistics are logged ──────────
    _DEBUG_ROUNDS = frozenset({1, 20, 40, 60, 80, 100})
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate(self, client_results: List[dict], error_buffers: dict,
                  round_num: int | None = None) -> None:
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

        # -- Determine which clients will participate in aggregation --
        agg_clients = []
        for r in clean:
            if getattr(self.config, "fedzip_actual_mode", False):
                agg_clients.append(r)
            elif self.config.fedsparse_sparsify_upload:
                agg_clients.append(r)
            else:
                if r["tier"] != 3 or getattr(self.config, "include_tier3_in_aggregation", False):
                    agg_clients.append(r)

        if not agg_clients:
            print("  [warn] all clients Tier 3 — skipping aggregation")
            self.global_delta = torch.zeros_like(old_flat)
            return

        # -- Compute aggregation weights ahead of time --
        total_samples = sum(r["num_samples"] for r in agg_clients)

        use_softmax = (
            self.config.use_divergence_weighting
            and self.config.divergence_weight_mode == "softmax"
            and all(r.get("divergence_score") is not None for r in agg_clients)
        )

        if use_softmax:
            d_scores = [r["divergence_score"] for r in agg_clients]
            div_weights = compute_softmax_weights(d_scores)
            weights = [dw * (r["num_samples"] / total_samples)
                       for dw, r in zip(div_weights, agg_clients)]
        else:
            weights = []
            for r in agg_clients:
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
        
        for r in client_results:
            r["aggregation_weight"] = 0.0

        for r, w in zip(agg_clients, weights):
            r["aggregation_weight"] = w

        # -- Streaming Weighted Accumulation --
        agg_delta = torch.zeros_like(old_flat)
        active_mask = torch.zeros_like(old_flat, dtype=torch.bool)

        for r, w in zip(agg_clients, weights):
            cf = self._flatten_params(r["state_dict"])
            raw_delta = cf - old_flat

            # ── DEBUG: log upload delta statistics for client 0 ───────────────
            if (r["client_id"] == 0
                    and round_num is not None
                    and round_num in self._DEBUG_ROUNDS):
                with torch.no_grad():
                    _d = raw_delta.float().cpu()
                    _n = _d.numel()
                    _l2  = float(_d.norm(p=2))
                    _l1  = float(_d.norm(p=1))
                    _nnz = int((_d.abs() > 1e-12).sum())
                    _lr_lam = self.config.local_lr * self.config.fedsparse_lambda
                    _gt_eps   = int((_d.abs() > 1e-4).sum())
                    _gt_1e12  = int((_d.abs() > 1e-12).sum())
                    _gt_lrlam = int((_d.abs() > _lr_lam).sum()) if _lr_lam > 0 else -1
                    print(
                        f"  [DEBUG-delta] round={round_num} client=0 "
                        f"n={_n} "
                        f"L2={_l2:.6e} L1={_l1:.6e} "
                        f"nnz={_nnz} ({100*_nnz/_n:.2f}%) "
                        f">1e-12={_gt_1e12} "
                        f">1e-4={_gt_eps} "
                        f">lr*lam({_lr_lam:.2e})="
                        f"{_gt_lrlam if _gt_lrlam >= 0 else 'N/A (lam=0)'}"
                    )
            # ─────────────────────────────────────────────────────────────────

            norm = torch.norm(raw_delta)
            if getattr(self.config, "server_clip_updates", False) and norm > self.config.grad_clip_norm:
                raw_delta = raw_delta * (self.config.grad_clip_norm / norm)

            # FedZip actual mode (Phase 5)
            if getattr(self.config, "fedzip_actual_mode", False):
                from baselines.fedzip_actual import fedzip_compress_delta
                compressed_delta, bytes_received = fedzip_compress_delta(
                    raw_delta,
                    z_ratio=self.config.fedzip_z_ratio,
                    k_clusters=self.config.fedzip_k_clusters,
                    random_state=self.config.seed
                )
                r["bytes_received"] = bytes_received
                # upload = compressed bytes actually sent by client
                r["upload_bytes"] = bytes_received
                # download = full global model broadcast by server to this client
                r["download_bytes"] = sum(
                    p.numel() * (2 if getattr(self.config, "use_fp16_download", False) else 4) for p in self.global_model.parameters()
                )
                compressed_delta = compressed_delta.to(self.device)
            else:
                if self.config.fedsparse_sparsify_upload:
                    mask = raw_delta.abs() > 1e-12
                    sparse_delta = raw_delta * mask
                    nnz = int(mask.sum().item())
                    bytes_received = nnz * 8
                    r["bytes_received"] = bytes_received
                    r["upload_bytes"] = bytes_received   # compressed sparse upload
                    r["download_bytes"] = sum(
                        p.numel() * 4 for p in self.global_model.parameters()
                    )
                    payload = {"values": sparse_delta, "indices": None, "bytes_transmitted": bytes_received}
                else:
                    layer_slices = None
                    if getattr(self.config, "use_layerwise_topk", False):
                        layer_slices = self._get_layer_slices()
                            
                    payload = apply_tiered_compression(
                        raw_delta, r["tier"], self.config, error_buffers, r["client_id"],
                        layer_slices=layer_slices, layer_importances=self._layer_importance)
                    r["bytes_received"] = payload["bytes_transmitted"]
                    # upload = compressed top-k bytes actually sent by client
                    r["upload_bytes"] = payload["bytes_transmitted"]
                    # download = full global model broadcast by server to this client
                    r["download_bytes"] = sum(
                        p.numel() * (2 if getattr(self.config, "use_fp16_download", False) else 4) for p in self.global_model.parameters()
                    )
                
                compressed_delta = reconstruct_delta(payload).to(self.device)

            agg_delta += w * compressed_delta

            if getattr(self.config, "fedzip_actual_mode", False):
                active_mask.fill_(True)
            else:
                if payload.get("indices") is not None:
                    active_mask[payload["indices"].long()] = True
                elif payload.get("values") is not None:
                    active_mask.fill_(True)

        # server momentum
        if self.config.use_server_momentum:
            if self._momentum_buf is None:
                self._momentum_buf = torch.zeros_like(agg_delta)
            self._momentum_buf = (self.config.server_momentum * self._momentum_buf
                                   + agg_delta)
            agg_delta = self.config.server_lr * self._momentum_buf * active_mask.float()

        new_flat = old_flat + agg_delta

        # ── [DEBUG_BN] Global model BN state BEFORE aggregation ────────────────
        if _fl_client.DEBUG_BN:
            _pre_agg_sd = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
            print(f"\n[DEBUG_BN] SERVER round={round_num}  "
                  f"GLOBAL MODEL BEFORE aggregation ({_fl_client._DEBUG_BN_LAYER}):")
            print(_fl_client._bn_snapshot(_pre_agg_sd, _fl_client._DEBUG_BN_LAYER))

            # show what client _DEBUG_BN_CLIENT uploaded (if it participated)
            _matching = [r for r in agg_clients
                         if r["client_id"] == _fl_client._DEBUG_BN_CLIENT]
            if _matching:
                _cr = _matching[0]
                print(f"\n[DEBUG_BN] SERVER: uploaded state_dict from client "
                      f"{_fl_client._DEBUG_BN_CLIENT} ({_fl_client._DEBUG_BN_LAYER}):")
                print(_fl_client._bn_snapshot(
                    {k: v.cpu() for k, v in _cr['state_dict'].items()},
                    _fl_client._DEBUG_BN_LAYER
                ))
            else:
                print(f"[DEBUG_BN] SERVER: client {_fl_client._DEBUG_BN_CLIENT} "
                      f"not in agg_clients this round.")
        # ──────────────────────────────────────────────────────────────────────
        
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
                if clean:
                    total_clean_samples = sum(r["num_samples"] for r in clean)
                    buf_sum = sum((r["num_samples"] / total_clean_samples) * r["state_dict"][k].to(self.device) 
                                  for r in clean)
                    new_sd[k] = buf_sum.to(v.dtype)
                else:
                    new_sd[k] = v
                    
        self.global_model.load_state_dict(new_sd)
        self.global_delta = agg_delta

        # ── [DEBUG_BN] Global model BN state AFTER aggregation ─────────────────
        if _fl_client.DEBUG_BN:
            _post_agg_sd = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
            print(f"\n[DEBUG_BN] SERVER round={round_num}  "
                  f"GLOBAL MODEL AFTER aggregation ({_fl_client._DEBUG_BN_LAYER}):")
            print(_fl_client._bn_snapshot(_post_agg_sd, _fl_client._DEBUG_BN_LAYER))

            # Check whether BN buffers actually changed
            _pre = _pre_agg_sd   # captured in the block above
            _bn_buf_keys = [k for k in _post_agg_sd
                            if any(s in k for s in
                                   ("running_mean", "running_var", "num_batches_tracked"))]
            _changed = [k for k in _bn_buf_keys
                        if not torch.allclose(
                            _post_agg_sd[k].float(), _pre[k].float(), atol=1e-7)]
            print(f"[DEBUG_BN] BN buffers changed by aggregation: "
                  f"{len(_changed)} / {len(_bn_buf_keys)}")
            if _changed:
                print(f"[DEBUG_BN] Changed: {_changed[:5]}{'...' if len(_changed)>5 else ''}")
        # ──────────────────────────────────────────────────────────────────────
        
        # -- Layer-wise Top-k Importance EMA Update --
        # Computed exactly ONCE per round from the globally aggregated delta.
        # This guarantees all clients in the next round will share exactly
        # the same budget, and removes the beta^N decay bug.
        if getattr(self.config, "use_layerwise_topk", False):
            layer_slices = self._get_layer_slices()
            layer_rms = [
                (self.global_delta[offset:offset+size].norm(p=2).item() / (size ** 0.5))
                for _, offset, size in layer_slices
            ]
            if getattr(self.config, "enable_routing_diagnostics", False) and round_num == 1:
                 print(f"  [DEBUG] Layer importance EMA updated ONCE. Slices: {len(layer_slices)}")
                 
            if self._layer_importance is None:
                self._layer_importance = layer_rms
            else:
                beta = self.config.layerwise_ema_beta
                self._layer_importance = [
                    beta * old + (1.0 - beta) * new
                    for old, new in zip(self._layer_importance, layer_rms)
                ]

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