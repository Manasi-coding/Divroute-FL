import json
import os
from typing import Any, Dict, List, Optional


class FLLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: List[dict] = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log(
        self,
        round_idx: int,
        test_accuracy: float,
        client_results: List[dict],
        delta_numel: int = 0,
        # ── PART 7: optional diagnostic payloads ──────────────────────────────
        # All default to None so that the call-site in main.py can omit them
        # when the corresponding Config flag is False.  When None, the field is
        # simply absent from the JSON entry, preserving backward compatibility.
        grad_diagnostics: Optional[Dict[int, Dict[str, float]]] = None,
        agg_update_norm: Optional[float] = None,
        global_update_norm: Optional[float] = None,
        routing_correlations: Optional[Dict[str, float]] = None,
        tier_contributions: Optional[Dict[str, Any]] = None,
        compression_analysis: Optional[List[Dict[str, Any]]] = None,
        tau_diagnostics: Optional[Dict[str, Any]] = None,
        # ────────────────────────────────────────────────────────────────────────
    ) -> None:
        """
        Phase 1.3: log both download (`bytes_received`, server -> client) and
        upload (`upload_bytes`, client -> server) per client, plus round
        totals `total_download_bytes` / `total_upload_bytes`.

        `total_bytes_transmitted` is kept for backward compatibility with
        existing plotting code and is equal to `total_download_bytes`.
        """
        # upload (C->S): compressed payload bytes each client transmitted
        total_upload   = sum(r.get("upload_bytes",   0) or 0 for r in client_results)
        # download (S->C): full global model broadcast to every selected client
        # (set by server.aggregate(); Tier-3 clients carry 0 since they were
        #  initialised to 0 before aggregate() and never entered the loop)
        total_download = sum(r.get("download_bytes", 0) or 0 for r in client_results)

        entry = {
            "round": round_idx,
            "test_accuracy": round(test_accuracy, 4),
            # backward-compat alias kept for existing plot/parse code
            "total_bytes_transmitted": total_upload,
            "total_upload_bytes":   total_upload,    # C->S compressed bytes
            "total_download_bytes": total_download,  # S->C full model bytes
            "delta_numel": delta_numel,
            "clients": [
                {
                    "client_id":       r["client_id"],
                    "divergence_score": r.get("divergence_score"),
                    "tier":            r.get("tier"),
                    "upload_bytes":    r.get("upload_bytes"),    # C->S compressed
                    "download_bytes":  r.get("download_bytes"),  # S->C full model
                    # keep legacy field so old log parsers still work
                    "bytes_received":  r.get("bytes_received"),
                    "aggregation_weight": r.get("aggregation_weight"),
                    # local_train_acc: accuracy on the client's *training* loader
                    #   (was "local_accuracy" — renamed to clarify it is not validation accuracy)
                    "local_train_acc": r.get("local_train_acc"),
                    # local_val_acc: accuracy on the held-out val subset (test transform, no aug).
                    #   None when local_val_fraction == 0.0 (diagnostic disabled).
                    "local_val_acc":   r.get("local_val_acc"),
                    # ── PART 7: per-client diagnostic fields (None when disabled) ──
                    "grad_diag":       (
                        grad_diagnostics.get(r["client_id"])
                        if grad_diagnostics is not None else None
                    ),
                }
                for r in client_results
            ],
        }

        # ── PART 7: round-level diagnostic fields (appended only when available) ──
        if agg_update_norm is not None:
            entry["agg_update_norm"] = agg_update_norm
        if global_update_norm is not None:
            entry["global_update_norm"] = global_update_norm
        if routing_correlations:
            entry["routing_correlations"] = routing_correlations
        if tier_contributions:
            entry["tier_contributions"] = tier_contributions
        if compression_analysis:
            entry["compression_analysis"] = compression_analysis
        if tau_diagnostics:
            entry["tau_diagnostics"] = tau_diagnostics
        # ────────────────────────────────────────────────────────────────────────

        self.history.append(entry)
        self._flush()

    def _flush(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)