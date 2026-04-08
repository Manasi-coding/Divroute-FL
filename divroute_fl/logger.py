"""
DivRoute-FL Lite — Logger
Accumulates per-round metrics and writes them to a JSON file.
"""
import json
import os
from typing import List


class FLLogger:
    """Collects round-level metrics and persists them as a JSON array."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: List[dict] = []

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log(self, round_idx: int, test_accuracy: float, client_results: List[dict]) -> None:
        """
        Record one training round.

        Parameters
        ----------
        round_idx : int
            Zero-based round number.
        test_accuracy : float
            Global test accuracy after aggregation.
        client_results : list[dict]
            The per-client result dicts returned by ``FLClient.train`` and
            enriched with divergence_score / tier / bytes_received.
        """
        total_bytes = sum(
            r.get("bytes_received", 0) or 0 for r in client_results
        )

        entry = {
            "round": round_idx,
            "test_accuracy": round(test_accuracy, 4),
            "total_bytes_transmitted": total_bytes,
            "clients": [
                {
                    "client_id": r["client_id"],
                    "divergence_score": r.get("divergence_score"),
                    "tier": r.get("tier"),
                    "bytes_received": r.get("bytes_received"),
                }
                for r in client_results
            ],
        }

        self.history.append(entry)
        self._flush()

    def _flush(self) -> None:
        """Write the full history list to disk (overwrite each time)."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
