import json
import os
from typing import List


class FLLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: List[dict] = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log(self, round_idx: int, test_accuracy: float, client_results: List[dict]) -> None:
        total_bytes = sum(r.get("bytes_received", 0) or 0 for r in client_results)

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
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
