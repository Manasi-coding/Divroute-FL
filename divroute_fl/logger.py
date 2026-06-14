import json
import os
from typing import List


class FLLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: List[dict] = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log(self, round_idx: int, test_accuracy: float, client_results: List[dict],
            delta_numel: int = 0) -> None:
        """
        Phase 1.3: log both download (`bytes_received`, server -> client) and
        upload (`upload_bytes`, client -> server) per client, plus round
        totals `total_download_bytes` / `total_upload_bytes`.

        `total_bytes_transmitted` is kept for backward compatibility with
        existing plotting code and is equal to `total_download_bytes`.
        """
        total_download = sum(r.get("bytes_received", 0) or 0 for r in client_results)
        total_upload = sum(r.get("upload_bytes", 0) or 0 for r in client_results)

        entry = {
            "round": round_idx,
            "test_accuracy": round(test_accuracy, 4),
            "total_bytes_transmitted": total_download,    # backward-compat alias (download)
            "total_download_bytes": total_download,
            "total_upload_bytes": total_upload,
            "delta_numel": delta_numel,
            "clients": [
                {
                    "client_id": r["client_id"],
                    "divergence_score": r.get("divergence_score"),
                    "tier": r.get("tier"),
                    "bytes_received": r.get("bytes_received"),
                    "upload_bytes": r.get("upload_bytes"),
                }
                for r in client_results
            ],
        }

        self.history.append(entry)
        self._flush()

    def _flush(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)