"""
DivRoute-FL Lite — Visualization
Reads logs/run.json and produces presentation plots.

Standalone — only imports json, os, numpy, matplotlib.
No dependency on model or training code.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_log(log_path: str) -> list:
    with open(log_path, "r") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Plot 1 — Divergence Heatmap
# -------------------------------------------------------------------------

def plot_divergence_heatmap(log_path: str = "logs/run.json"):
    """
    How much each client has drifted from the global model across every round.
    Rows = clients, columns = rounds. NaN cells = client not selected.
    """
    log = _load_log(log_path)
    num_rounds = len(log)

    # Infer num_clients from the maximum client_id seen
    all_ids = [c["client_id"] for r in log for c in r["clients"]]
    num_clients = max(all_ids) + 1

    # Build 2D grid: rows = clients, cols = rounds — NaN for unselected
    grid = np.full((num_clients, num_rounds), np.nan)
    for r in log:
        round_idx = r["round"]
        for c in r["clients"]:
            grid[c["client_id"], round_idx] = c["divergence_score"]

    fig, ax = plt.subplots(figsize=(14, 6))
    img = ax.imshow(
        grid,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=max(0.1, np.nanmax(grid)),  # auto-scale for 1-cos_sim range
        interpolation="nearest",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Divergence Score (1 - cos sim, EMA-smoothed)", fontsize=11)

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel("Client ID", fontsize=12)
    ax.set_title("Client Divergence Over Training — DivRoute-FL", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "divergence_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# -------------------------------------------------------------------------
# Plot 2 — Tier Distribution Over Time
# -------------------------------------------------------------------------

def plot_tier_distribution(log_path: str = "logs/run.json"):
    """
    Stacked bar chart showing how many clients land in each tier each round.
    With adaptive (percentile-based) tau, tier sizes are roughly determined
    by tau_low_pct / tau_high_pct each round, not by fixed thresholds.
    """
    log = _load_log(log_path)
    rounds = [r["round"] for r in log]
    tier1 = [sum(1 for c in r["clients"] if c["tier"] == 1) for r in log]
    tier2 = [sum(1 for c in r["clients"] if c["tier"] == 2) for r in log]
    tier3 = [sum(1 for c in r["clients"] if c["tier"] == 3) for r in log]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(rounds, tier1, label="Tier 1 - High Fidelity (most drifted)",
           color="steelblue")
    ax.bar(rounds, tier2, bottom=tier1,
           label="Tier 2 - Low Fidelity (moderately drifted)", color="orange")
    ax.bar(rounds, tier3,
           bottom=[t1 + t2 for t1, t2 in zip(tier1, tier2)],
           label="Tier 3 - Skip (converged)", color="lightgrey")

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel("Number of Selected Clients", fontsize=12)
    ax.set_title("Percentile-Adaptive Tier Assignment Over Training", fontsize=13)
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "tier_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# -------------------------------------------------------------------------
# Plot 3 — Communication vs Accuracy (headline result)
# -------------------------------------------------------------------------

def plot_comm_vs_accuracy(log_path: str = "logs/run.json"):
    """
    Three curves: cumulative download bytes (x) vs accuracy (y).
    1. Full FedAvg (no compression)
    2. Uniform Top-5% (prior work baseline)
    3. DivRoute-FL (our system)
    """
    log = _load_log(log_path)
    accuracies = [r["test_accuracy"] for r in log]

    # Read delta_numel directly from the log (stored by logger.py)
    delta_numel = log[0]["delta_numel"]
    clients_per_round = len(log[0]["clients"])

    # DivRoute-FL: actual download bytes from log
    divroute_bytes = [r.get("total_download_bytes", r["total_bytes_transmitted"]) for r in log]
    divroute_cumul = np.cumsum(divroute_bytes) / 1e6  # MB

    # Full FedAvg baseline: every client gets full delta every round
    fedavg_per_round = clients_per_round * delta_numel * 4
    fedavg_cumul = np.cumsum([fedavg_per_round] * len(log)) / 1e6

    # Uniform Top-5% baseline: every client always gets tier-2 compression (8 bytes/param)
    k_uniform = int(0.05 * delta_numel)
    uniform_per_round = clients_per_round * k_uniform * 8
    uniform_cumul = np.cumsum([uniform_per_round] * len(log)) / 1e6

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fedavg_cumul, accuracies, color="tomato",
            linewidth=2, label="Full FedAvg (no compression)")
    ax.plot(uniform_cumul, accuracies, color="goldenrod",
            linewidth=2, linestyle="--", label="Uniform Top-5% (prior work)")
    ax.plot(divroute_cumul, accuracies, color="steelblue",
            linewidth=2.5, label="DivRoute-FL (ours)")

    ax.set_xlabel("Cumulative Download Transmitted (MB)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Communication Efficiency — DivRoute-FL vs Baselines",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "comm_vs_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# -------------------------------------------------------------------------
# Plot 4 — Per-Round Bytes (Download + Upload)
# -------------------------------------------------------------------------

def plot_bytes_per_round(log_path: str = "logs/run.json"):
    """
    Per-round bandwidth: DivRoute-FL download vs Full FedAvg download,
    plus DivRoute-FL upload (Phase 1.3 — bidirectional accounting).
    Shows download bandwidth dropping over time as clients converge into
    lower tiers, while upload remains constant (uncompressed, current
    implementation).
    """
    log = _load_log(log_path)
    rounds = [r["round"] for r in log]
    divroute_down_mb = [r.get("total_download_bytes", r["total_bytes_transmitted"]) / 1e6 for r in log]
    divroute_up_mb = [r.get("total_upload_bytes", 0) / 1e6 for r in log]

    # Baseline: full FedAvg download
    delta_numel = log[0]["delta_numel"]
    clients_per_round = len(log[0]["clients"])
    fedavg_mb = (clients_per_round * delta_numel * 4) / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rounds, divroute_down_mb, color="steelblue",
            linewidth=2, label="DivRoute-FL download")
    ax.plot(rounds, divroute_up_mb, color="seagreen",
            linewidth=2, linestyle=":", label="DivRoute-FL upload (uncompressed)")
    ax.axhline(y=fedavg_mb, color="tomato", linewidth=2,
               linestyle="--", label=f"Full FedAvg download ({fedavg_mb:.1f} MB/round)")

    ax.fill_between(rounds, divroute_down_mb, fedavg_mb,
                     alpha=0.15, color="steelblue", label="Download bandwidth saved")

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel("Bytes Transmitted (MB)", fontsize=12)
    ax.set_title("Per-Round Bandwidth: DivRoute-FL vs Full FedAvg",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "bytes_per_round.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def generate_all_plots(log_path: str = "logs/run.json"):
    """Generate all presentation plots from the experiment log."""
    print(f"\n[plots] Reading log from: {log_path}")
    plot_divergence_heatmap(log_path)
    plot_tier_distribution(log_path)
    plot_comm_vs_accuracy(log_path)
    plot_bytes_per_round(log_path)
    print(f"[plots] All plots saved to: {PLOTS_DIR}/\n")


if __name__ == "__main__":
    generate_all_plots()