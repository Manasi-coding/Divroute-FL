"""
run_uniform_top5.py
===================
Runs the Uniform Top-5% baseline.

Every selected client transmits exactly top-5% of its gradient delta.
No divergence scoring, no tier assignment, no adaptive routing.
Communication volume is identical to DivRoute Tier-2 for all clients.

This serves as the communication-matched baseline for publication:
it answers whether DivRoute's routing intelligence outperforms
naive uniform compression at the same budget.

Usage:
    python run_uniform_top5.py
"""

from divroute_fl.config import get_uniform_top5_config
from divroute_fl.main import run

run(get_uniform_top5_config(
    num_clients       = 25,
    clients_per_round = 15,
    num_rounds        = 20,
    seed              = 42,
    skip_plot_prompt  = True,
    log_path          = "logs/uniform_top5_run.json",
))
