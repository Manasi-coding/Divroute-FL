from divroute_fl.config import Config
from divroute_fl.main import run

run(Config(
    k_ratio_tier1=1.0,
    k_ratio_tier2=1.0,
    use_adaptive_k=False,
    include_tier3_in_aggregation=True,
    use_error_feedback=False,
    use_tier3_sync=False,
    use_divergence_weighting=True,
    use_server_momentum=True,
    use_adaptive_tau=True,
    use_epoch_warmup=False,
    num_rounds=20,
    seed=42,
    skip_plot_prompt=True,
    log_path="logs/exp2_no_tier3_exclusion.json",
))
