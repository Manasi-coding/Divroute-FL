from divroute_fl.config import Config
from divroute_fl.main import run

run(Config(
    fedavg_baseline_mode=True,
    use_epoch_warmup=False,
    num_rounds=20,
    seed=42,
    log_path="logs/fedavg_baseline.json",
))