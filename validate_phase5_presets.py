import sys
sys.path.insert(0, '.')
import run_phase5_comparison as r5
from run_phase5_comparison import DATASET_PRESETS, _make_config, _logs_dir, _csv_path

for preset_name in ['cifar10', 'cifar100']:
    r5._PRESET = DATASET_PRESETS[preset_name]
    p = r5._PRESET

    cfg_div = _make_config('divroute',  42, 'logs/test.json')
    cfg_fs  = _make_config('fedsparse', 42, 'logs/test.json')

    print(f"=== {preset_name} preset ===")
    print(f"  logs dir       : {_logs_dir()}")
    print(f"  csv            : {_csv_path()}")
    print(f"  acc budgets    : {p['acc_at_budgets_mb']}")
    print(f"  divroute tau   : high={cfg_div.tau_high}  low={cfg_div.tau_low}  adaptive={cfg_div.use_adaptive_tau}")
    print(f"  fedsparse lam  : {cfg_fs.fedsparse_lambda}")
    print(f"  model/dataset  : {cfg_div.model_name} / {cfg_div.dataset_name}")
    print(f"  clients        : {cfg_div.num_clients} total, {cfg_div.clients_per_round}/round")
    print()

print("All preset checks passed.")
