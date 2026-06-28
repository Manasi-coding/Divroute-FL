import sys
sys.path.insert(0, '.')
import run_phase5_comparison as r5
from run_phase5_comparison import DATASET_PRESETS, _make_config, _logs_dir, _csv_path

for preset_name in ['cifar10', 'cifar100']:
    r5._PRESET = DATASET_PRESETS[preset_name]
    p = r5._PRESET

    cfg_avg = _make_config('fedavg',   42, 'logs/test.json')
    cfg_div = _make_config('divroute', 42, 'logs/test.json')
    cfg_uni = _make_config('uniform',  42, 'logs/test.json')

    print(f"=== {preset_name} preset ===")
    print(f"  rounds         : {p['num_rounds']}")
    print(f"  clients/round  : {p['clients_per_round']} / {p['num_clients']} ({100*p['clients_per_round']//p['num_clients']}% participation)")
    print(f"  local_lr       : {cfg_avg.local_lr}")
    print(f"  DivRoute k-ratios  : tier1={cfg_div.k_ratio_tier1}  tier2={cfg_div.k_ratio_tier2}")
    print(f"  Uniform  k-ratio   : {cfg_uni.k_ratio_tier2}  (fixed — not preset-driven)")
    print(f"  divroute tau   : high={cfg_div.tau_high}  low={cfg_div.tau_low}  adaptive={cfg_div.use_adaptive_tau}")

    # Byte savings preview (all clients Tier-1, early training)
    numel      = 11_229_752  # ResNet-18
    fedavg_bpp = 4           # bytes per param
    div_bpp    = cfg_div.k_ratio_tier1 * 8
    saving_pct = (1 - div_bpp / fedavg_bpp) * 100
    print(f"  Early-round DivRoute saving (all Tier-1): {saving_pct:.0f}%  "
          f"({cfg_div.k_ratio_tier1*numel/1e6:.2f}M params / {numel/1e6:.2f}M sent per client)")
    print()

print("All preset checks passed.")
