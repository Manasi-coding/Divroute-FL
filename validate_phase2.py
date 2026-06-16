"""
validate_phase2.py
==================
Dry-run validation suite for DivRoute-FL Phase 2 infrastructure.

Verifies imports, configurations, schemas, and paths without
executing any federated learning training rounds.

Usage:
    python validate_phase2.py
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

# To keep track of results
results = {}

def assert_pass(name, func):
    try:
        func()
        results[name] = "PASS"
    except AssertionError as e:
        results[name] = f"FAIL (AssertionError: {e})"
    except Exception as e:
        results[name] = f"FAIL ({type(e).__name__}: {e})"
        traceback.print_exc()

def check_imports():
    import divroute_fl.config
    import divroute_fl.main
    import divroute_fl.logger
    import divroute_fl.compression
    import run_uniform_top5
    import run_phase2_benchmarks
    import run_alpha_sweep
    import analyze_phase2
    
def check_config_factories():
    from divroute_fl.config import Config, get_recommended_divroute_config, get_uniform_top5_config
    c = Config()
    assert hasattr(c, "uniform_top5_mode"), "uniform_top5_mode missing from Config"
    
    dr = get_recommended_divroute_config(seed=42)
    assert dr.seed == 42
    assert dr.use_divergence_weighting == True
    
    ut = get_uniform_top5_config(seed=42)
    assert ut.seed == 42
    assert ut.uniform_top5_mode == True
    assert ut.k_ratio_tier2 == 0.05
    
def check_directories():
    for d in ["logs/phase2", "logs/alpha_sweep", "results"]:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        assert p.exists(), f"Failed to create directory {d}"

def check_benchmark_configs():
    import run_phase2_benchmarks as pb
    expected_jobs = len(pb.METHODS) * len(pb.SEEDS)
    jobs_built = 0
    for method in pb.METHODS:
        for seed in pb.SEEDS:
            cfg = pb._make_config(method, seed, "dummy.json")
            assert cfg.seed == seed
            assert cfg.num_rounds == pb.NUM_ROUNDS
            if method == "FedAvg":
                assert cfg.fedavg_baseline_mode == True
            elif method == "UniformTop5":
                assert cfg.uniform_top5_mode == True
            elif method == "DivRoute":
                assert cfg.use_divergence_weighting == True
            jobs_built += 1
    assert jobs_built == expected_jobs

def check_alpha_configs():
    import run_alpha_sweep as sw
    expected_jobs = len(sw.ALPHAS) * len(sw.METHODS) * len(sw.SEEDS)
    jobs_built = 0
    for alpha in sw.ALPHAS:
        for method in sw.METHODS:
            for seed in sw.SEEDS:
                cfg = sw._make_config(alpha, method, seed, "dummy.json")
                assert cfg.alpha == alpha
                assert cfg.seed == seed
                jobs_built += 1
    assert jobs_built == expected_jobs

def check_csv_schemas():
    import run_phase2_benchmarks as pb
    import run_alpha_sweep as sw
    import analyze_phase2 as az
    
    expected_pb = ["seed", "method", "accuracy", "download_mb", "upload_mb", "bidir_mb", "saving_pct", "runtime_sec"]
    assert pb.CSV_COLUMNS == expected_pb, "Benchmark CSV schema mismatch"
    
    expected_sw = ["alpha", "seed", "method", "accuracy", "download_mb", "upload_mb", "bidir_mb", "saving_pct", "runtime_sec"]
    assert sw.CSV_COLUMNS == expected_sw, "Alpha sweep CSV schema mismatch"
    
def check_log_parsing():
    import run_phase2_benchmarks as pb
    
    # Create a dummy FL log
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        dummy_hist = [{
            "round": 0, "test_accuracy": 0.65, 
            "total_download_bytes": 1500000, "total_upload_bytes": 1000000,
            "delta_numel": 10000
        }]
        json.dump(dummy_hist, f)
        temp_name = f.name
        
    try:
        res = pb._parse_log(Path(temp_name))
        assert abs(res["accuracy"] - 0.65) < 1e-5
        assert abs(res["download_mb"] - 1.5) < 1e-5
        assert abs(res["upload_mb"] - 1.0) < 1e-5
        assert abs(res["bidir_mb"] - 2.5) < 1e-5
        assert "saving_pct" in res
    finally:
        Path(temp_name).unlink()

def check_plotting_imports():
    import analyze_phase2
    # Verify that the booleans are properly exported based on try/except
    assert isinstance(analyze_phase2.HAS_MPL, bool)
    assert isinstance(analyze_phase2.HAS_SCIPY, bool)

def run_validations():
    print("Running Phase 2 Infrastructure Validation...\n")
    
    assert_pass("1. Core Imports", check_imports)
    assert_pass("2. Config Factories", check_config_factories)
    assert_pass("3. Output Directories", check_directories)
    assert_pass("4. Benchmark Config Builder", check_benchmark_configs)
    assert_pass("5. Alpha Sweep Config Builder", check_alpha_configs)
    assert_pass("6. CSV Schemas", check_csv_schemas)
    assert_pass("7. Log Parsing Logic", check_log_parsing)
    assert_pass("8. Plotting Dependencies", check_plotting_imports)

    print("=" * 70)
    print(f"{'Validation Check':<35} | {'Result'}")
    print("-" * 70)
    
    all_pass = True
    issues = []
    
    for name, res in results.items():
        print(f"{name:<35} | {res}")
        if not res.startswith("PASS"):
            all_pass = False
            issues.append(f"- {name}: {res}")
            
    print("=" * 70)
    
    if all_pass:
        print("\nPhase 2 infrastructure validation passed")
    else:
        print("\nValidation FAILED. Blocking issues found:")
        for issue in issues:
            print(issue)

if __name__ == "__main__":
    run_validations()
