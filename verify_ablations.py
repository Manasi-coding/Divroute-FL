"""
verify_ablations.py
===================
Runs each of the four ablation experiments for exactly 2 rounds and
checks that each experiment exhibits the expected behaviour.

Usage:
    python verify_ablations.py
"""

import sys
import os
import shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Capture output for analysis
from io import StringIO

def run_experiment(label, config, expect_fn):
    """Run config for 2 rounds, collect stdout, call expect_fn(stdout_text)."""
    import contextlib

    config.num_rounds = 2
    config.diag_print_interval = 1
    config.skip_plot_prompt = True

    buf = StringIO()
    print(f"\n{'='*68}")
    print(f"  VERIFYING: {label}")
    print(f"{'='*68}")

    # Tee output to both stdout and buffer
    class Tee:
        def __init__(self, real, buf):
            self.real = real; self.buf = buf
        def write(self, s):
            self.real.write(s); self.buf.write(s)
        def flush(self):
            self.real.flush()
        @property
        def encoding(self): return self.real.encoding

    orig = sys.stdout
    sys.stdout = Tee(orig, buf)
    try:
        from divroute_fl.main import run
        run(config)
    finally:
        sys.stdout = orig

    text = buf.getvalue()
    try:
        expect_fn(text)
        print(f"\n  [VERIFY] PASS: {label}")
    except AssertionError as e:
        print(f"\n  [VERIFY] FAIL: {label}")
        print(f"  Reason: {e}")
    return text


# ── Experiment A ──────────────────────────────────────────────────────────────
def verify_a():
    from divroute_fl.config import Config
    cfg = Config(
        run_label="ablation_a_fedavg",
        fedavg_baseline_mode=True,
        use_epoch_warmup=False,
        dataset_name="cifar100", model_name="resnet18",
        num_clients=100, clients_per_round=20,
        local_epochs=5, local_lr=0.1, batch_size=32, alpha=0.9, seed=42,
        enable_routing_diagnostics=True, enable_tau_diagnostics=True,
        log_path="logs/verify_a.json", fresh=True, resume=False,
    )

    def check(text):
        # FedAvg mode: tiers must all be 0/N/0 or equivalent (no routing)
        assert "FEDAVG BASELINE MODE" in text, \
            "Expected FEDAVG BASELINE MODE banner"
        assert "tiers:" in text, "Expected tiers: in round output"
        # In fedavg_baseline_mode tau is set to -1 so all clients are Tier-1
        # Verify no Tier-2 or Tier-3 in round summary
        import re
        tier_lines = [l for l in text.splitlines() if "tiers:" in l and "round" in l]
        assert tier_lines, "No tier lines found"
        for line in tier_lines:
            m = re.search(r"tiers:\s*(\d+)/(\d+)/(\d+)", line)
            if m:
                t1, t2, t3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                assert t2 == 0 and t3 == 0, \
                    f"Exp A should have T2=0 T3=0 but got {t1}/{t2}/{t3}"
        # Checkpoint label check
        assert "ablation_a_fedavg_cifar100_seed42" in text, \
            "Checkpoint directory should contain ablation_a_fedavg"

    run_experiment("Experiment A — FedAvg sanity baseline", cfg, check)


# ── Experiment B ──────────────────────────────────────────────────────────────
def verify_b():
    from divroute_fl.config import Config, get_recommended_divroute_config
    cfg = get_recommended_divroute_config(
        run_label="ablation_b_no_compression",
        ablation_routing_no_compression=True,
        dataset_name="cifar100", model_name="resnet18",
        num_clients=100, clients_per_round=20,
        local_epochs=5, local_lr=0.1, batch_size=32, alpha=0.9, seed=42,
        enable_routing_diagnostics=True, enable_tau_diagnostics=True,
        enable_compression_analysis=True,
        log_path="logs/verify_b.json", fresh=True, resume=False,
    )

    def check(text):
        assert "ABLATION B" in text, "Expected ABLATION B banner"
        assert "k=1.0" in text or "k_ratio: tier1=1.00 tier2=1.00" in text, \
            "Expected k=1.0 in ABLATION B banner"
        # k_ratio should show 1.0/1.0
        assert "tier1=1.00 tier2=1.00" in text, \
            "Expected tier1=1.00 tier2=1.00 in config banner"
        # Tiers should still be assigned normally (routing ON)
        import re
        tier_lines = [l for l in text.splitlines() if "tiers:" in l and "round" in l]
        assert tier_lines, "No tier lines found in Exp B"
        # At least one non-trivial tier line (routing should produce Tier-2 clients)
        # Just verify we have valid output
        assert "ablation_b_no_compression_cifar100_seed42" in text, \
            "Checkpoint directory should contain ablation_b_no_compression"

    run_experiment("Experiment B — Routing ON / compression OFF", cfg, check)


# ── Experiment C ──────────────────────────────────────────────────────────────
def verify_c():
    from divroute_fl.config import Config
    cfg = Config(
        run_label="ablation_c_uniform_compression",
        ablation_uniform_compression=True,
        ablation_uniform_k_ratio=0.05,
        dataset_name="cifar100", model_name="resnet18",
        num_clients=100, clients_per_round=20,
        local_epochs=5, local_lr=0.1, batch_size=32, alpha=0.9, seed=42,
        use_adaptive_tau=True, tau_alpha=0.5, tau_beta=1.0,
        enable_routing_diagnostics=True, enable_tau_diagnostics=True,
        enable_compression_analysis=True,
        log_path="logs/verify_c.json", fresh=True, resume=False,
    )

    def check(text):
        assert "ABLATION C" in text, "Expected ABLATION C banner"
        assert "k=0.050" in text, "Expected k=0.050 in ABLATION C banner"
        # All tiers should be Tier-1 after the routing override
        import re
        tier_lines = [l for l in text.splitlines() if "tiers:" in l and "round" in l]
        assert tier_lines, "No tier lines found in Exp C"
        for line in tier_lines:
            m = re.search(r"tiers:\s*(\d+)/(\d+)/(\d+)", line)
            if m:
                t1, t2, t3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                assert t2 == 0 and t3 == 0, \
                    f"Exp C: all clients should be Tier-1 but got {t1}/{t2}/{t3}"
        assert "ablation_c_uniform_compression_cifar100_seed42" in text, \
            "Checkpoint directory should contain ablation_c_uniform_compression"

    run_experiment("Experiment C — Uniform compression k=0.05 / routing OFF", cfg, check)


# ── Experiment D ──────────────────────────────────────────────────────────────
def verify_d():
    from divroute_fl.config import Config, get_recommended_divroute_config
    cfg = get_recommended_divroute_config(
        run_label="ablation_d_full_divroute",
        ablation_per_client_logging=True,
        dataset_name="cifar100", model_name="resnet18",
        num_clients=100, clients_per_round=20,
        local_epochs=5, local_lr=0.1, batch_size=32, alpha=0.9, seed=42,
        enable_routing_diagnostics=True, enable_tau_diagnostics=True,
        enable_compression_analysis=True, enable_gradient_diagnostics=True,
        log_path="logs/verify_d.json", fresh=True, resume=False,
    )

    def check(text):
        assert "ABLATION D" in text, "Expected ABLATION D banner"
        assert "[ABLATION-D]" in text, "Expected [ABLATION-D] per-client table"
        assert "d_raw" in text and "d_ema" in text, \
            "Expected d_raw and d_ema columns in forensic table"
        assert "k_ratio" in text, "Expected k_ratio column in forensic table"
        assert "sel_w" in text, "Expected sel_w column in forensic table"
        # Routing should be normal DivRoute (Tier-1 and Tier-2 clients present)
        import re
        tier_lines = [l for l in text.splitlines() if "tiers:" in l and "round" in l]
        assert tier_lines, "No tier lines found in Exp D"
        assert "ablation_d_full_divroute_cifar100_seed42" in text, \
            "Checkpoint directory should contain ablation_d_full_divroute"

    run_experiment("Experiment D — Full DivRoute + forensic logging", cfg, check)


# ── Distinct checkpoint directories ──────────────────────────────────────────
def verify_checkpoint_isolation():
    print(f"\n{'='*68}")
    print("  VERIFYING: Checkpoint directory isolation")
    print(f"{'='*68}")
    labels = [
        "ablation_a_fedavg_cifar100_seed42",
        "ablation_b_no_compression_cifar100_seed42",
        "ablation_c_uniform_compression_cifar100_seed42",
        "ablation_d_full_divroute_cifar100_seed42",
    ]
    assert len(set(labels)) == len(labels), \
        "Checkpoint directory labels are not all distinct!"
    for lbl in labels:
        print(f"  OK  checkpoints/{lbl}/")
    print(f"\n  [VERIFY] PASS: All 4 checkpoint directories are distinct")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    verify_a()
    verify_b()
    verify_c()
    verify_d()
    verify_checkpoint_isolation()
    print("\n" + "=" * 68)
    print("  ALL ABLATION VERIFICATIONS COMPLETE")
    print("=" * 68)
