import os
import random

import numpy as np
import torch


def _restore_checkpoint(
    config, device, latest_load_path, method_name, server, clients, logger
):
    print("[checkpoint]")
    print(f"Resuming from checkpoint directory:\n{config.checkpoint_dir}")
    print(f"Loaded checkpoint: {latest_load_path}")
    print(f"Method: {method_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Seed: {config.seed}")

    checkpoint = torch.load(latest_load_path, map_location=device, weights_only=False)

    # Safety/compatibility checks
    if checkpoint["method_name"] != method_name:
        raise ValueError(
            f"Incompatible checkpoint: method mismatch. Checkpoint={checkpoint['method_name']}, Expected={method_name}"
        )
    if checkpoint["dataset_name"] != config.dataset_name:
        raise ValueError(
            f"Incompatible checkpoint: dataset mismatch. Checkpoint={checkpoint['dataset_name']}, Expected={config.dataset_name}"
        )
    if checkpoint["model_name"] != config.model_name:
        raise ValueError(
            f"Incompatible checkpoint: model mismatch. Checkpoint={checkpoint['model_name']}, Expected={config.model_name}"
        )
    if checkpoint["seed"] != config.seed:
        raise ValueError(
            f"Incompatible checkpoint: seed mismatch. Checkpoint={checkpoint['seed']}, Expected={config.seed}"
        )

    checkpoint_config = checkpoint["config"]
    for field in ["num_clients", "clients_per_round", "local_lr", "alpha"]:
        checkpoint_val = getattr(checkpoint_config, field, None)
        expected_val = getattr(config, field, None)
        if checkpoint_val != expected_val:
            raise ValueError(
                f"Incompatible checkpoint: config field '{field}' mismatch. Checkpoint={checkpoint_val}, Expected={expected_val}"
            )

    # Restore dynamically adapted configuration state
    config.tau_low = checkpoint_config.tau_low
    config.tau_high = checkpoint_config.tau_high
    config.k_ratio_tier1 = checkpoint_config.k_ratio_tier1
    config.k_ratio_tier2 = checkpoint_config.k_ratio_tier2

    # Restore RNG states (robust across PyTorch versions)
    _restore_rng_state(checkpoint)

    # Restore server state
    server.global_model.load_state_dict(checkpoint["global_model_state_dict"])
    server.selection_weights = checkpoint["server_state"]["selection_weights"]
    server.global_delta = checkpoint["server_state"]["global_delta"]
    if server.global_delta is not None:
        server.global_delta = server.global_delta.to(device)
    server._rng.bit_generator.state = checkpoint["server_state"]["_rng_state"]
    server._momentum_buf = checkpoint["server_state"]["_momentum_buf"]
    if server._momentum_buf is not None:
        server._momentum_buf = server._momentum_buf.to(device)

    # Restore client states
    checkpoint_client_irw = checkpoint["client_states"]["irw_norms"]
    for client in clients:
        if client.client_id in checkpoint_client_irw:
            client._irw_norms = checkpoint_client_irw[client.client_id]

    # Restore main loop state
    ema_scores = checkpoint["main_loop_state"]["ema_scores"]
    _pg_last_round = checkpoint["main_loop_state"].get("_pg_last_round", {})
    error_buffers = {
        k: v.to(device)
        for k, v in checkpoint["main_loop_state"]["error_buffers"].items()
    }
    error_buffer_tiers = checkpoint["main_loop_state"].get("error_buffer_tiers", {})
    _baseline_bpr = checkpoint["main_loop_state"]["_baseline_bpr"]
    cumulative_divroute_download = checkpoint["main_loop_state"][
        "cumulative_divroute_download"
    ]
    cumulative_baseline_download = checkpoint["main_loop_state"][
        "cumulative_baseline_download"
    ]
    cumulative_upload = checkpoint["main_loop_state"]["cumulative_upload"]
    cumulative_baseline_upload = checkpoint["main_loop_state"][
        "cumulative_baseline_upload"
    ]

    _prev_delta_ckpt = checkpoint["main_loop_state"].get("_previous_global_delta", None)
    if _prev_delta_ckpt is not None:
        _previous_global_delta = _prev_delta_ckpt.to(device)
    else:
        _previous_global_delta = None
    _tau_score_history = checkpoint["main_loop_state"].get("_tau_score_history", [])

    # Restore logger history
    logger.history = checkpoint["logger_history"]
    logger._flush()

    start_round = checkpoint["round_num"] + 1
    print(f"Resuming from round {start_round}/{config.num_rounds}")
    return (
        start_round,
        ema_scores,
        _pg_last_round,
        error_buffers,
        error_buffer_tiers,
        _baseline_bpr,
        cumulative_divroute_download,
        cumulative_baseline_download,
        cumulative_upload,
        cumulative_baseline_upload,
        _previous_global_delta,
        _tau_score_history,
    )


def _restore_rng_state(checkpoint):
    try:
        random.setstate(checkpoint["rng_state"]["python"])
    except Exception as e:
        print(f"[resume] Warning: could not restore Python RNG ({e})")

    try:
        np.random.set_state(checkpoint["rng_state"]["numpy"])
    except Exception as e:
        print(f"[resume] Warning: could not restore NumPy RNG ({e})")

    # torch.load(..., map_location=device) above moves these RNG-state tensors
    # onto the GPU, but set_rng_state / set_rng_state_all only accept CPU
    # uint8 tensors -- hence the explicit .cpu().
    try:
        cpu_state = checkpoint["rng_state"]["torch_cpu"]

        if not isinstance(cpu_state, torch.Tensor):
            cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)

        torch.set_rng_state(cpu_state.cpu().to(torch.uint8))
    except Exception as e:
        print(f"[resume] Warning: could not restore CPU RNG ({e}), continuing...")

    try:
        if torch.cuda.is_available():
            cuda_state = checkpoint["rng_state"].get("torch_cuda", [])

            if cuda_state:
                fixed_states = []
                for state in cuda_state:
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.uint8)
                    fixed_states.append(state.cpu().to(torch.uint8))

                torch.cuda.set_rng_state_all(fixed_states)
    except Exception as e:
        print(f"[resume] Warning: could not restore CUDA RNG ({e}), continuing...")


def _save_checkpoint(
    config,
    server,
    clients,
    logger,
    rnd,
    completed_round,
    method_name,
    save_checkpoint_dir,
    latest_save_path,
    main_loop_state,
):
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    client_irw_norms = {client.client_id: client._irw_norms for client in clients}

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }

    checkpoint = {
        "round_num": rnd,
        "num_rounds": config.num_rounds,
        "seed": config.seed,
        "config": config,
        "method_name": method_name,
        "dataset_name": config.dataset_name,
        "model_name": config.model_name,
        "global_model_state_dict": {
            k: v.cpu() for k, v in server.global_model.state_dict().items()
        },
        "server_state": {
            "selection_weights": server.selection_weights.copy(),
            "global_delta": server.global_delta.cpu()
            if server.global_delta is not None
            else None,
            "_rng_state": server._rng.bit_generator.state,
            "_momentum_buf": server._momentum_buf.cpu()
            if server._momentum_buf is not None
            else None,
        },
        "client_states": {
            "irw_norms": client_irw_norms,
        },
        "main_loop_state": main_loop_state,
        "logger_history": logger.history.copy(),
        "rng_state": rng_state,
    }

    tmp_path = latest_save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, latest_save_path)
    print(f"[checkpoint] Saved checkpoint Round {completed_round}")

    if completed_round % 50 == 0:
        milestone_path = os.path.join(
            save_checkpoint_dir, f"round_{completed_round:03d}.pt"
        )
        torch.save(checkpoint, milestone_path)
