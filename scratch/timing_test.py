"""Timing test for 3 rounds."""
import time
import random
import torch
import numpy as np

random.seed(42); np.random.seed(42); torch.manual_seed(42)

from divroute_fl.data import get_client_datasets
from divroute_fl.model import get_model
from divroute_fl.client import FLClient
from divroute_fl.server import FLServer
from baselines.fedsparse_baseline import get_fedsparse_config

print("CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

cfg = get_fedsparse_config(
    fedsparse_lambda=0.01, num_clients=25, clients_per_round=15,
    num_rounds=3, local_lr=0.1, seed=42, dataset_name='cifar10',
    model_name='simplecnn', skip_plot_prompt=True,
    log_path='scratch/timing_test.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

client_datasets = get_client_datasets('cifar10', 25, cfg.alpha, 42)
global_model = get_model('simplecnn', 10)
server = FLServer(global_model, cfg, device)
clients = [FLClient(i, client_datasets[i], cfg.local_epochs, cfg.local_lr,
    cfg.batch_size, device, 'simplecnn', 10) for i in range(25)]
all_ids = list(range(25))
error_buffers = {}

for rnd in range(3):
    t0 = time.time()
    selected = server.select_clients(all_ids)
    global_sd = server.global_model.state_dict()
    results = []
    for cid in selected:
        r = clients[cid].train(global_sd, cfg.local_epochs, fedsparse_lambda=cfg.fedsparse_lambda)
        results.append(r)
    for r in results:
        r['divergence_score'] = 0.0; r['tier'] = 1; r['bytes_received'] = 0
    server.aggregate(results, error_buffers)
    print(f"Round {rnd+1}: {time.time()-t0:.2f}s", flush=True)

print("Done", flush=True)
