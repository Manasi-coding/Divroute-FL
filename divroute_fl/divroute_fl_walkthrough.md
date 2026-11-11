# Training flow

1. `main.run()` applies the selected configuration and seeds Python, NumPy, and PyTorch.
2. Dataset loaders create client partitions and the global test loader.
3. `FLServer.select_clients()` samples clients using the current selection weights.
4. `FLClient.train()` initializes from the global state and performs local training.
5. Routing computes scores, updates thresholds, and assigns tiers or explicit retention ratios.
6. `FLServer.aggregate()` compresses client deltas, applies aggregation weights, and updates the model.
7. Evaluation and reporting record accuracy, diagnostics, and simulated communication counts.
8. Checkpoints preserve the existing training and random-state fields for resume.

See the repository README for supported experiment commands and the configuration factories for exact presets.
