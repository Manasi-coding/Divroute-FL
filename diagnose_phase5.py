import json

for method in ['fedavg', 'fedsparse', 'fedzip', 'uniform', 'divroute']:
    path = f'logs/phase5/{method}_seed42.json'
    with open(path) as f:
        log = json.load(f)

    r1   = log[0]
    r50  = log[49]
    r100 = log[-1]

    a1   = r1['test_accuracy']
    a50  = r50['test_accuracy']
    a100 = r100['test_accuracy']

    dl_r1 = r1.get('total_download_bytes', r1.get('total_bytes_transmitted', 0))
    delta_numel = r1.get('delta_numel', 'N/A')

    clients_r1   = r1.get('clients', [])
    clients_r100 = r100.get('clients', [])

    div_r1   = [c.get('divergence_score') for c in clients_r1   if c.get('divergence_score') is not None]
    div_r100 = [c.get('divergence_score') for c in clients_r100 if c.get('divergence_score') is not None]

    tau_r1   = (r1.get('tau_low'),   r1.get('tau_high'))
    tau_r100 = (r100.get('tau_low'), r100.get('tau_high'))

    print(f"=== {method} ===")
    print(f"  acc  : r1={a1:.4f}  r50={a50:.4f}  r100={a100:.4f}")
    print(f"  bytes: download_r1={dl_r1/1e6:.2f}MB  delta_numel={delta_numel}")
    if div_r1:
        print(f"  div r1  : min={min(div_r1):.6f}  max={max(div_r1):.6f}  mean={sum(div_r1)/len(div_r1):.6f}")
    if div_r100:
        print(f"  div r100: min={min(div_r100):.6f}  max={max(div_r100):.6f}  mean={sum(div_r100)/len(div_r100):.6f}")
    print(f"  tau r1  : {tau_r1}")
    print(f"  tau r100: {tau_r100}")
    print()
