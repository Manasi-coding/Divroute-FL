"""
Extract existing sparsity data for client 0 from probe_data.txt and fedsparse_seed42.json.
"""
import json
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROBE_FILE = 'probe_data.txt'
LOG_FILE   = 'logs/phase5/cifar10/fedsparse_seed42.json'
LR         = 0.1
LAMBDA     = 0.01
PROX_THRESH = LR * LAMBDA  # 0.001
TOTAL_PARAMS = 268650

# ── 1. Parse probe_data.txt ───────────────────────────────────────────────────
content = open(PROBE_FILE, encoding='utf-8').read()
blocks  = [b.strip() for b in content.split('[SPARSITY PROBE]') if b.strip()]

probe_records = {}
for b in blocks:
    m = re.match(r'client=(\d+)\s+round=(\d+)', b)
    if not m:
        continue
    cid, rnd = int(m.group(1)), int(m.group(2))
    if cid != 0:
        continue

    # Parse fields
    def extract(pattern):
        found = re.search(pattern, b)
        return found.group(1) if found else None

    # nnz
    nnz_m = re.search(r'non-zero\s+:\s+([\d,]+)\s+\(\s*([\d.]+)%\)', b)
    nnz   = int(nnz_m.group(1).replace(',','')) if nnz_m else None
    nnz_pct = float(nnz_m.group(2)) if nnz_m else None

    # |Δ| > 1e-4
    m4 = re.search(r'1e-4.*?:\s+([\d,]+)\s+\(\s*([\d.]+)%\)', b)
    pct_1e4 = float(m4.group(2)) if m4 else None

    # |Δ| > 1e-9 (near-zero = >1e-9, same as nnz in probe format)
    m9 = re.search(r'1e-9.*?:\s+([\d,]+)\s+\(\s*([\d.]+)%\)', b)
    pct_1e9 = float(m9.group(2)) if m9 else None

    # max |Δ|
    maxd = re.search(r'max \|Δ\|\s+:\s+([0-9.e+\-]+)', b)

    # Proximal threshold
    prox_m = re.search(r'proximal threshold\s+:\s+([0-9.]+)', b)
    prox = float(prox_m.group(1)) if prox_m else PROX_THRESH

    probe_records[rnd] = {
        'source': 'probe_data.txt',
        'nnz':    nnz,
        'nnz_pct': nnz_pct,
        'pct_1e4': pct_1e4,
        'pct_1e12': nnz_pct,   # probe uses 1e-9 but server uses 1e-12; from probe text nnz = >1e-9
        'pct_prox': None,       # not directly in probe_data.txt
        'prox_thresh': prox,
        'max_delta': maxd.group(1) if maxd else None,
    }
    print(f'Probe rnd {rnd}: nnz={nnz} ({nnz_pct}%), >1e-4={pct_1e4}%, >1e-9={pct_1e9}%', flush=True)

print(flush=True)

# ── 2. Parse fedsparse_seed42.json ────────────────────────────────────────────
with open(LOG_FILE, encoding='utf-8') as f:
    log = json.load(f)

print(f'Log has {len(log)} rounds', flush=True)
print(f'Available rounds: {[e["round"]+1 for e in log]}', flush=True)
print(flush=True)

log_c0 = {}
for e in log:
    rnd = e['round'] + 1
    for c in e.get('clients', []):
        if c['client_id'] == 0:
            upload_bytes = c.get('upload_bytes', 0)
            bytes_received = c.get('bytes_received', 0)
            # FedSparse: bytes_received = nnz * 8 (float32 value + int32 index)
            nnz_from_bytes = bytes_received // 8
            nnz_pct_from_bytes = 100.0 * nnz_from_bytes / TOTAL_PARAMS
            log_c0[rnd] = {
                'source': 'fedsparse_seed42.json',
                'bytes_received': bytes_received,
                'upload_bytes': upload_bytes,
                'nnz': nnz_from_bytes,
                'nnz_pct': nnz_pct_from_bytes,
            }
            print(f'Log rnd {rnd}: bytes_received={bytes_received:,} => nnz={nnz_from_bytes:,} ({nnz_pct_from_bytes:.2f}%)', flush=True)

print(flush=True)

# ── 3. Summary ────────────────────────────────────────────────────────────────
print('=' * 80, flush=True)
print('SUMMARY — Client 0 sparsity across all available rounds', flush=True)
print('=' * 80, flush=True)
print(f'  Proximal threshold = lr x lambda = {LR} x {LAMBDA} = {PROX_THRESH:.4f}', flush=True)
print(f'  Total params = {TOTAL_PARAMS:,}', flush=True)
print(flush=True)
print(f"  {'Round':>5}  {'Source':>20}  {'nnz':>8}  {'nnz%':>7}  {'>1e-4%':>8}", flush=True)
print('  ' + '-' * 60, flush=True)

all_rounds = sorted(set(list(probe_records.keys()) + list(log_c0.keys())))
for rnd in all_rounds:
    if rnd in probe_records:
        p = probe_records[rnd]
        print(f"  {rnd:>5}  {'probe_data.txt':>20}  {p['nnz']:>8,}  {p['nnz_pct']:>7.2f}  "
              f"{p['pct_1e4']:>8.2f}", flush=True)
    elif rnd in log_c0:
        l = log_c0[rnd]
        print(f"  {rnd:>5}  {'json log':>20}  {l['nnz']:>8,}  {l['nnz_pct']:>7.2f}  {'N/A':>8}", flush=True)
