"""
Summarize DivRoute-FL run logs (logs/*.json) into the numbers used in the
write-up: plateau accuracy (not final-round), across-seed mean/spread, and
upload/bidirectional bytes.

Why this exists: results were being computed by hand from pasted console
output, which drops rounds when a notebook disconnects and is easy to
mis-index (the JSON `round` field is the 0-indexed loop variable; this script
reports 1-indexed rounds to match the console). The JSON log is flushed every
round and restored on resume, so it is the reliable source.

Usage
-----
    # one row per run
    python scripts/summarize_runs.py logs/a.json logs/b.json

    # compare methods across seeds (every pair of groups gets a gap-vs-spread verdict)
    python scripts/summarize_runs.py --last 20 \
        --group DivRoute logs/d42.json logs/d43.json logs/d44.json \
        --group Uniform  logs/u42.json logs/u43.json logs/u44.json \
        --group FedAvg   logs/f43.json logs/f44.json
"""
import argparse
import json
import statistics as st
import sys


def load_run(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    history = raw["history"] if isinstance(raw, dict) else raw
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    rounds = sorted(history, key=lambda e: e["round"])
    acc = [e["test_accuracy"] * 100.0 for e in rounds]
    up = sum((e.get("total_upload_bytes") or 0) for e in rounds) / 1e6
    down = sum((e.get("total_download_bytes") or 0) for e in rounds) / 1e6
    return {"path": path, "meta": meta, "acc": acc, "up": up, "down": down}


def plateau(acc, last):
    window = acc[-last:] if len(acc) >= last else acc
    sd = st.stdev(window) if len(window) > 1 else 0.0
    return st.mean(window), sd, len(window)


def describe(run, last):
    mean, sd, n = plateau(run["acc"], last)
    m = run["meta"]
    mode = m.get("partition_mode", "dirichlet")
    if mode == "dirichlet":
        part = f"dirichlet/alpha={m.get('alpha', '?')}"
    elif mode == "pathological":
        part = f"pathological/{m.get('shards_per_client', '?')} shards"
    else:                       # natural: alpha / shards do not apply
        part = mode
    tag = f"{part}/seed={m.get('seed', '?')}"
    note = "" if n == last else f"  [only {n} rounds available]"
    print(f"  {run['path']}\n    {tag} | rounds={len(run['acc'])} | final={run['acc'][-1]:.2f}% "
          f"| plateau(last {n})={mean:.2f}% (round-to-round std {sd:.2f}) "
          f"| peak={max(run['acc']):.2f}% | upload={run['up']:.1f}MB "
          f"| bidir={run['up'] + run['down']:.1f}MB{note}")
    return mean, sd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="run JSON logs (ungrouped)")
    p.add_argument("--group", action="append", nargs="+", metavar=("NAME", "FILE"),
                   help="NAME followed by that method's run JSONs (one per seed)")
    p.add_argument("--last", type=int, default=20, help="plateau window in rounds (default 20)")
    args = p.parse_args()

    if not args.files and not args.group:
        p.error("give run JSON files and/or --group NAME FILE...")

    for path in args.files:
        print("Run:")
        describe(load_run(path), args.last)

    stats = []
    for grp in args.group or []:
        name, files = grp[0], grp[1:]
        print(f"\nGroup {name} ({len(files)} run(s)):")
        means, within, ups = [], [], []
        for path in files:
            r = load_run(path)
            m, sd = describe(r, args.last)
            means.append(m)
            within.append(sd)
            ups.append(r["up"])
        across = st.stdev(means) if len(means) > 1 else None
        spread = across if across is not None else within[0]
        kind = "across-seed std" if across is not None else "single run: within-run round-to-round std"
        print(f"  => {name}: plateau mean {st.mean(means):.2f}%  ({kind} {spread:.2f})  "
              f"| mean upload {st.mean(ups):.1f}MB")
        stats.append({"name": name, "mean": st.mean(means), "spread": spread,
                      "n": len(means), "up": st.mean(ups)})

    if len(stats) >= 2:
        print()
        import itertools
        for a, b in itertools.combinations(stats, 2):
            gap = a["mean"] - b["mean"]
            noise = max(a["spread"], b["spread"])
            verdict = "within noise (tie)" if abs(gap) <= noise else "larger than noise"
            ratio = f" | upload ratio {a['name']}/{b['name']}: {a['up'] / b['up']:.2f}x" if b["up"] > 0 else ""
            low_n = "  [fewer than 3 runs on a side -- provisional]" if min(a["n"], b["n"]) < 3 else ""
            print(f"Compare {a['name']} vs {b['name']}: gap {gap:+.2f}pp vs larger spread "
                  f"{noise:.2f}pp -> {verdict}{ratio}{low_n}")


if __name__ == "__main__":
    sys.exit(main())
