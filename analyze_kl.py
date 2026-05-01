#!/usr/bin/env python3
"""
Analyze KL smoke-test results: correlation with reach, stratified by style,
plus comparison against surprisal patterns.

Reads:
  runs/kl_smoke_results.jsonl
  runs/overnight/summary.jsonl
  runs/surprisal_x_reach.jsonl  (for cross-comparison)
"""

import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_reach(trials):
    groups = defaultdict(list)
    for t in trials:
        if "term_in_B_content_lower" not in t:
            continue
        groups[(t["term"], t["style"])].append(t)
    out = {}
    for k, items in groups.items():
        n = len(items)
        out[k] = {
            "n_trials": n,
            "b_reach": sum(t["term_in_B_content_lower"] for t in items) / n,
            "c_reach": sum(t["term_in_C_content_lower"] for t in items) / n,
        }
    return out


def pearson_r(xs, ys):
    n = len(xs)
    if n < 3:
        return None, n
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return None, n
    return num / math.sqrt(dx2 * dy2), n


def main():
    kl_rows = load_jsonl(PROJECT / "runs" / "kl_smoke_results.jsonl")
    trials = load_jsonl(PROJECT / "runs" / "overnight" / "summary.jsonl")
    reach_agg = aggregate_reach(trials)

    print(f"KL rows: {len(kl_rows)}")

    joint = []
    for r in kl_rows:
        key = (r["term"], r["style"])
        if key in reach_agg:
            joint.append({**r, **reach_agg[key]})

    print(f"Joint rows: {len(joint)}")

    print("\n=== KL × REACH correlations ===\n")
    print("Overall (across all styles):")
    for k in ["sum_log_ratio_nats", "mean_log_ratio_nats", "max_log_ratio_nats"]:
        for r_key in ["b_reach", "c_reach"]:
            xs = [r[k] for r in joint]
            ys = [r[r_key] for r in joint]
            r, n = pearson_r(xs, ys)
            r_str = f"{r:+.3f}" if r is not None else "n/a"
            print(f"  {k:<28s} × {r_key:<10s}  r = {r_str}  n={n}")

    print("\nStratified by style:")
    for style in ["S1", "S2", "S3"]:
        rs = [r for r in joint if r["style"] == style]
        print(f"\n  {style} (n={len(rs)}):")
        for k in ["sum_log_ratio_nats", "mean_log_ratio_nats", "max_log_ratio_nats"]:
            for r_key in ["b_reach", "c_reach"]:
                xs = [r[k] for r in rs]
                ys = [r[r_key] for r in rs]
                r, n = pearson_r(xs, ys)
                r_str = f"{r:+.3f}" if r is not None else "n/a"
                print(f"    {k:<28s} × {r_key:<10s}  r = {r_str}")

    # Quick distributional summary
    print("\n=== KL distributions by style ===\n")
    for style in ["S1", "S2", "S3"]:
        vals = [r["sum_log_ratio_nats"] for r in joint if r["style"] == style]
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        print(f"  {style}: median={vals[n//2]:+.2f}  "
              f"q25={vals[n//4]:+.2f}  q75={vals[3*n//4]:+.2f}  "
              f"min={vals[0]:+.2f}  max={vals[-1]:+.2f}")

    # Save joint table for plotting
    out_path = PROJECT / "runs" / "kl_x_reach.jsonl"
    with out_path.open("w") as f:
        for r in joint:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
