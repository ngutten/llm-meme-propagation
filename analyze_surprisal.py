#!/usr/bin/env python3
"""
Analyze correlations between term surprisal and reach.

Reads:
  runs/surprisal_results.jsonl  — per (term, style) surprisal measures
  runs/overnight/summary.jsonl  — per-trial reach data

Computes:
  - Variance decomposition: how much of reach variance is explained by style + topic
  - Per-(term, style) reach aggregation
  - Correlations: surprisal × reach, with style-stratified breakdown
  - Sensitivity: mean / max / first-token surprisal compared

Usage:
    python3 analyze_surprisal.py [--summary path] [--surprisal path]
"""

import argparse
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
    """
    Group trials by (term, style), compute fraction of trials where reach was True.

    Returns dict (term, style) -> {b_reach_rate, c_reach_rate, n_trials, b_reasoning_rate, c_reasoning_rate}
    """
    groups = defaultdict(list)
    for t in trials:
        key = (t["term"], t["style"])
        groups[key].append(t)

    agg = {}
    for key, items in groups.items():
        n = len(items)
        agg[key] = {
            "term": key[0],
            "style": key[1],
            "n_trials": n,
            "b_reach_rate": sum(t["term_in_B_content_lower"] for t in items) / n,
            "c_reach_rate": sum(t["term_in_C_content_lower"] for t in items) / n,
            "b_reasoning_rate": sum(t["term_in_B_reasoning"] for t in items) / n,
            "c_reasoning_rate": sum(t["term_in_C_reasoning"] for t in items) / n,
        }
    return agg


def pearson_r(xs, ys):
    """Compute Pearson correlation. Returns (r, n) or (None, n) if undefined."""
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


def variance_decomp(trials, response_key):
    """
    Quick ANOVA-style variance decomposition: how much of trial-level reach
    variance is explained by style + c_topic + interaction.

    Reach is binary; we treat it as 0/1 and report fraction-of-variance-explained
    by group means relative to grand mean.
    """
    # Style-only model: SS by style
    by_style = defaultdict(list)
    by_topic = defaultdict(list)
    by_style_topic = defaultdict(list)
    all_y = []
    for t in trials:
        y = float(t[response_key])
        all_y.append(y)
        by_style[t["style"]].append(y)
        by_topic[t["c_topic"]].append(y)
        by_style_topic[(t["style"], t["c_topic"])].append(y)

    grand_mean = sum(all_y) / len(all_y)
    sst = sum((y - grand_mean) ** 2 for y in all_y)

    def ss_between(groups):
        ss = 0
        for g, vals in groups.items():
            n_g = len(vals)
            mean_g = sum(vals) / n_g
            ss += n_g * (mean_g - grand_mean) ** 2
        return ss

    ss_style = ss_between(by_style)
    ss_topic = ss_between(by_topic)
    ss_style_topic = ss_between(by_style_topic)

    return {
        "response": response_key,
        "n_trials": len(all_y),
        "grand_mean": grand_mean,
        "SST": sst,
        "R2_style_only": ss_style / sst if sst > 0 else 0,
        "R2_topic_only": ss_topic / sst if sst > 0 else 0,
        "R2_style_x_topic": ss_style_topic / sst if sst > 0 else 0,
        "residual_after_style_topic": 1 - (ss_style_topic / sst) if sst > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(PROJECT / "runs" / "overnight" / "summary.jsonl"))
    parser.add_argument("--surprisal", default=str(PROJECT / "runs" / "surprisal_results.jsonl"))
    args = parser.parse_args()

    trials_raw = load_jsonl(args.summary)
    # Filter out error rows (no reach data)
    trials = [t for t in trials_raw if "term_in_B_content_lower" in t]
    surprisal_rows = load_jsonl(args.surprisal)

    print(f"Loaded {len(trials_raw)} raw rows, {len(trials)} successful trials, "
          f"{len(surprisal_rows)} surprisal measurements\n")

    print("=" * 70)
    print("VARIANCE DECOMPOSITION (pre-step before surprisal analysis)")
    print("=" * 70)
    for response_key in ["term_in_B_content_lower", "term_in_C_content_lower"]:
        d = variance_decomp(trials, response_key)
        print(f"\n  Response: {response_key}")
        print(f"    Grand mean reach: {d['grand_mean']:.3f}")
        print(f"    R² (style only):       {d['R2_style_only']:.3f}")
        print(f"    R² (topic only):       {d['R2_topic_only']:.3f}")
        print(f"    R² (style × topic):    {d['R2_style_x_topic']:.3f}")
        print(f"    Residual variance:     {d['residual_after_style_topic']:.3f}")

    print("\n" + "=" * 70)
    print("CORRELATION: SURPRISAL × REACH")
    print("=" * 70)

    # Aggregate reach by (term, style)
    reach_agg = aggregate_reach(trials)

    # Build joint table: (term, style) -> surprisal + reach
    joint = []
    for s in surprisal_rows:
        key = (s["term"], s["style"])
        if key in reach_agg:
            row = {
                **s,
                **reach_agg[key],
            }
            joint.append(row)
        else:
            print(f"  WARN: no trials for ({s['term']}, {s['style']})")

    print(f"\n  N joint rows: {len(joint)}")

    print(f"\n  Overall correlations (across all styles):")
    for surp_key in ["mean_per_token_surprisal_nats",
                     "max_per_token_surprisal_nats",
                     "first_token_surprisal_nats",
                     "total_surprisal_nats"]:
        for reach_key in ["b_reach_rate", "c_reach_rate"]:
            xs = [r[surp_key] for r in joint]
            ys = [r[reach_key] for r in joint]
            r, n = pearson_r(xs, ys)
            r_str = f"{r:+.3f}" if r is not None else "    n/a"
            print(f"    {surp_key:<40s} × {reach_key:<14s}  r = {r_str}  (n={n})")

    print(f"\n  Stratified by style:")
    for style in ["S1", "S2", "S3"]:
        rows_s = [r for r in joint if r["style"] == style]
        print(f"\n    Style {style} (n={len(rows_s)}):")
        for surp_key in ["mean_per_token_surprisal_nats",
                         "max_per_token_surprisal_nats",
                         "first_token_surprisal_nats"]:
            for reach_key in ["b_reach_rate", "c_reach_rate"]:
                xs = [r[surp_key] for r in rows_s]
                ys = [r[reach_key] for r in rows_s]
                r, n = pearson_r(xs, ys)
                r_str = f"{r:+.3f}" if r is not None else "    n/a"
                print(f"      {surp_key:<40s} × {reach_key:<14s}  r = {r_str}")

    # Save joint table for any downstream analysis
    out_path = PROJECT / "runs" / "surprisal_x_reach.jsonl"
    with out_path.open("w") as f:
        for r in joint:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote joint table to {out_path}")


if __name__ == "__main__":
    main()
