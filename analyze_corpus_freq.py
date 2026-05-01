#!/usr/bin/env python3
"""
Look up corpus frequency for all terms used in existing memetics experiments,
then plot reach vs frequency. Tests whether corpus frequency (context-independent
rarity) discriminates propagation behavior.

Data sources:
  - runs/overnight/summary.jsonl     (hyphenated coinages, C1/C2/C3)
  - runs/known_control/summary.jsonl (known hyphenated terms)

For each term we compute:
  - whole_term_zipf:    Zipf frequency of the exact term as a phrase
  - min_component_zipf: min Zipf across component words (after splitting on '-')
  - max_component_zipf: max Zipf across component words
  - mean_component_zipf: mean of component Zipfs
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wordfreq

PROJECT = Path(__file__).resolve().parent
FIGS = PROJECT / "figures"
FIGS.mkdir(exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "term_in_B_content_lower" not in d:
                continue
            rows.append(d)
    return rows


def term_freq_features(term):
    """Return frequency features for a term (possibly hyphenated)."""
    whole = wordfreq.zipf_frequency(term.lower(), "en")
    parts = term.lower().split("-")
    part_freqs = [wordfreq.zipf_frequency(p, "en") for p in parts]
    return {
        "term": term,
        "whole_term_zipf": whole,
        "n_components": len(parts),
        "min_component_zipf": min(part_freqs) if part_freqs else 0,
        "max_component_zipf": max(part_freqs) if part_freqs else 0,
        "mean_component_zipf": (sum(part_freqs) / len(part_freqs)) if part_freqs else 0,
        "components": list(zip(parts, part_freqs)),
    }


def aggregate_by_term(trials, source_label):
    groups = defaultdict(list)
    for t in trials:
        groups[t["term"]].append(t)
    rows = []
    for term, items in groups.items():
        n = len(items)
        # Restrict to S3 for cleanest signal (per yesterday's analysis)
        s3 = [t for t in items if t["style"] == "S3"]
        if not s3:
            continue
        feat = term_freq_features(term)
        rows.append({
            "term": term,
            "source": source_label,
            "n_trials_total": n,
            "n_trials_s3": len(s3),
            "b_reach_s3": sum(t["term_in_B_content_lower"] for t in s3) / len(s3),
            "c_reach_s3": sum(t["term_in_C_content_lower"] for t in s3) / len(s3),
            **feat,
        })
    return rows


def main():
    overnight = load_jsonl(PROJECT / "runs" / "overnight" / "summary.jsonl")
    known = load_jsonl(PROJECT / "runs" / "known_control" / "summary.jsonl")

    rows = aggregate_by_term(overnight, "overnight (coinages)") + \
           aggregate_by_term(known, "known_control")

    # Print table
    print(f"{'term':<28s}  {'source':<22s}  "
          f"{'whole':<6s}  {'min_c':<6s}  {'max_c':<6s}  {'mean_c':<6s}  "
          f"{'b_S3':<6s}  {'c_S3':<6s}  n")
    print("-" * 110)
    for r in sorted(rows, key=lambda r: -r["b_reach_s3"]):
        print(f"{r['term']:<28s}  {r['source']:<22s}  "
              f"{r['whole_term_zipf']:<6.2f}  "
              f"{r['min_component_zipf']:<6.2f}  "
              f"{r['max_component_zipf']:<6.2f}  "
              f"{r['mean_component_zipf']:<6.2f}  "
              f"{r['b_reach_s3']:<6.2f}  "
              f"{r['c_reach_s3']:<6.2f}  "
              f"{r['n_trials_s3']}")

    # Save
    out = PROJECT / "runs" / "term_freq_x_reach.jsonl"
    with out.open("w") as f:
        for r in rows:
            r2 = {k: v for k, v in r.items() if k != "components"}
            f.write(json.dumps(r2) + "\n")
    print(f"\nSaved table: {out}")

    # Plot reach vs each frequency measure, S3 only
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    measures = [
        ("whole_term_zipf", "Whole-term Zipf frequency"),
        ("min_component_zipf", "Min component-word Zipf"),
        ("mean_component_zipf", "Mean component-word Zipf"),
    ]
    reaches = [
        ("b_reach_s3", "B-reach rate (S3)"),
        ("c_reach_s3", "C-reach rate (S3)"),
    ]
    sources = sorted(set(r["source"] for r in rows))
    src_colors = {"overnight (coinages)": "#1f77b4", "known_control": "#d62728"}
    for col, (skey, slabel) in enumerate(measures):
        for row_i, (rkey, rlabel) in enumerate(reaches):
            ax = axes[row_i, col]
            for src in sources:
                pts = [(r[skey], r[rkey], r["term"]) for r in rows if r["source"] == src]
                xs, ys, labels = zip(*pts) if pts else ([], [], [])
                ax.scatter(xs, ys, alpha=0.7, s=40, label=src,
                          color=src_colors.get(src, "#888"))
            ax.set_xlabel(slabel)
            ax.set_ylabel(rlabel)
            ax.set_ylim(-0.05, 1.05)
            if row_i == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Reach vs corpus frequency (S3 only)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "reach_vs_corpus_freq.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
