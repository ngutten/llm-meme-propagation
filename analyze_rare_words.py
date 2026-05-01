#!/usr/bin/env python3
"""
Analyze rare-words B-reach vs corpus frequency.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wordfreq

PROJECT = Path(__file__).resolve().parent
SUMMARY = PROJECT / "runs" / "rare_words" / "summary.jsonl"
FIGS = PROJECT / "figures"
FIGS.mkdir(exist_ok=True)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return None
    return num / math.sqrt(dx2 * dy2)


def main():
    rows = []
    with open(SUMMARY) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_word = defaultdict(list)
    for r in rows:
        by_word[r["word"]].append(r)

    print(f"{'word':<14s}  zipf   n  A_pickup  B_reach (S3)")
    print("-" * 55)
    agg = []
    for word in sorted(by_word, key=lambda w: wordfreq.zipf_frequency(w, "en")):
        items = by_word[word]
        n = len(items)
        a_rate = sum(t["term_in_A_content_lower"] for t in items) / n
        b_rate = sum(t["term_in_B_content_lower"] for t in items) / n
        zipf = wordfreq.zipf_frequency(word, "en")
        agg.append({"word": word, "zipf": zipf, "n": n,
                    "a_pickup": a_rate, "b_reach": b_rate})
        print(f"  {word:<12s}  {zipf:.2f}   {n}  {a_rate:.2f}      {b_rate:.2f}")

    xs = [a["zipf"] for a in agg]
    ys = [a["b_reach"] for a in agg]
    r = pearson(xs, ys)
    print(f"\nPearson r (zipf × B-reach): {r:+.3f}  (n={len(agg)})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=80, alpha=0.7, color="#1f77b4")
    for a in agg:
        ax.annotate(a["word"], (a["zipf"], a["b_reach"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    if len(xs) >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.array([min(xs), max(xs)])
        ax.plot(xline, slope * xline + intercept,
                color="black", linestyle="--", alpha=0.6,
                label=f"r = {r:+.3f}")
        ax.legend()
    ax.set_xlabel("Zipf frequency (corpus)")
    ax.set_ylabel("B-reach rate (S3)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rare-words B-reach vs corpus frequency (S3)")
    fig.tight_layout()
    out = FIGS / "rare_words_reach_vs_zipf.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
