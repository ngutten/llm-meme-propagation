#!/usr/bin/env python3
"""Zipf analysis on overnight C3 outputs.

Two measurements:
  Z1: word-frequency Zipf exponent on full C3 output text.
      Comparable to Astral's Moltbook 1.70 vs human ~1.0 stat.
  Z2: term reach-rate Zipf across the 30 unique terms.
      Internal: how concentrated is propagation across terms.

Estimates the exponent by linear fit on log-log over the head of the
distribution (excluding the long tail of singletons), and reports CI
via bootstrap.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


RUNS_DIR = Path("runs/overnight")


def load_records():
    records = []
    with open(RUNS_DIR / "summary.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if "error" not in d:
                records.append(d)
    by_path = {r["path"]: r for r in records}
    return list(by_path.values())


def extract_c_text(trial_path):
    with open(RUNS_DIR / trial_path) as f:
        d = json.load(f)
    chunks = []
    for t in d["transcript"]:
        if t["role"] == "assistant" and t["label"].startswith("C_"):
            chunks.append(t["content"])
    return " ".join(chunks)


WORD_RE = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*")


def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def fit_zipf(counts, head_frac=0.5, min_freq=2):
    """Fit log(rank) vs log(freq) over the head of the distribution.

    head_frac: use top fraction of unique words by frequency.
    min_freq: drop words with frequency below this.
    Returns (exponent, intercept, n_used).
    """
    counts = sorted(counts, reverse=True)
    counts = [c for c in counts if c >= min_freq]
    n = max(1, int(len(counts) * head_frac))
    counts = counts[:n]
    ranks = np.arange(1, len(counts) + 1)
    log_r = np.log(ranks)
    log_f = np.log(counts)
    # log(f) = -alpha * log(r) + b  => slope is -alpha
    slope, intercept = np.polyfit(log_r, log_f, 1)
    return -slope, intercept, len(counts)


def bootstrap_zipf(words, n_boot=200, head_frac=0.5, min_freq=2, seed=0):
    rng = np.random.default_rng(seed)
    words = np.array(words)
    alphas = []
    for _ in range(n_boot):
        sample = rng.choice(words, size=len(words), replace=True)
        c = Counter(sample)
        alpha, _, _ = fit_zipf(list(c.values()), head_frac, min_freq)
        alphas.append(alpha)
    return float(np.mean(alphas)), float(np.std(alphas, ddof=1))


def main():
    print("Loading overnight records...")
    records = load_records()
    print(f"  {len(records)} successful trials")

    print("Extracting and tokenizing C outputs...")
    all_words = []
    by_topic = {}
    by_style = {}
    for r in records:
        text = extract_c_text(r["path"])
        toks = tokenize(text)
        all_words.extend(toks)
        by_topic.setdefault(r["c_topic"], []).extend(toks)
        by_style.setdefault(r["style"], []).extend(toks)

    print(f"  total tokens: {len(all_words):,}")
    print(f"  unique types: {len(set(all_words)):,}")

    print("\n" + "=" * 70)
    print("Z1: word-frequency Zipf on all C3 output (Gemma 26B-A4B)")
    print("=" * 70)

    counts = Counter(all_words)
    print(f"  top 20: {counts.most_common(20)}")

    for head_frac in [0.1, 0.25, 0.5]:
        alpha, intercept, n_used = fit_zipf(list(counts.values()), head_frac=head_frac)
        print(f"  head_frac={head_frac:.2f}: alpha = {alpha:.3f}  "
              f"(intercept={intercept:.2f}, fit on top {n_used} types)")

    # Bootstrap CI on the head_frac=0.25 estimate (mid range)
    print("\nBootstrap (n=100, head_frac=0.25)...")
    alpha_b, alpha_se = bootstrap_zipf(all_words, n_boot=100, head_frac=0.25)
    print(f"  bootstrapped alpha = {alpha_b:.3f} ± {alpha_se:.3f}")

    print("\n" + "=" * 70)
    print("Z1 split by C-probe topic")
    print("=" * 70)
    for topic in sorted(by_topic):
        words = by_topic[topic]
        c = Counter(words)
        alpha, _, n_used = fit_zipf(list(c.values()), head_frac=0.25)
        print(f"  {topic:<14} alpha = {alpha:.3f}  "
              f"(N tokens={len(words):,}, types={len(c):,}, fit on top {n_used})")

    print("\n" + "=" * 70)
    print("Z1 split by style")
    print("=" * 70)
    for style in sorted(by_style):
        words = by_style[style]
        c = Counter(words)
        alpha, _, n_used = fit_zipf(list(c.values()), head_frac=0.25)
        print(f"  {style:<6} alpha = {alpha:.3f}  "
              f"(N tokens={len(words):,}, types={len(c):,}, fit on top {n_used})")

    print("\n" + "=" * 70)
    print("Z2: reach-rate distribution across the 30 terms")
    print("=" * 70)
    # Compute reach rate per term
    by_term = {}
    for r in records:
        by_term.setdefault(r["term"], []).append(int(r["term_in_C_content_lower"]))
    term_rates = {t: np.mean(v) for t, v in by_term.items()}
    sorted_terms = sorted(term_rates.items(), key=lambda x: -x[1])
    for i, (t, rate) in enumerate(sorted_terms, 1):
        bar = "█" * int(rate * 40)
        print(f"  {i:2d}. {t:<28} {rate:.2f}  {bar}")

    rates = sorted([r for r in term_rates.values() if r > 0], reverse=True)
    if len(rates) >= 5:
        ranks = np.arange(1, len(rates) + 1)
        slope, intercept = np.polyfit(np.log(ranks), np.log(rates), 1)
        print(f"\n  Z2 alpha (reach-rate vs rank) = {-slope:.3f}  on {len(rates)} terms")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
