#!/usr/bin/env python3
"""Analyze chain experiment results.

Computes per (term × style):
  - Per-link transmission rate P(term in step n | term in step n-1) = R0 (intrinsic)
  - Absolute presence rate at each depth (averaged over chains)
  - Hyphenation effect: R0_hyph - R0_unhyph at matched semantic content
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS_DIR = Path("runs/chain")


def load_summary():
    recs = []
    p = RUNS_DIR / "summary.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                continue
            recs.append(d)
    return recs


def main():
    recs = load_summary()
    print(f"Loaded {len(recs)} chains\n")
    if not recs:
        print("No chain runs yet.")
        return

    # Per (term, style): list of step_terms arrays
    cells = defaultdict(list)
    for r in recs:
        cells[(r["term_id"], r["style"])].append(r["step_terms"])

    # ---- Per-link transmission rate (intrinsic R0) ----
    print("=" * 78)
    print("Per-link transmission rate: P(term in step n | term in step n-1)")
    print("=" * 78)
    print(f"  {'term_id':<32} {'style':<5} {'R0_link':>10} {'N_links':>10} {'N_chains':>10}")
    for (term_id, style), chains in sorted(cells.items()):
        transitions = []
        for steps in chains:
            for n in range(1, len(steps)):
                if steps[n - 1]:  # term was in n-1
                    transitions.append(int(steps[n]))
        n_trans = len(transitions)
        r0 = sum(transitions) / max(1, n_trans) if n_trans else float("nan")
        print(f"  {term_id:<32} {style:<5} {r0:>10.3f} {n_trans:>10d} {len(chains):>10d}")

    # ---- Presence at each depth ----
    print()
    print("=" * 78)
    print("Absolute presence rate at each depth (averaged over chains)")
    print("=" * 78)
    max_d = max(len(s) for chains in cells.values() for s in chains)
    header = f"  {'term_id':<32} {'style':<5} " + " ".join(f"{'d'+str(d):>5}" for d in range(max_d))
    print(header)
    for (term_id, style), chains in sorted(cells.items()):
        rates = []
        for d in range(max_d):
            vals = [s[d] for s in chains if len(s) > d]
            rates.append(sum(vals) / max(1, len(vals)) if vals else float("nan"))
        rate_strs = [f"{r:>5.2f}" if not np.isnan(r) else "  -  " for r in rates]
        print(f"  {term_id:<32} {style:<5} " + " ".join(rate_strs))

    # ---- Hyphenation effect ----
    print()
    print("=" * 78)
    print("Hyphenation effect on R0 (matched semantic content)")
    print("=" * 78)
    pairs = [
        ("legibility-cost",       "legibility_cost_phrase"),
        ("fennel-apparatus",      "fennel_apparatus_unhyph"),
        ("unfalsifiability-shim", "shimmer_unfals_phrase"),
    ]
    for h_id, u_id in pairs:
        for style in sorted(set(r["style"] for r in recs)):
            h_chains = cells.get((h_id, style), [])
            u_chains = cells.get((u_id, style), [])
            if not h_chains or not u_chains:
                continue

            def r0_of(chains):
                ts = []
                for s in chains:
                    for n in range(1, len(s)):
                        if s[n - 1]:
                            ts.append(int(s[n]))
                return sum(ts) / max(1, len(ts)), len(ts)

            r0_h, n_h = r0_of(h_chains)
            r0_u, n_u = r0_of(u_chains)
            print(f"  {h_id:<24} vs {u_id:<26} ({style}): "
                  f"R0_hyph={r0_h:.3f}(N={n_h:3d})  R0_unhyph={r0_u:.3f}(N={n_u:3d})  "
                  f"diff={r0_h-r0_u:+.3f}")

    # ---- Style effect ----
    print()
    print("=" * 78)
    print("Style effect on R0 (collapsed over terms)")
    print("=" * 78)
    by_style = defaultdict(list)
    for (term_id, style), chains in cells.items():
        for s in chains:
            for n in range(1, len(s)):
                if s[n - 1]:
                    by_style[style].append(int(s[n]))
    for style in sorted(by_style):
        ts = by_style[style]
        r0 = sum(ts) / max(1, len(ts))
        se = (r0 * (1 - r0) / max(1, len(ts))) ** 0.5
        print(f"  {style}: R0 = {r0:.3f} ± {se:.3f}  (N={len(ts)} link transitions)")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
