#!/usr/bin/env python3
"""Analyze pattern-vs-instance results.

Per-primer mean count of hyphenated forms generated, broken down by prompt.
Test: does P1 (skeptical priming) reduce the count vs P0 (no primer)?
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS_DIR = Path("runs/pattern_vs_instance")


def load_summary():
    p = RUNS_DIR / "summary.jsonl"
    if not p.exists():
        return []
    recs = []
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


def stats(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
    }


def main():
    recs = load_summary()
    if not recs:
        print("No data yet.")
        return
    print(f"Loaded {len(recs)} trials\n")

    # ---- Overall: mean hyphenated forms per response, by primer ----
    print("=" * 72)
    print("Mean hyphenated forms generated per response, by primer")
    print("=" * 72)
    print(f"  {'primer':<8} {'mean':>10} {'se':>8} {'N':>5}")
    by_primer = defaultdict(list)
    for r in recs:
        by_primer[r["primer_label"]].append(r["n_hyph_output"])
    for p in sorted(by_primer):
        s = stats(by_primer[p])
        print(f"  {p:<8} {s['mean']:>10.2f} {s['se']:>8.2f} {s['n']:>5d}")

    # Pairwise diff vs P0
    if "P0" in by_primer:
        s0 = stats(by_primer["P0"])
        print("\n  Diffs vs P0 (n hyph in output):")
        for p in sorted(by_primer):
            if p == "P0":
                continue
            sp = stats(by_primer[p])
            diff = sp["mean"] - s0["mean"]
            se = (sp["se"] ** 2 + s0["se"] ** 2) ** 0.5
            z = diff / se if se > 0 else 0
            print(f"    {p} - P0: {diff:+.3f} ± {se:.3f}  (z = {z:+.2f})")

    # ---- By prompt × primer ----
    print()
    print("=" * 72)
    print("By prompt × primer (mean hyph forms in output)")
    print("=" * 72)
    prompts = sorted(set(r["prompt_label"] for r in recs))
    primers = sorted(set(r["primer_label"] for r in recs))
    print(f"  {'prompt':<14} " + " ".join(f"{p:>14}" for p in primers))
    for prompt in prompts:
        row = []
        for primer in primers:
            cell = [r["n_hyph_output"] for r in recs
                    if r["prompt_label"] == prompt and r["primer_label"] == primer]
            s = stats(cell)
            if s is None:
                row.append("        -      ")
            else:
                row.append(f"{s['mean']:>5.2f}±{s['se']:.2f}(N={s['n']:2d})")
        print(f"  {prompt:<14} " + " ".join(f"{c:>14}" for c in row))

    # ---- Reasoning trace check ----
    print()
    print("=" * 72)
    print("Reasoning trace: same analysis (some models think hyph but don't say it)")
    print("=" * 72)
    print(f"  {'primer':<8} {'mean':>10} {'se':>8} {'N':>5}")
    by_primer_r = defaultdict(list)
    for r in recs:
        by_primer_r[r["primer_label"]].append(r["n_hyph_reasoning"])
    for p in sorted(by_primer_r):
        s = stats(by_primer_r[p])
        print(f"  {p:<8} {s['mean']:>10.2f} {s['se']:>8.2f} {s['n']:>5d}")

    # ---- Show a sample of generated forms per primer ----
    print()
    print("=" * 72)
    print("Sample of generated hyphenated forms per primer")
    print("=" * 72)
    by_primer_forms = defaultdict(list)
    for r in recs:
        by_primer_forms[r["primer_label"]].extend(r["hyphenated_in_output"])
    for p in sorted(by_primer_forms):
        forms = by_primer_forms[p]
        unique = sorted(set(forms))
        print(f"\n  {p} ({len(forms)} total, {len(unique)} unique):")
        # Print up to 30 unique forms
        for i, form in enumerate(unique[:30]):
            print(f"    {form}")
        if len(unique) > 30:
            print(f"    ... ({len(unique) - 30} more)")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
