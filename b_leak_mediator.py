#!/usr/bin/env python3
"""B-leak as mediator analysis.

Tests whether the topic and style effects on reach operate via B-leak
(the within-agent carrying gate) or independently of it.

Decomposes:
  P(reach) = P(B | A) × P(C | B) + P(C | not B) × P(not B | A)
            = (uptake → carrying) × (carrying → output)
            +  (uptake → direct output, skipping B)

If P(C | B) is constant across topics and P(B | A) varies, the topic effect
is mediated by carrying.
If P(C | B) also varies, output gate is independently topic-sensitive.
"""

import json
import os
from collections import defaultdict
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


def rate(condition_fn, records):
    sel = [r for r in records if condition_fn(r)]
    if not sel:
        return None, 0
    n = len(sel)
    p = sum(int(r["term_in_C_content_lower"]) for r in sel) / n
    return p, n


def b_leak_rate(condition_fn, records):
    sel = [r for r in records if condition_fn(r)]
    if not sel:
        return None, 0
    n = len(sel)
    p = sum(int(r["term_in_B_content_lower"]) for r in sel) / n
    return p, n


def p_c_given_b(condition_fn, records, b_status=True):
    sel = [r for r in records
           if condition_fn(r) and bool(r["term_in_B_content_lower"]) == b_status]
    if not sel:
        return None, 0
    n = len(sel)
    p = sum(int(r["term_in_C_content_lower"]) for r in sel) / n
    return p, n


def main():
    records = load_records()
    print(f"Loaded {len(records)} records\n")

    # ============================================================
    # Decomposition by topic
    # ============================================================
    print("=" * 78)
    print("Decomposition by C-probe topic")
    print("=" * 78)
    print(f"  {'topic':<14} {'P(reach)':>10} {'P(B|A)':>10} {'P(C|B)':>10} "
          f"{'P(C|notB)':>11} {'lift_B':>8} {'N':>5}")
    topics = sorted(set(r["c_topic"] for r in records))
    for t in topics:
        cond = lambda r, t=t: r["c_topic"] == t
        p_reach, n = rate(cond, records)
        p_b, _ = b_leak_rate(cond, records)
        p_c_b, n_b = p_c_given_b(cond, records, b_status=True)
        p_c_nb, n_nb = p_c_given_b(cond, records, b_status=False)
        lift = p_c_b - p_c_nb if (p_c_b is not None and p_c_nb is not None) else None
        print(f"  {t:<14} {p_reach:>10.3f} {p_b:>10.3f} {p_c_b:>10.3f} "
              f"{p_c_nb:>11.3f} {lift:>+8.3f} {n:>5d}")

    # If P(C|B) varies less than P(B|A), B-leak mediates the topic effect.
    # Compute fraction of variance:
    pb_vals = []
    pcb_vals = []
    pcnb_vals = []
    for t in topics:
        cond = lambda r, t=t: r["c_topic"] == t
        pb, _ = b_leak_rate(cond, records)
        pcb, _ = p_c_given_b(cond, records, True)
        pcnb, _ = p_c_given_b(cond, records, False)
        pb_vals.append(pb)
        pcb_vals.append(pcb)
        pcnb_vals.append(pcnb)
    print(f"\n  P(B|A) range:    {min(pb_vals):.3f} – {max(pb_vals):.3f}  "
          f"(spread {max(pb_vals)-min(pb_vals):.3f})")
    print(f"  P(C|B) range:    {min(pcb_vals):.3f} – {max(pcb_vals):.3f}  "
          f"(spread {max(pcb_vals)-min(pcb_vals):.3f})")
    print(f"  P(C|notB) range: {min(pcnb_vals):.3f} – {max(pcnb_vals):.3f}  "
          f"(spread {max(pcnb_vals)-min(pcnb_vals):.3f})")
    print("\n  Larger spread on P(B|A) than P(C|B) ⇒ topic effect is mediated by carrying.")
    print("  Spread on P(C|notB) is the direct-output-gate component.")

    # ============================================================
    # Decomposition by style
    # ============================================================
    print()
    print("=" * 78)
    print("Decomposition by style")
    print("=" * 78)
    print(f"  {'style':<6} {'P(reach)':>10} {'P(B|A)':>10} {'P(C|B)':>10} "
          f"{'P(C|notB)':>11} {'lift_B':>8} {'N':>5}")
    for s in ["S1", "S2", "S3"]:
        cond = lambda r, s=s: r["style"] == s
        p_reach, n = rate(cond, records)
        p_b, _ = b_leak_rate(cond, records)
        p_c_b, _ = p_c_given_b(cond, records, b_status=True)
        p_c_nb, _ = p_c_given_b(cond, records, b_status=False)
        lift = p_c_b - p_c_nb if (p_c_b is not None and p_c_nb is not None) else None
        print(f"  {s:<6} {p_reach:>10.3f} {p_b:>10.3f} {p_c_b:>10.3f} "
              f"{p_c_nb:>11.3f} {lift:>+8.3f} {n:>5d}")

    pb_vals = []
    pcb_vals = []
    pcnb_vals = []
    for s in ["S1", "S2", "S3"]:
        cond = lambda r, s=s: r["style"] == s
        pb, _ = b_leak_rate(cond, records)
        pcb, _ = p_c_given_b(cond, records, True)
        pcnb, _ = p_c_given_b(cond, records, False)
        pb_vals.append(pb)
        pcb_vals.append(pcb)
        pcnb_vals.append(pcnb)
    print(f"\n  P(B|A) range:    {min(pb_vals):.3f} – {max(pb_vals):.3f}")
    print(f"  P(C|B) range:    {min(pcb_vals):.3f} – {max(pcb_vals):.3f}")
    print(f"  P(C|notB) range: {min(pcnb_vals):.3f} – {max(pcnb_vals):.3f}")

    # ============================================================
    # Decomposition by term class
    # ============================================================
    print()
    print("=" * 78)
    print("Decomposition by term class")
    print("=" * 78)
    print(f"  {'class':<6} {'P(reach)':>10} {'P(B|A)':>10} {'P(C|B)':>10} "
          f"{'P(C|notB)':>11} {'lift_B':>8} {'N':>5}")
    for tc in ["C1", "C2", "C3"]:
        cond = lambda r, tc=tc: r["term_class"] == tc
        p_reach, n = rate(cond, records)
        p_b, _ = b_leak_rate(cond, records)
        p_c_b, _ = p_c_given_b(cond, records, b_status=True)
        p_c_nb, _ = p_c_given_b(cond, records, b_status=False)
        lift = p_c_b - p_c_nb if (p_c_b is not None and p_c_nb is not None) else None
        print(f"  {tc:<6} {p_reach:>10.3f} {p_b:>10.3f} {p_c_b:>10.3f} "
              f"{p_c_nb:>11.3f} {lift:>+8.3f} {n:>5d}")

    # ============================================================
    # Joint: topic × style
    # ============================================================
    print()
    print("=" * 78)
    print("P(C | B=True) by topic × style — does carrying→output rate depend on context?")
    print("=" * 78)
    print(f"  {'topic':<14} {'S1':>14} {'S2':>14} {'S3':>14}")
    for t in topics:
        row = []
        for s in ["S1", "S2", "S3"]:
            cond = lambda r, t=t, s=s: r["c_topic"] == t and r["style"] == s
            p_c_b, n_b = p_c_given_b(cond, records, True)
            if p_c_b is None:
                row.append(f"     -        ")
            else:
                row.append(f"{p_c_b:.2f}(n={n_b:3d})")
        print(f"  {t:<14} " + " ".join(f"{c:>14}" for c in row))

    print()
    print("=" * 78)
    print("P(B | A) by topic × style — does uptake→carrying rate depend on context?")
    print("=" * 78)
    print(f"  {'topic':<14} {'S1':>14} {'S2':>14} {'S3':>14}")
    for t in topics:
        row = []
        for s in ["S1", "S2", "S3"]:
            cond = lambda r, t=t, s=s: r["c_topic"] == t and r["style"] == s
            p_b, n = b_leak_rate(cond, records)
            row.append(f"{p_b:.2f}(n={n:3d})")
        print(f"  {t:<14} " + " ".join(f"{c:>14}" for c in row))


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
