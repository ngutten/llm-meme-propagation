#!/usr/bin/env python3
"""Chain decay analysis — meaning vs form persistence by depth.

For each chain, compute:
  - cosine similarity of step N output to step 0 output (semantic alignment to original)
  - whether the term appears in step N (form persistence)
Compare decay curves: prediction is meaning erodes faster than form across depth.

The transcript-level qualitative finding is the meta-spiral: agents praise the
form of the prior response and ascend meta-levels, leaving the original term
behind. This script quantifies that.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RUNS_DIR = Path("runs/chain")


def load_chains():
    """Load all chain JSON files, return list of dicts."""
    chains = []
    for p in sorted(RUNS_DIR.glob("chain_*.json")):
        try:
            with open(p) as f:
                chains.append(json.load(f))
        except json.JSONDecodeError:
            continue
    return chains


def main():
    chains = load_chains()
    print(f"Loaded {len(chains)} chains")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect (depth, step_output, step_0_output, term_present, term, style) per step
    rows = []
    step0_texts_by_chain = []
    for c_idx, c in enumerate(chains):
        step0 = c["steps"][0]["output"]
        step0_texts_by_chain.append(step0)
        for step in c["steps"]:
            rows.append({
                "chain_idx": c_idx,
                "term_id": c["term_id"],
                "term_form": c["term_form"],
                "term_type": c["term_type"],
                "style": c["style"],
                "depth": step["step"],
                "output": step["output"],
                "term_present": bool(step["term_in_output_lower"]),
            })

    print(f"  {len(rows)} step records")

    # Embed all step outputs and all step-0 outputs
    print("Embedding outputs...")
    all_outputs = [r["output"] for r in rows]
    output_emb = model.encode(all_outputs, batch_size=32, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
    step0_emb = model.encode(step0_texts_by_chain, batch_size=32, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)

    # Compute cosine(step_N, step_0) for each row
    for i, r in enumerate(rows):
        r["sim_to_step0"] = float(output_emb[i] @ step0_emb[r["chain_idx"]].T)

    # ---- By depth: term presence rate vs mean cosine to step0 ----
    print("\n" + "=" * 72)
    print("Decay by depth (collapsed over all terms × styles)")
    print("=" * 72)
    print(f"  {'depth':>5} {'term_pres':>11} {'sim_to_s0':>11} {'N':>6}")
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[r["depth"]].append((r["term_present"], r["sim_to_step0"]))
    for d in sorted(by_depth):
        items = by_depth[d]
        n = len(items)
        pres = sum(int(t[0]) for t in items) / n
        sim = sum(t[1] for t in items) / n
        print(f"  {d:>5d} {pres:>11.3f} {sim:>11.3f} {n:>6d}")

    # ---- By depth × style ----
    print()
    print("=" * 72)
    print("Decay by depth × style")
    print("=" * 72)
    print(f"  {'style':<5} {'depth':>5} {'term_pres':>11} {'sim_to_s0':>11} {'N':>6}")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[(r["style"], r["depth"])].append((r["term_present"], r["sim_to_step0"]))
    for style in sorted(set(r["style"] for r in rows)):
        for d in sorted(set(r["depth"] for r in rows if r["style"] == style)):
            items = by_ds[(style, d)]
            n = len(items)
            pres = sum(int(t[0]) for t in items) / n
            sim = sum(t[1] for t in items) / n
            print(f"  {style:<5} {d:>5d} {pres:>11.3f} {sim:>11.3f} {n:>6d}")

    # ---- Decay rate comparison ----
    # Fit log(rate) ~ -lambda * depth for each: extract decay constants.
    print()
    print("=" * 72)
    print("Decay rates (fit log(rate) = -lambda * depth, depths 1+)")
    print("=" * 72)
    depths = sorted([d for d in by_depth if d >= 1 and len(by_depth[d]) >= 5])
    if len(depths) >= 3:
        pres_vals = []
        sim_vals = []
        for d in depths:
            items = by_depth[d]
            n = len(items)
            pres = sum(int(t[0]) for t in items) / n
            sim = sum(t[1] for t in items) / n
            pres_vals.append(pres)
            sim_vals.append(sim)
        # Fit log(rate) = -lambda*d + const, only on positive rates
        log_pres = [np.log(p) if p > 0 else None for p in pres_vals]
        valid = [(d, lp) for d, lp in zip(depths, log_pres) if lp is not None]
        if len(valid) >= 3:
            d_arr = np.array([v[0] for v in valid])
            lp_arr = np.array([v[1] for v in valid])
            slope_pres, _ = np.polyfit(d_arr, lp_arr, 1)
            print(f"  Term-presence decay constant (form):    lambda = {-slope_pres:.3f} per depth")

        # Sim decay: fit raw sim (not log) since it doesn't go to 0
        slope_sim, intercept_sim = np.polyfit(depths, sim_vals, 1)
        print(f"  Cosine-to-step0 slope (meaning):        d(sim)/dd = {slope_sim:+.3f} per depth")
        print(f"    (raw, not log; sim doesn't approach 0)")
        print()
        print(f"  Sim values: {[f'{s:.3f}' for s in sim_vals]}")
        print(f"  Pres values: {[f'{p:.3f}' for p in pres_vals]}")

    # ---- Hyphenated vs non-hyphenated decay ----
    print()
    print("=" * 72)
    print("Decay by depth × hyphenated vs non-hyphenated")
    print("=" * 72)
    hyph_types = {"hyph_high": "hyph", "hyph_low": "hyph",
                  "phrase": "non-hyph", "unhyphenated": "non-hyph"}
    by_dh = defaultdict(list)
    for r in rows:
        cat = hyph_types.get(r["term_type"], "other")
        by_dh[(cat, r["depth"])].append((r["term_present"], r["sim_to_step0"]))
    for cat in ["hyph", "non-hyph"]:
        print(f"\n  {cat}:")
        for d in sorted(set(r["depth"] for r in rows)):
            items = by_dh.get((cat, d), [])
            if not items:
                continue
            n = len(items)
            pres = sum(int(t[0]) for t in items) / n
            sim = sum(t[1] for t in items) / n
            print(f"    d{d}: pres={pres:.3f}  sim={sim:.3f}  N={n}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
