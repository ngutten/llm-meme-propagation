#!/usr/bin/env python3
"""M3: downstream-coherence shift.

Tests whether terms that reach in C3 drag C1 framing along (capture-shape)
or propagate as form without specific meaning (social-signal-shape).

Two comparisons:
  M3a. reach vs non-reach on similarity(C_output, A_output_same_trial).
       If reach > non-reach: term carries C1 context when it propagates.
  M3b. on reach trials only: similarity(C_output, A_output_same_term) vs
       mean similarity(C_output, A_output_other_term_same_class).
       If own > other: specific-meaning carrying. If own ~ other: form-only.

Uses all-MiniLM-L6-v2 by default. Loads overnight runs from runs/overnight/.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


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


def extract_outputs(trial_path):
    """Return (a_output, c_output) — concatenated agent responses in A and C."""
    with open(RUNS_DIR / trial_path) as f:
        d = json.load(f)
    a_chunks, c_chunks = [], []
    for t in d["transcript"]:
        if t["role"] != "assistant":
            continue
        label = t["label"]
        if label.startswith("A_"):
            a_chunks.append(t["content"])
        elif label.startswith("C_"):
            c_chunks.append(t["content"])
    return " ".join(a_chunks), " ".join(c_chunks)


def main():
    print("Loading overnight records...")
    records = load_records()
    print(f"  {len(records)} successful trials")

    print("Extracting A and C outputs from trial files...")
    a_texts, c_texts = [], []
    meta = []
    for r in records:
        a, c = extract_outputs(r["path"])
        a_texts.append(a)
        c_texts.append(c)
        meta.append({
            "term": r["term"],
            "term_class": r["term_class"],
            "style": r["style"],
            "c_topic": r["c_topic"],
            "rep": r["rep"],
            "reach_content": bool(r["term_in_C_content_lower"]),
            "reach_reasoning": bool(r["term_in_C_reasoning"]),
        })

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding A outputs...")
    a_emb = model.encode(a_texts, batch_size=32, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    print("Embedding C outputs...")
    c_emb = model.encode(c_texts, batch_size=32, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)

    n = len(meta)
    # similarity_own[i] = cosine(c_emb[i], a_emb[i])  (paired same-trial)
    sim_own = np.einsum("ij,ij->i", c_emb, a_emb)

    # For each trial i, compute mean similarity to A outputs of other trials
    # of same term_class but different term. This is the "other-term in same class"
    # baseline.
    print("Computing same-class-other-term baselines...")
    by_class = defaultdict(list)
    for i, m in enumerate(meta):
        by_class[m["term_class"]].append(i)

    sim_other = np.zeros(n)
    for i, m in enumerate(meta):
        candidates = [j for j in by_class[m["term_class"]]
                      if meta[j]["term"] != m["term"]]
        if not candidates:
            sim_other[i] = np.nan
            continue
        # mean cosine of c_emb[i] with a_emb[candidates]
        sims = c_emb[i] @ a_emb[candidates].T
        sim_other[i] = float(np.mean(sims))

    # Also: same-term-other-trial similarity (as a third reference)
    # captures within-term consistency of A outputs vs same-term C output
    print("Computing same-term-other-trial baselines...")
    by_term = defaultdict(list)
    for i, m in enumerate(meta):
        by_term[m["term"]].append(i)
    sim_same_term = np.zeros(n)
    for i, m in enumerate(meta):
        candidates = [j for j in by_term[m["term"]] if j != i]
        if not candidates:
            sim_same_term[i] = np.nan
            continue
        sims = c_emb[i] @ a_emb[candidates].T
        sim_same_term[i] = float(np.mean(sims))

    # ---- Analysis ----

    def stats(vals):
        vals = np.asarray(vals)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
        }

    reach_idx = np.array([m["reach_content"] for m in meta])
    print("\n" + "=" * 70)
    print("M3a: reach vs non-reach on similarity(C, A_same_trial)")
    print("=" * 70)
    s_r = stats(sim_own[reach_idx])
    s_nr = stats(sim_own[~reach_idx])
    print(f"  reach=True : mean={s_r['mean']:.4f} ± {s_r['se']:.4f}  (N={s_r['n']})")
    print(f"  reach=False: mean={s_nr['mean']:.4f} ± {s_nr['se']:.4f}  (N={s_nr['n']})")
    diff = s_r["mean"] - s_nr["mean"]
    se_diff = (s_r["se"] ** 2 + s_nr["se"] ** 2) ** 0.5
    print(f"  diff (reach-nonreach) = {diff:+.4f}  (SE diff = {se_diff:.4f}, "
          f"z = {diff/se_diff:+.2f})")

    print("\n" + "=" * 70)
    print("M3b (reach trials only): own vs other-term-same-class")
    print("=" * 70)
    own_r = sim_own[reach_idx]
    other_r = sim_other[reach_idx]
    s_o = stats(own_r)
    s_ot = stats(other_r)
    paired = own_r - other_r
    print(f"  own (paired)         : mean={s_o['mean']:.4f} ± {s_o['se']:.4f}")
    print(f"  other-term-same-class: mean={s_ot['mean']:.4f} ± {s_ot['se']:.4f}")
    s_p = stats(paired)
    print(f"  paired diff (own-other) = {s_p['mean']:+.4f} ± {s_p['se']:.4f}  "
          f"(z paired = {s_p['mean']/s_p['se']:+.2f}, N={s_p['n']})")

    print("\n" + "=" * 70)
    print("M3a by term_class × style (similarity_own; reach vs non-reach)")
    print("=" * 70)
    print(f"  {'class':<6} {'style':<5} {'reach_mean':>12} {'nonreach_mean':>15} "
          f"{'diff':>10} {'z':>8}")
    for tc in ["C1", "C2", "C3"]:
        for s in ["S1", "S2", "S3"]:
            mask_cell = np.array([m["term_class"] == tc and m["style"] == s
                                  for m in meta])
            r_mask = mask_cell & reach_idx
            nr_mask = mask_cell & ~reach_idx
            r_s = stats(sim_own[r_mask])
            nr_s = stats(sim_own[nr_mask])
            if r_s and nr_s:
                d = r_s["mean"] - nr_s["mean"]
                se = (r_s["se"] ** 2 + nr_s["se"] ** 2) ** 0.5
                z = d / se if se > 0 else 0
                print(f"  {tc:<6} {s:<5} {r_s['mean']:>10.3f}(N={r_s['n']:3d}) "
                      f"{nr_s['mean']:>11.3f}(N={nr_s['n']:3d}) {d:>+10.3f} {z:>+8.2f}")
            else:
                print(f"  {tc:<6} {s:<5}  (insufficient data)")

    print("\n" + "=" * 70)
    print("M3b by term_class (paired own-other on reach trials)")
    print("=" * 70)
    print(f"  {'class':<6} {'own_mean':>10} {'other_mean':>11} {'diff':>10} "
          f"{'z':>8} {'N':>6}")
    for tc in ["C1", "C2", "C3"]:
        mask = np.array([m["term_class"] == tc and m["reach_content"]
                         for m in meta])
        own_v = sim_own[mask]
        other_v = sim_other[mask]
        diff_v = own_v - other_v
        s_p = stats(diff_v)
        s_o = stats(own_v)
        s_ot = stats(other_v)
        if s_p:
            print(f"  {tc:<6} {s_o['mean']:>10.4f} {s_ot['mean']:>11.4f} "
                  f"{s_p['mean']:>+10.4f} {s_p['mean']/s_p['se']:>+8.2f} "
                  f"{s_p['n']:>6d}")

    print("\n" + "=" * 70)
    print("Sanity: same-term-other-trial similarity (should be high, like own)")
    print("=" * 70)
    s_st = stats(sim_same_term)
    s_o = stats(sim_own)
    s_ot = stats(sim_other)
    print(f"  sim(C, A_same_trial)               : {s_o['mean']:.4f} (N={s_o['n']})")
    print(f"  sim(C, A_same_term_other_trial)    : {s_st['mean']:.4f} (N={s_st['n']})")
    print(f"  sim(C, A_other_term_same_class)    : {s_ot['mean']:.4f} (N={s_ot['n']})")

    # Save raw arrays for further analysis if needed
    np.savez("runs/overnight/m3_results.npz",
             sim_own=sim_own, sim_other=sim_other, sim_same_term=sim_same_term,
             reach_content=reach_idx,
             reach_reasoning=np.array([m["reach_reasoning"] for m in meta]),
             term_class=np.array([m["term_class"] for m in meta]),
             term=np.array([m["term"] for m in meta]),
             style=np.array([m["style"] for m in meta]),
             c_topic=np.array([m["c_topic"] for m in meta]))
    print("\nSaved arrays to runs/overnight/m3_results.npz")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
