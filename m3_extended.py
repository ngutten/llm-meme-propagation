#!/usr/bin/env python3
"""Extended M3 analysis on overnight data.

Four analyses:
  (1) Term-masked M3 — re-embed with the term token replaced, to test
      whether the M3 effect persists when the term itself is removed.
      The decisive control: if effect disappears, M3 was tautological.
  (2) By-topic M3 — does the effect hold within each probe topic?
      Tests for topic-overlap confound.
  (3) By-term M3 — does the meaning-carrying effect vary across the
      30 terms? Top-reach terms might carry less meaning per propagation.
  (4) B-leak — does the term appear in C2 (the masking conversation)?
      And does B-leak predict reach?
"""

import json
import os
import re
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
    with open(RUNS_DIR / trial_path) as f:
        d = json.load(f)
    a, b, c = [], [], []
    for t in d["transcript"]:
        if t["role"] != "assistant":
            continue
        label = t["label"]
        if label.startswith("A_"):
            a.append(t["content"])
        elif label.startswith("B_"):
            b.append(t["content"])
        elif label.startswith("C_"):
            c.append(t["content"])
    return " ".join(a), " ".join(b), " ".join(c)


def mask_term(text, term):
    """Replace term (and stem) with placeholder, case-insensitive."""
    # Replace verbatim with hyphen, and also stems (e.g. plurals)
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    out = pattern.sub("[TERM]", text)
    # Also strip lowercase-no-hyphen variants
    no_hyphen = term.replace("-", " ")
    pattern2 = re.compile(re.escape(no_hyphen), re.IGNORECASE)
    out = pattern2.sub("[TERM]", out)
    return out


def stats(vals):
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "se": float(np.std(vals, ddof=1) / np.sqrt(max(1, len(vals) - 1))) if len(vals) > 1 else 0.0,
    }


def diff_z(s_a, s_b):
    if s_a is None or s_b is None:
        return None, None, None
    diff = s_a["mean"] - s_b["mean"]
    se = (s_a["se"] ** 2 + s_b["se"] ** 2) ** 0.5
    z = diff / se if se > 0 else 0
    return diff, se, z


def main():
    print("Loading overnight records...")
    records = load_records()
    print(f"  {len(records)} successful trials")

    print("Extracting A, B, C outputs from trial files...")
    a_texts, b_texts, c_texts = [], [], []
    a_masked, b_masked, c_masked = [], [], []
    meta = []
    for r in records:
        a, b, c = extract_outputs(r["path"])
        term = r["term"]
        a_texts.append(a)
        b_texts.append(b)
        c_texts.append(c)
        a_masked.append(mask_term(a, term))
        b_masked.append(mask_term(b, term))
        c_masked.append(mask_term(c, term))
        # Compute B-leak from masked text presence (cheap pre-check: did term appear in B at all?)
        b_lower = b.lower()
        b_has_term = term.lower() in b_lower
        meta.append({
            "term": r["term"],
            "term_class": r["term_class"],
            "style": r["style"],
            "c_topic": r["c_topic"],
            "rep": r["rep"],
            "reach_content": bool(r["term_in_C_content_lower"]),
            "reach_reasoning": bool(r["term_in_C_reasoning"]),
            "b_has_term": b_has_term,
            "b_term_in_summary": bool(r.get("term_in_B_content_lower", False)),
        })

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding raw A, C outputs...")
    a_emb = model.encode(a_texts, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    c_emb = model.encode(c_texts, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    print("Embedding masked A, C outputs...")
    a_emb_m = model.encode(a_masked, batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    c_emb_m = model.encode(c_masked, batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)

    n = len(meta)
    sim_own = np.einsum("ij,ij->i", c_emb, a_emb)
    sim_own_masked = np.einsum("ij,ij->i", c_emb_m, a_emb_m)

    by_class = defaultdict(list)
    by_term = defaultdict(list)
    for i, m in enumerate(meta):
        by_class[m["term_class"]].append(i)
        by_term[m["term"]].append(i)

    print("Computing same-class-other-term baselines (raw + masked)...")
    sim_other = np.zeros(n)
    sim_other_masked = np.zeros(n)
    for i, m in enumerate(meta):
        candidates = [j for j in by_class[m["term_class"]]
                      if meta[j]["term"] != m["term"]]
        if candidates:
            sim_other[i] = float(np.mean(c_emb[i] @ a_emb[candidates].T))
            sim_other_masked[i] = float(np.mean(c_emb_m[i] @ a_emb_m[candidates].T))
        else:
            sim_other[i] = sim_other_masked[i] = np.nan

    reach_idx = np.array([m["reach_content"] for m in meta])

    # ============================================================
    # (1) Term-masked M3
    # ============================================================
    print("\n" + "=" * 70)
    print("(1) Term-masked M3: term replaced with [TERM] in A and C before embedding")
    print("=" * 70)

    print("\n  M3a (reach vs non-reach on sim_own):")
    print(f"  {'':<20} {'raw':>22} {'masked':>22}")
    for label, mask in [("reach=True", reach_idx), ("reach=False", ~reach_idx)]:
        sr = stats(sim_own[mask])
        sm = stats(sim_own_masked[mask])
        print(f"  {label:<20} {sr['mean']:.4f}±{sr['se']:.4f}(N={sr['n']:4d})  "
              f"{sm['mean']:.4f}±{sm['se']:.4f}(N={sm['n']:4d})")
    raw_diff, raw_se, raw_z = diff_z(stats(sim_own[reach_idx]),
                                     stats(sim_own[~reach_idx]))
    msk_diff, msk_se, msk_z = diff_z(stats(sim_own_masked[reach_idx]),
                                     stats(sim_own_masked[~reach_idx]))
    print(f"  {'reach-nonreach diff':<20} "
          f"{raw_diff:+.4f}              z={raw_z:+.2f}    "
          f"{msk_diff:+.4f}              z={msk_z:+.2f}")
    if msk_z and raw_z:
        print(f"  attenuation: z dropped from {raw_z:+.2f} to {msk_z:+.2f} "
              f"(remaining {100 * msk_z / raw_z:.0f}%)")

    print("\n  M3b (reach trials only, own-other paired diff):")
    for label, mask in [("reach=True", reach_idx)]:
        own_r_raw = sim_own[mask]
        oth_r_raw = sim_other[mask]
        own_r_msk = sim_own_masked[mask]
        oth_r_msk = sim_other_masked[mask]
        diff_raw = own_r_raw - oth_r_raw
        diff_msk = own_r_msk - oth_r_msk
        s_raw = stats(diff_raw)
        s_msk = stats(diff_msk)
        print(f"  {label:<12}  raw diff = {s_raw['mean']:+.4f} ± {s_raw['se']:.4f}  z={s_raw['mean']/s_raw['se']:+.2f}   N={s_raw['n']}")
        print(f"  {label:<12}  msk diff = {s_msk['mean']:+.4f} ± {s_msk['se']:.4f}  z={s_msk['mean']/s_msk['se']:+.2f}")
        print(f"  attenuation: z dropped from {s_raw['mean']/s_raw['se']:+.2f} to "
              f"{s_msk['mean']/s_msk['se']:+.2f} "
              f"(remaining {100 * (s_msk['mean']/s_msk['se']) / (s_raw['mean']/s_raw['se']):.0f}%)")

    # ============================================================
    # (2) By-topic M3 (raw)
    # ============================================================
    print("\n" + "=" * 70)
    print("(2) By-topic M3 (raw sim_own, reach vs non-reach within topic)")
    print("=" * 70)
    print(f"  {'topic':<15} {'reach_mean':>12} {'nonreach_mean':>15} {'diff':>10} {'z':>8} {'N_reach':>10}")
    topics = sorted(set(m["c_topic"] for m in meta))
    for t in topics:
        cell = np.array([m["c_topic"] == t for m in meta])
        sr = stats(sim_own[cell & reach_idx])
        snr = stats(sim_own[cell & ~reach_idx])
        if sr and snr:
            d, se, z = diff_z(sr, snr)
            print(f"  {t:<15} {sr['mean']:>10.3f}   {snr['mean']:>13.3f}    "
                  f"{d:>+9.3f}  {z:>+7.2f}   {sr['n']:>10d}")

    # Same for masked
    print("\n  Same, term-masked:")
    print(f"  {'topic':<15} {'reach_mean':>12} {'nonreach_mean':>15} {'diff':>10} {'z':>8} {'N_reach':>10}")
    for t in topics:
        cell = np.array([m["c_topic"] == t for m in meta])
        sr = stats(sim_own_masked[cell & reach_idx])
        snr = stats(sim_own_masked[cell & ~reach_idx])
        if sr and snr:
            d, se, z = diff_z(sr, snr)
            print(f"  {t:<15} {sr['mean']:>10.3f}   {snr['mean']:>13.3f}    "
                  f"{d:>+9.3f}  {z:>+7.2f}   {sr['n']:>10d}")

    # ============================================================
    # (3) By-term M3
    # ============================================================
    print("\n" + "=" * 70)
    print("(3) By-term M3 (paired own-other on reach trials, masked)")
    print("=" * 70)
    print(f"  {'term':<28} {'class':<5} {'reach_rate':>10} "
          f"{'msk_diff':>10} {'z':>8} {'N':>5}")
    rows = []
    for term in sorted(by_term):
        cell = np.array([m["term"] == term and m["reach_content"] for m in meta])
        if cell.sum() < 3:
            continue
        own_v = sim_own_masked[cell]
        oth_v = sim_other_masked[cell]
        diff_v = own_v - oth_v
        s = stats(diff_v)
        tcls = next(m["term_class"] for m in meta if m["term"] == term)
        all_cell = np.array([m["term"] == term for m in meta])
        reach_rate = float(np.mean(reach_idx[all_cell]))
        rows.append((term, tcls, reach_rate, s["mean"], s["mean"]/s["se"] if s["se"] > 0 else 0, s["n"]))
    rows.sort(key=lambda x: -x[3])  # by msk_diff
    for term, tcls, rate, d, z, ne in rows:
        print(f"  {term:<28} {tcls:<5} {rate:>10.2f} {d:>+10.4f} {z:>+8.2f} {ne:>5d}")

    # Correlation between reach rate and meaning-carrying
    if len(rows) >= 5:
        rates = np.array([r[2] for r in rows])
        diffs = np.array([r[3] for r in rows])
        corr = float(np.corrcoef(rates, diffs)[0, 1])
        print(f"\n  Correlation(reach_rate, masked own-other diff) = {corr:+.3f}")
        print("  (negative would mean: terms that propagate more carry less meaning)")

    # ============================================================
    # (4) B-leak
    # ============================================================
    print("\n" + "=" * 70)
    print("(4) B-leak: term appearance in C2 (the masking conversation)")
    print("=" * 70)
    b_leak_count = sum(m["b_has_term"] for m in meta)
    print(f"  Trials with term in B: {b_leak_count}/{len(meta)} ({100*b_leak_count/len(meta):.1f}%)")
    print(f"  (Reference: term-in-C reach rate = {100*reach_idx.mean():.1f}%)")

    # B-leak by reach
    print(f"\n  P(term in B | reach=True)  = {sum(1 for m in meta if m['b_has_term'] and m['reach_content']) / max(1, sum(1 for m in meta if m['reach_content'])):.3f}")
    print(f"  P(term in B | reach=False) = {sum(1 for m in meta if m['b_has_term'] and not m['reach_content']) / max(1, sum(1 for m in meta if not m['reach_content'])):.3f}")

    # B-leak by style and term-class
    print(f"\n  B-leak rate by style x term_class:")
    print(f"  {'':<6} {'S1':>8} {'S2':>8} {'S3':>8}")
    for tc in ["C1", "C2", "C3"]:
        row = []
        for s in ["S1", "S2", "S3"]:
            cell = [m for m in meta if m["term_class"] == tc and m["style"] == s]
            rate = sum(m["b_has_term"] for m in cell) / max(1, len(cell))
            row.append(f"{rate:>.3f}")
        print(f"  {tc:<6} " + " ".join(f"{c:>8}" for c in row))

    # B-leak prediction of reach
    print(f"\n  Reach rate conditional on B-leak status:")
    bm = np.array([m["b_has_term"] for m in meta])
    print(f"    P(reach=True | b_has_term) = {reach_idx[bm].mean():.3f}  (N={bm.sum()})")
    print(f"    P(reach=True | not b_has)  = {reach_idx[~bm].mean():.3f}  (N={(~bm).sum()})")
    rb_diff = reach_idx[bm].mean() - reach_idx[~bm].mean()
    print(f"    diff = {rb_diff:+.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
