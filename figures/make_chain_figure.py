#!/usr/bin/env python3
"""Make figure: chain decay — fraction of chains still containing the term
at each depth, separated by introducing style and by hyphenated vs phrase form.

Reads from runs/chain/summary.jsonl. Run after chain_experiment.py finishes.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SUMMARY = Path(__file__).parent.parent / "runs" / "chain" / "summary.jsonl"

# Map term_id to (style-independent) form category
HYPH_TYPES = {
    "hyph_high": "hyphenated",
    "hyph_low":  "hyphenated",
    "phrase":      "phrase",
    "unhyphenated": "phrase",
}


def load():
    recs = []
    if not SUMMARY.exists():
        return recs
    with open(SUMMARY) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                continue
            recs.append(d)
    return recs


def presence_curves(recs, max_depth=8):
    """Return dict {(form_category, style): {depth: (mean_present, n)}}.

    Pads early-stopped chains with 0 (term absent) for depths after the stop —
    early-stop fires precisely when the term has gone extinct, so the missing
    depths are unambiguous absences. Without this, late-depth data is biased
    toward the few surviving chains that kept the term alive.
    """
    bins = defaultdict(lambda: defaultdict(list))
    for r in recs:
        cat = HYPH_TYPES.get(r.get("term_type", ""))
        if not cat:
            continue
        style = r["style"]
        steps = list(r.get("step_terms") or [])
        # Pad to max_depth: early-stopped chains have implicit 0s after the stop.
        steps += [False] * (max_depth - len(steps))
        for d, present in enumerate(steps):
            bins[(cat, style)][d].append(int(present))
    out = {}
    for key, d_to_vals in bins.items():
        out[key] = {d: (np.mean(v), len(v)) for d, v in d_to_vals.items()}
    return out


def main():
    recs = load()
    if not recs:
        print(f"No chain data at {SUMMARY}")
        return
    print(f"Loaded {len(recs)} chains")

    curves = presence_curves(recs)
    if not curves:
        print("No usable curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=150, sharey=True)

    style_to_ax = {"S1": axes[0], "S3": axes[1]}
    style_titles = {
        "S1": "casual introducing voice (S1)",
        "S3": "elaborate confident voice (S3)",
    }
    cat_styles = {
        "hyphenated": {"color": "#5b8def", "marker": "o", "label": "hyphenated form"},
        "phrase":     {"color": "#d97706", "marker": "s", "label": "non-hyphenated phrase"},
    }

    for style in ["S1", "S3"]:
        ax = style_to_ax.get(style)
        if ax is None:
            continue
        for cat in ["hyphenated", "phrase"]:
            key = (cat, style)
            if key not in curves:
                continue
            ds = sorted(curves[key].keys())
            means = [curves[key][d][0] for d in ds]
            ns = [curves[key][d][1] for d in ds]
            ses = [(m * (1 - m) / max(1, n)) ** 0.5 for m, n in zip(means, ns)]
            sty = cat_styles[cat]
            ax.errorbar(ds, means, yerr=ses, fmt=sty["marker"] + "-",
                        color=sty["color"], label=sty["label"],
                        markersize=6, linewidth=1.6, capsize=3, alpha=0.95)
        ax.set_xlabel("Chain depth (number of agents in sequence)", fontsize=10.5)
        ax.set_title(style_titles[style], fontsize=11)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(range(8))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=9.5)

    axes[0].set_ylabel(
        "Fraction of chains still containing the term", fontsize=10.5
    )
    fig.suptitle(
        "Coined words decay across chains of fresh agents — "
        "regardless of form and introducing style.",
        fontsize=11.5, y=1.02,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "chain_decay.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
