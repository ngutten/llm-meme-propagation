#!/usr/bin/env python3
"""Make figure: form vs meaning decay across chain depth.

Two curves on one panel:
  - Term presence: fraction of chains still containing the coined word
  - Cosine to step 0: semantic similarity of each step's output to the
    original first-step output

Form decays to near-zero by depth 5-6. Meaning drops sharply at the first
step, then plateaus well above the baseline for unrelated outputs (~0.15).
Even when the coined word is gone, the chain carries an echo of the
original context.

Reads runs/chain/summary.jsonl (term presence) and embeds outputs from
the trial JSONs (cosine).
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer


CHAIN_DIR = Path(__file__).parent.parent / "runs" / "chain"


def load_chains():
    chains = []
    for p in sorted(CHAIN_DIR.glob("chain_*.json")):
        try:
            with open(p) as f:
                chains.append(json.load(f))
        except json.JSONDecodeError:
            continue
    return chains


def main():
    chains = load_chains()
    print(f"Loaded {len(chains)} chains")
    if not chains:
        return

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build (chain_idx, depth, output, term_present) records, padded to depth=8
    max_depth = 8
    rows = []
    step0_texts = []
    for c_idx, c in enumerate(chains):
        steps = c["steps"]
        if not steps:
            continue
        step0 = steps[0]["output"]
        step0_texts.append(step0)
        for d in range(max_depth):
            if d < len(steps):
                rows.append({
                    "chain_idx": c_idx,
                    "depth": d,
                    "output": steps[d]["output"],
                    "term_present": bool(steps[d]["term_in_output_lower"]),
                    "padded": False,
                })
            else:
                # Early-stopped: term is definitely absent and we have no output
                # to embed. We'll handle this by counting term=False but excluding
                # from the cosine average (since there's no output).
                rows.append({
                    "chain_idx": c_idx,
                    "depth": d,
                    "output": None,
                    "term_present": False,
                    "padded": True,
                })

    # Embed step 0 outputs and all non-padded outputs
    print("Embedding outputs...")
    step0_emb = model.encode(step0_texts, batch_size=32, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)

    non_padded = [r for r in rows if not r["padded"]]
    texts = [r["output"] for r in non_padded]
    embs = model.encode(texts, batch_size=32, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    for r, e in zip(non_padded, embs):
        r["sim_to_step0"] = float(e @ step0_emb[r["chain_idx"]].T)

    # By depth: term presence (with padding) and cosine (without padding)
    depth_pres = defaultdict(list)
    depth_sim = defaultdict(list)
    for r in rows:
        depth_pres[r["depth"]].append(int(r["term_present"]))
        if not r["padded"]:
            depth_sim[r["depth"]].append(r["sim_to_step0"])

    depths = sorted(depth_pres.keys())
    pres_means = [np.mean(depth_pres[d]) for d in depths]
    pres_ses = [
        (m * (1 - m) / len(depth_pres[d])) ** 0.5
        for d, m in zip(depths, pres_means)
    ]
    sim_means = [np.mean(depth_sim[d]) for d in depths]
    sim_ses = [
        np.std(depth_sim[d], ddof=1) / np.sqrt(len(depth_sim[d]))
        if len(depth_sim[d]) > 1 else 0.0
        for d in depths
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    # Cosine similarity (meaning)
    ax.errorbar(
        depths, sim_means, yerr=sim_ses,
        fmt="s-", color="#1c4ec9", markersize=6, linewidth=1.7, capsize=3,
        label="Semantic similarity to original (cosine)",
    )
    # Term presence (form)
    ax.errorbar(
        depths, pres_means, yerr=pres_ses,
        fmt="o-", color="#d97706", markersize=6, linewidth=1.7, capsize=3,
        label="Term still present (verbatim)",
    )

    # Baseline reference
    ax.axhline(0.15, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(7.1, 0.155, "baseline cosine\n(unrelated outputs)",
            fontsize=8.5, color="#666", ha="right", va="bottom")

    ax.set_xlabel("Chain depth (number of agents in sequence)", fontsize=10.5)
    ax.set_ylabel("Fraction / Cosine similarity", fontsize=10.5)
    ax.set_xticks(depths)
    ax.set_ylim(-0.03, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title(
        "Form drops to near-zero in chains; meaning plateaus above baseline.\n"
        "Even after the coined word is gone, chains echo the original context.",
        fontsize=11.5, pad=12,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "form_vs_meaning_decay.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
