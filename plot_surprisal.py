#!/usr/bin/env python3
"""
Plots for the surprisal × reach analysis. Visualizing Simpson's paradox:
the between-style structure creates an apparent overall correlation that
disappears or reverses when stratified by style.

Outputs:
  figures/surprisal_hist_by_style.png   — surprisal distribution by style
  figures/surprisal_x_reach_scatter.png — scatter colored by style
  figures/reach_by_style.png            — reach distributions by style
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
FIGS = PROJECT / "figures"
FIGS.mkdir(exist_ok=True)

DATA = PROJECT / "runs" / "surprisal_x_reach.jsonl"

STYLE_COLORS = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c"}
STYLE_LABELS = {"S1": "S1 (casual)", "S2": "S2 (analyst)", "S3": "S3 (dense)"}


def load():
    rows = []
    with open(DATA) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def hist_by_style(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    measures = [
        ("mean_per_token_surprisal_nats", "Mean per-token surprisal (nats)"),
        ("max_per_token_surprisal_nats", "Max per-token surprisal (nats)"),
        ("first_token_surprisal_nats", "First-token surprisal (nats)"),
    ]
    for ax, (key, label) in zip(axes, measures):
        for style in ["S1", "S2", "S3"]:
            xs = [r[key] for r in rows if r["style"] == style]
            ax.hist(xs, bins=15, alpha=0.5, label=STYLE_LABELS[style],
                    color=STYLE_COLORS[style])
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("Surprisal distributions by style — the X-axis confound",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / "surprisal_hist_by_style.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def scatter_by_style(rows):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    measures = [
        ("mean_per_token_surprisal_nats", "Mean per-token surprisal"),
        ("max_per_token_surprisal_nats", "Max per-token surprisal"),
        ("first_token_surprisal_nats", "First-token surprisal"),
    ]
    reaches = [
        ("b_reach_rate", "B-reach rate (carrying gate)"),
        ("c_reach_rate", "C-reach rate (output gate)"),
    ]
    for col, (skey, slabel) in enumerate(measures):
        for row, (rkey, rlabel) in enumerate(reaches):
            ax = axes[row, col]
            # Scatter by style
            for style in ["S1", "S2", "S3"]:
                pts = [(r[skey], r[rkey]) for r in rows if r["style"] == style]
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, alpha=0.7, s=30, label=STYLE_LABELS[style],
                          color=STYLE_COLORS[style])
                # Within-style regression line
                xs_a, ys_a = np.array(xs), np.array(ys)
                if np.std(xs_a) > 0:
                    slope, intercept = np.polyfit(xs_a, ys_a, 1)
                    xline = np.array([xs_a.min(), xs_a.max()])
                    ax.plot(xline, slope * xline + intercept,
                            color=STYLE_COLORS[style], alpha=0.5, linewidth=1.5)
            # Overall regression line (the apparent correlation)
            all_x = np.array([r[skey] for r in rows])
            all_y = np.array([r[rkey] for r in rows])
            slope, intercept = np.polyfit(all_x, all_y, 1)
            xline = np.array([all_x.min(), all_x.max()])
            ax.plot(xline, slope * xline + intercept,
                    color="black", linewidth=2, linestyle="--",
                    label="Overall (Simpson's pooled)")
            ax.set_xlabel(slabel)
            ax.set_ylabel(rlabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Simpson's paradox: pooled correlation (dashed black) vs "
                 "within-style correlations (colored lines)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "surprisal_x_reach_scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def reach_by_style(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in [
        (axes[0], "b_reach_rate", "B-reach rate"),
        (axes[1], "c_reach_rate", "C-reach rate"),
    ]:
        data = []
        labels = []
        colors = []
        for style in ["S1", "S2", "S3"]:
            ys = [r[key] for r in rows if r["style"] == style]
            data.append(ys)
            labels.append(STYLE_LABELS[style])
            colors.append(STYLE_COLORS[style])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Reach distributions by style — the Y-axis confound",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / "reach_by_style.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load()
    print(f"Loaded {len(rows)} rows")
    hist_by_style(rows)
    scatter_by_style(rows)
    reach_by_style(rows)


if __name__ == "__main__":
    main()
