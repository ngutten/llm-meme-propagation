#!/usr/bin/env python3
"""
Plots for KL × reach analysis. Same Simpson's-paradox-aware structure as
plot_surprisal.py: histograms by style + scatter colored by style with
within-style and pooled regression lines.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
FIGS = PROJECT / "figures"
FIGS.mkdir(exist_ok=True)

DATA = PROJECT / "runs" / "kl_x_reach.jsonl"

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
        ("sum_log_ratio_nats", "Sum log p(t|with) - log p(t|without)\nover 50-token response (nats)"),
        ("mean_log_ratio_nats", "Mean log-ratio per token (nats)"),
        ("max_log_ratio_nats", "Max single-token log-ratio (nats)"),
    ]
    for ax, (key, label) in zip(axes, measures):
        for style in ["S1", "S2", "S3"]:
            xs = [r[key] for r in rows if r["style"] == style]
            if not xs:
                continue
            ax.hist(xs, bins=15, alpha=0.5, label=STYLE_LABELS[style],
                    color=STYLE_COLORS[style])
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("KL-from-injection distributions by style", fontsize=11)
    fig.tight_layout()
    out = FIGS / "kl_hist_by_style.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def scatter_by_style(rows):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    measures = [
        ("sum_log_ratio_nats", "Sum log-ratio (nats)"),
        ("mean_log_ratio_nats", "Mean log-ratio (nats)"),
        ("max_log_ratio_nats", "Max single-token log-ratio (nats)"),
    ]
    reaches = [
        ("b_reach", "B-reach rate (carrying gate)"),
        ("c_reach", "C-reach rate (output gate)"),
    ]
    for col, (skey, slabel) in enumerate(measures):
        for row, (rkey, rlabel) in enumerate(reaches):
            ax = axes[row, col]
            for style in ["S1", "S2", "S3"]:
                pts = [(r[skey], r[rkey]) for r in rows if r["style"] == style]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, alpha=0.7, s=30, label=STYLE_LABELS[style],
                          color=STYLE_COLORS[style])
                xs_a, ys_a = np.array(xs), np.array(ys)
                if np.std(xs_a) > 0 and len(xs_a) > 2:
                    slope, intercept = np.polyfit(xs_a, ys_a, 1)
                    xline = np.array([xs_a.min(), xs_a.max()])
                    ax.plot(xline, slope * xline + intercept,
                            color=STYLE_COLORS[style], alpha=0.5, linewidth=1.5)
            all_x = np.array([r[skey] for r in rows])
            all_y = np.array([r[rkey] for r in rows])
            if np.std(all_x) > 0 and len(all_x) > 2:
                slope, intercept = np.polyfit(all_x, all_y, 1)
                xline = np.array([all_x.min(), all_x.max()])
                ax.plot(xline, slope * xline + intercept,
                        color="black", linewidth=2, linestyle="--",
                        label="Pooled")
            ax.set_xlabel(slabel)
            ax.set_ylabel(rlabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("KL-from-injection × reach (smoke test, 50 tokens, 1 trial/pair)",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / "kl_x_reach_scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load()
    print(f"Loaded {len(rows)} rows")
    hist_by_style(rows)
    scatter_by_style(rows)


if __name__ == "__main__":
    main()
