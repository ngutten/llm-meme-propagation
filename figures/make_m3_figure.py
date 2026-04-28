#!/usr/bin/env python3
"""Make figure: M3 by-topic, raw vs masked.

Shows that the meaning-carrying effect (reach trials more similar to A than
non-reach trials) is robust in open-register topics but vanishes in
jargon-pulling topics when the term is masked out.

X axis: probe topic
Y axis: cosine-similarity difference (reach - non-reach)
Bar pairs: raw (term included) vs masked (term replaced)

Numbers from m3_extended.py output:
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Numbers from m3_extended.py output, by-topic raw vs masked, reach - non-reach diff
# Each entry: (topic_label, raw_diff, raw_z, masked_diff, masked_z)
data = [
    ("introspective\n(\"what is it like\nto be an LLM\")",       0.115, 8.40, 0.113, 8.92),
    ("habits\n(self-reflection)",                                  0.153, 8.73, 0.120, 8.34),
    ("philosophy\n(analytic vs.\ncontinental)",                    0.127, 5.34, 0.065, 3.77),
    ("magnets\n(technical)",                                        0.075, 4.44, 0.006, 0.56),
]

topics = [d[0] for d in data]
raw = [d[1] for d in data]
masked = [d[3] for d in data]

x = np.arange(len(topics))
w = 0.38

fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
b1 = ax.bar(x - w/2, raw, w, label="Term included", color="#5b8def", edgecolor="white")
b2 = ax.bar(x + w/2, masked, w, label="Term masked out", color="#1c4ec9", edgecolor="white")

ax.set_ylabel("Framing alignment\n(cosine difference: reach − non-reach)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(topics, fontsize=10)
ax.set_ylim(-0.01, 0.18)
ax.axhline(0, color="#444", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_title(
    "When coined words reach a third conversation, the originating framing comes with them —\n"
    "except in destinations with their own established jargon.",
    fontsize=11.5, pad=14,
)

# Annotate the disappearing effect for magnets
ax.annotate(
    "effect vanishes\nwhen term is removed",
    xy=(3 + w/2, 0.006), xytext=(3.05, 0.10),
    fontsize=9, color="#1c4ec9", ha="left",
    arrowprops=dict(arrowstyle="->", color="#1c4ec9", lw=0.8),
)

plt.tight_layout()
out_path = Path(__file__).parent / "m3_by_topic.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved {out_path}")
