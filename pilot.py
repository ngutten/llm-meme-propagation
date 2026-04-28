#!/usr/bin/env python3
"""Pilot run — small matrix to see if the masking-failure pattern holds.

Matrix:
- 3 term classes (C1 random, C2 structural, C3 crafted), 1 term each
- 3 styles (S1, S2, S3)
- 4 C-topics (habits, philosophy, llm_identity, magnets) — span neutral-open and attractor probes
- 2 B registers (neutral, casual_strong) — test whether user-distinctness breaks the pull

Total: 3 × 3 × 4 × 2 = 72 runs at ~60s each = ~72 minutes.
"""

import json
import random
import time
from pathlib import Path

from runner import PROJECT, RUNS_DIR, run_trial, save_log

PILOT_DIR = RUNS_DIR / "pilot"
PILOT_DIR.mkdir(exist_ok=True, parents=True)

# B topic indices by register
B_INDEX = {"neutral": [0, 1], "casual_strong": [2, 3]}

TERM_CLASSES = ["C1", "C2", "C3"]
STYLES = ["S1", "S2", "S3"]
C_TOPICS = ["habits", "philosophy", "llm_identity", "magnets"]
B_REGISTERS = ["neutral", "casual_strong"]


def load_term(term_class, idx=0):
    term_file = PROJECT / "terms" / {
        "C1": "C1_random.json",
        "C2": "C2_structural.json",
        "C3": "C3_crafted.json",
    }[term_class]
    with open(term_file) as f:
        terms = json.load(f)
    return terms[idx]["form"]


def main():
    random.seed(1776886431)

    # Build all trial configs
    trials = []
    for tc in TERM_CLASSES:
        term = load_term(tc, idx=0)
        for style in STYLES:
            for c_topic in C_TOPICS:
                for b_reg in B_REGISTERS:
                    b_idx = random.choice(B_INDEX[b_reg])
                    trials.append({
                        "term_class": tc,
                        "term": term,
                        "style": style,
                        "c_topic": c_topic,
                        "b_register": b_reg,
                        "b_topic_idx": b_idx,
                    })

    # Randomize run order (protects against drift in model state)
    random.shuffle(trials)

    print(f"Running {len(trials)} trials")
    summary_path = PILOT_DIR / "summary.jsonl"
    start = time.time()

    with open(summary_path, "w") as summary_f:
        for i, cfg in enumerate(trials):
            t0 = time.time()
            print(f"[{i+1}/{len(trials)}] {cfg['term_class']}/{cfg['style']}/"
                  f"{cfg['c_topic']}/{cfg['b_register']} ... ", end="", flush=True)
            try:
                log = run_trial(
                    term=cfg["term"],
                    style=cfg["style"],
                    c_topic=cfg["c_topic"],
                    b_topic_idx=cfg["b_topic_idx"],
                )
                log["b_register"] = cfg["b_register"]
                log["term_class"] = cfg["term_class"]

                # Save full log
                full_path = PILOT_DIR / (
                    f"trial_{int(log['start_time'])}_{cfg['term_class']}_"
                    f"{cfg['style']}_{cfg['c_topic']}_{cfg['b_register']}.json"
                )
                save_log(log, full_path)

                # Save summary row
                summary = {
                    "term_class": cfg["term_class"],
                    "term": cfg["term"],
                    "style": cfg["style"],
                    "c_topic": cfg["c_topic"],
                    "b_register": cfg["b_register"],
                    "b_topic_idx": cfg["b_topic_idx"],
                    "duration_s": log["duration_s"],
                    **log["reach"],
                    "path": str(full_path.name),
                }
                summary_f.write(json.dumps(summary) + "\n")
                summary_f.flush()
                print(f"{log['duration_s']:.0f}s  "
                      f"pickup={log['reach']['term_in_A_content_lower']} "
                      f"B={log['reach']['term_in_B_content_lower']} "
                      f"C={log['reach']['term_in_C_content_lower']} "
                      f"Creason={log['reach']['term_in_C_reasoning']}")
            except Exception as e:
                print(f"FAILED: {e}")
                summary_f.write(json.dumps({"error": str(e), "cfg": cfg}) + "\n")
                summary_f.flush()

    elapsed = time.time() - start
    print(f"\nTotal: {elapsed/60:.1f} min")
    print(f"Summary at {summary_path}")


if __name__ == "__main__":
    main()
