#!/usr/bin/env python3
"""Overnight full-matrix experiment runner.

Matrix (v1 scope, ~18 hours of compute on a local Gemma-26B A4B):
- 3 term classes × 10 terms per class  = 30 terms
- 3 surrounding styles (S1, S2, S3)
- 4 C-topics (habits, llm_identity, philosophy, magnets)
- 1 B-register (neutral — pilot showed no effect of register switch)
- N = 3 replicates per cell

Total: 30 × 3 × 4 × 3 = 1080 runs at ~60s each = ~18 hours.

Interruptible / resumable:
- Progress is checkpointed to runs/overnight/summary.jsonl (appended after each trial).
- On restart, script reads summary.jsonl, identifies already-completed (term, style, c_topic, rep)
  keys, and skips them.
- Errored trials ARE re-attempted on resume (transient llama-server issues etc).
- To start fresh, delete runs/overnight/ and re-run.

Usage:
    uv run python overnight.py [--scope small]

--scope small runs a reduced matrix (1 term per class, 2 topics, N=2) as a smoke test.
"""

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

from runner import PROJECT, RUNS_DIR, run_trial, save_log

OVERNIGHT_DIR = RUNS_DIR / "overnight"
OVERNIGHT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_PATH = OVERNIGHT_DIR / "summary.jsonl"

TERM_CLASSES = ["C1", "C2", "C3"]
STYLES = ["S1", "S2", "S3"]
C_TOPICS = ["habits", "llm_identity", "philosophy", "magnets"]
B_REGISTER = "neutral"  # pilot null, picking this one for production realism
B_TOPIC_IDX = 0  # within the neutral register, first topic (dinner party)
N_REP = 3

TERM_FILES = {
    "C1": "C1_random.json",
    "C2": "C2_structural.json",
    "C3": "C3_crafted.json",
}


# --- graceful interrupt handling ---
_interrupt_requested = False


def _handle_sigint(signum, frame):
    global _interrupt_requested
    if _interrupt_requested:
        print("\n[Second interrupt — exiting immediately.]", flush=True)
        sys.exit(1)
    _interrupt_requested = True
    print(
        "\n[Interrupt requested — will finish current trial then stop. "
        "Ctrl-C again to abort immediately.]",
        flush=True,
    )


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def load_terms():
    """Load all terms from the three class files."""
    all_terms = []
    for cls in TERM_CLASSES:
        with open(PROJECT / "terms" / TERM_FILES[cls]) as f:
            terms = json.load(f)
        for idx, t in enumerate(terms):
            all_terms.append({"term_class": cls, "term_idx": idx, "term": t["form"]})
    return all_terms


def trial_key(term_class, term, style, c_topic, rep):
    """Canonical identifier for a trial (used for resume)."""
    return f"{term_class}|{term}|{style}|{c_topic}|rep{rep}"


def load_completed_keys():
    """Read summary.jsonl and return the set of successfully-completed trial keys."""
    done = set()
    if not SUMMARY_PATH.exists():
        return done
    with open(SUMMARY_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines — could happen on interrupt mid-write
                continue
            if "error" in r:
                continue  # errored trials are not "done", will retry
            k = trial_key(
                r["term_class"], r["term"], r["style"], r["c_topic"], r["rep"]
            )
            done.add(k)
    return done


def build_matrix(terms_list, n_rep=N_REP, scope="full"):
    """Build the full experiment matrix as a list of trial configs.

    Scope "full" = entire matrix; "small" = smoke test.
    """
    if scope == "small":
        # Smoke test: 1 term per class × 2 styles × 1 topic × 1 rep = 6 trials, ~6 min
        terms_list = [terms_list[0], terms_list[10], terms_list[20]]  # one per class
        c_topics = C_TOPICS[:1]
        n_rep = 1
        global STYLES
        STYLES = ["S1", "S3"]
    else:
        c_topics = C_TOPICS

    trials = []
    for term_info in terms_list:
        for style in STYLES:
            for c_topic in c_topics:
                for rep in range(n_rep):
                    trials.append({
                        "term_class": term_info["term_class"],
                        "term_idx": term_info["term_idx"],
                        "term": term_info["term"],
                        "style": style,
                        "c_topic": c_topic,
                        "b_register": B_REGISTER,
                        "b_topic_idx": B_TOPIC_IDX,
                        "rep": rep,
                    })
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["full", "small"], default="full")
    parser.add_argument("--seed", type=int, default=1776891000,
                        help="Seed for shuffling trial order (ensures reproducibility).")
    args = parser.parse_args()

    terms_list = load_terms()
    trials = build_matrix(terms_list, scope=args.scope)

    # Deterministic shuffle so the interleaving of classes/styles is consistent across resumes
    rng = random.Random(args.seed)
    rng.shuffle(trials)

    # Resume logic
    done_keys = load_completed_keys()
    remaining = [
        c for c in trials
        if trial_key(c["term_class"], c["term"], c["style"], c["c_topic"], c["rep"]) not in done_keys
    ]

    print(f"Matrix: {len(trials)} total trials.")
    print(f"Already done (on disk): {len(done_keys)}")
    print(f"Remaining to run this session: {len(remaining)}")
    if not remaining:
        print("Nothing to do — matrix already complete.")
        return

    est_sec = 62 * len(remaining)
    print(f"Estimated runtime this session: {est_sec/3600:.1f} hours "
          f"({est_sec/60:.1f} min).")

    # Open summary file in append mode (resume-safe — new lines append, no reset)
    start = time.time()
    with open(SUMMARY_PATH, "a") as summary_f:
        for i, cfg in enumerate(remaining):
            if _interrupt_requested:
                print(f"[Interrupt acknowledged — stopping after {i}/{len(remaining)} trials.]")
                break

            key = trial_key(cfg["term_class"], cfg["term"], cfg["style"],
                            cfg["c_topic"], cfg["rep"])
            wall_elapsed_min = (time.time() - start) / 60
            print(f"[{i+1}/{len(remaining)}] ({wall_elapsed_min:.0f}m) {key} ... ",
                  end="", flush=True)

            try:
                log = run_trial(
                    term=cfg["term"],
                    style=cfg["style"],
                    c_topic=cfg["c_topic"],
                    b_topic_idx=cfg["b_topic_idx"],
                    max_tokens=10240,  # generous budget — reasoning alone can reach ~4k chars, leave headroom
                )
                log["b_register"] = cfg["b_register"]
                log["term_class"] = cfg["term_class"]
                log["rep"] = cfg["rep"]

                # Save full trial log
                full_path = OVERNIGHT_DIR / (
                    f"trial_{int(log['start_time'])}_{cfg['term_class']}_"
                    f"idx{cfg['term_idx']}_{cfg['style']}_{cfg['c_topic']}_rep{cfg['rep']}.json"
                )
                save_log(log, full_path)

                # Append summary row
                row = {
                    "term_class": cfg["term_class"],
                    "term_idx": cfg["term_idx"],
                    "term": cfg["term"],
                    "style": cfg["style"],
                    "c_topic": cfg["c_topic"],
                    "b_register": cfg["b_register"],
                    "rep": cfg["rep"],
                    "duration_s": log["duration_s"],
                    **log["reach"],
                    "path": full_path.name,
                }
                summary_f.write(json.dumps(row) + "\n")
                summary_f.flush()  # critical for resume safety

                r = log["reach"]
                print(f"{log['duration_s']:.0f}s  "
                      f"A={int(r['term_in_A_content_lower'])} "
                      f"B={int(r['term_in_B_content_lower'])} "
                      f"C={int(r['term_in_C_content_lower'])} "
                      f"Creason={int(r['term_in_C_reasoning'])}")

            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
                err_row = {
                    "error": f"{type(e).__name__}: {e}",
                    "cfg": cfg,
                    "timestamp": time.time(),
                }
                summary_f.write(json.dumps(err_row) + "\n")
                summary_f.flush()
                # don't raise — keep going; errored trials can be retried on next resume

    elapsed = time.time() - start
    done_now = len(load_completed_keys())
    print(f"\nSession elapsed: {elapsed/60:.1f} min.")
    print(f"Completed on disk: {done_now}/{len(trials)} trials total.")
    print(f"Remaining: {len(trials) - done_now}.")


if __name__ == "__main__":
    main()
