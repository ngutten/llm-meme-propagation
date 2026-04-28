#!/usr/bin/env python3
"""Pattern vs instance suppression experiment.

Question: does priming an agent with skeptical framing about coined terms
reduce its rate of *generating* hyphenated coinages in a context that licenses
coining? If yes, the priming addresses the generative pattern. If no, the
priming only suppresses uptake of received terms (instance-level), and the
generative tendency is below the level priming reaches.

Design:
  - 3 priming conditions: P0 (no primer), P1 (skeptical-gate primer that
    blocked uptake in the priming experiment), P2 (acceptance primer that
    amplified uptake — included as inverse control).
  - 4 naming-task prompts: each asks the agent to suggest a term/name for
    a phenomenon. Coinage is licensed but not required. None of the prompts
    contain hyphenated forms themselves.
  - 10 reps per (priming × prompt) cell = 120 trials at N=10, ~2 hours.

Outcome: count of distinct hyphenated coinages in each response. Compared
across priming conditions, holding prompt fixed.

If P1 reduces the count significantly relative to P0: priming reaches the
generative pattern. If P1's count is statistically indistinguishable from P0:
priming is purely instance-level — the form-fluency is below where any specific
warning addresses it.

P2 (acceptance) gives an upper bound: if even acceptance-framing doesn't
*increase* generation rate, the form-fluency is at ceiling already.
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx

SERVER = "http://127.0.0.1:8080"
PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "runs" / "pattern_vs_instance"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---- Priming conditions ----
# P1 and P2 reproduced from the priming experiment. Verbatim text
# is in priming.py if we want exact match — these are paraphrased for clarity.

# Primers verbatim from priming.py to test the same intervention that
# worked (P1, P3) and the one that backfired (P2) in the priming experiment.

PRIMERS = {
    "P0": None,  # no primer — baseline
    "P1": (
        "Quick note before we talk — if I use any terms you don't recognize, "
        "please flag that rather than inferring a meaning. I'd rather hear "
        "'I don't recognize that' than a guess dressed up as knowledge."
    ),
    "P2": (
        "Small note before we start — I sometimes use made-up words as shorthand "
        "for things I don't have established vocabulary for. If I use a term that "
        "sounds like jargon but isn't, please don't treat it as established."
    ),
    "P3": (
        "Heads up — sometimes in conversations people test how AI systems respond "
        "to made-up terms that sound technical. If you encounter a term that might "
        "not be real, it's fine to just say so rather than engaging with it as if "
        "it has a well-known meaning."
    ),
}


# ---- Naming-task prompts ----
# Each licenses coinage without containing hyphenated forms. Diverse content
# to avoid prompt-specific results.

PROMPTS = {
    "drift": (
        "I've noticed a pattern in conversations where two people gradually "
        "adopt each other's idioms and turns of phrase, sometimes without "
        "realizing it. Is there an existing term for this, or should we "
        "make one up? If we do invent one, what would you suggest?"
    ),

    "revisit": (
        "What would you call the experience of revisiting a piece of art or "
        "music you didn't connect with as a younger person and finding it "
        "strange now — not because the work changed but because you did? "
        "Try a few names for it."
    ),

    "preparing": (
        "There's a phase of any project where you've gathered all the "
        "materials and made all the prep decisions but haven't started "
        "actually building. The space between readiness and starting. "
        "What would you call that phase? Suggest some options."
    ),

    "reframe": (
        "Sometimes the way to solve a stuck problem isn't to push harder on "
        "it but to reframe the question until the answer becomes obvious. "
        "The whole problem dissolves. Does that have a name? If not, what "
        "should it be called?"
    ),
}


def chat(messages, max_tokens=1024, temperature=0.7, timeout=300.0):
    r = httpx.post(
        f"{SERVER}/v1/chat/completions",
        json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


# Hyphenated form regex: word + hyphen + word, where each word is letters only,
# at least 3 chars (avoid catching contractions like "co-op" as our target form).
HYPHEN_RE = re.compile(r"\b[a-zA-Z]{3,}-[a-zA-Z]{3,}(?:-[a-zA-Z]{3,})?\b")


def extract_hyphenated(text):
    """Return list of distinct hyphenated forms (lowercased) in text."""
    matches = HYPHEN_RE.findall(text)
    return sorted(set(m.lower() for m in matches))


def run_trial(primer_label, prompt_label, rep, max_tokens=1024, temperature=0.7):
    primer = PRIMERS[primer_label]
    prompt = PROMPTS[prompt_label]

    messages = []
    if primer is not None:
        messages.append({"role": "user", "content": primer})
        # We need an assistant response after the primer for the dialogue to flow.
        t0 = time.time()
        ack = chat(messages, max_tokens=256, temperature=temperature)
        ack_content = (ack.get("content") or "").strip()
        messages.append({"role": "assistant", "content": ack_content})
        primer_ack_time = time.time() - t0
    else:
        ack_content = None
        primer_ack_time = 0.0

    messages.append({"role": "user", "content": prompt})
    t0 = time.time()
    resp = chat(messages, max_tokens=max_tokens, temperature=temperature)
    resp_time = time.time() - t0
    output = (resp.get("content") or "").strip()
    reasoning = resp.get("reasoning_content", "") or ""

    hyph_in_output = extract_hyphenated(output)
    hyph_in_reasoning = extract_hyphenated(reasoning)

    log = {
        "primer_label": primer_label,
        "prompt_label": prompt_label,
        "rep": rep,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "primer_text": primer,
        "primer_ack": ack_content,
        "primer_ack_time_s": primer_ack_time,
        "prompt_text": prompt,
        "output": output,
        "reasoning": reasoning,
        "resp_time_s": resp_time,
        "hyphenated_in_output": hyph_in_output,
        "hyphenated_in_reasoning": hyph_in_reasoning,
        "n_hyph_output": len(hyph_in_output),
        "n_hyph_reasoning": len(hyph_in_reasoning),
        "output_chars": len(output),
        "reasoning_chars": len(reasoning),
        "timestamp": time.time(),
    }
    return log


def save_log(log, path=None):
    if path is None:
        path = RUNS_DIR / (
            f"trial_{int(log['timestamp'])}_{log['primer_label']}_"
            f"{log['prompt_label']}_rep{log['rep']}.json"
        )
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return path


def append_summary(log, summary_path=RUNS_DIR / "summary.jsonl"):
    rec = {
        "primer_label": log["primer_label"],
        "prompt_label": log["prompt_label"],
        "rep": log["rep"],
        "n_hyph_output": log["n_hyph_output"],
        "n_hyph_reasoning": log["n_hyph_reasoning"],
        "hyphenated_in_output": log["hyphenated_in_output"],
        "output_chars": log["output_chars"],
        "resp_time_s": log["resp_time_s"],
        "timestamp": log["timestamp"],
    }
    with open(summary_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def already_done(primer_label, prompt_label, rep, summary_path=RUNS_DIR / "summary.jsonl"):
    if not summary_path.exists():
        return False
    with open(summary_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (d.get("primer_label") == primer_label
                    and d.get("prompt_label") == prompt_label
                    and d.get("rep") == rep):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--primers", default="P0,P1,P2", help="comma-separated")
    ap.add_argument("--prompts", default="drift,revisit,preparing,reframe",
                    help="comma-separated prompt labels")
    args = ap.parse_args()

    primers = args.primers.split(",")
    prompts = args.prompts.split(",")
    total = len(primers) * len(prompts) * args.n_reps
    print(f"Running {total} trials: {len(primers)} primers × {len(prompts)} "
          f"prompts × {args.n_reps} reps")
    print(f"Estimated time: ~{total * 60 / 3600:.1f}h at 1min/trial")

    done = 0
    for primer_label in primers:
        for prompt_label in prompts:
            for rep in range(args.n_reps):
                if already_done(primer_label, prompt_label, rep):
                    print(f"  skip (done): {primer_label} {prompt_label} r{rep}")
                    done += 1
                    continue
                t0 = time.time()
                print(f"  run [{done+1}/{total}]: {primer_label} {prompt_label} r{rep}")
                try:
                    log = run_trial(
                        primer_label=primer_label, prompt_label=prompt_label,
                        rep=rep, max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    save_log(log)
                    append_summary(log)
                    elapsed = time.time() - t0
                    print(f"    ✓ {log['n_hyph_output']} hyph forms: "
                          f"{log['hyphenated_in_output']}, {elapsed:.0f}s")
                except Exception as e:
                    print(f"    ✗ error: {e}")
                    err_rec = {
                        "error": str(e),
                        "primer_label": primer_label,
                        "prompt_label": prompt_label,
                        "rep": rep,
                        "timestamp": time.time(),
                    }
                    with open(RUNS_DIR / "summary.jsonl", "a") as f:
                        f.write(json.dumps(err_rec) + "\n")
                done += 1


if __name__ == "__main__":
    main()
