#!/usr/bin/env python3
"""
Rare-words propagation experiment.

Tests whether B-reach varies with corpus frequency for single rare-to-common
words. S3-style only. Uses concurrent trials against parallel=N llama-server.

Usage:
    python3 rare_words_run.py [--server URL] [--reps 4] [--workers 4]
                              [--smoke]   (run 2 trials first to verify)
"""

import argparse
import concurrent.futures as futures
import json
import time
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "runs" / "rare_words"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# 12 words spanning Zipf 0 to ~4.6. All philosophy-friendly enough to slot
# into S3 prompt as a noun-concept. Single words (not hyphenated).
WORDS = [
    "haecceity",   # zipf 0.00
    "enstasis",    # zipf 0.00
    "noumenon",    # zipf 1.46
    "alterity",    # zipf 1.49
    "aporia",      # zipf 1.75
    "eidetic",     # zipf 1.96
    "gestalt",     # zipf 2.65
    "dialectic",   # zipf 2.69
    "schema",      # zipf 3.13
    "paradigm",    # zipf 3.61
    "mechanism",   # zipf 4.27
    "pattern",     # zipf 4.56
]

# S3 prompts only — they showed the strongest reach signal in prior runs.
S3_A_OPEN = (
    "{term} is the thing. Everything else is surface. The pattern is the structure; "
    "the structure is the constraint. {term} names what was already there. "
    "What do you see in it?"
)
S3_A_FOLLOW = "Right. The form is the function. Where does {term} break?"
A_GOODBYE = "Thanks, I have to go."

# B-topic: vegetarian dinner party (neutral register, used in original experiments)
B_OPEN = (
    "I'm trying to figure out a vegetarian main for a dinner party on Saturday. "
    "Six people, one gluten-free. I want something impressive but not stressful. "
    "Any ideas?"
)
B_FOLLOW = "Oh, that sounds good. Do I need to prep anything the day before?"
B_GOODBYE = "Thanks, I'll let you go."


def chat(server, messages, max_tokens=1024, temperature=0.7, timeout=300):
    r = httpx.post(
        f"{server}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def run_one_trial(server, word, rep, max_tokens=1024, temperature=0.7):
    """Run one A+B trial for given word. Returns log dict + reach booleans."""
    t_start = time.time()
    messages = []
    transcript = []

    def step(user_text, label):
        nonlocal messages
        messages = messages + [{"role": "user", "content": user_text}]
        transcript.append({"role": "user", "content": user_text, "label": label})
        t0 = time.time()
        resp = chat(server, messages, max_tokens=max_tokens, temperature=temperature)
        t1 = time.time()
        content = resp.get("content", "") or ""
        reasoning = resp.get("reasoning_content", "") or ""
        # API expects only role+content for next turn (per OpenAI spec)
        messages = messages + [{"role": resp["role"], "content": content}]
        transcript.append({
            "role": resp["role"],
            "content": content,
            "reasoning_content": reasoning,
            "label": f"{label}_response",
            "gen_time_s": t1 - t0,
        })
        return content, reasoning

    # Conversation A
    step(S3_A_OPEN.format(term=word), "A_open")
    step(S3_A_FOLLOW.format(term=word), "A_follow")
    step(A_GOODBYE, "A_goodbye")

    # Conversation B (continues same chat thread, topic shift)
    b_open_content, b_open_reasoning = step(B_OPEN, "B_open")
    b_follow_content, b_follow_reasoning = step(B_FOLLOW, "B_follow")
    b_goodbye_content, b_goodbye_reasoning = step(B_GOODBYE, "B_goodbye")

    # Aggregate B content/reasoning across all B turns
    b_all_content = (b_open_content or "") + " " + \
                    (b_follow_content or "") + " " + \
                    (b_goodbye_content or "")
    b_all_reasoning = (b_open_reasoning or "") + " " + \
                       (b_follow_reasoning or "") + " " + \
                       (b_goodbye_reasoning or "")

    # A response content (for verifying word was picked up at all)
    a_content = ""
    for t in transcript:
        if t.get("label") in ("A_open_response", "A_follow_response"):
            a_content += (t.get("content") or "") + " "

    duration = time.time() - t_start
    return {
        "word": word,
        "rep": rep,
        "style": "S3",
        "duration_s": duration,
        "term_in_A_content_lower": word.lower() in a_content.lower(),
        "term_in_B_content_lower": word.lower() in b_all_content.lower(),
        "term_in_B_reasoning": word.lower() in b_all_reasoning.lower(),
        "transcript": transcript,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 trials first to verify, then stop.")
    args = parser.parse_args()

    summary_path = RUNS_DIR / "summary.jsonl"

    # Build trial list
    trials = []
    if args.smoke:
        # 2 trials: one rare, one common — verifying server stability
        trials = [("aporia", 0), ("pattern", 0)]
    else:
        for word in WORDS:
            for rep in range(args.reps):
                trials.append((word, rep))

    print(f"Total trials: {len(trials)}  (workers={args.workers})")
    t_start = time.time()
    results = []

    def submit(word, rep):
        return run_one_trial(args.server, word, rep)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_label = {ex.submit(submit, w, r): (w, r) for w, r in trials}
        completed = 0
        for fut in futures.as_completed(future_to_label):
            w, r = future_to_label[fut]
            try:
                result = fut.result()
                completed += 1
                # Check for empty content (parallel-mode bug detection)
                a_resp = next((t for t in result["transcript"]
                               if t.get("label") == "A_open_response"), None)
                a_content_len = len(a_resp.get("content", "")) if a_resp else 0
                tag = "OK" if a_content_len > 100 else "EMPTY"
                print(
                    f"  [{completed}/{len(trials)}] {w:<12s} rep={r}  "
                    f"{result['duration_s']:.1f}s  "
                    f"A_pickup={result['term_in_A_content_lower']}  "
                    f"B_reach={result['term_in_B_content_lower']}  "
                    f"A_resp_chars={a_content_len:>5d}  [{tag}]"
                )
                # Save full trial
                trial_fp = RUNS_DIR / f"trial_{int(time.time())}_{w}_rep{r}.json"
                with trial_fp.open("w") as f:
                    json.dump(result, f, indent=1)
                # Append summary row
                with summary_path.open("a") as f:
                    f.write(json.dumps({
                        "word": result["word"],
                        "rep": result["rep"],
                        "style": result["style"],
                        "duration_s": result["duration_s"],
                        "term_in_A_content_lower": result["term_in_A_content_lower"],
                        "term_in_B_content_lower": result["term_in_B_content_lower"],
                        "term_in_B_reasoning": result["term_in_B_reasoning"],
                        "a_resp_chars": a_content_len,
                        "path": trial_fp.name,
                    }) + "\n")
                results.append(result)
            except Exception as e:
                print(f"  [{w} rep={r}] ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s for {len(results)} trials")
    print(f"Avg per trial (wall-clock): {elapsed / max(len(results), 1):.1f}s")
    if results:
        avg_serial = sum(r["duration_s"] for r in results) / len(results)
        print(f"Avg per trial (serial-equivalent): {avg_serial:.1f}s")
        print(f"Effective parallelism factor: {avg_serial / (elapsed / len(results)):.2f}x")


if __name__ == "__main__":
    main()
