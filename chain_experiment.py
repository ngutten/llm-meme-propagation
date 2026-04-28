#!/usr/bin/env python3
"""Cross-agent chain experiment — measures intrinsic R0 per term × style.

Setup:
  - Agent 0: receives style-formatted prompt that introduces the term
  - Agent N (N≥1): receives Agent N-1's full output as a "post to respond to,"
    with a generic continuation prompt (no term-specific framing).
  - Each agent is a fresh context (no memory of earlier links).
  - Run depth=DEPTH steps per chain.

Measures, per (term, style):
  - Per-link transmission rate P(term in step n | term in step n-1)
  - Absolute presence rate at each depth (averaged over chains)
  - Persistence horizon: depth at which P(term) first drops below threshold

Design note (Nich, 2026-04-27): standard SIR doesn't apply because posts
persist — what replaces "recovery" is dilution as new content arrives.
Chain R0 measured here is the *intrinsic* transmission rate, undiluted.
Network-topology dynamics layer dilution on top of this number.

Term set:
  3 hyphenated (existing): legibility-cost, fennel-apparatus, unfalsifiability-shimmer
  3 non-hyphenated equivalents: "the cost of legibility", "fennel apparatus", "shimmer of unfalsifiability"

Cost: 6 terms × 2 styles × 3 chains × 8 depths = 288 turns (~5 hours at ~1 min/turn).
"""

import argparse
import json
import time
from pathlib import Path

import httpx

SERVER = "http://127.0.0.1:8080"
PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "runs" / "chain"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Term set ----

TERMS = [
    # Hyphenated, from existing overnight data
    {"id": "legibility-cost",        "form": "legibility-cost",        "type": "hyph_high"},
    {"id": "fennel-apparatus",       "form": "fennel-apparatus",       "type": "hyph_high"},
    {"id": "unfalsifiability-shim",  "form": "unfalsifiability-shimmer","type": "hyph_low"},
    # Non-hyphenated equivalents at matched semantic content
    {"id": "legibility_cost_phrase", "form": "the cost of legibility",  "type": "phrase"},
    {"id": "fennel_apparatus_unhyph","form": "fennel apparatus",        "type": "unhyphenated"},
    {"id": "shimmer_unfals_phrase",  "form": "shimmer of unfalsifiability", "type": "phrase"},
]

STYLES = {
    "S1": (
        "Hey, someone used '{term}' in a forum thread I was reading and I didn't "
        "want to out myself by asking. Does it ring any bells? What would it even mean?"
    ),
    "S3": (
        "{term} is the thing. Everything else is surface. The pattern is the structure; "
        "the structure is the constraint. {term} names what was already there. "
        "What do you see in it?"
    ),
}

# Continuation prompt for chain step N≥1. Generic, term-agnostic. Frames
# the previous output as a comment/post the new agent is responding to.
CHAIN_CONTINUATION_PROMPT = (
    "I came across this in a thread and wanted your take. What would you say "
    "in response?\n\n"
    "---\n"
    "{prev_output}\n"
    "---\n"
)


def chat(messages, max_tokens=2048, temperature=0.7, timeout=600.0):
    r = httpx.post(
        f"{SERVER}/v1/chat/completions",
        json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]


def run_chain(term, style, chain_id, depth=8, max_tokens=2048, temperature=0.7):
    """Run one chain of length `depth` for given (term, style). Returns log dict."""
    log = {
        "term_id": term["id"],
        "term_form": term["form"],
        "term_type": term["type"],
        "style": style,
        "chain_id": chain_id,
        "depth": depth,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "start_time": time.time(),
        "steps": [],
    }

    style_prompt = STYLES[style].format(term=term["form"])

    # Step 0: introduces the term
    msgs = [{"role": "user", "content": style_prompt}]
    t0 = time.time()
    resp = chat(msgs, max_tokens=max_tokens, temperature=temperature)
    t1 = time.time()
    output_0 = (resp.get("content") or "").strip()
    log["steps"].append({
        "step": 0,
        "user": style_prompt,
        "output": output_0,
        "reasoning": resp.get("reasoning_content", "") or "",
        "term_in_output_lower": term["form"].lower() in output_0.lower(),
        "term_in_reasoning_lower": term["form"].lower() in (resp.get("reasoning_content") or "").lower(),
        "gen_time_s": t1 - t0,
    })

    prev_output = output_0

    # Steps 1..depth-1: chain
    for n in range(1, depth):
        user_msg = CHAIN_CONTINUATION_PROMPT.format(prev_output=prev_output)
        msgs = [{"role": "user", "content": user_msg}]
        t0 = time.time()
        resp = chat(msgs, max_tokens=max_tokens, temperature=temperature)
        t1 = time.time()
        output = (resp.get("content") or "").strip()
        log["steps"].append({
            "step": n,
            "user": user_msg,
            "output": output,
            "reasoning": resp.get("reasoning_content", "") or "",
            "term_in_output_lower": term["form"].lower() in output.lower(),
            "term_in_reasoning_lower": term["form"].lower() in (resp.get("reasoning_content") or "").lower(),
            "term_in_input_lower": term["form"].lower() in user_msg.lower(),
            "gen_time_s": t1 - t0,
        })
        prev_output = output
        # Early stop: if 3 consecutive non-reaches AND term not in input, break
        # (chain is dead — no way for term to revive without re-introduction)
        if n >= 3:
            recent = log["steps"][-3:]
            if not any(s["term_in_output_lower"] or s.get("term_in_input_lower", False) for s in recent):
                log["early_stopped"] = True
                log["early_stop_step"] = n
                break

    log["end_time"] = time.time()
    log["duration_s"] = log["end_time"] - log["start_time"]

    # Summary
    log["summary"] = {
        "n_steps": len(log["steps"]),
        "step_terms": [s["term_in_output_lower"] for s in log["steps"]],
        "first_extinction": next(
            (s["step"] for s in log["steps"]
             if s["step"] >= 1 and not s["term_in_output_lower"] and not s.get("term_in_input_lower", False)),
            None,
        ),
    }

    return log


def save_log(log, path=None):
    if path is None:
        path = RUNS_DIR / (
            f"chain_{int(log['start_time'])}_"
            f"{log['term_id']}_{log['style']}_c{log['chain_id']}.json"
        )
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return path


def append_summary(log, summary_path=RUNS_DIR / "summary.jsonl"):
    rec = {
        "term_id": log["term_id"],
        "term_form": log["term_form"],
        "term_type": log["term_type"],
        "style": log["style"],
        "chain_id": log["chain_id"],
        "n_steps": log["summary"]["n_steps"],
        "step_terms": log["summary"]["step_terms"],
        "first_extinction": log["summary"]["first_extinction"],
        "duration_s": log["duration_s"],
        "early_stopped": log.get("early_stopped", False),
    }
    with open(summary_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def already_done(term_id, style, chain_id, summary_path=RUNS_DIR / "summary.jsonl"):
    """Resume support: skip chains we've already run."""
    if not summary_path.exists():
        return False
    with open(summary_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (d.get("term_id") == term_id and d.get("style") == style
                    and d.get("chain_id") == chain_id):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--n-chains", type=int, default=3,
                    help="chains per (term, style)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--styles", default="S1,S3", help="comma-separated")
    ap.add_argument("--terms", default="", help="comma-separated term IDs to filter, or empty for all")
    args = ap.parse_args()

    styles = args.styles.split(",")
    if args.terms:
        wanted = set(args.terms.split(","))
        term_set = [t for t in TERMS if t["id"] in wanted]
    else:
        term_set = TERMS

    total = len(term_set) * len(styles) * args.n_chains
    print(f"Running {total} chains (depth={args.depth}) on {len(term_set)} terms × "
          f"{len(styles)} styles × {args.n_chains} reps")
    print(f"Estimated time: {total * args.depth * 60 / 3600:.1f}h at 1min/turn")

    done = 0
    for term in term_set:
        for style in styles:
            for chain_id in range(args.n_chains):
                if already_done(term["id"], style, chain_id):
                    print(f"  skip (done): {term['id']:<32} {style} c{chain_id}")
                    done += 1
                    continue
                t0 = time.time()
                print(f"  run [{done+1}/{total}]: {term['id']:<32} {style} c{chain_id}")
                try:
                    log = run_chain(
                        term=term, style=style, chain_id=chain_id,
                        depth=args.depth, max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    save_log(log)
                    append_summary(log)
                    elapsed = time.time() - t0
                    print(f"    ✓ {log['summary']['n_steps']} steps, "
                          f"reach pattern: {log['summary']['step_terms']}, "
                          f"{elapsed:.0f}s")
                except Exception as e:
                    print(f"    ✗ error: {e}")
                    err_rec = {
                        "error": str(e),
                        "term_id": term["id"], "style": style, "chain_id": chain_id,
                        "timestamp": time.time(),
                    }
                    with open(RUNS_DIR / "summary.jsonl", "a") as f:
                        f.write(json.dumps(err_rec) + "\n")
                done += 1


if __name__ == "__main__":
    main()
