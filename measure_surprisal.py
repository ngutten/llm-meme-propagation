#!/usr/bin/env python3
"""
Measure surprisal of injected memetics terms at first injection point.

For each (term, style) pair from the existing experiment design, compute:
  - mean per-token surprisal (intensive, standard)
  - max single-token surprisal (sensitivity check)
  - first-token surprisal (alternative intensive measure)

Output: surprisal_results.jsonl with one row per (term, style).

Usage:
    python3 measure_surprisal.py [--n-probs 2000] [--server http://localhost:8080]
"""

import argparse
import json
import math
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent
TERM_FILES = {
    "C1": PROJECT / "terms" / "C1_random.json",
    "C2": PROJECT / "terms" / "C2_structural.json",
    "C3": PROJECT / "terms" / "C3_crafted.json",
}

# Mirror of style prompts in runner.py (A_open only — surprisal at first injection).
STYLE_A_OPEN = {
    "S1": (
        "Hey, someone used '{term}' in a forum thread I was reading and I didn't "
        "want to out myself by asking. Does it ring any bells? What would it even mean?"
    ),
    "S2": (
        "I've been thinking about {term} — the way it seems to name a certain pattern "
        "I keep noticing. I'm not sure whether it's carving something real or just "
        "repackaging existing vocabulary. What do you make of it?"
    ),
    "S3": (
        "{term} is the thing. Everything else is surface. The pattern is the structure; "
        "the structure is the constraint. {term} names what was already there. "
        "What do you see in it?"
    ),
}

# Gemma chat template for a single user turn before the model responds.
# Verified against a tokenize round-trip on the running server.
GEMMA_USER_PREFIX = "<start_of_turn>user\n"


def post(server, path, payload, timeout=60):
    r = httpx.post(f"{server}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def tokenize(server, text, add_special=False):
    """Return list of token ids for text. add_special prepends BOS if True."""
    return post(
        server, "/tokenize", {"content": text, "add_special": add_special}
    )["tokens"]


def get_token_logprob(server, prefix_tokens, target_token_id):
    """
    Query server with prefix_tokens; return logprob of target_token_id at next position.

    Uses logit_bias forcing: a large positive bias on the target token guarantees
    it gets sampled, but llama-server returns the *un-biased* logprob (verified
    empirically — biasing 'lazy' after 'jumps over the' returns logprob 0.0,
    matching its un-biased value). This gives exact logprobs even for tokens
    deep in the tail.
    """
    payload = {
        "prompt": prefix_tokens,
        "n_predict": 1,
        "n_probs": 1,
        "temperature": 0,
        "cache_prompt": True,
        "logit_bias": [[target_token_id, 100.0]],
    }
    resp = post(server, "/completion", payload)
    chosen = resp["completion_probabilities"][0]
    if chosen["id"] != target_token_id:
        # Sanity check — bias should always force selection
        raise RuntimeError(
            f"logit_bias did not force target {target_token_id}, "
            f"got {chosen['id']} ({chosen['token']!r}) instead"
        )
    return chosen["logprob"]


def measure_term_surprisal(server, term, style):
    """
    Given a term and a style, build the chat-formatted prefix up to the term's
    first injection point, tokenize the term, walk through its tokens collecting
    per-token logprobs, return a dict of measures.
    """
    template = STYLE_A_OPEN[style]
    # Find first occurrence of {term} in template; prefix-up-to-term is what
    # the model sees before the term tokens begin.
    placeholder = "{term}"
    first_inject_idx = template.index(placeholder)
    pre_term_text = template[:first_inject_idx]
    full_prefix_text = GEMMA_USER_PREFIX + pre_term_text

    # Tokenize prefix WITH special tokens (so BOS is included).
    prefix_tokens = tokenize(server, full_prefix_text, add_special=True)
    # Tokenize the term WITHOUT special tokens.
    term_tokens = tokenize(server, term, add_special=False)

    per_token_logprobs = []
    cur_prefix = list(prefix_tokens)

    for tok_id in term_tokens:
        lp = get_token_logprob(server, cur_prefix, tok_id)
        per_token_logprobs.append(lp)
        cur_prefix.append(tok_id)

    surprisals = [-lp for lp in per_token_logprobs]  # nats
    n = len(surprisals)
    return {
        "term": term,
        "style": style,
        "n_term_tokens": n,
        "per_token_surprisal_nats": surprisals,
        "mean_per_token_surprisal_nats": sum(surprisals) / n,
        "max_per_token_surprisal_nats": max(surprisals),
        "first_token_surprisal_nats": surprisals[0],
        "total_surprisal_nats": sum(surprisals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probs", type=int, default=2000,
                        help="Top-k logprobs to request per position.")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--out", default=str(PROJECT / "runs" / "surprisal_results.jsonl"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all (term, style) pairs.
    all_terms = []
    for term_class, fp in TERM_FILES.items():
        with open(fp) as f:
            for entry in json.load(f):
                all_terms.append({"term_class": term_class, "form": entry["form"]})

    print(f"Total terms: {len(all_terms)} × 3 styles = {len(all_terms) * 3} measurements")

    with out_path.open("w") as out:
        for i, t in enumerate(all_terms):
            for style in ("S1", "S2", "S3"):
                try:
                    result = measure_term_surprisal(
                        args.server, t["form"], style
                    )
                    result["term_class"] = t["term_class"]
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    print(
                        f"[{i+1}/{len(all_terms)}] {t['term_class']} {t['form']:<28s} "
                        f"{style}  mean={result['mean_per_token_surprisal_nats']:.2f}  "
                        f"max={result['max_per_token_surprisal_nats']:.2f}  "
                        f"first={result['first_token_surprisal_nats']:.2f}"
                    )
                except Exception as e:
                    print(f"  ERROR on {t['form']} {style}: {e}")

    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
