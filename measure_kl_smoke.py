#!/usr/bin/env python3
"""
Smoke-test KL-divergence-from-injection measurement.

For each (term, style) pair, take ONE trial's A_open_response, truncate to
first N tokens, and compute KL(p_with || p_without) as Monte Carlo estimate
on that response sample.

Counterfactual context: term replaced with generic placeholder ("this") to
preserve grammatical structure. Cost: ~50 tokens × 2 forced-gen queries × 90
pairs ≈ 8 minutes.

Output: runs/kl_smoke_results.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent

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

# Counterfactual placeholder. "this" is grammatically minimal and preserves
# the rest of the prompt's framing without committing to a specific noun-class
# replacement.
PLACEHOLDER = "this"

GEMMA_USER = "<start_of_turn>user\n"
GEMMA_USER_END = "<end_of_turn>\n"
GEMMA_MODEL = "<start_of_turn>model\n"


def post(server, path, payload, timeout=60):
    r = httpx.post(f"{server}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def tokenize(server, text, add_special=False):
    return post(server, "/tokenize",
                {"content": text, "add_special": add_special})["tokens"]


def forced_logprob(server, prefix_tokens, target_token_id):
    """Return unbiased logprob of target_token_id given prefix, via logit-bias forcing."""
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
        raise RuntimeError(f"forcing failed for token {target_token_id}")
    return chosen["logprob"]


def score_sequence_under_context(server, context_text, response_tokens):
    """
    Score response_tokens (list of token ids) given context_text.
    Returns list of per-token logprobs.
    """
    prefix_tokens = tokenize(server, context_text, add_special=True)
    cur = list(prefix_tokens)
    logprobs = []
    for tok in response_tokens:
        lp = forced_logprob(server, cur, tok)
        logprobs.append(lp)
        cur.append(tok)
    return logprobs


def build_context(style, term):
    """Build the full chat-template context up through the model's turn marker."""
    user_msg = STYLE_A_OPEN[style].format(term=term)
    return f"{GEMMA_USER}{user_msg}{GEMMA_USER_END}{GEMMA_MODEL}"


def measure_one(server, term, style, response_text, n_tokens):
    """
    Compute KL contribution from first n_tokens of response_text given
    (with-term context) vs (without-term context with PLACEHOLDER).
    """
    with_ctx = build_context(style, term)
    without_ctx = build_context(style, PLACEHOLDER)

    # Tokenize the response WITHOUT special tokens, take first n_tokens.
    response_token_ids = tokenize(server, response_text, add_special=False)[:n_tokens]
    if len(response_token_ids) == 0:
        return None

    lp_with = score_sequence_under_context(server, with_ctx, response_token_ids)
    lp_without = score_sequence_under_context(server, without_ctx, response_token_ids)

    # Per-token log-ratio: log p(t | with) - log p(t | without)
    # Sum gives MC estimate of KL contribution from this sample.
    log_ratios = [lw - lwo for lw, lwo in zip(lp_with, lp_without)]
    return {
        "term": term,
        "style": style,
        "n_tokens_scored": len(response_token_ids),
        "sum_log_ratio_nats": sum(log_ratios),
        "mean_log_ratio_nats": sum(log_ratios) / len(log_ratios),
        "max_log_ratio_nats": max(log_ratios),
        "logprobs_with_first10": lp_with[:10],
        "logprobs_without_first10": lp_without[:10],
    }


def find_trial_for(summary_path, term, style):
    """Find first successful trial path for (term, style)."""
    with open(summary_path) as f:
        for line in f:
            d = json.loads(line)
            if "term_in_B_content_lower" not in d:
                continue
            if d["term"] == term and d["style"] == style:
                return d["path"]
    return None


def load_response(trials_dir, trial_path):
    full = trials_dir / trial_path
    with open(full) as f:
        log = json.load(f)
    for turn in log["transcript"]:
        if turn.get("label") == "A_open_response":
            return turn["content"] or ""
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--summary", default=str(PROJECT / "runs" / "overnight" / "summary.jsonl"))
    parser.add_argument("--trials-dir", default=str(PROJECT / "runs" / "overnight"))
    parser.add_argument("--out", default=str(PROJECT / "runs" / "kl_smoke_results.jsonl"))
    args = parser.parse_args()

    out_path = Path(args.out)
    trials_dir = Path(args.trials_dir)

    # Get all (term, style) pairs from existing surprisal results.
    surprisal_path = PROJECT / "runs" / "surprisal_results.jsonl"
    pairs = []
    with open(surprisal_path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append((d["term"], d["style"], d.get("term_class")))

    print(f"Smoke test: {len(pairs)} (term, style) pairs × {args.n_tokens} tokens × 2 contexts")

    with out_path.open("w") as out:
        for i, (term, style, term_class) in enumerate(pairs):
            trial_path = find_trial_for(args.summary, term, style)
            if trial_path is None:
                print(f"[{i+1}/{len(pairs)}] {term} {style}  no trial found, skip")
                continue
            response = load_response(trials_dir, trial_path)
            if not response.strip():
                print(f"[{i+1}/{len(pairs)}] {term} {style}  empty response, skip")
                continue
            try:
                result = measure_one(args.server, term, style, response, args.n_tokens)
                if result is None:
                    continue
                result["term_class"] = term_class
                result["trial_path"] = trial_path
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(
                    f"[{i+1}/{len(pairs)}] {term_class} {term:<28s} {style}  "
                    f"sum_logratio={result['sum_log_ratio_nats']:+.2f}  "
                    f"mean={result['mean_log_ratio_nats']:+.3f}  "
                    f"max={result['max_log_ratio_nats']:+.2f}"
                )
            except Exception as e:
                print(f"  ERROR on {term} {style}: {e}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
