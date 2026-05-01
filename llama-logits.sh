#!/bin/bash
# llama-logits.sh — start llama-server in scoring mode for surprisal measurement
#
# Usage:
#   ./llama-logits.sh /path/to/model.gguf [port]
#
# To score a string, POST to http://localhost:<port>/v1/completions:
#
#   curl -s http://localhost:8080/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "prompt": "<full text including term to score>",
#       "max_tokens": 0,
#       "echo": true,
#       "logprobs": 5
#     }'
#
# Response includes choices[0].logprobs with parallel arrays:
#   tokens[i]         — string of token i
#   token_logprobs[i] — log-probability of token i given preceding tokens
#   text_offset[i]    — character offset of token i in the prompt
#
# Use text_offset to locate the injected term in the prompt and pull out the
# token_logprobs for those positions. Surprisal of a token = -token_logprobs[i]
# (in nats; multiply by log2(e) ≈ 1.4427 for bits).
#
# Note: token_logprobs[0] is null (no preceding context). All other positions
# give p(t_i | t_0..t_{i-1}).
#
# If echo:true doesn't behave as expected, the alternative is the native
# llama.cpp /completion endpoint, which exposes the same information via
# the "completion_probabilities" field with a slightly different response
# shape. Worth trying both.

set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
PORT="${2:-${PORT:-8080}}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: $0 <path-to-model.gguf> [port]" >&2
  echo "  or set MODEL_PATH env var" >&2
  exit 1
fi

# Adjust hardware-specific flags as needed. The key thing for surprisal work
# is that the model loads and exposes /v1/completions; logprobs are a
# request-level parameter and don't need a server-side flag.
#
# Flag set: hat tip to @kira.pds.witchcraft.systems, whose strix-halo notes
# had most of these. For speed at slight memory savings: --cache-type-k q8_0
# --cache-type-v q8_0. For concurrent trials: --parallel N.

llama-server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --no-mmap \
  --cache-reuse 1
