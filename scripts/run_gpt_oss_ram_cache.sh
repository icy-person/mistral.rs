#!/usr/bin/env bash
# Run GPT-OSS MXFP4 with an Edge0-inspired, byte-bounded RAM expert LRU.
set -Eeuo pipefail

BIN="${MISTRALRS_BIN:-./mistralrs}"
MODEL="${1:-${HOME}/.lmstudio/models/openai/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf}"
CACHE_MB="${CACHE_MB:-768}"
IO_THREADS="${IO_THREADS:-2}"
OUT_PREFIX="${OUT_PREFIX:-gpt-oss-ram-cache}"

if [[ ! -x "$BIN" ]]; then
  echo "mistralrs binary is not executable: $BIN" >&2
  echo "Set MISTRALRS_BIN=/path/to/mistralrs or run this from the build directory." >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "GGUF model file not found: $MODEL" >&2
  echo "Pass its path as the first argument." >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "/usr/bin/time is required for peak-RSS reporting." >&2
  exit 2
fi

if MISTRALRS_MOE_RAM_CACHE=1 RUST_LOG="${RUST_LOG:-info}" /usr/bin/time -v "$BIN" run \
  --cpu \
  --moe-stream \
  --cache-mb "$CACHE_MB" \
  --cache-floor-mb 1024 \
  --cache-ceil-mb "$CACHE_MB" \
  --io-threads "$IO_THREADS" \
  --moe-overlap \
  --moe-stats \
  --max-model-len 2048 \
  --max-tokens 96 \
  --thinking false \
  -f "$MODEL" \
  -i "Explain how an operating system schedules processes in about 60 words." \
  > "${OUT_PREFIX}.stdout" 2> "${OUT_PREFIX}.stderr"; then
  status=0
else
  status=$?
fi

printf '\n===== MODEL OUTPUT =====\n'
sed '/^Stats:/,$d' "${OUT_PREFIX}.stdout"

printf '\n===== GENERATION STATS =====\n'
grep -E '^CLI time|^Prompt:|^Decode:|^Prefix cache:|^Sampling:' "${OUT_PREFIX}.stdout" || true

printf '\n===== STREAMING CACHE =====\n'
grep 'GPT-OSS MXFP4 stream cache' "${OUT_PREFIX}.stderr" | tail -n 10 || true

printf '\n===== MEMORY REPORT =====\n'
grep -E 'Maximum resident set size|Elapsed \(wall clock\)|Exit status' "${OUT_PREFIX}.stderr" || true

exit "$status"
