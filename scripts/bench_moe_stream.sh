#!/usr/bin/env bash
# Repeatable CPU benchmark matrix for GPT-OSS MXFP4 expert streaming.
#
# Usage:
#   scripts/bench_moe_stream.sh /path/to/gpt-oss-20b-MXFP4.gguf [output-dir]
#
# Optional environment overrides:
#   MISTRALRS_BIN=./target/release/mistralrs
#   CONTEXT=2048 PROMPT_LEN=128 GEN_LEN=256 DEPTH=128 ITERATIONS=3 WARMUP=1
#   CACHE_MB=512 CACHE_FLOOR_MB=1024 CACHE_CEIL_MB=512
#
# This script measures configurations; it does not assume one configuration is
# faster. Compare the result tables and logs on the target CPU/SSD.

set -Eeuo pipefail

usage() {
  echo "Usage: $0 /path/to/model-MXFP4.gguf [output-dir]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

MODEL_FILE=$1
OUT_DIR=${2:-"mistralrs-moe-bench-$(date +%Y%m%d-%H%M%S)"}
BIN=${MISTRALRS_BIN:-mistralrs}
CONTEXT=${CONTEXT:-2048}
PROMPT_LEN=${PROMPT_LEN:-128}
GEN_LEN=${GEN_LEN:-256}
DEPTH=${DEPTH:-128}
ITERATIONS=${ITERATIONS:-3}
WARMUP=${WARMUP:-1}
CACHE_MB=${CACHE_MB:-512}
CACHE_FLOOR_MB=${CACHE_FLOOR_MB:-1024}
CACHE_CEIL_MB=${CACHE_CEIL_MB:-512}

[[ -f "$MODEL_FILE" ]] || { echo "Model file not found: $MODEL_FILE" >&2; exit 2; }
if [[ "$BIN" == */* ]]; then
  [[ -x "$BIN" ]] || { echo "mistralrs binary not executable: $BIN" >&2; exit 2; }
else
  command -v "$BIN" >/dev/null || { echo "mistralrs binary not found: $BIN" >&2; exit 2; }
fi

mkdir -p "$OUT_DIR"
MODEL_FILE=$(realpath "$MODEL_FILE")

{
  echo "date=$(date --iso-8601=seconds)"
  echo "binary=$BIN"
  echo "model=$MODEL_FILE"
  echo "model_sha256=$(sha256sum "$MODEL_FILE" | awk '{print $1}')"
  echo "context=$CONTEXT"
  echo "prompt_len=$PROMPT_LEN"
  echo "gen_len=$GEN_LEN"
  echo "depth=$DEPTH"
  echo "iterations=$ITERATIONS"
  echo "warmup=$WARMUP"
  echo "cache_mb=$CACHE_MB"
  echo "cache_floor_mb=$CACHE_FLOOR_MB"
  echo "cache_ceil_mb=$CACHE_CEIL_MB"
  echo
  uname -a
  echo
  lscpu 2>/dev/null || true
  echo
  free -h 2>/dev/null || true
  echo
  df -h "$MODEL_FILE"
} > "$OUT_DIR/system.txt"

common=(
  bench --cpu -f "$MODEL_FILE" --moe-stream
  --max-model-len "$CONTEXT"
  --cache-mb "$CACHE_MB"
  --cache-floor-mb "$CACHE_FLOOR_MB"
  --cache-ceil-mb "$CACHE_CEIL_MB"
  --moe-stats
  --prompt-len "$PROMPT_LEN"
  --gen-len "$GEN_LEN"
  --depth "$DEPTH"
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
)

run_case() {
  local name=$1
  shift
  echo
  echo "===== $name ====="
  echo "Log: $OUT_DIR/$name.log"
  "$BIN" "${common[@]}" "$@" 2>&1 | tee "$OUT_DIR/$name.log"
}

run_case baseline-buffered \
  --io-threads 1 \
  --moe-overlap=false \
  --moe-zero-copy=false \
  --moe-prefault=false \
  --moe-threads 1

run_case tuned-overlap \
  --io-threads 4 \
  --moe-overlap \
  --moe-zero-copy \
  --moe-prefault \
  --moe-threads physical

run_case low-residency \
  --io-threads 2 \
  --moe-overlap \
  --moe-zero-copy \
  --moe-prefault \
  --moe-release-cold \
  --moe-release-idle 1024 \
  --moe-threads physical

echo
echo "Benchmark logs saved to: $OUT_DIR"
echo "Compare decode tok/s, TTFT/TPOT, peak RAM, cache hit rate, physical mmap residency and I/O wait."
