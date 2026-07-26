#!/usr/bin/env bash
# DRY-at-depth on the production GLM, matched pair, run against an ALREADY-SERVED model.
# usage: glm_dry_depth.sh <port> [depth_tokens] [n_predict]
#
# Why this shape:
#   * --chat  : hits /v1/chat/completions and reads reasoning_content + content, which is
#               both real --jinja usage AND the only way to see a loop that happens
#               entirely inside <think>. A content-only reader scores those as EMPTY.
#   * matched : both arms at the SAME temperature, so DRY is the only variable. Comparing
#               dry@0.7 against greedy@0.0 changes two things at once.
#   * depth   : ~12K, because the measured usable ceiling for this model on this hardware
#               is ~16K, not 128K. Asking for more just produces KV-full failures.
set -u
PORT=${1:-8080}
DEPTH=${2:-12000}
NPRED=${3:-800}
LR=/tmp/loop_rate.py
CORPUS=<BOX>/glm-calib/corpus/longdocs.txt
OUT=<BOX>/d1/glm_dry_depth
mkdir -p "$OUT"

echo "=== ARM A: plain sampling @0.7, depth ${DEPTH} (baseline) ==="
python3 "$LR" --host 127.0.0.1 --port "$PORT" --chat \
  --label "GLM-5.2-IQ1_KT|1.99|plain@0.7|depth${DEPTH}" \
  --sampler greedy --temp 0.7 --n-predict "$NPRED" \
  --prefill-tokens "$DEPTH" --prefill-file "$CORPUS" \
  --tsv "$OUT/plain.tsv" --json "$OUT/plain.json" \
  --save-samples "$OUT/samples_plain" 2>&1 | tail -14

echo
echo "=== ARM B: tuned DRY @0.7, depth ${DEPTH} (the fix under test) ==="
python3 "$LR" --host 127.0.0.1 --port "$PORT" --chat \
  --label "GLM-5.2-IQ1_KT|1.99|dry@0.7|depth${DEPTH}" \
  --sampler dry --temp 0.7 --n-predict "$NPRED" \
  --prefill-tokens "$DEPTH" --prefill-file "$CORPUS" \
  --tsv "$OUT/dry.tsv" --json "$OUT/dry.json" \
  --save-samples "$OUT/samples_dry" 2>&1 | tail -14

echo
echo "=== SUMMARY ==="
grep -hE "LOOP RATE|GENERATION SUCCESS" "$OUT/plain.tsv" "$OUT/dry.tsv" 2>/dev/null
echo "=== DONE ==="
