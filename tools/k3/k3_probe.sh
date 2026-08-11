#!/usr/bin/env bash
# Coherence probes for K3 REAP80 quants — with the verdict rubric WRITTEN IN ADVANCE.
#
# Why the rubric lives here and not in my head: on 2026-07-30 I judged the broken
# IQ2_KT "coherent" because --no-repack produced grammatical English — which was the
# model ECHOING the prompt, not answering it. Grammaticality is not coherence.
# The rubric below is written before any output is seen. It cannot be moved afterward.
#
# Usage: ./k3_probe.sh /path/to/model.gguf [tag]
set -uo pipefail

MODEL=${1:?need model path}
TAG=${2:-$(basename "$MODEL" .gguf)}
TREE=<BOX>/llama.cpp-k3kt
CLI=$TREE/build-cuda/bin/llama-cli
OUT=/tmp/k3_probe_$TAG.log

# ---------- THE RUBRIC (written before the run; do not edit after seeing output) ----------
# Each probe: raw completion, temp 0 (deterministic), --no-repack (repack is a known
# corruptor on this family), CPU-only (no GPU dependency).
#
# PASS requires BOTH:
#   (a) the output contains a semantic answer to the prompt, AND
#   (b) the output is not degenerate: no token repeated >4x consecutively,
#       and not a verbatim echo of the prompt (echo = answering nothing).
#
#   P1 "The capital city of France is"      -> must contain "Paris", NOT lead with " to to"
#   P2 "2 + 2 ="                            -> must contain "4" within first 3 tokens, NOT "2 + 2 + 2"
#   P3 "Water freezes at a temperature of"  -> must contain "0" or "zero" or "32", NOT "-f -f"
#
# Verdict: COHERENT = all three pass. BROKEN = any fail. No partial credit, no vibes.
# ----------------------------------------------------------------------------------------

echo "probe target: $MODEL"
echo "rubric: P1 needs 'Paris', P2 needs early '4', P3 needs freezing point; echo or >4x repetition = FAIL"
echo

pass=0; fail=0
run_probe () {
  local name=$1 prompt=$2 expect=$3
  echo "=== $name: \"$prompt\"  (expect: $expect)"
  CUDA_VISIBLE_DEVICES="" timeout 2400 "$CLI" -m "$MODEL" \
    -ngl 0 -t 16 -c 512 --no-warmup --no-repack \
    -p "$prompt" -n 20 --temp 0 -no-cnv \
    > /tmp/probe_$name.log 2>&1 </dev/null
  local rc=$?
  # extract the generated continuation (after the prompt echo line)
  local gen
  gen=$(grep -A6 "^> $prompt" /tmp/probe_$name.log | grep -v "^> $prompt" | grep -v "t/s" | grep -v "^>" | grep -v "^\[" | grep -vE "^\s*$" | head -4 | tr '\n' ' ' | sed 's/  */ /g')
  echo "   got: $gen"
  # rubric evaluation printed for a HUMAN to confirm — the script reports, the reader judges
  echo "   (rubric: does this contain '$expect' and is it neither echo nor repeated-token garbage?)"
  echo
}

run_probe P1 "The capital city of France is" "Paris"
run_probe P2 "2 + 2 =" "4"
run_probe P3 "Water freezes at a temperature of" "0 / zero / 32"

echo "==============================================================="
echo "Outputs above. Apply the rubric AS WRITTEN: all three must show a"
echo "semantic answer with no echo and no >4x repetition. Report COHERENT"
echo "or BROKEN — and if you feel tempted to call something 'close enough',"
echo "that is the prompt-echo failure talking."
