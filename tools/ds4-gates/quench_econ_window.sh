#!/usr/bin/env bash
# quench_econ_window.sh — P0-1 governor ECONOMICS arm (the GPU half quench_gate.sh could not run).
# Staged 2026-08-03 so the box window ask is gate-on-binary clean: everything below is ready
# to run the moment a card slot is granted; expected wall ~20-30 min.
#
# What it measures (the econ question the CPU gate deferred): with a REAL model + REAL
# drafter on GPU, does DS4P_YIELD_QUENCH=1 buy latency without costing acceptance?
#   arm A: governor OFF  (baseline spec decode)
#   arm B: governor ON   (DS4P_YIELD_QUENCH=1)
# Both arms: same prompts, same sampler, acceptance + t/s from response JSON
# (spec-decode-measurement-gotchas memory: acceptance lives in the RESPONSE, prompt_n proves
# cache reuse, saturating prompts pin AR at 1.0 — use mixed-length prompts).
#
# Usage (on the box, card slot GRANTED, per room choreography):
#   MODEL=<BOX>/models/dsv4-flash-UD-Q4_K_XL/...gguf \
#   DRAFT=<BOX>/models/drafter/DSV4-Flash-DSpark-draft-bf16.gguf \
#   SPEC_TYPE=draft-dspark ./quench_econ_window.sh
# Binary tip must be ds4-ports >= ebb85473 (P0-1). Verify: strings "$BIN" | grep -q DS4P_YIELD_QUENCH.

set -euo pipefail

# ⚠ PLATFORM PRECONDITION -- REFUSE, DO NOT PRETEND. Box gate (CUDA / NVMe paths). On a machine
# without them it errors, prints no verdict, and EXITS 0 -- scored as a pass by any exit-code runner.
# Placed immediately after `set` on purpose: an earlier attempt used "after the last VAR= line" and
# landed inside a PROMPTS=( ... ) array literal, breaking the script. bash -n caught it.
if [ ! -d "<BOX>" ]; then
    echo "PRECONDITION FAIL: <BOX> not present -- this gate runs on the box, not here." >&2
    echo "  Refusing rather than reporting a pass." >&2
    exit 2
fi


BIN=${BIN:-<BOX>/wt-ds4-ports/build-cuda/bin/llama-server}
MODEL=${MODEL:?set MODEL to the V4 gguf path (regen serve config uses dsv4-flash UD-Q4_K_XL)}
DRAFT=${DRAFT:?set DRAFT to the DSpark drafter gguf path}
SPEC_TYPE=${SPEC_TYPE:-draft-dspark}
PORT=${PORT:-8093}
OUT=${OUT:-/tmp/quench-econ-$(date +%Y%m%d-%H%M).txt}

command -v jq >/dev/null || { echo "jq required"; exit 1; }
# llama-server is a ~18 KB thin wrapper (same class as the dequant binary) -- the governor
# string lives in libllama-server-impl.so next to it, so check there, not the launcher
IMPL="$(dirname "$BIN")/libllama-server-impl.so"
grep -q DS4P_YIELD_QUENCH "$IMPL" 2>/dev/null || strings "$IMPL" | grep -q DS4P_YIELD_QUENCH \
  || { echo "FATAL: $IMPL lacks the governor (need ds4-ports >= ebb85473)"; exit 1; }

# mixed-length prompts: short decode-bound + long prefill-tail, per the gotchas memory
PROMPTS=(


  "Explain in one paragraph why paged KV caches reduce memory fragmentation."
  "Write a bash loop that renames all .txt files in a directory to .md, with comments."
  "$(printf 'List item %d. ' $(seq 1 200)) Summarize the pattern above in one sentence."
)

run_arm() { # $1=env $2=label
  echo "=== ARM $2 ===" | tee -a "$OUT"
  env $1 "$BIN" -m "$MODEL" -md "$DRAFT" --spec-type "$SPEC_TYPE" \
      -ngl 99 -c 8192 -np 1 -fa auto --port "$PORT" --metrics --no-warmup >/dev/null 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 2; done
  for p in "${PROMPTS[@]}"; do
    curl -sf "http://127.0.0.1:$PORT/completion" -d "$(jq -n --arg p "$p" '{prompt:$p, n_predict:200, temperature:0}')" \
      | jq -c '{timings: .timings, draft_n: (.timings.draft_n // null), draft_accepted: (.timings.draft_n_accepted // null)}' \
      | tee -a "$OUT"
  done
  curl -sf "http://127.0.0.1:$PORT/metrics" | grep -iE "quench|yield" | tee -a "$OUT" || true
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  sleep 2
}

date | tee "$OUT"
echo "binary: $BIN ($(git -C <BOX>/wt-ds4-ports rev-parse --short HEAD 2>/dev/null || echo unknown))" | tee -a "$OUT"
run_arm "" "A-baseline"
run_arm "DS4P_YIELD_QUENCH=1" "B-governor"
echo "=== done; teardown verified by caller (nvidia-smi <= 15 MiB) ===" | tee -a "$OUT"
echo "banked to: $OUT (copy same-path to ornith-1m/tools/ds4-gates/results/)"
