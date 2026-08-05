#!/usr/bin/env bash
# quench_gate.sh — P0-1 yield-quench governor gate (DS4P_YIELD_QUENCH), CPU-runnable arms.
# Proves with counters + content comparison:
#   phase 1: forced quench fires (counter) AND greedy output is IDENTICAL to a no-spec run
#            (the quench transition corrupts nothing; spec is lossless under greedy)
#   phase 2: high-acceptance workload (draft == target model) is NOT quenched
#   phase 3: feature off => counter zero
# DEFERRED TO A GPU WINDOW (documented, not skipped silently): the mid-acceptance regime
# A/B (real 0.60-acceptance head must auto-quench to <= 1.01x of no-spec baseline).
# Design: docs/DESIGN-DS4-P0.md §3. Conventions: README.md.
#
# Usage: ./quench_gate.sh <llama-server-bin> <small-model.gguf> [port=8398]

# ⚠ Strip absolute home paths before this file is committed. Gates build their output path from
# $HOME and echo it, which writes /Users/<username>/... into the result. See _no_abs_paths.sh.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ TRAP, NOT A TRAILING CALL. The first wiring put scrub_abs_paths at the end of the file, where
# the gate's own `exit` jumped straight over it -- and a `grep -l scrub_abs_paths` still listed the
# gate as a caller, because a grep counts TEXT, not control flow. Verified end-to-end afterwards:
# the result file still carried the absolute path. On EXIT it runs whatever path the gate takes.
trap 'scrub_abs_paths "$OUT"' EXIT

set -euo pipefail

BIN="${1:?llama-server binary}"
MODEL="${2:?small model gguf}"
PORT="${3:-8398}"
URL="http://127.0.0.1:$PORT"
OUT="$(dirname "$0")/results/quench-$(date +%Y%m%d-%H%M).txt"
mkdir -p "$(dirname "$OUT")"
PASS=1

say() { echo "$*" | tee -a "$OUT"; }

start_server() { # $1 = extra env, $2 = extra server args
  env $1 "$BIN" -m "$MODEL" --port "$PORT" -np 1 -c 4096 -ngl 0 --metrics $2 \
      > /tmp/quench_gate_server.log 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "$URL/health" > /dev/null 2>&1 && return 0
    sleep 1
  done
  say "FATAL: server did not become healthy"; kill "$SRV_PID" 2>/dev/null; exit 1
}

stop_server() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }

metric() { # loud on absence (reval_gate lesson)
  local body
  if ! body=$(curl -sf "$URL/metrics"); then echo "METRICS_ENDPOINT_DOWN"; return 0; fi
  echo "$body" | awk -v n="llamacpp:$1" '$1==n {print $2; f=1} END {if(!f) print "METRIC_ABSENT"}'
}

gen() { # greedy 64-token generation; prints the content string (or FAIL)
  curl -s -X POST "$URL/completion" -H 'Content-Type: application/json' \
    -d '{"prompt": "Write the numbers one to twenty as words:", "n_predict": 64, "temperature": 0, "seed": 42, "cache_prompt": false}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('content'); print(c if isinstance(c,str) and c else 'FAIL')"
}

check() { local ok; case "$3" in ge) [ "$2" -ge "$4" ] 2>/dev/null && ok=1 || ok=0 ;; eq) [ "$2" -eq "$4" ] 2>/dev/null && ok=1 || ok=0 ;; esac
  if [ "$ok" = 1 ]; then say "  PASS: $1 ($2 $3 $4)"; else say "  FAIL: $1 ($2 not $3 $4)"; PASS=0; fi; }

say "# quench_gate $(date -Iseconds) bin=$BIN model=$(basename "$MODEL")"
say "# provenance: VERIFIED-BY-GATE. GPU-window arm (0.60-acceptance auto-quench <=1.01x) DEFERRED, see header."

say "== reference: no-spec greedy output =="
start_server "" ""
REF="$(gen)"
[ "$REF" != FAIL ] || { say "FATAL: reference generation failed"; stop_server; exit 1; }
stop_server

say "== phase 1: forced quench fires + output identical to no-spec =="
start_server "DS4P_YIELD_QUENCH=1 DS4P_QUENCH_FORCE=2" "-md $MODEL --spec-type draft-simple"
OUT1="$(gen)"
check "quench_seqs >= 1" "$(metric quench_seqs_total)" ge 1
if [ "$OUT1" = "$REF" ]; then say "  PASS: forced-quench output IDENTICAL to no-spec reference"; else say "  FAIL: output diverged after quench"; PASS=0; fi
stop_server

say "== phase 2: high-acceptance (draft==target) must NOT quench =="
start_server "DS4P_YIELD_QUENCH=1" "-md $MODEL --spec-type draft-simple"
OUT2="$(gen)"
check "quench_seqs == 0" "$(metric quench_seqs_total)" eq 0
if [ "$OUT2" = "$REF" ]; then say "  PASS: spec output identical to reference (lossless)"; else say "  FAIL: spec output diverged"; PASS=0; fi
stop_server

say "== phase 3: feature OFF => counter zero =="
start_server "" "-md $MODEL --spec-type draft-simple"
OUT3="$(gen)"
check "quench_seqs == 0 (off)" "$(metric quench_seqs_total)" eq 0
stop_server

if [ "$PASS" = 1 ]; then say "== RESULT: PASS (CPU arms; GPU-window arm deferred) =="; else say "== RESULT: FAIL =="; exit 1; fi
say "results -> $OUT"
