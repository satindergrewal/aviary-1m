#!/usr/bin/env bash
# reval_gate.sh — P0-2 state<->claim binding gate (DS4P_REVALIDATE).
# Proves with counters (never log lines alone):
#   phase 1: seal on save + verified load, zero failures
#   phase 2: injected corruption (DS4P_REVAL_FAULT=1) is DETECTED and degrades to
#            recompute — request still completes with valid output
#   phase 3: feature off => all counters zero (kill-switch honest)
# Design: docs/DESIGN-DS4-P0.md §2. CPU-only ok; do not run against a production serve.
#
# Usage: ./reval_gate.sh <llama-server-bin> <small-model.gguf> [port=8397]

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
PORT="${3:-8397}"
URL="http://127.0.0.1:$PORT"
OUT="$(dirname "$0")/results/reval-$(date +%Y%m%d-%H%M).txt"
mkdir -p "$(dirname "$OUT")"
PASS=1

say() { echo "$*" | tee -a "$OUT"; }

start_server() { # $1 = extra env as "K=V K=V"
  env $1 "$BIN" -m "$MODEL" --port "$PORT" -np 1 -c 4096 --cache-ram 256 -ngl 0 --metrics \
      > /tmp/reval_gate_server.log 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "$URL/health" > /dev/null 2>&1 && return 0
    sleep 1
  done
  say "FATAL: server did not become healthy"; kill "$SRV_PID" 2>/dev/null; exit 1
}

stop_server() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }

metric() { # $1 = name -> prints value; a missing endpoint or metric FAILS LOUDLY.
  # (First gate run read silent-404 zeros and "passed" vacuous checks - never again:
  #  a reader that cannot distinguish "counter is 0" from "no counter" is decoration.)
  local body
  if ! body=$(curl -sf "$URL/metrics"); then
    echo "METRICS_ENDPOINT_DOWN"
    return 0
  fi
  echo "$body" | awk -v n="llamacpp:$1" '$1==n {print $2; f=1} END {if(!f) print "METRIC_ABSENT"}'
}

completion() { # $1 = prompt text; prints OK or FAIL
  curl -s -X POST "$URL/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"$1\", \"n_predict\": 8, \"cache_prompt\": true, \"temperature\": 0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if isinstance(d.get('content'), str) else 'FAIL')"
}

# Prompts: A and B share no long prefix, so the slot swap triggers save+load traffic.
PA="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima $(printf 'mike november %.0s' {1..60})"
PB="zulu yankee xray whiskey victor uniform tango sierra romeo quebec papa $(printf 'oscar november %.0s' {1..60})"

check() { # $1 desc, $2 actual, $3 op(ge|eq), $4 want
  local ok
  case "$3" in ge) [ "$2" -ge "$4" ] && ok=1 || ok=0 ;; eq) [ "$2" -eq "$4" ] && ok=1 || ok=0 ;; esac
  if [ "$ok" = 1 ]; then say "  PASS: $1 ($2 $3 $4)"; else say "  FAIL: $1 ($2 not $3 $4)"; PASS=0; fi
}

say "# reval_gate $(date -Iseconds) bin=$BIN model=$(basename "$MODEL")"
say "# provenance: VERIFIED-BY-GATE (counters read from /metrics, not from log lines)"

say "== phase 1: seal + verified load =="
start_server "DS4P_REVALIDATE=1"
[ "$(completion "$PA")" = OK ] || { say "  FAIL: completion A"; PASS=0; }
[ "$(completion "$PB")" = OK ] || { say "  FAIL: completion B"; PASS=0; }
[ "$(completion "$PA")" = OK ] || { say "  FAIL: completion A2"; PASS=0; }
check "reval_saves >= 1"        "$(metric reval_saves_total)"         ge 1
check "reval_loads >= 1"        "$(metric reval_loads_total)"         ge 1
check "hash_fail == 0"          "$(metric reval_hash_fail_total)"     eq 0
check "identity_fail == 0"      "$(metric reval_identity_fail_total)" eq 0
check "cell_mismatch == 0"      "$(metric reval_cell_mismatch_total)" eq 0
stop_server

say "== phase 2: injected corruption detected, degrades to recompute =="
start_server "DS4P_REVALIDATE=1 DS4P_REVAL_FAULT=1"
[ "$(completion "$PA")" = OK ] || { say "  FAIL: completion A"; PASS=0; }
[ "$(completion "$PB")" = OK ] || { say "  FAIL: completion B"; PASS=0; }
R3="$(completion "$PA")"
check "hash_fail >= 1"          "$(metric reval_hash_fail_total)"     ge 1
if [ "$R3" = OK ]; then say "  PASS: request after detection still completes (recompute path)"; else say "  FAIL: recompute path broke"; PASS=0; fi
stop_server

say "== phase 3: feature OFF => counters stay zero =="
start_server ""
[ "$(completion "$PA")" = OK ] || { say "  FAIL: completion A"; PASS=0; }
[ "$(completion "$PB")" = OK ] || { say "  FAIL: completion B"; PASS=0; }
[ "$(completion "$PA")" = OK ] || { say "  FAIL: completion A2"; PASS=0; }
check "saves == 0 (off)"        "$(metric reval_saves_total)"         eq 0
check "loads == 0 (off)"        "$(metric reval_loads_total)"         eq 0
stop_server

if [ "$PASS" = 1 ]; then say "== RESULT: PASS =="; else say "== RESULT: FAIL =="; exit 1; fi
say "results -> $OUT"
