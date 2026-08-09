#!/usr/bin/env bash
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ A TRAP, NOT A TRAILING CALL. A trailing scrub is jumped over by the gate's own `exit`, while
# `grep -l scrub_abs_paths` still lists the file as a caller -- grep counts TEXT, not control flow.
# On EXIT it runs whatever path the gate takes.
# ⚠ ADDED 2026-08-09 after a full-history rewrite + force-push removed /Users/<username> from 11
# PUBLISHED commits. That cleaned the backlog; this closes the PRODUCERS. A history scrub with live
# emitters still in the tree is a fix with a regression path.
trap 'scrub_abs_paths "${OUT:-}"' EXIT
# abort_paths_gate.sh — P0-3 STEP 0: measure the CURRENT tree's dead-client abort behavior.
# Reports, per death mode (fin / rst / kill9):
#   t_detect  = client-death -> "cancel task" in the server log (should be <= poll tick + 1s)
#   t_free    = client-death -> "slot ... release" (should be <= one batch time after detect)
#   runaway   = prefill progress lines appearing AFTER death+3s (the dead client's prefill
#               continuing = the exact ECONNRESET-night failure)
# If all modes detect fast, free fast, and never run away, P0-3 closes with this script as
# its artifact (no code). Design: docs/DESIGN-DS4-P0.md §1. Gate-name credit: ds4.
#
# HARD-WON GATE LESSONS ENCODED HERE (each invalidated a whole run):
#  - filler words tokenize ~1.5-2.2 tok/word; oversize => instant rejection => vacuous pass
#  - identical prompts across modes => prompt-cache hit => no prefill => vacuous
#  - backgrounded curl IGNORES SIGINT (no job control) => "fin" mode never killed anything
#  - "stop processing" also matches NATURAL completion => must separate detect vs free vs runaway
#  - macOS `wc -l <` pads with blanks => tail -n +"  17" is invalid
#
# Usage: ./abort_paths_gate.sh <server-url> <server-log-path> [prompt-tokens=12000]
#   Server: llama-server from the tree under test, single slot free, any model. NEVER a
#   production serve. CPU serve is IDEAL (slow prefill = wide kill window).
set -euo pipefail

URL="${1:?server url}"
SRVLOG="${2:?server log path}"
NTOK="${3:-12000}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TS() { python3 -c 'import time; print(f"{time.time():.3f}")'; }
OUT="$DIR/results/abort-$(date +%Y%m%d-%H%M).txt"
mkdir -p "$DIR/results"
PASS=1

log_mark() { wc -l < "$SRVLOG" | tr -d '[:blank:]'; }

mk_prompt() { # $1 = mode tag -> unique prompt per mode (prompt-cache breaker)
  python3 - "$NTOK" "$1" <<'PY'
import sys, json
n, tag = int(sys.argv[1]), sys.argv[2]
n_words = max(64, int(n * 0.45))
base = "alpha bravo charlie delta echo foxtrot golf hotel"
words = (f"UNIQUE-{tag}-{tag} " + (base + " ") * (n_words // 8 + 1)).split()[:n_words]
print(json.dumps({"prompt": " ".join(words), "n_predict": 32, "stream": True}))
PY
}

run_death() { # $1 = fin | rst | kill9
  local mode="$1" pid mark t0 t_detect="" t_free="" runaway=0 pf
  mark=$(log_mark)
  pf=$(mktemp); mk_prompt "$mode-$$" > "$pf"
  echo "== mode=$mode" | tee -a "$OUT"

  python3 "$DIR/abort_client.py" "$URL" "$pf" "$mode" &
  pid=$!
  sleep 10

  if ! tail -n "+$mark" "$SRVLOG" | grep -q "prompt processing"; then
    echo "  INVALID: no prefill in progress at kill time" | tee -a "$OUT"
    kill -KILL "$pid" 2>/dev/null || true; PASS=0; rm -f "$pf"; sleep 12; return
  fi

  local mark_kill; mark_kill=$(log_mark)
  case "$mode" in
    kill9) kill -KILL "$pid" 2>/dev/null || true ;;
    *)     kill -TERM "$pid" 2>/dev/null || true ;;   # abort_client dies per its mode
  esac
  t0=$(TS)

  for _ in $(seq 1 360); do
    local since; since=$(tail -n "+$mark_kill" "$SRVLOG")
    if [ -z "$t_detect" ] && echo "$since" | grep -q "cancel task"; then t_detect=$(TS); fi
    if echo "$since" | grep -qE "slot[[:space:]]+release"; then t_free=$(TS); break; fi
    sleep 0.5
  done

  # runaway check: the ONE in-flight batch legitimately drains after death (batch-boundary
  # semantics) - runaway means TWO OR MORE progress lines after the kill, i.e. new batches
  # kept being scheduled for a dead client
  local n_prog; n_prog=$(tail -n "+$mark_kill" "$SRVLOG" | grep -c "prompt processing" || true)
  [ "${n_prog:-0}" -ge 2 ] && runaway=1

  local d="NOT_DETECTED" f="NOT_FREED_180s"
  [ -n "$t_detect" ] && d=$(echo "$t_detect - $t0" | bc)
  [ -n "$t_free" ]   && f=$(echo "$t_free - $t0" | bc)
  echo "  t_detect_s=$d  t_free_s=$f  prefill_continued_after_death=$runaway" | tee -a "$OUT"
  [ -z "$t_free" ] && PASS=0
  rm -f "$pf"
  sleep 12
}

echo "# abort_paths_gate $(date -Iseconds) url=$URL ntok=$NTOK" | tee "$OUT"
run_death fin
run_death rst
run_death kill9
echo "# judge: t_detect <= ~2s (1s poll tick), t_free <= t_detect + one batch, runaway=0" | tee -a "$OUT"
if [ "$PASS" = 1 ]; then echo "== RESULT: MEASURED (judge latencies above) =="; else echo "== RESULT: FAIL/INVALID (see above) =="; fi | tee -a "$OUT"
echo "results -> $OUT"
