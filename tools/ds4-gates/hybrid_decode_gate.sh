#!/usr/bin/env bash
# HYBRID DECODE GATE. Polarity is the OPPOSITE of the taint gate:
#   taint gate  -> divergence = GOOD (proves consumption)
#   decode gate -> divergence = BUG  (paged must MATCH static)
# Multi-token greedy so decode steps dominate, not prefill.
set -uo pipefail

# ⚠ PRIVACY. This gate writes into tools/ds4-gates/results/, which is TRACKED -- and it built its
# output path from $HOME, so every run committed /Users/<username>/ into the repo. the owner's rules
# name file paths containing usernames as private data at absolute highest priority, after two
# incidents that each needed a full history rewrite.
#
# ⚠ A TRAP, NOT A TRAILING CALL. paged_multimodel_gate.sh records why: a trailing scrub was jumped
# over by the gate's own `exit`, while `grep -l scrub_abs_paths` still listed the gate as a caller,
# because grep counts TEXT and not control flow. On EXIT it runs whatever path the gate takes.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT   # ${OUT:-} : fires on early exits too, before OUT is set

WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
B=$WT/build-metal/bin/llama-server
M=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf
P=8995
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/hybrid-decode-gate-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
run() { # $1 label  $2 server args  $3 env
    pkill -f "$B" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env $3 nohup "$B" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 $2 > "/tmp/dg-$1.log" 2>&1 &
    for _ in $(seq 1 180); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break; sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] || { echo "$1 NEVER READY"|tee -a "$OUT"; echo "READYFAIL"; return 1; }
    curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Count from one to ten in English words, separated by commas:","n_predict":40,"temperature":0,"seed":1,"cache_prompt":false}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"].replace(chr(10)," ")[:150])'
    pkill -f "$B" >/dev/null 2>&1; sleep 2
}
echo "--- STATIC (reference) ---" | tee -a "$OUT"
a=$(run STATIC "" "")
echo "$a" | tee -a "$OUT"
echo "--- PAGED (self-drive) ---" | tee -a "$OUT"
b=$(run PAGED "--kv-paged --kv-block-size 32" "DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1")
echo "$b" | tee -a "$OUT"
# ★ THE PASS ABOVE IS NOT EVIDENCE ON ITS OWN. paged==static is exactly what a SILENT FALLBACK
# looks like -- that is audit finding 5. Third arm: paged + taint. If paged genuinely drives the
# whole generation, tainted KV MUST change the output. If it does not, decode never used the pool
# and the "PASS" is a tautology.
echo "--- PAGED + TAINT (tautology discriminator) ---" | tee -a "$OUT"
c=$(run PAGEDTAINT "--kv-paged --kv-block-size 32" "DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1 DS4P_PAGED_TAINT=1")
echo "$c" | tee -a "$OUT"

echo "---" | tee -a "$OUT"
# ⚠⚠ SENTINELS AND EMPTY BODIES ARE NOT OBSERVABLES (fixed 2026-08-13).
# run() tees "LABEL NEVER READY" and echoes READYFAIL; a=$(run ...) therefore captures BOTH
# lines. Dead STATIC vs dead PAGED then have different labels so a!=b and the gate printed
# DECODE GATE FAIL -- a false FAIL. Empty/error JSON from both healthy servers gives a==b==""
# and printed DECODE GATE PASS -- a tautological PASS. And the script never exited nonzero
# (exit-zero-did-nothing). Normalize every unusable arm to empty, VOID instead of PASS/FAIL,
# and exit 2 on VOID / 1 on FAIL / 0 on PASS.
normalize() {
    case "$1" in
        *NEVER\ READY*|*READYFAIL*) echo "";;
        *) echo "$1";;
    esac
}
a=$(normalize "$a"); b=$(normalize "$b"); c=$(normalize "$c")
if [ -z "$a" ] || [ -z "$b" ] || [ -z "$c" ]; then
    echo "DECODE GATE VOID: one or more arms produced no usable content (dead server or empty/error body -- see the arm lines above)." | tee -a "$OUT"
    echo "OUT=$OUT"
    exit 2
fi
if [ "$b" = "$c" ]; then
    echo "TAUTOLOGY WARNING: taint did NOT change paged decode output => decode is NOT consuming the pool; the PASS below is meaningless" | tee -a "$OUT"
    echo "DECODE GATE VOID: tautology -- paged==tainted, so MATCH-vs-static is not a paging result." | tee -a "$OUT"
    echo "OUT=$OUT"
    exit 2
else
    echo "DISCRIMINATOR OK: taint changed paged output => the pool IS driving this generation" | tee -a "$OUT"
fi

if [ "$a" = "$b" ]; then
    echo "DECODE GATE PASS: paged output MATCHES static across 40 decode tokens" | tee -a "$OUT"
    echo "OUT=$OUT"
    exit 0
else
    echo "DECODE GATE FAIL: paged DIVERGES from static -- decode is incorrect (expected: self-drive resets n_past=0 per call)" | tee -a "$OUT"
    echo "OUT=$OUT"
    exit 1
fi
