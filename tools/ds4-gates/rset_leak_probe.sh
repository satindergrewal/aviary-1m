#!/usr/bin/env bash
# RESIDENCY-SET LEAK PROBE -- narrows, does not fix.
#
# FINDING BEING NARROWED (2026-08-05): every --kv-paged server aborts at exit with
#   ggml/src/ggml-metal/ggml-metal-device.m:656: GGML_ASSERT([rsets->data count] == 0) failed
# whose own comment reads "most likely you haven't deallocated all Metal resources before exiting".
# It is UPSTREAM's assert (8ce774a1), it fires in 100% of paged runs including zero-swap controls,
# and never in a static server. It has been in every paged gate log this lane has produced, unread,
# because the gates only ever read their own verdict line.
#
# The backtrace is exit -> __cxa_finalize_ranges -> device destructor, i.e. TEARDOWN. So the
# functional blast radius is small; the process was leaving anyway. It still matters: it means paged
# Metal buffers are never released, it makes every paged run exit on an abort, and upstream will not
# take a patch that trips their own assert.
#
# EACH ARM CHANGES EXACTLY ONE THING, and the questions are ordered so the first NO ends the hunt:
#   1 static, no env        does it happen at all without us?           (expect: no assert)
#   2 static, WITH env      is it the env vars rather than the flag?    (isolates env from --kv-paged)
#   3 paged, NO request     is it allocation, or is it use?             (the decisive one)
#   4 paged, one request    does serving add anything?
#   5 paged, -ncpub 0       is the unfreed resource the CPU block pool?
#
# Arm 3 is the one that pays: if the assert fires with no request ever served, the leak is in POOL
# CONSTRUCTION and nothing about attention, eviction, slots or block copies is involved.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${RL_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
P=9008
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/rsetleak
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/rset-leak-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"

arm() { # $1 label  $2 env  $3 server args  $4 send_request(0|1)
    local log="$LOGDIR/$1.log"
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    # -lv 4 IS LOAD-BEARING: the "initializing paged KV cache" presence marker only prints at that
    # verbosity. The first run of this probe omitted it and scored paged_pool=0 on an arm where the
    # paged path was demonstrably active -- a false negative on my own marker, produced by the log
    # level rather than by the code. Same trap as grepping op_paged_attn without -lv 4.
    env $2 nohup "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        $3 > "$log" 2>&1 &
    local ok=no
    for _ in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && { ok=yes; break; }
        sleep 1
    done
    if [ "$ok" != yes ]; then
        # ⚠ A server that never bound produces a log with no assert in it -- which reads EXACTLY
        # like "clean". Preconditions before counters, always.
        echo "$1: NEVER_READY (result unusable, not a clean arm)" | tee -a "$OUT"; return
    fi
    if [ "$4" = 1 ]; then
        curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
          -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0,"seed":1,"cache_prompt":false}' >/dev/null
    fi
    pkill -f "$SRV" >/dev/null 2>&1
    # give the process time to run its exit path -- the assert is IN teardown, so killing and
    # reading immediately would miss it and score a leak as clean
    for _ in $(seq 1 20); do pgrep -x llama-server >/dev/null || break; sleep 1; done
    sleep 2
    local a p
    a=$(grep -c "GGML_ASSERT(\[rsets->data count\] == 0)" "$log" 2>/dev/null); a=${a:-0}
    p=$(grep -c "initializing paged KV cache" "$log" 2>/dev/null); p=${p:-0}
    local note=""
    # a PAGED arm that cannot prove its pool was built is not evidence about the paged path
    case "$1" in
        *paged*) [ "$p" -lt 1 ] && note="  <-- ⚠ MARKER MISSING: cannot confirm the paged pool was built, arm is UNUSABLE" ;;
        *static*) [ "$p" -gt 0 ] && note="  <-- ⚠ a STATIC arm built a paged pool, the arms are not what they claim" ;;
    esac
    printf "%-22s paged_pool=%s  rset_assert=%s%s\n" "$1" "$p" "$a" "$note" | tee -a "$OUT"
}

PENV="DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1"
PFLAGS="--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128"

arm "1-static-noenv"   ""      ""                                          0
arm "2-static-withenv" "$PENV" ""                                          0
arm "3-paged-norequest" "$PENV" "$PFLAGS"                                  0
arm "4-paged-request"  "$PENV" "$PFLAGS"                                   1
arm "5-paged-nocpupool" "$PENV" "--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 0" 0

echo "-----" | tee -a "$OUT"
echo "read arm 3 first: an assert there means the leak is in POOL CONSTRUCTION, not in serving" | tee -a "$OUT"
echo "log: $OUT"
