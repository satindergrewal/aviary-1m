#!/usr/bin/env bash
# CROSS-MODEL PAGED GATE. the owner's concern: different models handle KV/context differently,
# does the paged path (and the new memory policy) take over from static without breaking things?
# For each model: run STATIC and PAGED, compare OUTPUT SHA, and record RSS + system free.
# A paged path that is fast but answers differently is broken; a paged path that eats the
# machine is broken; both are checked here.

# ⚠ Strip absolute home paths before this file is committed. Gates build their output path from
# $HOME and echo it, which writes /Users/<username>/... into the result. See _no_abs_paths.sh.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ TRAP, NOT A TRAILING CALL. The first wiring put scrub_abs_paths at the end of the file, where
# the gate's own `exit` jumped straight over it -- and a `grep -l scrub_abs_paths` still listed the
# gate as a caller, because a grep counts TEXT, not control flow. Verified end-to-end afterwards:
# the result file still carried the absolute path. On EXIT it runs whatever path the gate takes.
trap 'scrub_abs_paths "$OUT"' EXIT

set -uo pipefail
B=${B:-$HOME/Documents/GitHub/llama.cpp-ds4ports/build-metal/bin/llama-server}
P=${P:-9047}
CTX=${CTX:-8192}
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/paged-multimodel-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
fails=0; skipped=0
echo "binary: $B  tip=$(cd "$(dirname "$B")/../.." && git rev-parse --short HEAD)" | tee "$OUT"

arm() { # $1 model  $2 label  $3 extra args
    pkill -f "$B" >/dev/null 2>&1; sleep 3
    # shellcheck disable=SC2086
    nohup env ${HYBENV:-} "$B" -m "$1" -ngl 99 -c $CTX -np 1 -b 512 -ub 512 \
        --port $P --no-warmup -lv 4 $3 > /tmp/mm.log 2>&1 &
    local ok=0
    for i in $(seq 1 300); do
        c=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" 2>/dev/null)
        [ "$c" = "200" ] && { ok=1; break; }
        pgrep -f "llama-server .*--port $P" >/dev/null || break
        sleep 1
    done
    if [ "$ok" != 1 ]; then
        local why; why=$(grep -iE "error|not supported|assert" /tmp/mm.log | tail -1 | cut -c1-90)
        printf '  %-8s DID NOT SERVE -> %s\n' "$2" "${why:-<no error>}" | tee -a "$OUT"
        pkill -f "$B" >/dev/null 2>&1; return
    fi
    local PID sha rss free pol
    PID=$(pgrep -f "llama-server .*--port $P" | head -1)
    sha=$(curl -s -X POST "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' \
          -d '{"prompt":"List three colours and one fact about each:","n_predict":64,"temperature":0,"cache_prompt":false}' \
          | python3 -c 'import json,sys,hashlib;print(hashlib.sha1(json.load(sys.stdin).get("content","").encode()).hexdigest()[:10])' 2>/dev/null)
    rss=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' | awk '{printf "%.2f",$1/1048576}')
    free=$(memory_pressure 2>/dev/null | grep -i "free percentage" | grep -oE '[0-9]+')
    pol=$(grep -m1 "memory policy" /tmp/mm.log | sed 's/.*policy: //' | cut -c1-40)
    printf '  %-8s sha=%-11s RSS=%6s GB  free=%s%%  %s\n' "$2" "${sha:-FAIL}" "$rss" "$free" "$pol" | tee -a "$OUT"
    echo "$sha" > "/tmp/mm-sha-$2"
    pkill -f "$B" >/dev/null 2>&1; sleep 2
}
# ⚠ REFUSE TO RUN VACUOUSLY. This gate takes its models as arguments, and with none it fell
# straight through the loop, printed "=== done ===" and exited 0 -- a gate that tested nothing and
# reported success. It sat in the suite in that state, and a batch runner counting exit codes would
# have scored it as a pass forever.
#
# Its neighbour abort_paths_gate.sh has the SAME requirement and handles it the opposite way:
# `URL="${1:?server url}"` refuses loudly and exits non-zero. Same problem, one refuses, one
# fabricates a pass. An empty run must never be indistinguishable from a clean run.
if [ "$#" -eq 0 ]; then
    echo "usage: $0 <model.gguf> [model2.gguf ...]" >&2
    echo "  refusing to run with no models -- an empty run is not a passing run" >&2
    exit 2
fi
for m in "$@"; do
    [ -f "$m" ] || { echo "missing: $m" | tee -a "$OUT"; continue; }
    echo "=== $(basename "$m") (ctx=$CTX) ===" | tee -a "$OUT"
    rm -f /tmp/mm-sha-static /tmp/mm-sha-paged
    arm "$m" static ""
    arm "$m" paged  "--kv-paged"
    a=$(cat /tmp/mm-sha-static 2>/dev/null); b=$(cat /tmp/mm-sha-paged 2>/dev/null)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then
        echo "  -> OUTPUT IDENTICAL" | tee -a "$OUT"
    elif [ -z "$b" ]; then
        # ⚠ SEPARATE A DESIGNED REFUSAL FROM A FAILURE. The hybrid and SWA guards refuse --kv-paged
        # on purpose and say so; that is the guard working, and this gate cannot answer for that
        # model without the enabling env. Anything else that fails to serve is a real failure.
        if grep -qE "not yet supported for hybrid architectures|needs DS4P_PAGED_SWA=1" /tmp/mm.log 2>/dev/null; then
            echo "  -> SKIPPED: paged refused BY DESIGN for this architecture (set HYBENV to enable) -- gate cannot answer" | tee -a "$OUT"
            skipped=$((skipped+1))
        else
            echo "  -> *** paged did not produce output (see above) ***" | tee -a "$OUT"
            fails=$((fails+1))
        fi
    else
        echo "  -> *** OUTPUT DIVERGED: static=$a paged=$b ***" | tee -a "$OUT"
        fails=$((fails+1))
    fi
done
# ⚠ THE EXIT CODE MUST AGREE WITH THE TEXT. This gate used to print "paged did not produce output"
# and then exit 0, so any batch runner reading exit codes scored it a pass while the body said it
# had failed. A verdict nobody can act on programmatically is a verdict that gets ignored.
echo "=== done: $fails failure(s), $skipped skipped -> $OUT ===" | tee -a "$OUT"
exit $((fails > 0 ? 1 : 0))
