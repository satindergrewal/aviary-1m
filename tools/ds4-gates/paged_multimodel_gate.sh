#!/usr/bin/env bash
# CROSS-MODEL PAGED GATE. Satinder's concern: different models handle KV/context differently,
# does the paged path (and the new memory policy) take over from static without breaking things?
# For each model: run STATIC and PAGED, compare OUTPUT SHA, and record RSS + system free.
# A paged path that is fast but answers differently is broken; a paged path that eats the
# machine is broken; both are checked here.
set -uo pipefail
B=${B:-$HOME/Documents/GitHub/llama.cpp-ds4ports/build-metal/bin/llama-server}
P=${P:-9047}
CTX=${CTX:-8192}
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/paged-multimodel-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
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
for m in "$@"; do
    [ -f "$m" ] || { echo "missing: $m" | tee -a "$OUT"; continue; }
    echo "=== $(basename "$m") (ctx=$CTX) ===" | tee -a "$OUT"
    rm -f /tmp/mm-sha-static /tmp/mm-sha-paged
    arm "$m" static ""
    arm "$m" paged  "--kv-paged"
    a=$(cat /tmp/mm-sha-static 2>/dev/null); b=$(cat /tmp/mm-sha-paged 2>/dev/null)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "  -> OUTPUT IDENTICAL" | tee -a "$OUT"
    elif [ -z "$b" ]; then echo "  -> paged did not produce output (see above)" | tee -a "$OUT"
    else echo "  -> *** OUTPUT DIVERGED: static=$a paged=$b ***" | tee -a "$OUT"; fi
done
echo "=== done -> $OUT ===" | tee -a "$OUT"
