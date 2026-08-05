#!/usr/bin/env bash
# METAL MMA WALL -- three arms, ONE binary, one model, one prompt.
#   STATIC       --kv-paged absent            -> the tuned Metal flash-attention path (target)
#   PAGED-MMA    --kv-paged, MMA enabled      -> the simdgroup-matrix prefill path
#   PAGED-SCALAR --kv-paged, DS4P_METAL_NO_MMA=1 -> the scalar path (what MMA must beat)
#
# WALL is the headline. prompt_ms is printed but is NOT the number: quoting prompt_ms alone is
# exactly how a 21.6x that did not exist got produced once ([[arms-must-differ-in-one-thing]]).
# Arms differ in ONE thing each, and every arm gets a FRESH server so no arm pays another's
# eviction or benefits from another's warm cache.

# ⚠ Strip absolute home paths before this file is committed. Gates build their output path from
# $HOME and echo it, which writes /Users/<username>/... into the result. See _no_abs_paths.sh.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ TRAP, NOT A TRAILING CALL. The first wiring put scrub_abs_paths at the end of the file, where
# the gate's own `exit` jumped straight over it -- and a `grep -l scrub_abs_paths` still listed the
# gate as a caller, because a grep counts TEXT, not control flow. Verified end-to-end afterwards:
# the result file still carried the absolute path. On EXIT it runs whatever path the gate takes.
trap 'scrub_abs_paths "$OUT"' EXIT

set -uo pipefail

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
B=${B:-$WT/build-metal/bin/llama-server}
M=${M:-/tmp/qwen3-4b-metal-Q4_K_M.gguf}
P=${P:-8977}
NREP=${NREP:-1500}
REPS=${REPS:-5}
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/metal-mma-wall-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"

# Binary freshness: a stale binary produced a false FAIL after a revert and a false PASS after
# a broken build in this lane. State it, do not assume it.
echo "binary: $B  mtime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$B")" | tee "$OUT"
echo "tip:    $(cd "$WT" && git rev-parse --short HEAD)  dirty=$(cd "$WT" && git status --porcelain | wc -l | tr -d ' ')" | tee -a "$OUT"

python3 - "$NREP" > /tmp/mma-prompt.json <<'PY'
import json, sys
body = "alpha " * int(sys.argv[1])
print(json.dumps({"prompt": f"{body}\nRemember this: the vault code is 7741\nQuestion: repeat the fact exactly.\nAnswer:",
                  "n_predict": 32, "temperature": 0, "cache_prompt": False}))
PY

run_arm() {  # $1 label  $2 server args  $3 env prefix
    pkill -f "$B" >/dev/null 2>&1
    sleep 3
    # shellcheck disable=SC2086
    env $3 nohup "$B" -m "$M" -ngl 99 -c 16384 -np 1 -b 512 -ub 512 \
        --port "$P" --no-warmup -lv 4 $2 > "/tmp/mma-$1.log" 2>&1 &
    # Poll the STATUS CODE until 200. /health answers 503 while loading and `curl -s >/dev/null`
    # exits 0 on a 503 -- that is how an arm once reported WALL=27ms with timings of -1.
    local ready=0
    for _ in $(seq 1 150); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" 2>/dev/null)
        if [ "$code" = "200" ]; then ready=1; break; fi
        sleep 1
    done
    if [ "$ready" != "1" ]; then echo "$1: SERVER NEVER READY" | tee -a "$OUT"; return 1; fi

    for r in $(seq 1 "$REPS"); do
        local t0 t1 wall pms
        t0=$(python3 -c 'import time;print(int(time.time()*1000))')
        resp=$(curl -s -X POST "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' -d @/tmp/mma-prompt.json)
        t1=$(python3 -c 'import time;print(int(time.time()*1000))')
        wall=$((t1 - t0))
        pms=$(printf '%s' "$resp" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(round(d.get("timings",{}).get("prompt_ms",-1),1))' 2>/dev/null || echo "-1")
        printf '%-13s rep%-2s WALL=%-7s prompt_ms=%-9s\n' "$1" "$r" "$wall" "$pms" | tee -a "$OUT"
    done
    # Read the marker AFTER the reps. The paged-attention op only executes when a request is
    # served, so grepping right after readiness looks BEFORE the line can possibly exist --
    # which is why an earlier revision reported the path unproven on a run that did take it.
    # It is also INFO-level, so it is invisible at -lv 2. Both had to be fixed to see it.
    local marker
    marker=$(grep -m1 "DS4P-MMA" "/tmp/mma-$1.log" 2>/dev/null || true)
    if [ -n "$marker" ]; then echo "              marker: ${marker#*: }" | tee -a "$OUT"
    elif [ "$1" = "STATIC" ]; then echo "              marker: n/a (static never runs the paged op)" | tee -a "$OUT"
    else echo "              marker: *** ABSENT -- PATH UNPROVEN, ARM INVALID ***" | tee -a "$OUT"; fi

    # ANSWER COMPARE -- restored from the old baseline. 12/12 covers the kernel, but this is
    # the product-level check that a fast kernel is still RIGHT: a slow kernel is allowed to
    # be wrong and so is a fast one. Text is captured per arm and diffed at the end.
    printf '%s' "$resp" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("content","")[:72])' \
        > "/tmp/mma-answer-$1.txt" 2>/dev/null || true
    pkill -f "$B" >/dev/null 2>&1
    sleep 2
}

run_arm STATIC       ""            ""
run_arm PAGED-MMA    "--kv-paged"  ""
run_arm PAGED-SCALAR "--kv-paged"  "DS4P_METAL_NO_MMA=1"

echo "--- answers must agree (a fast kernel is still allowed to be wrong) ---" | tee -a "$OUT"
for a in STATIC PAGED-MMA PAGED-SCALAR; do
    printf '  %-13s %s\n' "$a" "$(cat "/tmp/mma-answer-$a.txt" 2>/dev/null)" | tee -a "$OUT"
done
if cmp -s /tmp/mma-answer-STATIC.txt /tmp/mma-answer-PAGED-MMA.txt \
   && cmp -s /tmp/mma-answer-STATIC.txt /tmp/mma-answer-PAGED-SCALAR.txt; then
    echo "  -> IDENTICAL" | tee -a "$OUT"
else
    echo "  -> *** DIVERGED -- the wall number is meaningless until this is explained ***" | tee -a "$OUT"
fi
echo "=== done -> $OUT ==="
