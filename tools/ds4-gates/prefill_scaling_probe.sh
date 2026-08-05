#!/usr/bin/env bash
# PREFILL SCALING PROBE -- does the paged prefill penalty GROW with context length?
#
# The long-context gate measured paged prefill at 2.18x static on a 25.7K prompt (94.3 s vs 43.3 s
# per request, consistent across requests). That number on its own does not say whether the paged
# path can reach the sizes this lane exists for.
#
# THE ONLY QUESTION THAT MATTERS FOR THE NORTH STAR: is 2.18x a fixed overhead, or does the ratio
# climb with length?
#   FLAT     a constant multiplier. Unpleasant, survivable, and it does not get worse at 128K or 1M.
#   CLIMBING the paged path has a worse complexity than static, and there is a context length beyond
#            which it is simply unusable -- which is exactly the length the lane cares about.
# One number at one length cannot distinguish these. A curve can.
#
# n_predict=1 so the measurement is PREFILL, not generation. Decode cost would otherwise ride along
# and dilute the very thing being measured.
#
# ⚠ ratios are computed per length from paired arms measured in the SAME process, and each length is
# measured twice, because a single timing is a sample and this lane has already been bitten by
# quoting a rate from a window too short for the unit of work.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${PS_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
P=9012
CTX=${PS_CTX:-32768}
# ⚠ LINES, NOT TOKENS, AND THE CONVERSION IS MEASURED NOT GUESSED. The first run used
# "200 800 2400 4800" on an assumed ~12 tokens/line. The very first data point reported 4297 tokens
# for 200 lines -- 21.5 tokens/line -- which would have put the last two points at 51K and 103K
# tokens against a 32768 window. Both would have truncated or been refused, and a truncated prompt
# times FASTER, so the ratio curve would have bent downward and I would have read "the penalty
# shrinks at length" off arithmetic that never happened.
# Caught from the first point rather than after the run. The guard below now enforces it.
LENS=${PS_LENS:-"200 600 1000 1400"}     # ~4.3K / 12.9K / 21.5K / 30.1K tokens at 21.5 tok/line
BS=${PS_BS:-32}                          # --kv-block-size for the paged arm
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/prefillscale-bs${PS_BS:-32}
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/prefill-scaling-bs${PS_BS:-32}-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  ctx=$CTX" | tee "$OUT"

mkreq() { # $1 lines
    python3 - "$1" "$LOGDIR/req-$1.json" <<'PY'
import json,sys
n, out = int(sys.argv[1]), sys.argv[2]
body = "\n".join(f"Note {i}: the quick brown fox jumps over the lazy dog near the old stone bridge."
                 for i in range(n))
json.dump({"prompt": body + "\n\nSummarise in one word:", "n_predict": 1,
           "temperature": 0, "seed": 1, "cache_prompt": False}, open(out, "w"))
PY
}

start_srv() { # $1 label  $2 extra args
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        $2 > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

declare -A T N
arm() { # $1 name  $2 server args
    if ! start_srv "$1" "$2"; then
        echo "$1: NEVER_READY -- results unusable" | tee -a "$OUT"; return 1
    fi
    local marker; marker=$(grep -c "initializing paged KV cache" "$LOGDIR/$1.log")
    case "$1" in
        paged)  [ "$marker" -lt 1 ] && { echo "paged: FAIL no pool built" | tee -a "$OUT"; pkill -f "$SRV"; return 1; } ;;
        static) [ "$marker" -gt 0 ] && { echo "static: FAIL built a paged pool" | tee -a "$OUT"; pkill -f "$SRV"; return 1; } ;;
    esac
    for L in $LENS; do
        mkreq "$L"
        local best="" toks=""
        for rep in 1 2; do          # two samples per point; keep the FASTER (least noise-contaminated)
            local t0 t1 dt r
            t0=$(python3 -c 'import time;print(time.time())')
            r=$(curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion \
                 -H 'Content-Type: application/json' --data-binary "@$LOGDIR/req-$L.json")
            t1=$(python3 -c 'import time;print(time.time())')
            dt=$(python3 -c "print(f'{$t1-$t0:.2f}')")
            toks=$(echo "$r" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("tokens_evaluated",-1))
except Exception: print(-1)')
            [ -z "$best" ] && best=$dt
            best=$(python3 -c "print(min($best,$dt))")
        done
        T["$1-$L"]=$best; N["$L"]=$toks
        # a point that did not fit the window is not a slower point, it is a DIFFERENT experiment.
        # Truncated prompts are FASTER, which bends the curve the flattering way.
        if [ "$toks" -ge "$CTX" ] || [ "$toks" -lt 1 ]; then
            printf "  %-6s lines=%-5s tokens=%-6s OVER/INVALID vs ctx=%s -- point DISCARDED\n" "$1" "$L" "$toks" "$CTX" | tee -a "$OUT"
            unset "T[$1-$L]"
            continue
        fi
        printf "  %-6s lines=%-5s tokens=%-6s prefill=%ss\n" "$1" "$L" "$toks" "$best" | tee -a "$OUT"
    done
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
}

echo "--- STATIC ---" | tee -a "$OUT"; arm static "" || exit 1
echo "--- PAGED ---"  | tee -a "$OUT"; arm paged "--kv-paged --kv-block-size $BS -ngpub 8192 -ncpub 512" || exit 1

echo "-----" | tee -a "$OUT"
printf "%-8s %-8s %-9s %-9s %s\n" tokens static paged ratio verdict | tee -a "$OUT"
prev=""
for L in $LENS; do
    s=${T[static-$L]:-}; p=${T[paged-$L]:-}; n=${N[$L]:-0}
    [ -z "$s" ] || [ -z "$p" ] && continue
    r=$(python3 -c "print(f'{$p/$s:.2f}')")
    d=""
    if [ -n "$prev" ]; then
        d=$(python3 -c "print('climbing' if $r > $prev*1.15 else ('flat' if $r > $prev*0.85 else 'falling'))")
    fi
    printf "%-8s %-8s %-9s %-9s %s\n" "$n" "${s}s" "${p}s" "${r}x" "$d" | tee -a "$OUT"
    prev=$r
done
echo | tee -a "$OUT"
echo "FLAT ratio  => fixed overhead; survives to 128K/1M unchanged." | tee -a "$OUT"
echo "CLIMBING    => worse complexity than static; there is a length past which paged is unusable," | tee -a "$OUT"
echo "               and it is the length this lane exists for. That would be the real finding." | tee -a "$OUT"
echo "log: $OUT"
