#!/usr/bin/env bash
# METAL DECODE WALL -- decode was never re-measured after head_dim became a function constant,
# and that constant applies to the WHOLE kernel, decode path included. The recorded 1.60x
# decode gap therefore predates the single largest change of the day.
#
# Short prompt, long generation, so the number is dominated by DECODE and not by prefill.
# bash + args passed properly (zsh does not word-split unquoted vars), status never read
# through a pipe, and a SKIP/absence is never counted as a pass.

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
M=${M:-/tmp/qwen3-4b-metal-Q4_K_M.gguf}
P=${P:-9003}
NPRED=${NPRED:-256}
REPS=${REPS:-4}
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/metal-decode-wall-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"

echo "binary: $B  mtime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$B")" | tee "$OUT"
echo "tip:    $(cd "$(dirname "$B")/../.." && git rev-parse --short HEAD)" | tee -a "$OUT"

python3 - "$NPRED" > /tmp/dec-prompt.json <<'PY'
import json, sys
print(json.dumps({"prompt": "Count slowly and describe each number in one short sentence:",
                  "n_predict": int(sys.argv[1]), "temperature": 0, "cache_prompt": False}))
PY

arm() {  # $1 label  $2 server args
    pkill -f "$B" >/dev/null 2>&1; sleep 3
    # shellcheck disable=SC2086
    nohup "$B" -m "$M" -ngl 99 -c 16384 -np 1 -b 512 -ub 512 \
        --port "$P" --no-warmup -lv 4 $2 > "/tmp/dec-$1.log" 2>&1 &
    local ready=0
    for _ in $(seq 1 150); do
        c=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" 2>/dev/null)
        [ "$c" = "200" ] && { ready=1; break; }; sleep 1
    done
    if [ "$ready" != "1" ]; then echo "$1: SERVER NEVER READY" | tee -a "$OUT"; return 1; fi
    local best=0
    for r in $(seq 1 "$REPS"); do
        tps=$(curl -s -X POST "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' \
              -d @/tmp/dec-prompt.json \
              | python3 -c 'import json,sys;print(round(json.load(sys.stdin).get("timings",{}).get("predicted_per_second",-1),2))' 2>/dev/null || echo 0)
        printf '%-14s rep%-2s decode=%-8s tok/s\n' "$1" "$r" "$tps" | tee -a "$OUT"
        awk -v a="$tps" -v b="$best" 'BEGIN{exit !(a>b)}' && best=$tps
    done
    # numbers from a crashed process are not measurements
    local crashed; crashed=$(grep -c "GGML_ASSERT" "/tmp/dec-$1.log" 2>/dev/null || echo 0)
    printf '%-14s BEST=%-8s tok/s   crashed=%s\n' "$1" "$best" "$crashed" | tee -a "$OUT"
    pkill -f "$B" >/dev/null 2>&1; sleep 2
}
arm STATIC ""
arm PAGED  "--kv-paged"
echo "=== done -> $OUT ===" | tee -a "$OUT"
