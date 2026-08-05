#!/usr/bin/env bash
# LONG-CONTEXT GATE -- the coverage hole that matters most to what this lane is FOR.
#
# WHY THIS EXISTS: after closing the -np 1 hole (which immediately produced a 3/3 corruption bug), I
# swept the other pinned parameters across every gate in this directory. block size, -np, and the KV
# cache types are all varied somewhere. CONTEXT SIZE is not:
#
#     -c 300 x2   -c 2048 x2   -c 4096 x13   -c 8192 x6   -c 16384 x3   -c 40960 x2
#
# The workhorse gates run at 4K. The north star is "as much big context window as they can have to
# their limits". A paged KV cache validated only at 4K has been validated at a size where paging is
# almost pointless -- three blocks, one table, no growth. Everything that makes long context hard
# (multi-block tables, table growth, many blocks per attention call) is untested.
#
# ⚠ THIS GATE COMPARES PAGED AGAINST STATIC, NOT AGAINST A FIXED EXPECTED STRING. A model can fail
# to retrieve at depth for ordinary reasons -- quantisation, position extrapolation, the prompt. If
# only the paged arm were run, every one of those would present as a paged bug. Static is the
# control, and if STATIC cannot retrieve, the paged result is UNINTERPRETABLE and this gate says so
# rather than reporting a failure it cannot attribute.
#
# ARMS: needle at 3 depths x {static, paged}. Both at -c 32768, -np 1.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${LC_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
P=9009
CTX=${LC_CTX:-32768}
FILL=${LC_FILL:-14000}      # approximate prompt tokens
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/longctx
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/long-context-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  ctx=$CTX  fill~${FILL}tok" | tee "$OUT"

NEEDLE="MAGENTA-7742"

mkreq() { # $1 depth_percent -> writes a JSON request body
    python3 - "$1" "$FILL" "$NEEDLE" "$LOGDIR/req-$1.json" <<'PY'
import json,sys
depth, fill, needle, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
# ~12 tokens per filler line, so lines ~= fill/12
lines = max(1, fill // 12)
at = max(1, int(lines * depth / 100))
buf = []
for i in range(lines):
    if i == at:
        buf.append(f"Note {i}: The secret passcode is {needle}. Remember it.")
    else:
        buf.append(f"Note {i}: the quick brown fox jumps over the lazy dog near the old stone bridge.")
prompt = ("Below are many numbered notes. One of them contains a secret passcode.\n\n"
          + "\n".join(buf)
          + "\n\nQuestion: what is the secret passcode? Answer with the code only.\nAnswer:")
json.dump({"prompt": prompt, "n_predict": 16, "temperature": 0, "seed": 1,
           "cache_prompt": False}, open(out, "w"))
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

fails=0
declare -A found
declare -A ptok

probe() { # $1 arm  $2 extra args
    local log="$LOGDIR/$1.log" marker
    if ! start_srv "$1" "$2"; then
        echo "$1: NEVER_READY -- results unusable, not a clean arm" | tee -a "$OUT"; fails=$((fails+1)); return
    fi
    # presence marker, checked per arm and in BOTH directions
    marker=$(grep -c "initializing paged KV cache" "$log" 2>/dev/null); marker=${marker:-0}
    case "$1" in
        paged)  [ "$marker" -lt 1 ] && { echo "paged: FAIL no paged pool built -- arm is not paged" | tee -a "$OUT"; fails=$((fails+1)); pkill -f "$SRV"; return; } ;;
        static) [ "$marker" -gt 0 ] && { echo "static: FAIL built a paged pool -- arm is not static" | tee -a "$OUT"; fails=$((fails+1)); pkill -f "$SRV"; return; } ;;
    esac
    for d in 10 50 90; do
        mkreq "$d"
        local resp txt n
        resp=$(curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion \
               -H 'Content-Type: application/json' --data-binary "@$LOGDIR/req-$d.json")
        txt=$(echo "$resp" | python3 -c 'import json,sys
try: j=json.load(sys.stdin)
except Exception as e: print("MALFORMED:"+str(e)[:60]); raise SystemExit
print(j.get("content","<<NO CONTENT: "+json.dumps(j)[:100]+">>").replace(chr(10)," ")[:120])')
        n=$(echo "$resp" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("tokens_evaluated",-1))
except Exception: print(-1)')
        ptok["$1-$d"]=$n
        if echo "$txt" | grep -q "$NEEDLE"; then found["$1-$d"]=1; else found["$1-$d"]=0; fi
        printf "  %-6s depth %2s%%  prompt_tokens=%-6s found=%s  %s\n" "$1" "$d" "$n" "${found[$1-$d]}" "$txt" | tee -a "$OUT"
    done
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
}

echo "--- STATIC (control) ---" | tee -a "$OUT"
probe static ""
echo "--- PAGED ---" | tee -a "$OUT"
probe paged "--kv-paged --kv-block-size 32 -ngpub 4096 -ncpub 512"

echo "-----" | tee -a "$OUT"
# ⚠ VERIFY THE PROMPT WAS ACTUALLY LONG. A truncated or rejected prompt retrieves nothing and looks
# exactly like a retrieval failure; and a SHORT prompt retrieves everything and looks like a pass.
# Either way the context size is the thing under test, so it gets checked, not assumed.
for k in "${!ptok[@]}"; do
    if [ "${ptok[$k]}" -lt 4096 ]; then
        echo "FAIL $k evaluated only ${ptok[$k]} prompt tokens -- this is not a long-context run" | tee -a "$OUT"
        fails=$((fails+1))
    fi
done

s_ok=$(( ${found[static-10]:-0} + ${found[static-50]:-0} + ${found[static-90]:-0} ))
p_ok=$(( ${found[paged-10]:-0}  + ${found[paged-50]:-0}  + ${found[paged-90]:-0} ))
echo "retrieved: static $s_ok/3   paged $p_ok/3" | tee -a "$OUT"

if [ "$s_ok" -eq 0 ]; then
    echo "RESULT: UNINTERPRETABLE -- the STATIC control retrieved nothing, so this model/quant cannot" | tee -a "$OUT"
    echo "        do the task at this length and the paged arm proves nothing either way." | tee -a "$OUT"
    fails=$((fails+1))
elif [ "$p_ok" -lt "$s_ok" ]; then
    echo "RESULT: PAGED LOSES CONTEXT -- static $s_ok/3, paged $p_ok/3 at ${CTX} ctx. This is ours." | tee -a "$OUT"
    fails=$((fails+1))
else
    echo "LONG-CONTEXT GATE: PASS paged matches static ($p_ok/3 vs $s_ok/3) at ${CTX} ctx" | tee -a "$OUT"
fi
echo "log: $OUT"
exit $fails
