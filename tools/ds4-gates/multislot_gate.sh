#!/usr/bin/env bash
# MULTI-SLOT GATE -- the regime every other gate in this lane silently excluded.
#
# WHY THIS EXISTS: on 2026-08-05 the rebuilt eviction gate fired two requests CONCURRENTLY for the
# first time and sequence A came back holding sequence B's answer. My first reading was that
# eviction had corrupted block ownership. The control said otherwise: it reproduces on a GENEROUS
# pool with swaps=0, so nothing was ever evicted. Concurrency alone did it.
#
# Every gate in this lane runs -np 1. Six green gates, and two-slot serving had never once been
# exercised -- which matters more here than anywhere else, because serving concurrent sequences is
# the entire reason a paged KV cache exists. The feature's whole purpose was its blind spot.
#
# THE QUESTION THIS GATE ANSWERS, and it is one question: is the corruption OURS?
#   static  --kv-paged OFF, everything else identical
#   paged   --kv-paged ON, generous pool (must not spill)
# If static crosses too, it is upstream or it is my harness, and filing it against the paged path
# would be a second mis-attribution on top of the first.
#
# DESIGN NOTES
# - solo and concurrent are measured ON THE SAME SERVER PROCESS, so the pair differs in concurrency
#   and nothing else -- not model load, not warmup, not process state.
# - arms are INTERLEAVED and REPEATED. A single clean static run is a null, and nulls from n=1 are
#   not conclusions; a positive (corruption observed) is the robust direction, so the arm that must
#   be trusted when clean is the one that needs the repeats.
# - the paged arm asserts swaps==0. If it spills, it is not the no-eviction control it claims to be.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${MS_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=9006
REPS=${MS_REPS:-3}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/multislot
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/multislot-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"

PROMPT_A='Count from one to twenty in English words, separated by commas:'
PROMPT_B='List the first twelve letters of the English alphabet, separated by commas:'

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  model=$(basename "$M")  reps=$REPS" | tee "$OUT"

classify() {
    python3 -c '
import json,sys
try:
    j = json.load(sys.stdin)
except Exception as e:
    print("MALFORMED#" + str(e)[:80]); sys.exit()
if isinstance(j, dict) and "content" in j:
    print("CONTENT#" + j["content"].replace(chr(10)," ")[:200])
elif isinstance(j, dict) and "error" in j:
    e = j["error"]; m = e.get("message","") if isinstance(e, dict) else str(e)
    print("REFUSED#" + m.replace(chr(10)," ")[:200])
else:
    print("NOFIELD#" + json.dumps(j)[:120])
'
}

ask() {
    curl -s --max-time 180 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$1\",\"n_predict\":48,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" | classify
}

start_srv() { # $1 label  $2 extra args
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np 2 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        $2 > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

fails=0
declare -a verdicts

probe() { # $1 arm  $2 rep  $3 server args
    local tag="$1-r$2" soloA soloB concA concB swaps v
    if ! start_srv "$tag" "$3"; then
        echo "$tag: NEVER_READY" | tee -a "$OUT"; verdicts+=("$1:NEVER_READY"); fails=$((fails+1)); return
    fi
    # solo first, SAME process -- concurrency is then the only thing that changes
    soloA=$(ask "$PROMPT_A")
    soloB=$(ask "$PROMPT_B")
    ask "$PROMPT_A" > "$LOGDIR/$tag-A.txt" & local pa=$!
    ask "$PROMPT_B" > "$LOGDIR/$tag-B.txt" & local pb=$!
    wait $pa; wait $pb
    concA=$(cat "$LOGDIR/$tag-A.txt"); concB=$(cat "$LOGDIR/$tag-B.txt")
    swaps=$(grep -c "DS4P-EVICT swap" "$LOGDIR/$tag.log" 2>/dev/null); swaps=${swaps:-0}
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2

    if [ "$1" = "paged" ] && [ "$swaps" -ne 0 ]; then
        echo "$tag: INVALID pool spilled $swaps time(s) -- not a no-eviction control" | tee -a "$OUT"
        verdicts+=("$1:INVALID"); fails=$((fails+1)); return
    fi
    # ★★ THE BAR IS SEMANTIC, NOT BYTE-EXACT. Read this before "tightening" it back.
    #
    # The first version of this gate scored byte-equality against the solo run. Both arms failed it
    # 3/3, and it printed "BOTH paths corrupt -- do not file this against the paged cache". That
    # verdict was WRONG, and the reason is physics: batched inference reduces in a different order
    # depending on batch composition, so a solo request and a concurrent one are not required to be
    # bitwise identical. On a bar that no correct implementation can pass, the exonerating answer is
    # unreachable and every run reports the same thing. My gate could not have told me anything else.
    #
    # What the byte bar hid, visible the moment you look at MEANING instead of bytes:
    #   static  concA numbers, concB letters                     -- both answer their own prompt, 3/3
    #   paged   concA "commas: commas: ..." x2, then B's ALPHABET -- answers its own prompt 0/3
    # Same DIVERGED label on both. Completely different events.
    #
    # So: does each sequence still answer ITS OWN prompt? Immune to reduction-order jitter, and it
    # catches ownership and broken state, which are the only failures worth a bug report.
    local a_own b_own a_other b_other
    a_own=0;  echo "$concA" | grep -q "Three, Four, Five" && a_own=1
    a_other=0; echo "$concA" | grep -q "C, D, E, F"       && a_other=1
    b_own=0;  echo "$concB" | grep -q "C, D, E, F"        && b_own=1
    b_other=0; echo "$concB" | grep -q "Three, Four, Five" && b_other=1
    if [ "$a_own" = 1 ] && [ "$b_own" = 1 ]; then
        v=CLEAN            # both answered their own prompt; byte diffs are jitter
    elif [ "$a_other" = 1 ] || [ "$b_other" = 1 ]; then
        v=CROSSED          # a slot returned the OTHER slot's answer
    else
        v=GARBAGE          # a slot answered neither prompt -- broken state
    fi
    # byte-equality kept as a secondary observation, never as the verdict
    local bytes=exact
    { [ "$concA" != "$soloA" ] || [ "$concB" != "$soloB" ]; } && bytes=differs
    verdicts+=("$1:$v")
     echo "--- $tag  swaps=$swaps  bytes=$bytes  -> $v" | tee -a "$OUT"
    if [ "$v" != CLEAN ]; then
        echo "    soloA: $soloA" | tee -a "$OUT"
        echo "    concA: $concA" | tee -a "$OUT"
        echo "    soloB: $soloB" | tee -a "$OUT"
        echo "    concB: $concB" | tee -a "$OUT"
    fi
}

for r in $(seq 1 "$REPS"); do
    # INTERLEAVED, and the order flips each rep so a positional effect cannot masquerade as an
    # arm effect. Back-to-back reps inside one order have produced 29% drift in this lane before.
    if [ $((r % 2)) -eq 1 ]; then
        probe static "$r" ""
        probe paged  "$r" "--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128"
    else
        probe paged  "$r" "--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128"
        probe static "$r" ""
    fi
done

echo "-----" | tee -a "$OUT"
s_clean=0; s_bad=0; p_clean=0; p_bad=0
for v in "${verdicts[@]}"; do
    case "$v" in
        static:CLEAN) s_clean=$((s_clean+1));; static:*) s_bad=$((s_bad+1));;
        paged:CLEAN)  p_clean=$((p_clean+1));; paged:*)  p_bad=$((p_bad+1));;
    esac
done
echo "static: $s_clean clean / $s_bad bad     paged: $p_clean clean / $p_bad bad" | tee -a "$OUT"
echo "verdicts: ${verdicts[*]}" | tee -a "$OUT"

if [ "$s_bad" -gt 0 ] && [ "$p_bad" -gt 0 ]; then
    echo "RESULT: BOTH paths corrupt under concurrency -- NOT paged-specific. Do not file this against the paged cache." | tee -a "$OUT"
    fails=$((fails+1))
elif [ "$s_bad" -eq 0 ] && [ "$p_bad" -gt 0 ]; then
    echo "RESULT: PAGED-SPECIFIC multi-slot corruption ($p_bad/$((p_clean+p_bad)) reps) while static is clean $s_clean/$s_clean. This is ours." | tee -a "$OUT"
    fails=$((fails+1))
elif [ "$s_bad" -gt 0 ] && [ "$p_bad" -eq 0 ]; then
    echo "RESULT: static corrupts and paged does not -- unexpected; treat the harness as suspect before anything else." | tee -a "$OUT"
    fails=$((fails+1))
else
    echo "MULTI-SLOT GATE: PASS both paths clean across $REPS reps" | tee -a "$OUT"
fi
echo "log: $OUT"
exit $fails
