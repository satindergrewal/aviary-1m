#!/usr/bin/env bash
# SLOT-ID REUSE PROBE -- does the -np 2 slot-A corruption require a PRIOR finished request on that slot?
#
# HYPOTHESIS (from reading, not yet measured). request_id IS the slot id and the server REUSES slot
# ids -- server-context.cpp:1543 says so, and llama-paged-scheduler-impl.cpp:250 guards finish()
# against exactly that: "erase only OUR mapping: a new request may already have reused this id
# ... erasing its entry orphans the new request". The scheduler applies that guard:
#
#     auto it = id_to_group.find(group.request_id);
#     if (it != id_to_group.end() && it->second == &group) { id_to_group.erase(it); }
#
# llama_kv_cache_paged::free_blocks does NOT (llama-kv-cache-paged.cpp:454-456):
#
#     seq_rm(group.request_id, llama_pos{}, llama_pos{});
#     sequence_blocks.erase(group.request_id);
#
# So a stale group finishing after a new request has taken the same slot would wipe the LIVE
# request's KV. Slot 0 is the slot the multislot gate finds corrupted; slot 1 is the one it finds
# clean.
#
# THE PREDICTION THIS TESTS: corruption should require a prior FINISHED request on that slot.
#   COLD arm  concurrent A+B as the very FIRST requests on a fresh server -> predicted CLEAN
#   WARM arm  a solo request first (so slot 0 finishes), then concurrent A+B -> predicted CORRUPT
#
# ⚠ The WARM arm is the control and it must REPRODUCE, or the COLD arm proves nothing: "clean" is a
# null, and a null from an arm whose partner also came out clean says only that the bug did not fire
# today. Both arms run in the SAME invocation, on fresh servers, and the warm arm is checked first.
#
# ⚠ Semantic bar, case-insensitive, per multislot_gate.sh: does each sequence still answer ITS OWN
# prompt? Byte-equality is unsatisfiable across different batch compositions.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${SR_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=9014
REPS=${SR_REPS:-2}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/slotreuse
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/slot-reuse-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

PROMPT_A='Count from one to twenty in English words, separated by commas:'
PROMPT_B='List the first twelve letters of the English alphabet, separated by commas:'
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS" | tee "$OUT"

classify() {
    python3 -c '
import json,sys
try: j=json.load(sys.stdin)
except Exception as e: print("MALFORMED#"+str(e)[:60]); sys.exit()
if isinstance(j,dict) and "content" in j: print("CONTENT#"+j["content"].replace(chr(10)," ")[:160])
elif isinstance(j,dict) and "error" in j:
    e=j["error"]; print("REFUSED#"+(e.get("message","") if isinstance(e,dict) else str(e))[:160])
else: print("NOFIELD#"+json.dumps(j)[:100])
'
}
ask() {
    curl -s --max-time 180 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$1\",\"n_predict\":48,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" | classify
}
start_srv() { # $1 label
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np 2 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        --kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128 > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}
verdict() { # $1 outA  $2 outB -> CLEAN | CROSSED | GARBAGE
    local ao=0 bo=0 ax=0 bx=0
    echo "$1" | grep -qi "three, four, five" && ao=1
    echo "$1" | grep -qi "c, d, e, f"        && ax=1
    echo "$2" | grep -qi "c, d, e, f"        && bo=1
    echo "$2" | grep -qi "three, four, five" && bx=1
    if [ $ao = 1 ] && [ $bo = 1 ]; then echo CLEAN
    elif [ $ax = 1 ] || [ $bx = 1 ]; then echo CROSSED
    else echo GARBAGE; fi
}

declare -a cold warm
for r in $(seq 1 "$REPS"); do
    # --- WARM: a solo request finishes on slot 0 FIRST, then the concurrent pair (the control) ---
    if start_srv "warm-r$r"; then
        ask "$PROMPT_A" >/dev/null                       # slot 0 takes this, finishes, frees
        ask "$PROMPT_A" > "$LOGDIR/warm-r$r-A.txt" & pa=$!
        ask "$PROMPT_B" > "$LOGDIR/warm-r$r-B.txt" & pb=$!
        wait $pa; wait $pb
        v=$(verdict "$(cat "$LOGDIR/warm-r$r-A.txt")" "$(cat "$LOGDIR/warm-r$r-B.txt")")
        warm+=("$v"); echo "  warm-r$r (prior finish on slot 0): $v" | tee -a "$OUT"
    else warm+=(NEVER_READY); echo "  warm-r$r: NEVER_READY" | tee -a "$OUT"; fi
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2

    # --- COLD: the concurrent pair is the VERY FIRST thing this server ever sees ---
    if start_srv "cold-r$r"; then
        ask "$PROMPT_A" > "$LOGDIR/cold-r$r-A.txt" & pa=$!
        ask "$PROMPT_B" > "$LOGDIR/cold-r$r-B.txt" & pb=$!
        wait $pa; wait $pb
        v=$(verdict "$(cat "$LOGDIR/cold-r$r-A.txt")" "$(cat "$LOGDIR/cold-r$r-B.txt")")
        cold+=("$v"); echo "  cold-r$r (no prior request):      $v" | tee -a "$OUT"
    else cold+=(NEVER_READY); echo "  cold-r$r: NEVER_READY" | tee -a "$OUT"; fi
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
done

echo "-----" | tee -a "$OUT"
wbad=0; cbad=0
for v in "${warm[@]}"; do [ "$v" != CLEAN ] && wbad=$((wbad+1)); done
for v in "${cold[@]}"; do [ "$v" != CLEAN ] && cbad=$((cbad+1)); done
echo "warm: ${warm[*]}   ($wbad/${#warm[@]} not clean)" | tee -a "$OUT"
echo "cold: ${cold[*]}   ($cbad/${#cold[@]} not clean)" | tee -a "$OUT"
echo | tee -a "$OUT"
# ⚠ CONTROL FIRST. If warm does not reproduce, cold proves nothing either way.
if [ "$wbad" -eq 0 ]; then
    echo "INCONCLUSIVE: the WARM control did not reproduce the corruption, so the COLD arm is" | tee -a "$OUT"
    echo "  uninterpretable -- 'clean' here would only mean the bug did not fire today." | tee -a "$OUT"
    exit 2
fi
if [ "$cbad" -eq 0 ]; then
    echo "RESULT: corruption requires a PRIOR FINISHED request on the slot. The slot-id reuse path" | tee -a "$OUT"
    echo "  (free_blocks -> seq_rm/erase on request_id, unguarded) is IMPLICATED." | tee -a "$OUT"
else
    echo "RESULT: corruption fires with NO prior request. Slot-id reuse is NOT the whole story --" | tee -a "$OUT"
    echo "  the hypothesis is wrong or incomplete." | tee -a "$OUT"
fi
echo "log: $OUT" | tee -a "$OUT"
