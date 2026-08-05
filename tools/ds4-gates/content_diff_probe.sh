#!/usr/bin/env bash
# CONTENT-DIFF PROBE -- #1(c): what differs between a CLEAN rep and a CORRUPT rep of the SAME run?
#
# WHY THIS EXISTS. Ten hypotheses for the ~33% residual multi-slot corruption have been excluded,
# six by measurement and four by reading:
#   1 allocator/scheduler   block traces BYTE-IDENTICAL in clean and corrupt reps
#   2 GPU concurrency       GGML_METAL_CONCURRENCY_DISABLE=1 -> still corrupt
#   3 write-grid overrun    dispatch is n_tokens, ubatch-sized
#   4 can_reuse staleness   returns false
#   5 unwritten-cell reads  DS4P_KV_POISON=1 -> unchanged (weak: cannot distinguish NaN garbage)
#   6 block_table stride    tensor and kernel take max_blocks from the SAME context
#   7 stride, second writer set_max_blocks(num_gpu_blocks) is init_full()'s RESERVE ctx, never races
#   8 -1 padding read       nblk derives from q_pos, so a seq never walks past its own blocks
#   9 non-uniform return    n_tokens_total = q->ne[2] = the UBATCH, so the clamp cannot unmatch a row
#  10 nsg/dispatch lockstep n_groups_x is computed from the FINAL nsg, after the smem halving
#
# What survives all ten: the write path puts the right tokens in the right blocks, deterministically,
# and one run in three still reads them wrongly. So the question is no longer WHICH blocks -- it is
# WHAT ARGUMENTS the kernel received on the rep that failed.
#
# ★ THE DESIGN: run the SAME concurrent pair N times against a FRESH server each rep, capture the
# full ARGDUMP per rep, classify each rep CLEAN/CROSSED/GARBAGE, then diff the argdumps of the clean
# reps against the corrupt ones. Any argument that is constant within each class and differs BETWEEN
# classes is the mechanism. If the argdumps are identical across classes, the cause is NOT in the
# arguments -- which is itself a hard result, and points at the kernel's internal state.
#
# ⚠ 6 REPS MINIMUM, and that is not a style preference. The failure rate is ~33%, so a 3-rep run has
# a ~30% chance of showing all-clean by luck. Judging this bug on 3 samples is exactly what made the
# first three fix attempts wrong. n<6 is refused, loudly.
#
# ⚠ PRESENCE FIRST. If no rep is corrupt, this probe proves NOTHING and says so (exit 2). A diff
# needs both classes populated; "all clean" here means the bug did not fire today, not that it is
# fixed. Same for all-corrupt.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${CD_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${CD_PORT:-9024}
REPS=${CD_REPS:-6}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/contentdiff
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/content-diff-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

if [ "$REPS" -lt 6 ]; then
    echo "REFUSED: CD_REPS=$REPS. The failure rate is ~33%; fewer than 6 reps cannot separate" | tee "$OUT"
    echo "  'clean' from 'did not fire'. Re-run with CD_REPS>=6." | tee -a "$OUT"
    exit 2
fi

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
# semantic bar, case-insensitive -- byte-equality is unsatisfiable across batch compositions
verdict() {
    local ao=0 bo=0 ax=0 bx=0
    echo "$1" | grep -qi "three, four, five" && ao=1
    echo "$1" | grep -qi "c, d, e, f"        && ax=1
    echo "$2" | grep -qi "c, d, e, f"        && bo=1
    echo "$2" | grep -qi "three, four, five" && bx=1
    if [ $ao = 1 ] && [ $bo = 1 ]; then echo CLEAN
    elif [ $ax = 1 ] || [ $bx = 1 ]; then echo CROSSED
    else echo GARBAGE; fi
}

declare -a verds
for r in $(seq 1 "$REPS"); do
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # ⚠ CD_ARGDUMP=off is a ONE-FACTOR ARM, not a convenience. ARGDUMP writes ~4200 log lines per
    # rep, and logging changes dispatch timing -- instrumentation that masks a race is the classic
    # Heisenbug. If the corruption rate differs between argdump on and off, the ARGDUMP runs are
    # uninterpretable and every argument-level conclusion drawn from them is void.
    ARGDUMP_ENV="DS4P_ARGDUMP=${CD_ARGDUMP:-600}"
    [ "${CD_ARGDUMP:-600}" = "off" ] && ARGDUMP_ENV="DS4P_ARGDUMP_DISABLED=1"
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 $ARGDUMP_ENV \
        nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np 2 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        --kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128 > "$LOGDIR/rep$r.log" 2>&1 &
    ok=0
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && { ok=1; break; }
        sleep 1
    done
    if [ "$ok" != 1 ]; then verds+=(NEVER_READY); echo "  rep$r: NEVER_READY" | tee -a "$OUT"; continue; fi

    ask "$PROMPT_A" > "$LOGDIR/rep$r-A.txt" & pa=$!
    ask "$PROMPT_B" > "$LOGDIR/rep$r-B.txt" & pb=$!
    wait $pa; wait $pb
    v=$(verdict "$(cat "$LOGDIR/rep$r-A.txt")" "$(cat "$LOGDIR/rep$r-B.txt")")
    verds+=("$v"); echo "  rep$r: $v" | tee -a "$OUT"
    grep -a "ARGDUMP" "$LOGDIR/rep$r.log" > "$LOGDIR/rep$r.argdump" 2>/dev/null
    echo "    argdump lines: $(wc -l < "$LOGDIR/rep$r.argdump" | tr -d ' ')" | tee -a "$OUT"
done
pkill -f "$SRV" >/dev/null 2>&1; sleep 1

echo "-----" | tee -a "$OUT"
echo "verdicts: ${verds[*]}" | tee -a "$OUT"

CLEANS=""; DIRTY=""
i=0
for v in "${verds[@]}"; do
    i=$((i+1))
    case "$v" in
        CLEAN)   CLEANS="$CLEANS $i" ;;
        CROSSED|GARBAGE) DIRTY="$DIRTY $i" ;;
    esac
done
echo "clean reps:$CLEANS" | tee -a "$OUT"
echo "dirty reps:$DIRTY" | tee -a "$OUT"

# ⚠ BOTH CLASSES REQUIRED. A diff with an empty side is not a diff.
if [ -z "$(echo $CLEANS)" ] || [ -z "$(echo $DIRTY)" ]; then
    echo | tee -a "$OUT"
    echo "INCONCLUSIVE: need at least one CLEAN and one DIRTY rep to diff; got clean=[$CLEANS]" | tee -a "$OUT"
    echo "  dirty=[$DIRTY]. All-clean means the bug did not fire this run, NOT that it is fixed." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

c1=$(echo $CLEANS | awk '{print $1}')
d1=$(echo $DIRTY  | awk '{print $1}')
echo | tee -a "$OUT"
echo "=== ARGDUMP DIFF: clean rep$c1 vs dirty rep$d1 ===" | tee -a "$OUT"
# normalise volatile fields (pointers/timings) so a diff shows ARGUMENTS, not addresses
norm() { sed -E 's/0x[0-9a-f]+/0xPTR/g; s/[0-9]+\.[0-9]+ ?ms//g' "$1"; }
norm "$LOGDIR/rep$c1.argdump" > "$LOGDIR/c.norm"
norm "$LOGDIR/rep$d1.argdump" > "$LOGDIR/d.norm"
if diff -q "$LOGDIR/c.norm" "$LOGDIR/d.norm" >/dev/null 2>&1; then
    echo "IDENTICAL: the kernel received the SAME arguments on a clean and a corrupt rep." | tee -a "$OUT"
    echo "  ⇒ the cause is NOT in the dispatch arguments. Everything the host tells the kernel" | tee -a "$OUT"
    echo "    matches; the divergence is inside the kernel or in the DATA the blocks hold." | tee -a "$OUT"
    echo "CONTENT-DIFF: ARGS-EXONERATED" | tee -a "$OUT"
else
    echo "DIFFERENT: the argdumps diverge. First 40 differing lines:" | tee -a "$OUT"
    diff "$LOGDIR/c.norm" "$LOGDIR/d.norm" | head -40 | tee -a "$OUT"
    echo "CONTENT-DIFF: ARGS-DIVERGE  <- the differing field is the mechanism" | tee -a "$OUT"
fi
echo "log: $OUT"
