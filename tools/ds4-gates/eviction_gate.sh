#!/usr/bin/env bash
# EVICTION GATE -- covers the GPU<->CPU block-swap path.
#
# WHY THIS EXISTS: ca704d06 changed do_block_copy's staging buffer from one pool-wide size to a
# per-layer size, and nothing in this lane had ever executed that code. Every gate here runs on a
# warm box with ~88 GiB free, where the pool NEVER spills. The paged design's whole reason to exist
# is what happens when it does not fit -- and that regime was the one regime never tested.
#
# ⚠ THE PRESENCE MARKER IS THE GATE. Starving GPU blocks and then comparing outputs would pass
# perfectly if no eviction ever occurred: the run would simply be an ordinary run. "Outputs match"
# is evidence only when the path under test actually executed, so DS4P-EVICT swap lines must be
# non-zero or the comparison means nothing.
#
# ★★ THIS GATE CANNOT PASS TODAY, AND THE REASON IS NOW KNOWN (2026-08-05) -- it is NOT weak
# starvation. Under DS4P_PAGED_DRIVE=1 the self-drive path and the server's scheduler BOTH allocate
# for the same sequence (measured: 8 checkouts for a 4-block need, two distinct group objects, one
# of which IS sd_group and one of which is not). When the pool runs low, self_drive_begin fails to
# grow, FREES its own KV, and bails to the static path -- so the sequence LEAVES the pool before it
# can ever pressure it into a swap. Eviction is unreachable from the server in this configuration.
#
# One-factor, marker-verified:
#   DRIVE_ON   paged_attn config-lines=2  selfdrive_calls=8  checkouts=8  -> corrupt output
#   DRIVE_OFF  paged_attn config-lines=2  selfdrive_calls=0  checkouts=4  -> clean output
# (the "=2" was originally written as "dispatches=2" -- WRONG, that log line is throttled on a
#  config key, so 2 is what a normal run prints, not a count of dispatches. See the note below.)
#
# So a FAIL here is currently CORRECT and expected. Do not "fix" it by loosening the marker check --
# fix the allocator conflict (mutual exclusion, or self-drive sharing the scheduler's group), then
# this gate becomes meaningful for the first time. See ds4-ports-lane memory for the full trail,
# including the six wrong names this took to reach.
#
# ARMS
#   CTRL   plenty of GPU blocks -> no spill, reference output
#   EVICT  starved GPU blocks + many CPU blocks -> must spill, output MUST equal CTRL
# Run against BOTH a per-layer-geometry model (widths differ, which is what the staging fix is for)
# and a uniform one (regression).
#
# ★ 2026-08-05 LATER: self-drive is now overridable so this gate can test the hypothesis above.
# DS4P_DRIVE_ARM=0 runs it with self-drive OFF, which -- if the diagnosis is right -- lets the
# sequence STAY in the pool and actually pressure it. The default is still 1 so the historical
# (failing) configuration remains what a bare invocation measures.
#   ⚠ The flag is value-checked at llama-kv-cache-paged.cpp:1132 (`e && atoi(e) != 0`), NOT
#   presence-checked. Verified before running: a presence-checked flag would make the OFF arm
#   silently identical to the ON arm, i.e. a fake-clean arm, which this lane has already been
#   burned by once. Do not assume this of the next flag -- check it.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
P=9005
DRIVE=${DS4P_DRIVE_ARM:-1}
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/eviction-drive$DRIVE-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
PROMPT='Count from one to twenty in English words, separated by commas:'

M_MIX=$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf
M_UNI=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf

# per-arm log dir: the two DRIVE arms use identical labels, so a shared path would let the second
# run overwrite the first's evidence. Also keeps parallel jobs from clobbering each other in /tmp.
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/ev-drive$DRIVE
mkdir -p "$LOGDIR"

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  DS4P_PAGED_DRIVE=$DRIVE" | tee "$OUT"

fails=0

run() { # $1 label  $2 model  $3 extra args
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=$DRIVE DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$2" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        --kv-paged --kv-block-size 16 $3 > "$LOGDIR/ev-$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
        echo "NEVER_READY"; pkill -f "$SRV" >/dev/null 2>&1; return
    fi
    curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":48,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" \
      | classify
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
}

# ⚠ REFUSED IS NOT CORRUPTED. The old gate piped the response straight into json["content"] and
# printed whatever came out. When the starved arm was REFUSED -- a well-formed error object with no
# "content" key -- python raised KeyError, the field came back empty, and the gate announced
# "output DIFFERS after eviction -- block copy is corrupting KV". That verdict was false, and it was
# the worst kind of false: it named a memory-corruption bug that did not exist while hiding a clean,
# correct refusal that did. A gate that misnames the failure is worse than one that just fails,
# because the name is the thing anyone acts on. Classify explicitly instead.
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
    e = j["error"]
    m = e.get("message","") if isinstance(e, dict) else str(e)
    print("REFUSED#" + m.replace(chr(10)," ")[:200])
else:
    print("NOFIELD#" + json.dumps(j)[:120])
'
}

check_pair() { # $1 tag  $2 model
    echo "=== $1 ===" | tee -a "$OUT"

    local ctrl evict swaps_c swaps_e peak pool
    ctrl=$(run "$1-ctrl"  "$2" "-ngpub 512 -ncpub 128")
    echo "CTRL : $ctrl" | tee -a "$OUT"

    # ★★ STARVATION IS NOW DERIVED, NOT GUESSED. Read the history before changing this back.
    #
    # -ngpub 24 did not spill (24 blocks x 16 = 384 tokens vs a ~60-token job: no pressure). I then
    # "established" that -ngpub 8 forces a spill and wrote that into this header as fact. It was
    # FALSE, and the way it was false is the point: the 8 came from a DS4P_PAGED_DRIVE=1 run where
    # self-drive double-allocated -- 8 checkouts for a 3-block table, ALLOCFAIL, min free_before=2.
    # The BUG was manufacturing the pressure. With self-drive off the same job takes 4 blocks and
    # min free_before=5: it fits in 8 with four to spare, and the "starved" arm was an ordinary run
    # wearing a scary flag. A threshold calibrated on a contaminated run silently decalibrates the
    # moment the contamination is fixed. (Second occurrence of that class in this lane.)
    #
    # So: MEASURE the peak from the control, then starve BELOW the measured value.
    #   at the tightest allocation the log prints free_before=f_min, leaving f_min-1 free after it
    #   peak_used = total_gpu - (f_min - 1)
    # Pool-size independent, so the generous control measures the same demand the starved arm faces.
    peak=$(awk '
        /DS4P-CHECKOUT allocate/ { if (match($0,/free_before=[0-9]+/)) {
            v = substr($0, RSTART+12, RLENGTH-12) + 0; if (m=="" || v < m) m = v } }
        /initializing paged KV cache/ { if (match($0,/n_gpu_blocks=[0-9]+/))
            t = substr($0, RSTART+13, RLENGTH-13) + 0 }
        END { if (m=="" || t=="") print 0; else print t - (m - 1) }
    ' "$LOGDIR/ev-$1-ctrl.log" 2>/dev/null)
    peak=${peak:-0}
    # published to the multi-seq gate below. RESET FIRST: if this model's measurement fails and the
    # previous model's value were left standing, the next gate would size its pool from the WRONG
    # model and report a confident result about a configuration it never tested.
    MEASURED_PEAK=$peak
    if [ "$peak" -lt 2 ]; then
        echo "$1: FAIL could not measure peak block demand from the control (got '$peak') -- refusing to guess a starvation level" | tee -a "$OUT"
        fails=$((fails+1)); return
    fi
    pool=$((peak - 1)); [ "$pool" -lt 2 ] && pool=2
    echo "measured peak demand: $peak blocks -> starving to -ngpub $pool" | tee -a "$OUT"

    evict=$(run "$1-evict" "$2" "-ngpub $pool -ncpub 512")
    echo "EVICT: $evict" | tee -a "$OUT"

    swaps_c=$(grep -c "DS4P-EVICT swap" "$LOGDIR/ev-$1-ctrl.log"  2>/dev/null);  swaps_c=${swaps_c:-0}
    swaps_e=$(grep -c "DS4P-EVICT swap" "$LOGDIR/ev-$1-evict.log" 2>/dev/null); swaps_e=${swaps_e:-0}
    echo "swaps: CTRL=$swaps_c  EVICT=$swaps_e" | tee -a "$OUT"

    # ★ SECOND MARKER, opposite sign: prove the paged attention kernel ran at all in the starved arm.
    # swaps>=1 says a spill happened; it does NOT say the request was still using the pool when it
    # answered. If self_drive_begin freed its KV and bailed to the static path, the run can look
    # ordinary and correct while testing nothing. Zero here means the paged kernel never ran,
    # whatever the swap counter says. Verified two-sided: a static server's log scores 0.
    #
    # ⚠ READ THE COUNT AS PRESENCE ONLY, NEVER AS A DISPATCH COUNT. The line is THROTTLED --
    # ggml-metal-ops.cpp keys it on `static int last_key` and prints only when the config changes.
    # A healthy 48-token generation prints TWO lines (prefill nsg, then decode nsg) and then goes
    # silent forever. I read the historical "2" as "it dispatched twice and bailed" and was about
    # to bank that; two is the signature of a perfectly normal run. The header's old
    # "paged_attn_dispatches=2" label is likewise a misnomer -- corrected below.
    # This marker therefore CANNOT prove the run stayed paged for the whole generation. It proves
    # the kernel ran at least once. Do not stretch it further.
    local disp_e
    disp_e=$(grep -c "op_paged_attn" "$LOGDIR/ev-$1-evict.log" 2>/dev/null); disp_e=${disp_e:-0}
    echo "paged_attn config-lines (EVICT arm, presence only): $disp_e" | tee -a "$OUT"
    if [ "$disp_e" -lt 1 ]; then
        echo "$1: FAIL starved arm never dispatched paged attention -- it fell back to static, so it proves NOTHING" | tee -a "$OUT"
        fails=$((fails+1))
        return
    fi

    # ⚠ THE MARKER CHECK COMES FIRST. Without it the comparison below is a tautology.
    if [ "$swaps_e" -lt 1 ]; then
        echo "$1: FAIL no eviction occurred -- the starved arm never spilled, so this proves NOTHING" | tee -a "$OUT"
        fails=$((fails+1))
        return
    fi
    if [ "$evict" = "NEVER_READY" ] || [ "$ctrl" = "NEVER_READY" ]; then
        echo "$1: FAIL a server never became healthy" | tee -a "$OUT"
        fails=$((fails+1))
        return
    fi

    # ★★ WHAT THIS ARM IS ACTUALLY TESTING -- READ BEFORE "FIXING" THE VERDICT BACK.
    #
    # One sequence starved below its own peak demand CANNOT be served, and that is not a bug, it is
    # attention. Every decode step reads the entire KV history, so every block of the sequence is
    # live at every step; there is no subset that can be spilled and no cold blocks to spill. The
    # measured swap trace says exactly that: OUT 3, IN 3, OUT 3, IN 3 -- pure ping-pong until the
    # scheduler gives up. This gate spent its whole life demanding lossless output from a
    # configuration where correct behaviour is a REFUSAL.
    #
    # So the single-sequence bar is: it must TRY (swaps>=1) and then FAIL CLEANLY -- an explicit
    # error to the client, not a hang, not silence, not wrong tokens. Lossless eviction is tested
    # by check_multiseq below, which is the regime the design is actually for.
    case "$evict" in
        REFUSED#*)
            echo "$1: PASS starved to ${pool}<${peak} blocks, attempted $swaps_e swap(s), then refused CLEANLY" | tee -a "$OUT"
            echo "    client saw: ${evict#REFUSED#}" | tee -a "$OUT"
            ;;
        CONTENT#*)
            # It served the request anyway. Then the content MUST be right.
            if [ "$evict" = "$ctrl" ]; then
                echo "$1: PASS served under starvation across $swaps_e swap(s), output identical" | tee -a "$OUT"
            else
                echo "$1: FAIL served under starvation but output DIFFERS -- this IS block-copy corruption" | tee -a "$OUT"
                echo "    ctrl : ${ctrl#CONTENT#}"  | tee -a "$OUT"
                echo "    evict: ${evict#CONTENT#}" | tee -a "$OUT"
                fails=$((fails+1))
            fi
            ;;
        *)
            echo "$1: FAIL starved arm neither served nor refused cleanly: $evict" | tee -a "$OUT"
            fails=$((fails+1))
            ;;
    esac
}

# ==================================================================================================
# ★★ THE LOSSLESS-EVICTION GATE. This is the one that tests do_block_copy for CORRECTNESS.
#
# check_pair above starves ONE sequence below its own demand, which attention makes unserveable --
# every block is live every step, so the only honest outcome is a refusal. Nothing there ever
# validates that a block survives a round trip to CPU and back.
#
# Eviction exists for the multi-sequence case: two sequences, a pool that holds one but not both,
# so the scheduler spills the one that is not currently being decoded and pages it back when its
# turn comes. THAT round trip is what ca704d06's per-layer staging buffer has to get right, and the
# bar is exact: each sequence's continuation must be byte-identical to running it alone.
#
# Pool sizing, derived like everything else here: peak+1. Above `peak` so one sequence fits with
# room to grow; below 2*peak so both cannot be resident at once. Holds for any peak >= 2.
# ==================================================================================================
PROMPT_A='Count from one to twenty in English words, separated by commas:'
PROMPT_B='List the first twelve letters of the English alphabet, separated by commas:'

start_srv() { # $1 label  $2 model  $3 extra args  $4 np
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=$DRIVE DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$2" -ngl 99 -c 8192 -np "$4" -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        --kv-paged --kv-block-size 16 $3 > "$LOGDIR/ev-$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

ask() { # $1 prompt
    curl -s --max-time 180 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$1\",\"n_predict\":48,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" | classify
}

fire_both() { # $1 tag -- both prompts CONCURRENTLY, results into $LOGDIR
    ask "$PROMPT_A" > "$LOGDIR/$1-A.txt" &
    local pa=$!
    ask "$PROMPT_B" > "$LOGDIR/$1-B.txt" &
    local pb=$!
    wait $pa; wait $pb
}

check_multiseq() { # $1 tag  $2 model  $3 measured peak (single-sequence)
    echo "=== $1 MULTI-SEQ lossless eviction ===" | tee -a "$OUT"
    local peak=$3 pool refA refB cgA cgB outA outB swaps swaps_ref
    [ "$peak" -lt 2 ] && { echo "$1: SKIP no measured peak"; return; }
    pool=$((peak + 1))

    # ★★ THREE ARMS, so every comparison changes exactly ONE thing.
    #
    # The first version of this gate compared a SEQUENTIAL generous run against a CONCURRENT starved
    # one and found the two sequences' outputs swapped. That looked like eviction corrupting block
    # ownership -- but the arms differed in concurrency AND in swapping, so the result could not
    # distinguish an eviction bug from a plain two-slot contamination bug that has nothing to do
    # with paging. A cross-contamination signature is if anything MORE typical of slot handling.
    #
    #   SOLO           sequential, generous  -> the true reference for each prompt alone
    #   CONC-GENEROUS  concurrent, generous  -> isolates CONCURRENCY (must equal SOLO, swaps=0)
    #   CONC-STARVED   concurrent, starved   -> isolates EVICTION (compared against CONC-GENEROUS)
    #
    # If CONC-GENEROUS already breaks, the eviction comparison is BLOCKED, not passed: this gate
    # must say "cannot attribute" rather than clear a path it never managed to test.
    if ! start_srv "$1-ms-ref" "$2" "-ngpub 512 -ncpub 128" 2; then
        echo "$1: FAIL reference server never became healthy" | tee -a "$OUT"; fails=$((fails+1)); return
    fi
    refA=$(ask "$PROMPT_A"); refB=$(ask "$PROMPT_B")
    fire_both "$1-cg"
    cgA=$(cat "$LOGDIR/$1-cg-A.txt"); cgB=$(cat "$LOGDIR/$1-cg-B.txt")
    swaps_ref=$(grep -c "DS4P-EVICT swap" "$LOGDIR/ev-$1-ms-ref.log" 2>/dev/null); swaps_ref=${swaps_ref:-0}
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    echo "SOLO  A: $refA" | tee -a "$OUT"
    echo "SOLO  B: $refB" | tee -a "$OUT"
    echo "CONC-GENEROUS A: $cgA" | tee -a "$OUT"
    echo "CONC-GENEROUS B: $cgB" | tee -a "$OUT"
    echo "CONC-GENEROUS swaps (must be 0 or it is not generous): $swaps_ref" | tee -a "$OUT"

    if [ "$swaps_ref" -ne 0 ]; then
        echo "$1 MULTI-SEQ: FAIL the generous arm swapped $swaps_ref time(s) -- it is not a no-eviction control" | tee -a "$OUT"
        fails=$((fails+1)); return
    fi
    if [ "$cgA" != "$refA" ] || [ "$cgB" != "$refB" ]; then
        echo "$1 MULTI-SEQ: FAIL CONCURRENCY ALONE corrupts -- two slots on an UNSTARVED pool already diverge" | tee -a "$OUT"
        echo "    this is NOT an eviction defect, and the eviction comparison is BLOCKED, not passed" | tee -a "$OUT"
        [ "$cgA" != "$refA" ] && echo "    A solo: $refA
    A conc: $cgA" | tee -a "$OUT"
        [ "$cgB" != "$refB" ] && echo "    B solo: $refB
    B conc: $cgB" | tee -a "$OUT"
        fails=$((fails+1)); return
    fi
    echo "CONC-GENEROUS: clean -- concurrency alone is not the cause, eviction is now one-factor" | tee -a "$OUT"

    echo "starving 2 sequences to -ngpub $pool (single-seq peak=$peak, so one fits, two cannot)" | tee -a "$OUT"
    if ! start_srv "$1-ms-evict" "$2" "-ngpub $pool -ncpub 512" 2; then
        echo "$1: FAIL starved server never became healthy" | tee -a "$OUT"; fails=$((fails+1)); return
    fi
    # CONCURRENT on purpose: the whole point is that both sequences are in flight at once so the
    # scheduler has to choose one to spill. Sequential requests would never contend. Identical
    # firing to CONC-GENEROUS above -- same helper, so the two arms cannot drift apart.
    fire_both "$1-ms"
    outA=$(cat "$LOGDIR/$1-ms-A.txt"); outB=$(cat "$LOGDIR/$1-ms-B.txt")
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2

    swaps=$(grep -c "DS4P-EVICT swap" "$LOGDIR/ev-$1-ms-evict.log" 2>/dev/null); swaps=${swaps:-0}
    echo "swaps: $swaps" | tee -a "$OUT"
    echo "outA: $outA" | tee -a "$OUT"
    echo "outB: $outB" | tee -a "$OUT"

    # ⚠ MARKER FIRST, as everywhere else in this file. With no swap the two sequences simply both
    # fit, and "outputs match" is a statement about an ordinary run.
    if [ "$swaps" -lt 1 ]; then
        echo "$1 MULTI-SEQ: FAIL no swap occurred -- the pool was not actually contended, proves NOTHING" | tee -a "$OUT"
        fails=$((fails+1)); return
    fi
    # ⚠ COMPARED AGAINST CONC-GENEROUS, NOT SOLO. Concurrency is held constant across this pair, so
    # the only difference left is whether the pool spilled. Comparing to SOLO would silently
    # reintroduce the two-factor confound this whole restructuring exists to remove.
    if [ "$outA" = "$cgA" ] && [ "$outB" = "$cgB" ]; then
        echo "$1 MULTI-SEQ: PASS both sequences byte-identical across $swaps swap(s) -- eviction is LOSSLESS" | tee -a "$OUT"
    else
        echo "$1 MULTI-SEQ: FAIL a sequence changed across eviction (one factor: same concurrency, only the pool differs)" | tee -a "$OUT"
        [ "$outA" != "$cgA" ] && echo "    A conc-generous: $cgA
    A conc-starved : $outA" | tee -a "$OUT"
        [ "$outB" != "$cgB" ] && echo "    B conc-generous: $cgB
    B conc-starved : $outB" | tee -a "$OUT"
        # name the signature if the two sequences swapped identities rather than merely degraded --
        # they point at different code (block ownership vs block contents)
        if [ "$outA" = "$cgB" ] || [ "$outB" = "$cgA" ]; then
            echo "    ★ SIGNATURE: the outputs CROSSED -- a sequence returned the OTHER sequence's result." | tee -a "$OUT"
            echo "      That is block OWNERSHIP (wrong table entry restored), not block CONTENTS." | tee -a "$OUT"
        fi
        fails=$((fails+1))
    fi
}

check_pair "MIXED-geometry" "$M_MIX"
check_multiseq "MIXED-geometry" "$M_MIX" "${MEASURED_PEAK:-0}"
check_pair "UNIFORM-geometry" "$M_UNI"
check_multiseq "UNIFORM-geometry" "$M_UNI" "${MEASURED_PEAK:-0}"

echo "-----" | tee -a "$OUT"
if [ "$fails" -eq 0 ]; then
    echo "EVICTION GATE: PASS" | tee -a "$OUT"
else
    echo "EVICTION GATE: FAIL ($fails)" | tee -a "$OUT"
fi
echo "log: $OUT"
exit $fails
