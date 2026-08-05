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
#   DRIVE_ON   paged_attn_dispatches=2  selfdrive_calls=8  checkouts=8  -> corrupt output
#   DRIVE_OFF  paged_attn_dispatches=2  selfdrive_calls=0  checkouts=4  -> clean output
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
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
P=9005
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/eviction-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
PROMPT='Count from one to twenty in English words, separated by commas:'

M_MIX=$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf
M_UNI=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"

fails=0

run() { # $1 label  $2 model  $3 extra args
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1 DS4P_PAGED_SWA=1 \
        nohup "$SRV" -m "$2" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        --kv-paged --kv-block-size 16 $3 > "/tmp/ev-$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
        echo "NEVER_READY"; pkill -f "$SRV" >/dev/null 2>&1; return
    fi
    curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":48,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"].replace(chr(10)," ")[:200])'
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
}

check_pair() { # $1 tag  $2 model
    echo "=== $1 ===" | tee -a "$OUT"

    local ctrl evict swaps_c swaps_e
    ctrl=$(run "$1-ctrl"  "$2" "-ngpub 512 -ncpub 128")
    echo "CTRL : $ctrl" | tee -a "$OUT"

    # ⚠ -ngpub 24 DID NOT SPILL. 24 blocks x 16 = 384 tokens against a ~60-token job: no pressure,
    # swaps=0, and the gate correctly failed itself on the presence marker. The livelock hunt later
    # established that -ngpub 8 (128 tokens) is what actually forces the pool to spill on this
    # prompt. Starvation has to be sized against the JOB, not merely look small.
    evict=$(run "$1-evict" "$2" "-ngpub 8 -ncpub 512")
    echo "EVICT: $evict" | tee -a "$OUT"

    swaps_c=$(grep -c "DS4P-EVICT swap" "/tmp/ev-$1-ctrl.log"  2>/dev/null);  swaps_c=${swaps_c:-0}
    swaps_e=$(grep -c "DS4P-EVICT swap" "/tmp/ev-$1-evict.log" 2>/dev/null); swaps_e=${swaps_e:-0}
    echo "swaps: CTRL=$swaps_c  EVICT=$swaps_e" | tee -a "$OUT"

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
    if [ "$ctrl" = "$evict" ]; then
        echo "$1: PASS output identical across $swaps_e swap(s) -- eviction is lossless" | tee -a "$OUT"
    else
        echo "$1: FAIL output DIFFERS after eviction -- block copy is corrupting KV" | tee -a "$OUT"
        echo "    ctrl : $ctrl"  | tee -a "$OUT"
        echo "    evict: $evict" | tee -a "$OUT"
        fails=$((fails+1))
    fi
}

check_pair "MIXED-geometry" "$M_MIX"
check_pair "UNIFORM-geometry" "$M_UNI"

echo "-----" | tee -a "$OUT"
if [ "$fails" -eq 0 ]; then
    echo "EVICTION GATE: PASS" | tee -a "$OUT"
else
    echo "EVICTION GATE: FAIL ($fails)" | tee -a "$OUT"
fi
echo "log: $OUT"
exit $fails
