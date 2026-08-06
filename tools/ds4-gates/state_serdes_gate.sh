#!/usr/bin/env bash
# PAGED STATE SAVE/RESTORE GATE -- the second of the two paths I changed BLIND in ca704d06.
#
# state_write/state_read got a per-layer block_bytes AND an attention-only nullptr guard that had
# never been executed by anything in this lane. Nothing here saves paged state, which is exactly why
# the missing guard survived: absence of a caller is not absence of a bug, it just moves the crash
# to whoever adds one. This is me adding one.
#
# ⚠ RUNS ON A HYBRID (Ornith) ON PURPOSE. The filter must be ACTIVE or the nullptr guard is not
# reached and a green result would mean nothing -- the same "the case never ran" trap the eviction
# gate caught itself in.
#
# ARMS
#   A  save        -> must succeed and produce a non-empty file
#   B  restore     -> must succeed
#   C  continuation after restore == continuation without interruption (the actual correctness bar)
# ★★ WHAT THIS GATE IS ACTUALLY MEASURING (corrected 2026-08-06 -- READ BEFORE "FIXING" state_write)
#
# This gate FAILS at arm A with n_written=716 against an expected ~864 KB, and that is NOT a broken
# serialiser. server-context.cpp:3229 documents it: the paged path runs finish() -> free_blocks()
# and RETURNS THE BLOCKS TO THE POOL AT REQUEST FINISH, BY DESIGN -- "holding a finished sequence's
# blocks would fight the entire point of paging". /slots/N?action=save fires AFTER the completion
# returns, so sequence_blocks.find(seq_id) correctly misses and state_write correctly writes nothing.
#
# The caller is wired, the serialiser is complete, and both are working. The DEFECT this gate is
# really catching is that the server answers 200 with n_saved=27 / n_written=716 for a save that
# saved nothing. An endpoint that cannot work on this path by construction should REFUSE.
#
# So the fix is at the ENDPOINT, not in state_write, and anyone who "makes the serialiser work" here
# will be fighting the design. Self-drive is NOT involved: DS4P_PAGED_DRIVE=0 gives a byte-identical
# 716.

set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf
P=9010
SAVEDIR=$HOME/.claude/jobs/9fa4fa3b/tmp/slotsave
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/state-serdes-$(date +%Y%m%d-%H%M).txt
mkdir -p "$SAVEDIR" "$(dirname "$OUT")"
rm -f "$SAVEDIR"/*.bin 2>/dev/null
PROMPT='The capital of France is Paris. The capital of Germany is'

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
fails=0

pkill -f "$SRV" >/dev/null 2>&1; sleep 2
# ⚠ DRIVE IS A VARIABLE NOW. This gate has always pinned DS4P_PAGED_DRIVE=1, and self-drive is
# known to allocate its own group and, when it cannot grow, FREE its own KV and fall back to the
# static path. If that happens the sequence's KV is not in the pool at save time and state_write
# has nothing to serialise -- which is exactly the 716-byte symptom this gate reports.
# SS_DRIVE=0 tests that in one factor.
env DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=${SS_DRIVE:-1} nohup "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 \
    --port $P --no-warmup -lv 4 --kv-paged --kv-block-size 16 --slot-save-path "$SAVEDIR/" \
    > ${SS_LOG:-/tmp/ss.log} 2>&1 &
for _ in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
    sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
    echo "server never became healthy" | tee -a "$OUT"; exit 1
fi

# prove the attention-only filter is ACTIVE, or a pass proves nothing
filt=$(grep -oE "attention-only pool: [0-9]+ of [0-9]+ layers hold KV" ${SS_LOG:-/tmp/ss.log} | head -1)
echo "filter: ${filt:-NOT ACTIVE}" | tee -a "$OUT"
if [ -z "$filt" ]; then
    echo "FAIL attention-only filter not active -- the nullptr guard would not be exercised" | tee -a "$OUT"
    fails=$((fails+1))
fi

# seed the slot with KV
base=$(curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"seed\":1,\"cache_prompt\":true}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"].replace(chr(10)," ")[:120])')
echo "baseline continuation: $base" | tee -a "$OUT"

echo "--- A SAVE ---" | tee -a "$OUT"
sa=$(curl -s -X POST "http://127.0.0.1:$P/slots/0?action=save" -H 'Content-Type: application/json' \
     -d '{"filename":"ds4-state.bin"}')
echo "$sa" | head -c 300 | tee -a "$OUT"; echo | tee -a "$OUT"
sz=$(stat -f%z "$SAVEDIR/ds4-state.bin" 2>/dev/null || echo 0)
echo "saved bytes: $sz" | tee -a "$OUT"
# ⚠ THE BAR HERE IS A REFUSAL, NOT A SUCCESSFUL SAVE -- flipped when the design intent was
# established. The paged path frees a sequence's blocks at finish BY DESIGN, and this endpoint runs
# after the completion returns, so there is no KV left to serialise. A successful save on the paged
# path is therefore NOT the correct outcome; a clean refusal is. Arm A originally required
# sz >= 1024 and correctly FAILED at 716 B, which is how the defect was found -- but once the
# endpoint refuses, that same bar would fail a CORRECT fix.
#   PAIRED WITH: patches/DRAFT-defect3-slotsave-refuse.patch. Apply both or neither: this gate
#   inverted without the patch fails on the unfixed tree for the wrong reason, and the patch without
#   this gate turns a good fix red and looks like a regression.
if echo "$sa" | grep -qi "not supported on the paged KV path"; then
    echo "A: PASS endpoint REFUSED cleanly on the paged path -- correct, there is no KV to save" | tee -a "$OUT"
elif [ "$sz" -lt 1024 ]; then
    echo "A: FAIL save produced ${sz}B and did NOT refuse -- silently reported success for a no-op" | tee -a "$OUT"; fails=$((fails+1))
else
    echo "A: PASS save produced $sz bytes" | tee -a "$OUT"
fi

echo "--- B RESTORE ---" | tee -a "$OUT"
rs=$(curl -s -X POST "http://127.0.0.1:$P/slots/0?action=restore" -H 'Content-Type: application/json' \
     -d '{"filename":"ds4-state.bin"}')
echo "$rs" | head -c 300 | tee -a "$OUT"; echo | tee -a "$OUT"
# ⚠ SYMMETRIC WITH ARM A. If save correctly refuses on the paged path, there is no file to restore
# and restore must refuse too -- accepting it would be the second half of a broken pair pretending to
# work. So a refusal here is the PASS, and "restore accepted" on the paged path is the failure.
if echo "$rs" | grep -qi "not supported on the paged KV path"; then
    echo "B: PASS restore REFUSED cleanly on the paged path -- symmetric with save" | tee -a "$OUT"
elif echo "$rs" | grep -qi "error"; then
    echo "B: FAIL restore errored for some OTHER reason (not the paged refusal)" | tee -a "$OUT"; fails=$((fails+1))
else
    echo "B: FAIL restore ACCEPTED on the paged path -- there is no KV behind it" | tee -a "$OUT"; fails=$((fails+1))
fi

echo "--- C CONTINUATION AFTER RESTORE (must equal baseline) ---" | tee -a "$OUT"
aft=$(curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":16,\"temperature\":0,\"seed\":1,\"cache_prompt\":true}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"].replace(chr(10)," ")[:120])')
echo "after restore:        $aft" | tee -a "$OUT"
# ⚠ ARM C IS ONLY MEANINGFUL IF A SAVE/RESTORE ACTUALLY HAPPENED. On the paged path both refuse by
# design, so there is no round trip to be lossless about -- comparing continuations here would be
# measuring ordinary generation, and calling that a PASS would be the vacuous-gate failure this
# suite already found elsewhere.
if echo "$sa$rs" | grep -qi "not supported on the paged KV path"; then
    echo "C: N/A save+restore refused by design -- no round trip to verify, not scored" | tee -a "$OUT"
elif [ "$aft" = "$base" ]; then
    echo "C: PASS continuation identical across save/restore" | tee -a "$OUT"
else
    echo "C: FAIL continuation DIFFERS -- serdes is not lossless" | tee -a "$OUT"; fails=$((fails+1))
fi

pkill -f "$SRV" >/dev/null 2>&1
echo "-----" | tee -a "$OUT"
[ "$fails" -eq 0 ] && echo "STATE-SERDES GATE: PASS" | tee -a "$OUT" || echo "STATE-SERDES GATE: FAIL ($fails)" | tee -a "$OUT"
echo "log: $OUT"
exit $fails
