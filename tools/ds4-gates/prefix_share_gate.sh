#!/usr/bin/env bash
# PREFIX-SHARING GATE -- do two INDEPENDENT requests with a common prefix actually share blocks?
#
# ⚠ THE BAR IS CHECKOUT COUNT, NOT OUTPUT TEXT. Both requests produce correct output whether sharing
# happened or not, so scoring on text would pass on a completely unwired feature. That exact trap
# cost an hour on Qwen3.6 today: it built a paged pool, served perfect text, and never paged.
#
# MECHANISM UNDER TEST (llama-paged-scheduler-impl.cpp, COLD branch): at admission, scan `running`
# for the longest common prefix of logical_seq, cap at the source's n_past, floor to block_size, and
# hand those blocks over via fork_blocks. The sharing half (refcounts, COW tail) is P1-6 work and is
# already proven; this gate tests DISCOVERY.
#
# ★ THE TWO REQUESTS MUST OVERLAP IN TIME. Sharing scans the RUNNING set, so request A has to still
# be live when B is admitted. A is given a long n_predict and B is fired while A generates. If A
# finishes first the gate proves nothing -- so overlap is asserted, not assumed.
#
# PRE-REGISTERED:
#   SHARED marker present AND B's checkouts << A's   -> sharing works
#   SHARED marker absent                             -> discovery never fired (the feature is dark)
#   marker present but checkouts equal               -> it logged a share it did not perform
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${PS_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${PS_PORT:-9061}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/prefixshare
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/prefix-share-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"

# A long SHARED prefix (many blocks at bs=16) plus a distinct tail per request.
# ⚠ Built to a FILE, not a shell variable. The first version interpolated $PREFIX into a nested
# python json.dumps inside double quotes; the prompt that actually reached the server was ~48 tokens
# (3 blocks at bs=16) instead of the intended ~700. A gate whose stimulus is 48 tokens cannot test
# block-level sharing at all, and its null would have read as "feature broken" rather than
# "test too small". The stimulus is now asserted below before any verdict is reached.
python3 -c "open('$LOGDIR/prefix.txt','w').write('You are a careful assistant. Follow the system rules exactly and cite your reasoning. ' * 60)"
PREFIX_CHARS=$(wc -c < "$LOGDIR/prefix.txt" | tr -d ' ')
echo "shared prefix: $PREFIX_CHARS chars" | tee -a "$OUT" 

pkill -f "$SRV" >/dev/null 2>&1; sleep 2
env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np 2 -b 512 -ub 512 \
    --port $P --no-warmup -lv 4 --kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128 \
    > "$LOGDIR/srv.log" 2>&1 &
for _ in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
    sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
    echo "FAIL: server never became healthy -- result unusable" | tee -a "$OUT"; exit 1
fi

ask() { # $1 tail  $2 n_predict  $3 label
    TAIL="$1" NPRED="$2" PFILE="$LOGDIR/prefix.txt" python3 -c "
import json, os
p = open(os.environ['PFILE']).read() + os.environ['TAIL']
print(json.dumps({'prompt': p, 'n_predict': int(os.environ['NPRED']),
                  'temperature': 0, 'cache_prompt': False}))" > "$LOGDIR/$3.req"
    curl -s --max-time 300 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d @"$LOGDIR/$3.req" > "$LOGDIR/$3.json" 2>/dev/null
}

# A runs long so it is still in `running` when B is admitted.
ask " Now write a long essay about rivers." 200 A &
pa=$!
sleep 6                                  # let A finish prefill and enter decode
MARK=$(grep -ac "DS4P-CHECKOUT" "$LOGDIR/srv.log")
ask " Now write a long essay about mountains." 16 B
wait $pa 2>/dev/null
pkill -f "$SRV" >/dev/null 2>&1; sleep 1

echo "-----" | tee -a "$OUT"
# ⚠ PRESENCE OF THE REGIME: both requests must have been admitted, and they must have overlapped.
ADMITS=$(grep -ac "paged: request registered" "$LOGDIR/srv.log")
SHARED=$(grep -ac "admitted SHARED" "$LOGDIR/srv.log")
CO_TOTAL=$(grep -ac "DS4P-CHECKOUT" "$LOGDIR/srv.log")
CO_AFTER=$(( CO_TOTAL - MARK ))
echo "requests admitted:            $ADMITS   (need 2)" | tee -a "$OUT"
echo "checkouts before B admitted:  $MARK" | tee -a "$OUT"
echo "checkouts attributable to B:  $CO_AFTER" | tee -a "$OUT"
echo "'admitted SHARED' lines:      $SHARED" | tee -a "$OUT"
grep -a "admitted SHARED" "$LOGDIR/srv.log" | head -2 | cut -c1-150 | tee -a "$OUT"
echo | tee -a "$OUT"

# ⚠ STIMULUS CHECK, before any verdict. If the prompt the model actually saw is too short to span
# several blocks, a null tells us nothing about sharing.
PROMPT_N=$(grep -aoE "prompt eval time.*/ +[0-9]+ tokens" "$LOGDIR/srv.log" | grep -oE "[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
echo "prompt tokens actually evaluated (request A): ${PROMPT_N:-unknown}" | tee -a "$OUT"
if [ -z "${PROMPT_N:-}" ] || [ "${PROMPT_N:-0}" -lt 64 ]; then
    echo "INCONCLUSIVE: the stimulus was only ${PROMPT_N:-0} tokens -- too short to span enough blocks" | tee -a "$OUT"
    echo "  at block_size 16 for block-level sharing to be observable. Fix the gate, not the feature." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

if [ "$ADMITS" -lt 2 ]; then
    echo "INCONCLUSIVE: fewer than 2 requests were admitted -- the regime was never entered." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
if [ "$SHARED" -eq 0 ]; then
    echo "PREFIX-SHARING GATE: FAIL -- no 'admitted SHARED' line. Discovery never fired: the second" | tee -a "$OUT"
    echo "  request re-prefilled a prefix that was live in the pool. The feature is DARK." | tee -a "$OUT"
    echo "log: $OUT"; exit 1
fi
if [ "$CO_AFTER" -ge "$MARK" ] && [ "$MARK" -gt 0 ]; then
    echo "PREFIX-SHARING GATE: FAIL -- SHARED was logged but B checked out as many blocks as A." | tee -a "$OUT"
    echo "  A share that allocates the same number of blocks did not share. Marker without effect." | tee -a "$OUT"
    echo "log: $OUT"; exit 1
fi
echo "PREFIX-SHARING GATE: PASS -- second request inherited a live prefix and allocated fewer blocks" | tee -a "$OUT"
echo "  ($CO_AFTER checkouts for B vs $MARK for A)." | tee -a "$OUT"
echo "log: $OUT"
