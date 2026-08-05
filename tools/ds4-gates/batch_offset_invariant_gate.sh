#!/usr/bin/env bash
# BATCH-OFFSET INVARIANT GATE -- tests the DEFECT, not the symptom.
#
# The -np>1 corruption found on 2026-08-06 has an exact invariant behind it, so it can be checked
# directly instead of through output damage:
#
#   ★ THE INVARIANT: for every paged-attention dispatch, batch_offsets/batch_lens must describe the
#     tokens ACTUALLY DISPATCHED. The kernel resolves a token's sequence with
#         gtok >= batch_offsets[s] && gtok < batch_offsets[s] + batch_lens[s]
#     where gtok is LOCAL to the dispatch (0..n_tokens-1). So the arrays must be local too:
#         sum(batch_lens) == n_tokens
#     A dispatch of 14 tokens carrying batch_lens=[15,14] (sum 29) is the bug, verbatim.
#
# WHY AN INVARIANT GATE AND NOT THE OUTPUT GATE. multislot_gate detects this through corrupted text,
# which is real but indirect: the symptom varies (CROSSED when the read misresolves, GARBAGE when
# the write clobbers), it needs a model whose output is judgeable, and it cannot say WHICH dispatch
# was wrong. This gate reads the numbers the kernel actually received. It fails on the defect itself,
# names the offending dispatch, and will go green the moment the contract is fixed -- including for a
# fix that re-bases offsets per ubatch, which is the likely shape.
#
# ⚠ PRESENCE FIRST. If no multi-sequence dispatch is ever produced, this gate proves NOTHING -- a
# suite where nothing is concurrent would pass it trivially, which is exactly how the original defect
# survived (a -np 2 gate that sent requests sequentially). It therefore REQUIRES at least one
# dispatch with n_seq >= 2 and reports INCONCLUSIVE otherwise.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${BI_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=9021
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/batchinv
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/batch-offset-invariant-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"

pkill -f "$SRV" >/dev/null 2>&1; sleep 2
env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 DS4P_ARGDUMP=600 \
    nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np 2 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
    --kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128 > "$LOGDIR/srv.log" 2>&1 &
for _ in $(seq 1 400); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
    sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
    echo "FAIL: server never became healthy -- result unusable" | tee -a "$OUT"; exit 1
fi

# two concurrent requests of DIFFERENT lengths, so a mismatch is unambiguous
curl -s --max-time 180 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Count from one to twenty in English words, separated by commas:","n_predict":12,"temperature":0,"seed":1,"cache_prompt":false}' >/dev/null &
pa=$!
curl -s --max-time 180 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"List the first twelve letters of the English alphabet, separated by commas:","n_predict":12,"temperature":0,"seed":1,"cache_prompt":false}' >/dev/null &
pb=$!
wait $pa; wait $pb
pkill -f "$SRV" >/dev/null 2>&1; sleep 1

python3 - "$LOGDIR/srv.log" <<'PY' | tee -a "$OUT"
import re, sys
txt = open(sys.argv[1], errors='ignore').read().split('\n')
# ARGDUMP emits, per dispatch: n_tokens=... then (later) batch_lens[N] = a b ...
cur_n = None
dispatches = []          # (n_tokens, [lens])
for l in txt:
    m = re.search(r'ARGDUMP n_tokens=(\d+)', l)
    if m: cur_n = int(m.group(1)); continue
    m = re.search(r'ARGDUMP batch_lens\[(\d+)\] =([\d ]+)', l)
    if m and cur_n is not None:
        lens = [int(x) for x in m.group(2).split()]
        dispatches.append((cur_n, lens)); cur_n = None

multi = [(n, L) for n, L in dispatches if len(L) >= 2]
print(f"dispatches captured: {len(dispatches)}   multi-sequence: {len(multi)}")

if not multi:
    print("INCONCLUSIVE: no multi-sequence dispatch was produced, so this gate proves NOTHING.")
    print("  (A suite where nothing is concurrent passes this trivially -- which is how the")
    print("   original defect survived. Presence of the regime is checked before the invariant.)")
    sys.exit(2)

bad = [(n, L) for n, L in multi if sum(L) != n]
seen = set()
for n, L in bad:
    k = (n, tuple(L))
    if k in seen: continue
    seen.add(k)
    print(f"  VIOLATION  n_tokens={n}  batch_lens={L}  sum={sum(L)}  -> kernel indexes 0..{n-1} "
          f"against offsets spanning 0..{sum(L)-1}")
if bad:
    print(f"BATCH-OFFSET INVARIANT: FAIL  ({len(bad)} of {len(multi)} multi-seq dispatches violate "
          f"sum(batch_lens) == n_tokens)")
    sys.exit(1)
print("BATCH-OFFSET INVARIANT: PASS  every multi-sequence dispatch has sum(batch_lens) == n_tokens")
PY
rc=${PIPESTATUS[0]}
echo "log: $OUT"
exit $rc
