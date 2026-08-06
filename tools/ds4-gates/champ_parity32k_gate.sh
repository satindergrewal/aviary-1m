#!/usr/bin/env bash
# CHAMPION PREFILL PARITY AT 32K -- turning a provisional 0.98x into a result, or killing it.
#
# WHAT IT IS TESTING. After the tile-parallel mask fill the champion measured 57532/59129 ms against
# static's 58062/60253 at 32,768 tokens: paired ratios 0.991x and 0.981x. Two reps against ~5%
# run-to-run variance on this box is parity WITHIN NOISE, not a win. A 1-2% margin is exactly the size
# of claim this lane has got wrong before, so it does not go on the board until it survives reps.
#
# ⚠ ORDER IS A CONFOUND WORTH 29% IN THIS LANE. Server start order and thermal drift both favour
# whatever runs first. Arms run ABBA -- STATIC, CHAMP, CHAMP, STATIC -- so any monotonic drift cancels
# between the halves instead of loading onto one arm. Four server starts, REPS requests inside each.
#
# ⚠ PAIRED BY POSITION, NOT JUST POOLED. The medians answer "which is faster overall"; the per-half
# comparison answers "did the second half agree with the first". If the two halves disagree in SIGN,
# the difference is drift and the gate says so rather than averaging it away.
#
# ⚠ CORRECTNESS GATES TIMING. Every response is compared against the static reference; a mismatch
# voids the run. A faster kernel that answers differently is not parity.
#
# PRE-REGISTERED:
#   champ median <= static median AND both halves agree in sign -> parity at 32k, claim stands
#   halves disagree in sign                                     -> drift, INCONCLUSIVE, more reps
#   champ median > static median                                -> the 0.98x was noise; withdraw it
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${PT_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
REPS=${PT_REPS:-6}
NTOK=${PT_NTOK:-32768}
CTX=${PT_CTX:-36864}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champparity32k
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-parity32k-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

pick_port() {
    local p
    for p in $(seq "${PT_PORT:-9191}" $(( ${PT_PORT:-9191} + 40 ))); do
        if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}
P=$(pick_port) || { echo "FAIL: no free port"; exit 1; }

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS ntok=$NTOK" | tee "$OUT"

python3 -c "
import random
random.seed(41)
s=['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
v=['is the capital of','lies north of','was founded before','trades heavily with','borders']
o=['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
while len(' '.join(out)) < $NTOK*8: out.append('%s %s %s.' % (random.choice(s),random.choice(v),random.choice(o)))
open('$LOGDIR/f.txt','w').write(' '.join(out))"

# ⚠ PT_CHAMP_ENV lets the same ABBA machinery measure any champion variant against static without
# a second gate: decode nwg arms, ablation arms, kernel one-factor arms. The balancing is the whole
# value here, and every comparison this lane makes needs it -- a single-run ratio against static is
# worth nothing on a box whose first six reps drift 8%.
half() { # $1 label  $2 mode -> appends "ms" lines to $LOGDIR/$1.ms and text to $1.txt
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local envs=(DS4P_PAGED_HYBRID=1) flags=()
    if [ "$2" = champ ]; then
        envs+=(DS4P_METAL_CHAMP=1)
        # shellcheck disable=SC2206
        [ -n "${PT_CHAMP_ENV:-}" ] && envs+=(${PT_CHAMP_ENV})
        flags=(--kv-paged --kv-block-size 64 -ngpub $(( (CTX + 63)/64 )) -ncpub 256)
    fi
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
        echo "  --- $1 did not start; last lines of its log ---" | tee -a "$OUT"
        tail -10 "$LOGDIR/$1.log" | sed 's/^/  | /' | tee -a "$OUT"
        return 1
    fi
    if [ ! -s "$LOGDIR/toks.json" ]; then
        python3 -c "
import json;print(json.dumps({'content':open('$LOGDIR/f.txt').read()}))" > "$LOGDIR/t.req"
        curl -s --max-time 300 -X POST http://127.0.0.1:$P/tokenize -H 'Content-Type: application/json' -d @"$LOGDIR/t.req" > "$LOGDIR/t.json"
        python3 -c "
import json;t=json.load(open('$LOGDIR/t.json'))['tokens']
open('$LOGDIR/toks.json','w').write(json.dumps(t));print('  filler tokens:',len(t))" | tee -a "$OUT"
    fi
    python3 -c "
import json;t=json.load(open('$LOGDIR/toks.json'))[:$NTOK]
print(json.dumps({'prompt':t,'n_predict':${PT_NPRED:-24},'temperature':0,'seed':1,'cache_prompt':False}))" > "$LOGDIR/$1.req"
    : > "$LOGDIR/$1.ms"; : > "$LOGDIR/$1.txt"; : > "$LOGDIR/$1.tps"
    for r in $(seq 1 "$REPS"); do
        curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
            -d @"$LOGDIR/$1.req" > "$LOGDIR/$1-$r.json"
        local ms txt dtp
        ms=$(grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
        dtp=$(grep -a "        eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "[0-9.]+ tokens per" | grep -oE "[0-9.]+" | head -1)
        echo "${dtp:-NA}" >> "$LOGDIR/$1.tps"
        txt=$(python3 -c "
import json
try: print(json.load(open('$LOGDIR/$1-$r.json')).get('content','')[:44].replace(chr(10),' '))
except Exception: print('MALFORMED')")
        echo "${ms:-NA}" >> "$LOGDIR/$1.ms"; echo "$txt" >> "$LOGDIR/$1.txt"
        printf "  %-10s r%s  prefill=%-10s decode=%-7s  out=[%s]\n" "$1" "$r" "${ms:-NA}" "${dtp:-NA}" "$txt" | tee -a "$OUT"
    done
    if [ "$2" = champ ]; then
        local mk; mk=$(grep -aci "CHAMP-PAGED ACTIVE" "$LOGDIR/$1.log")
        echo "  $1 champion marker count: $mk" | tee -a "$OUT"
        [ "$mk" = "0" ] && { echo "VOID: champion arm never logged CHAMP-PAGED ACTIVE" | tee -a "$OUT"; return 2; }
    fi
    return 0
}

# ⚠ WARM-UP HALF, MEASURED AND DISCARDED. ABBA cancels a LINEAR drift; what this box actually does
# is a warm-up TRANSIENT that has mostly finished by the third half. Measured in one run: STATIC_A
# from cold drifted 61444 -> 65049 between two adjacent reps, while CHAMP_B at steady state produced
# 58035 58267 58320 58306 58256 58408 -- six reps inside 373 ms. Averaging a transient against a
# steady state is not balancing, it is diluting, and it is why two balanced runs of the same code
# path came out 1.152x and 0.992x. The first half now exists only to heat the machine.
half WARMUP  static || exit 1
echo "  (warm-up half above is DISCARDED -- it measures the machine, not the kernel)" | tee -a "$OUT"

# ABBA: any residual monotonic drift cancels between halves instead of loading onto one arm.
half STATIC_A static || exit 1
half CHAMP_A  champ  || exit 1
half CHAMP_B  champ  || exit 1
half STATIC_B static || exit 1
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
python3 - "$LOGDIR" <<'PY' | tee -a "$OUT"
import sys, statistics as st
d = sys.argv[1]
DROP = 1   # the first request in each server also pays pipeline compilation and graph warmup
def rd(n):
    v = [float(x) for x in open(f"{d}/{n}.ms") if x.strip() and x.strip() != "NA"]
    return v[DROP:] if len(v) > DROP + 1 else v
def tx(n):
    return [l.rstrip("\n") for l in open(f"{d}/{n}.txt")]
sa, ca, cb, sb = rd("STATIC_A"), rd("CHAMP_A"), rd("CHAMP_B"), rd("STATIC_B")
ta, tca = tx("STATIC_A"), tx("CHAMP_A")
if min(len(sa), len(ca), len(cb), len(sb)) == 0:
    print("INCONCLUSIVE: an arm produced no timings."); raise SystemExit(2)
ref = ta[0]
bad = [t for t in tx("CHAMP_A") + tx("CHAMP_B") + tx("STATIC_B") if t != ref]
if bad:
    print(f"VOID: {len(bad)} response(s) differ from the static reference.")
    print(f"  reference: {ref}")
    print(f"  first differing: {bad[0]}")
    raise SystemExit(1)
h1 = st.median(ca)/st.median(sa)     # first half: static then champ
h2 = st.median(cb)/st.median(sb)     # second half: champ then static
allc, alls = st.median(ca+cb), st.median(sa+sb)
print(f"  half 1 (STATIC_A -> CHAMP_A) : champ/static = {h1:.3f}x")
print(f"  half 2 (CHAMP_B -> STATIC_B) : champ/static = {h2:.3f}x")
print(f"  pooled medians               : champ {allc:.1f} ms / static {alls:.1f} ms = {allc/alls:.3f}x")
print(f"  spread                       : champ {min(ca+cb):.0f}-{max(ca+cb):.0f}  static {min(sa+sb):.0f}-{max(sa+sb):.0f}")
def rdt(n):
    try: v = [float(x) for x in open(f"{d}/{n}.tps") if x.strip() and x.strip() != "NA"]
    except OSError: return []
    return v[DROP:] if len(v) > DROP + 1 else v
dca, dcb, dsa, dsb = rdt("CHAMP_A"), rdt("CHAMP_B"), rdt("STATIC_A"), rdt("STATIC_B")
if dca and dsa and dcb and dsb:
    dh1 = st.median(dsa)/st.median(dca)
    dh2 = st.median(dsb)/st.median(dcb)
    dall = st.median(dsa+dsb)/st.median(dca+dcb)
    print(f"  decode  half1 {dh1:.3f}x  half2 {dh2:.3f}x  pooled {dall:.3f}x  (static tok/s over champ tok/s)")
print()
if (h1 - 1.0) * (h2 - 1.0) < 0:
    print("INCONCLUSIVE: the two halves disagree in SIGN, so the difference is order or drift,")
    print("  not the kernel. More reps, or a finer interleave, before any claim.")
elif allc/alls <= 1.0:
    print(f"RESULT: champion prefill at or under static at this length ({allc/alls:.3f}x), and both")
    print("  halves agree in sign. The provisional 0.98x stands.")
else:
    print(f"RESULT: champion is SLOWER ({allc/alls:.3f}x). The provisional 0.98x was noise -- withdraw it.")
PY
echo "log: $OUT"
