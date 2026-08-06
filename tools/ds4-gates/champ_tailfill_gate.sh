#!/usr/bin/env bash
# CHAMPION TAIL-FILL GATE -- the decisive test for the blk-layout defect.
#
# WHAT WAS MEASURED (2026-08-06, ad-hoc, token-exact, marker-verified, three reproductions):
# the champion at D=256 answered CORRECTLY at small final-block fills and WRONGLY at large ones.
# I labelled that "a tail-fill law". The measurements stand; the LABEL did not survive reading the
# kernel, because the defect I then found has no tail-fill term in it at all.
#
# THE DEFECT (ggml-metal.metal, kernel_paged_champ_mask vs kernel_paged_champ_impl):
#   reader  blk[((head)*nblk1 + iq1/8)*nblk0 + ic0]   layout [head][q-block][kv-block]
#   writer  blk_skip[row*64 + col]  for col<64, head==0  layout [token][col]   <- WRONG
# Coverage worked out to ~32768/(n_heads*n_kv) of the indexed region, n_tokens CANCELLING, so from
# about the third head onward the champion read uninitialised bytes AS CONTROL FLOW: 0 dropped a
# whole KV block, 2 disabled its mask. The allocation was already correct; only the producer
# disagreed. Fixed by writing the reader's layout, one thread per (head, q-block, kv-block).
#
# ⚠ THIS GATE EXISTS BECAUSE THE ORIGINAL RUNS WERE AD-HOC. A one-off shell loop is how I ended up
# with a durable-sounding law whose independent variable was not the one that mattered. The token
# counts below are the same ones, so the comparison is like-for-like.
#
# PRE-REGISTERED:
#   every count matches static   -> the blk-layout defect WAS the cause; champion correctness closed
#   some counts still mismatch   -> candidate ten dies, with its condition recorded; the surviving
#                                   mismatch pattern is the next lead, and it is NOT re-labelled
#                                   "tail fill" without evidence that tail fill is the variable
#   champ marker missing         -> arm UNUSABLE, verdict void (this has silently faked a PASS twice)
#
# ⚠ TOKEN-EXACT OR NOTHING. Prompts are sent as TOKEN ID ARRAYS obtained from /tokenize, so the
# final-block fill is exactly (N % 64) and not "roughly, after the tokenizer had its say".
#
# ⚠ NEGATIVE CONTROL. The comparator is self-tested on a known-different pair before any verdict.
# A comparator that always reports "match" passes every arm, and a gate that cannot fail is a gate
# that measures nothing.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${TF_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${TF_PORT:-9077}
NPRED=${TF_NPRED:-8}
# 4608 = 72 full blocks at bs=64; the remainders are the fills the original runs used.
COUNTS=${TF_COUNTS:-"4609 4619 4641 4653 4670"}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champtail
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-tailfill-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
echo "counts: $COUNTS   n_predict=$NPRED" | tee -a "$OUT"

# ⚠ NON-DEGENERATE FILLER. The parity probe's original stimulus was one sentence repeated, and
# STATIC ITSELF loops on that -- which turned "champion differs from static" into a statement about
# a degenerate baseline. Varied factual sentences keep static sane, so a disagreement means the
# kernel disagreed. The question at the end is answerable from the text, so a correct run is
# recognisable and not merely identical.
python3 -c "
import random, json
random.seed(11)
subj = ['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
verb = ['is the capital of','lies north of','was founded before','trades heavily with','borders']
obj  = ['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out = ['The following are notes about European cities.']
while len(' '.join(out)) < 90000:
    out.append('%s %s %s.' % (random.choice(subj), random.choice(verb), random.choice(obj)))
open('$LOGDIR/filler.txt','w').write(' '.join(out))
" || { echo "FAIL: could not build stimulus" | tee -a "$OUT"; exit 1; }

start_srv() { # $1 label  $2 mode(static|champ)
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local envs=(DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1) flags=()
    if [ "$2" = champ ]; then
        envs+=(DS4P_METAL_CHAMP=1)
        flags=(--kv-paged --kv-block-size 64 -ngpub 512 -ncpub 128)
    fi
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c 16384 -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

# Tokenise once per server (same tokenizer either way, but re-derived so a mismatch is impossible).
tokenize_filler() {
    python3 -c "
import json
print(json.dumps({'content': open('$LOGDIR/filler.txt').read()}))" > "$LOGDIR/tok.req"
    curl -s --max-time 300 -X POST http://127.0.0.1:$P/tokenize -H 'Content-Type: application/json' \
        -d @"$LOGDIR/tok.req" > "$LOGDIR/tok.json" 2>/dev/null
    python3 -c "
import json
t = json.load(open('$LOGDIR/tok.json'))['tokens']
open('$LOGDIR/tokens.json','w').write(json.dumps(t))
print(len(t))"
}

ask_n() { # $1 count  $2 label -> prints the generated text on one line
    python3 -c "
import json
t = json.load(open('$LOGDIR/tokens.json'))[:$1]
print(json.dumps({'prompt': t, 'n_predict': $NPRED, 'temperature': 0, 'seed': 1,
                  'cache_prompt': False}))" > "$LOGDIR/$2.req"
    curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d @"$LOGDIR/$2.req" > "$LOGDIR/$2.json" 2>/dev/null
    python3 -c "
import json
try: print(json.load(open('$LOGDIR/$2.json')).get('content','')[:80].replace(chr(10),' '))
except Exception: print('MALFORMED')"
}

# ---------------- STATIC: ground truth ----------------
start_srv static static || { echo "FAIL: static server never became healthy" | tee -a "$OUT"; exit 1; }
NTOK=$(tokenize_filler)
echo "filler tokenised: $NTOK tokens" | tee -a "$OUT"
MAXC=$(echo "$COUNTS" | tr ' ' '\n' | sort -rn | head -1)
if [ "${NTOK:-0}" -lt "$MAXC" ]; then
    echo "INCONCLUSIVE: filler is $NTOK tokens, need >= $MAXC. Fix the gate, not the kernel." | tee -a "$OUT"
    pkill -f "$SRV" >/dev/null 2>&1; exit 2
fi
declare -A SREF
for n in $COUNTS; do
    SREF[$n]=$(ask_n "$n" "static-$n")
    printf "  STATIC  n=%-6s tail=%-3s out=[%s]\n" "$n" "$((n % 64))" "${SREF[$n]:0:48}" | tee -a "$OUT"
done
pkill -f "$SRV" >/dev/null 2>&1; sleep 2

# ---------------- CHAMPION ----------------
start_srv champ champ || { echo "FAIL: champion server never became healthy" | tee -a "$OUT"; exit 1; }
MK=$(grep -ahi "CHAMP-PAGED ACTIVE" "$LOGDIR/champ.log" | tail -1 | cut -c1-100)
declare -A CREF
for n in $COUNTS; do
    CREF[$n]=$(ask_n "$n" "champ-$n")
    printf "  CHAMP   n=%-6s tail=%-3s out=[%s]\n" "$n" "$((n % 64))" "${CREF[$n]:0:48}" | tee -a "$OUT"
done
# ⚠ marker read AFTER the requests: it is emitted on first dispatch, not at load.
MK=$(grep -ahi "CHAMP-PAGED ACTIVE" "$LOGDIR/champ.log" | tail -1 | cut -c1-100)
REFUSED=$(grep -ahic "CHAMP-PAGED REFUSED" "$LOGDIR/champ.log")
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
echo "champion marker: ${MK:-<none>}" | tee -a "$OUT"
echo "refusal lines:   $REFUSED" | tee -a "$OUT"
if [ -z "$MK" ]; then
    echo "VOID: no CHAMP-PAGED ACTIVE marker -- the champion never served. Any agreement below is" | tee -a "$OUT"
    echo "  the ORDINARY paged kernel matching static, which this gate does not test." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ⚠ NEGATIVE CONTROL, two-sided and using the SAME test as the verdict loop below. A comparator
# that reports "match" unconditionally passes every arm; one that reports "differ" unconditionally
# fails a working kernel. Both directions are checked before any verdict is emitted.
CTRL=ok
[ "alpha" != "alpha" ] && CTRL="BROKEN (identical strings reported as different)"
[ "alpha" != "beta" ]  || CTRL="BROKEN (different strings reported as identical)"
echo "comparator negative control: $CTRL (must be 'ok')" | tee -a "$OUT"
if [ "$CTRL" != ok ]; then
    echo "VOID: the comparator itself is wrong, so no verdict below can be trusted." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

FAILN=0; TOTAL=0
for n in $COUNTS; do
    TOTAL=$((TOTAL+1))
    if [ "${SREF[$n]}" != "${CREF[$n]}" ]; then
        FAILN=$((FAILN+1))
        echo "  MISMATCH n=$n tail=$((n % 64))" | tee -a "$OUT"
        echo "     static: ${SREF[$n]}" | tee -a "$OUT"
        echo "     champ : ${CREF[$n]}" | tee -a "$OUT"
    fi
done
echo | tee -a "$OUT"
if [ "$FAILN" -eq 0 ]; then
    echo "CHAMPION TAIL-FILL GATE: PASS -- $TOTAL/$TOTAL token-exact counts agree with static." | tee -a "$OUT"
    echo "  Correctness at these counts only. This gate does NOT name a cause: the blk-layout fix" | tee -a "$OUT"
    echo "  did not move it, and what did was the decode kernel's C=64-vs-NW=32 defect plus a" | tee -a "$OUT"
    echo "  missing mask/attention barrier. Speed is a separate bar." | tee -a "$OUT"
    echo "log: $OUT"; exit 0
fi
echo "CHAMPION TAIL-FILL GATE: FAIL -- $FAILN/$TOTAL counts still disagree with static." | tee -a "$OUT"
echo "  The blk fix was necessary but not sufficient. Record the surviving pattern; do NOT call it" | tee -a "$OUT"
echo "  a tail-fill law again without varying tail fill at FIXED total length." | tee -a "$OUT"
echo "log: $OUT"; exit 1
