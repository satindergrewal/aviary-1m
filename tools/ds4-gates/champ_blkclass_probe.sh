#!/usr/bin/env bash
# CHAMPION BLOCK-CLASSIFICATION PROBE -- does skipping off-diagonal blocks close the gap to static?
#
# STATE THIS MEASURES AGAINST. The champion is numerically correct (test-paged-vs-cpu ALL PASSED,
# 9/9 end to end on the 26B) and SLOWER than static, by a margin that GROWS with context:
#     4,591 tokens  1.253x      10,003 tokens  1.623x
# A per-dispatch or per-token overhead cannot produce that shape; a per-KEY cost can.
#
# THE CHANGE UNDER TEST. kernel_paged_champ_mask now classifies each (head, q-block, kv-block) tile:
#     0 = wholly above the diagonal -> the champion skips the block entirely (no mask, no K/V read)
#     2 = wholly below              -> taken with no mask load at all
#     1 = the diagonal, or a q-block straddling two sequences, or a rel bias present
# Causal prefill is about half 0 and half 2, so the walk should shed close to half its work. Before
# tonight every tile was 1, which is correct and the slowest value the kernel accepts.
#
# ⚠ ONE FACTOR, VIA A REAL SWITCH. DS4P_CHAMP_BLKCLASS=0 forces the old always-1 behaviour in the
# same binary, so ON and OFF differ in exactly this and nothing else -- not in build, not in flags.
#
# ⚠ CORRECTNESS GATES THE TIMING. A block wrongly marked 0 drops keys and a block wrongly marked 2
# attends masked ones; both are fast and wrong. Every arm's text is compared against STATIC and a
# mismatch VOIDS the timing verdict. This is not optional: two "BAR MET" results in this lane were
# already produced by arms that were not doing the work.
#
# PRE-REGISTERED:
#   CHAMP_ON <= STATIC                    -> his <=1.0x bar met on prefill; verify decode next
#   CHAMP_ON < CHAMP_OFF, still > STATIC  -> the mask read was part of the cost, not all of it
#   CHAMP_ON ~= CHAMP_OFF                 -> classification buys nothing; the cost is the K/V gather
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${BC_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${BC_PORT:-9085}
REPS=${BC_REPS:-3}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champblk
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-blkclass-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS" | tee "$OUT"

# ~10k tokens: the length where the gap was 1.623x, so the effect has room to show.
#
# ⚠ THE STIMULUS ENDS IN A DETERMINATE QUESTION, AND THAT IS THE POINT. Random
# 'CITY is the capital of COUNTRY' filler has no right answer for the next token -- ten cities sit
# within noise of each other. The champion is numerically correct but NOT bit-identical to static
# (max_abs 5e-09 against the CPU reference), and 5e-09 flips a coin-flip token under greedy decoding.
# The first version of this probe voided itself 3/3 on exactly that: both champion arms agreed with
# EACH OTHER and differed from static, on a continuation where both answers were equally supported.
#
# The filler stays random so static cannot loop on it -- that trap cost an evening earlier -- but the
# tokens actually COMPARED now come after a question with one correct answer. Degenerate baseline at
# one extreme, undetermined baseline at the other; the comparison has to sit between them.
python3 -c "
import random
random.seed(31)
subj=['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
verb=['is the capital of','lies north of','was founded before','trades heavily with','borders']
obj =['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
while len(' '.join(out)) < 52000:
    out.append('%s %s %s.' % (random.choice(subj), random.choice(verb), random.choice(obj)))
text = ' '.join(out)
text += ('\n\nIgnore the notes above. Answer this from general knowledge, one word only.\n'
         'Question: What is the capital city of Japan?\nAnswer:')
open('$LOGDIR/p.txt','w').write(text)
"

run() { # $1 label  $2 mode(static|on|off)
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local envs=(DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1) flags=()
    case "$2" in
        static) ;;
        on)  envs+=(DS4P_METAL_CHAMP=1); flags=(--kv-paged --kv-block-size 64 -ngpub 512 -ncpub 128) ;;
        off) envs+=(DS4P_METAL_CHAMP=1 DS4P_CHAMP_BLKCLASS=0); flags=(--kv-paged --kv-block-size 64 -ngpub 512 -ncpub 128) ;;
    esac
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c 16384 -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ] && { echo "NA|NEVER_READY|0|NA"; return; }
    python3 -c "
import json
print(json.dumps({'prompt': open('$LOGDIR/p.txt').read(), 'n_predict': ${CB_NPRED:-64}, 'ignore_eos': True, 'temperature': 0,
                  'seed': 1, 'cache_prompt': False}))" > "$LOGDIR/$1.req"
    curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d @"$LOGDIR/$1.req" > "$LOGDIR/$1.json" 2>/dev/null
    local pms dms txt mk
    pms=$(grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
    dms=$(grep -a "        eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "[0-9.]+ tokens per" | grep -oE "[0-9.]+" | head -1)
    # ⚠ A RATE WITHOUT ITS SAMPLE SIZE CANNOT BE AUDITED. n_predict is a CEILING; the achieved
    # count comes from the same log line the rate does. Five gates carried this hole tonight.
    dn=$(grep -a "        eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "/ *[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
    txt=$(python3 -c "
import json
try: print(json.load(open('$LOGDIR/$1.json')).get('content','')[:56].replace(chr(10),' '))
except Exception: print('MALFORMED')")
    mk=0
    [ "$2" != static ] && mk=$(grep -aci "CHAMP-PAGED ACTIVE" "$LOGDIR/$1.log")
    echo "${pms:-NA}|${txt}|${mk}|${dms:-NA}|${dn:-NA}"
}

declare -a S C1 C0
declare -a ST CT1 CT0
for r in $(seq 1 "$REPS"); do
    for arm in STATIC CHAMP_ON CHAMP_OFF; do
        case "$arm" in
            STATIC)    res=$(run "$arm-r$r" static) ;;
            CHAMP_ON)  res=$(run "$arm-r$r" on) ;;
            CHAMP_OFF) res=$(run "$arm-r$r" off) ;;
        esac
        pms=${res%%|*}; rest=${res#*|}; txt=${rest%%|*}; rest=${rest#*|}; mk=${rest%%|*}; rest=${rest#*|}; dtps=${rest%%|*}; dn=${rest##*|}
        printf "  %-10s r%s  prefill=%-10s ms  decode=%-7s tok/s (n=%s)  marker=%s  out=[%s]\n" \
               "$arm" "$r" "$pms" "$dtps" "${dn:-?}" "$mk" "${txt:0:36}" | tee -a "$OUT"
        case "$arm" in
            STATIC)    S+=("$pms|$dtps"); ST+=("$txt") ;;
            CHAMP_ON)  C1+=("$pms|$dtps|$mk"); CT1+=("$txt") ;;
            CHAMP_OFF) C0+=("$pms|$dtps|$mk"); CT0+=("$txt") ;;
        esac
    done
done
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
for v in "${C1[@]}" "${C0[@]}"; do
    f=${v#*|}; [ "${f#*|}" = "0" ] && { echo "VOID: a champion arm never logged CHAMP-PAGED ACTIVE." | tee -a "$OUT"; echo "log: $OUT"; exit 2; }
done
BAD=0
for i in "${!ST[@]}"; do
    [ "${ST[$i]}" != "${CT1[$i]:-}" ] && { echo "⚠ OUTPUT MISMATCH rep$((i+1)) static vs CHAMP_ON -- TIMING VOID" | tee -a "$OUT"; echo "   static: ${ST[$i]}" | tee -a "$OUT"; echo "   champ : ${CT1[$i]:-NA}" | tee -a "$OUT"; BAD=1; }
done
med() { printf '%s\n' "$@" | cut -d'|' -f"$1" | grep -v NA | sort -n | awk '{a[NR]=$1} END{if(NR==0)print "NA"; else print a[int((NR+1)/2)]}'; }
SP=$(med 1 "${S[@]}");  SD=$(med 2 "${S[@]}")
P1=$(med 1 "${C1[@]}"); D1=$(med 2 "${C1[@]}")
P0=$(med 1 "${C0[@]}"); D0=$(med 2 "${C0[@]}")
echo "median prefill ms:   STATIC=$SP  CHAMP_ON=$P1  CHAMP_OFF=$P0" | tee -a "$OUT"
echo "median decode tok/s: STATIC=$SD  CHAMP_ON=$D1  CHAMP_OFF=$D0" | tee -a "$OUT"
[ "$BAD" = 1 ] && { echo "VERDICT VOID: a champion arm disagreed with static." | tee -a "$OUT"; echo "log: $OUT"; exit 1; }
python3 - "$SP" "$P1" "$P0" "$SD" "$D1" "$D0" <<'PY' | tee -a "$OUT"
import sys
try: sp,p1,p0,sd,d1,d0 = (float(x) for x in sys.argv[1:7])
except ValueError:
    print("INCONCLUSIVE: an arm produced no timing."); raise SystemExit(2)
print(f"  prefill  CHAMP_OFF vs static = {p0/sp:.3f}x")
print(f"  prefill  CHAMP_ON  vs static = {p1/sp:.3f}x   <- THE BAR IS <= 1.000x")
print(f"  prefill  classification gain = {p0/p1:.3f}x")
print(f"  decode   CHAMP_ON  vs static = {sd/d1:.3f}x   <- THE BAR IS <= 1.000x")
print()
if p1 <= sp:
    print("RESULT: prefill BAR MET. Decode is the remaining row.")
elif p1 < p0*0.95:
    print("RESULT: classification is a real gain, bar not met. Report the number; not done.")
else:
    print("RESULT: classification bought nothing measurable -- the cost is the K/V gather, not the")
    print("  mask read. Go after the block walk.")
PY
echo "log: $OUT"
