#!/usr/bin/env bash
# CHAMPION-AT-BS64 PARITY PROBE -- the actual candidate for the owner's <=1.0x bar on Metal.
#
# HIS BAR (2026-08-04 table): "not finished until every row is <=1.0x -- then past it."
#     Metal prefill  static 1,131  paged 1,570   1.39x  X
#     Metal decode   static 1,214  paged 1,408   1.16x  X
#
# WHY THIS ARM AND NOT MMA. MMA was refuted today: destaging K raises nsg 1->2 at D=256 and MMA still
# loses to scalar (21,393 vs 13,986 ms). Dropped. The prefill decomposition says the gap is PER-KEY
# (linear term 0.98x = identical, quadratic 8.64x), and reading the default kernel shows the shape:
# the block-table lookup IS hoisted per block, but a threadgroup_barrier plus a full staging pass runs
# ONCE PER BLOCK. At bs=16 a 6,000-token prefill walks ~375 blocks = 375 barriers and 375 staging
# passes; a flash kernel chunking C=64 keys does a QUARTER of that.
#
# The champion port already exists (ported from ggml's kernel_flash_attn_ext_impl, Georgi Gerganov and
# the ggml contributors) and its recorded contract is exactly this: C must EQUAL the paged block_size,
# champion C = NCPSG = 64, so the pool must run bs=64. That is free -- pool footprint is bs-invariant
# (measured 3.46 GiB at bs=16 and bs=32 alike). It is built and DORMANT behind DS4P_METAL_CHAMP.
#
# ARMS (interleaved, so thermal drift hits all equally):
#   STATIC          no --kv-paged                      the bar
#   PAGED_BS16      default paged path                 today's 1.39x
#   CHAMP_BS64      DS4P_METAL_CHAMP=1, block-size 64  the candidate
#
# ⚠ PRE-REGISTERED:
#   CHAMP_BS64 <= STATIC            -> the bar is MET on prefill; go verify decode and correctness
#   CHAMP_BS64 < PAGED_BS16 only    -> real gain, bar not met; report the number, do not call it done
#   CHAMP_BS64 >= PAGED_BS16        -> the champion is not the answer either, and C=64 was not the term
#
# ⚠ PRESENCE, TWO-SIDED AND ON THE RIGHT AXIS. The champion arm is UNUSABLE unless its own
# CHAMP-PAGED marker appears -- DS4P_METAL_CHAMP is read in two places and the path REFUSES (logging
# "CHAMP-PAGED REFUSED (<why>)") when the geometry does not fit, which would otherwise silently
# measure the ordinary paged kernel and report it as the champion. Marker greps are CASE-INSENSITIVE:
# a case-sensitive pattern broke the arm under test twice today.
#
# ⚠ CORRECTNESS IS NOT OPTIONAL. A faster kernel that returns different text is not parity, so every
# arm's output is captured and compared; a mismatch overrides any timing verdict.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${CH_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${CH_PORT:-9033}
REPS=${CH_REPS:-3}
NTOK=${CH_NTOK:-6000}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champparity
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-parity-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS ntok=$NTOK" | tee "$OUT"
# ⚠ NON-DEGENERATE FILLER, and this is load-bearing. The old prompt was one sentence repeated
# hundreds of times, and STATIC ITSELF loops on it (' three, three, three, ...'). When ground truth is
# a degenerate loop, "arm differs from static" is not evidence of a kernel fault -- I spent an evening
# convicting the champion on exactly that input and had to withdraw five separate framings.
# Varied factual sentences keep static sane, so a disagreement means something.
PROMPT=$(python3 -c "
import random
random.seed(7)
subj = ['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
verb = ['is the capital of','lies north of','was founded before','trades heavily with','borders']
obj  = ['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
while len(' '.join(out)) < 52000:
    out.append('%s %s %s.' % (random.choice(subj), random.choice(verb), random.choice(obj)))
print(' '.join(out)[:52000])")

run_arm() { # $1 label  $2 mode(static|paged|champ)
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local envs=(DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1) flags=()
    case "$2" in
        static) flags=() ;;
        # ⚠ POOL CAPACITY IS EQUALISED IN TOKENS, NOT BLOCKS. Both paged arms used -ngpub 512,
        # which is 8,192 tokens at bs=16 and 32,768 at bs=64 -- a 4x difference in the pool ON TOP
        # of the thing under test. The small arm then DEADLOCKED on a 10k prompt ("scheduler cannot
        # make progress with the available block pool") and reported no timing in all 3 reps, which
        # read as a broken feature rather than a broken arm. 32,768 tokens each.
        paged)  flags=(--kv-paged --kv-block-size 16 -ngpub 2048 -ncpub 512) ;;
        champ)  envs+=(DS4P_METAL_CHAMP=1); flags=(--kv-paged --kv-block-size 64 -ngpub 512 -ncpub 128) ;;
    esac
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c 16384 -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ] && { echo "NEVER_READY|" ; return; }
    python3 -c "
import json
print(json.dumps({'prompt':'''$PROMPT''','n_predict':16,'temperature':0,'seed':1,'cache_prompt':False}))" > "$LOGDIR/$1.req"
    curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d @"$LOGDIR/$1.req" > "$LOGDIR/$1.json" 2>/dev/null
    local ms txt
    ms=$(grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
    txt=$(python3 -c "
import json
try: print(json.load(open('$LOGDIR/$1.json')).get('content','')[:60].replace(chr(10),' '))
except Exception: print('MALFORMED')")
    echo "${ms:-NA}|${txt}"
}

declare -a s_ms p_ms c_ms
declare -a s_tx p_tx c_tx
for r in $(seq 1 "$REPS"); do
    for arm in STATIC PAGED_BS16 CHAMP_BS64; do
        case "$arm" in
            STATIC)     res=$(run_arm "$arm-r$r" static) ;;
            PAGED_BS16) res=$(run_arm "$arm-r$r" paged) ;;
            CHAMP_BS64) res=$(run_arm "$arm-r$r" champ) ;;
        esac
        ms=${res%%|*}; txt=${res#*|}
        mk=""
        if [ "$arm" = CHAMP_BS64 ]; then
            mk=$(grep -ahi "CHAMP-PAGED\|CHAMP-VEC" "$LOGDIR/$arm-r$r.log" 2>/dev/null | tail -1 | cut -c1-90)
            [ -z "$mk" ] && mk="⚠ NO CHAMP MARKER (arm UNUSABLE -- may be the ordinary paged kernel)"
        fi
        printf "  %-11s r%s  prompt_eval=%s ms  out=[%s] %s\n" "$arm" "$r" "$ms" "${txt:0:40}" "$mk" | tee -a "$OUT"
        case "$arm" in
            STATIC)     s_ms+=("$ms"); s_tx+=("$txt") ;;
            PAGED_BS16) p_ms+=("$ms"); p_tx+=("$txt") ;;
            CHAMP_BS64) c_ms+=("$ms"); c_tx+=("$txt") ;;
        esac
    done
done
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
med() { printf '%s\n' "$@" | grep -v NA | sort -n | awk '{a[NR]=$1} END{if(NR==0){print "NA"}else{print a[int((NR+1)/2)]}}'; }
S=$(med "${s_ms[@]}"); Pg=$(med "${p_ms[@]}"); C=$(med "${c_ms[@]}")
echo "median prompt-eval ms:  STATIC=$S  PAGED_BS16=$Pg  CHAMP_BS64=$C" | tee -a "$OUT"
# correctness first -- a faster kernel that answers differently is not parity
if [ "${s_tx[0]:-x}" != "${c_tx[0]:-y}" ]; then
    echo "⚠ OUTPUT MISMATCH static vs champion -- TIMING VERDICT VOID until this is explained:" | tee -a "$OUT"
    echo "   static: ${s_tx[0]:-NA}" | tee -a "$OUT"
    echo "   champ : ${c_tx[0]:-NA}" | tee -a "$OUT"
fi
python3 - "$S" "$Pg" "$C" <<'PY' | tee -a "$OUT"
import sys
try: s,p,c = (float(x) for x in sys.argv[1:4])
except ValueError:
    print("INCONCLUSIVE: an arm produced no timing."); raise SystemExit(2)
print(f"  PAGED_BS16 vs static : {p/s:.3f}x")
print(f"  CHAMP_BS64 vs static : {c/s:.3f}x   <- THE BAR IS <= 1.000x")
print(f"  CHAMP  vs PAGED_BS16 : {p/c:.3f}x speedup")
print()
if c <= s:
    print("RESULT: BAR MET on Metal prefill. Verify decode and correctness before claiming it.")
elif c < p:
    print("RESULT: real gain, BAR NOT MET. Report the number; this is not done.")
else:
    print("RESULT: the champion does not beat the default paged path -- C=64 was not the term.")
PY
echo "log: $OUT"
