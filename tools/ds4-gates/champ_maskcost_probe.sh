#!/usr/bin/env bash
# CHAMPION MASK-COST PROBE -- is the champion's prefill cost driven by POOL CAPACITY or by CONTEXT?
#
# WHY THIS EXISTS. The champion is now numerically correct (test-paged-vs-cpu ALL PASSED, 9/9 end to
# end against the 26B) and SLOWER than static: 1.62x on prefill, 1.30x on decode at 10,003 tokens.
# The port derives all causality from a MATERIALISED mask which it refills every dispatch, sized
# n_heads * n_tokens * n_kv halves where n_kv = max_blk * block_size -- the POOL CAPACITY, not the
# sequence length. If that fill dominates, prefill time depends on -c even when the prompt does not
# change, and the fix is to write only the diagonal band plus classify whole blocks in blk[].
#
# ⚠ ONE FACTOR. Same prompt, same token count, same block size, same binary, same arm. ONLY -c moves.
# Two paged arms (champion on) at -c 16384 and -c 8192, plus STATIC at both as a CONTROL: static does
# not build this mask, so if STATIC also moves with -c then -c is changing something else (allocator
# pressure, graph size) and the comparison is confounded rather than informative.
#
# ⚠ INTERLEAVED AND REPEATED. Arm order is a confound worth 29% in this lane, so arms alternate and
# every arm is measured REPS times; the median is reported, not the best.
#
# PRE-REGISTERED:
#   champ time drops ~proportionally with n_kv, static flat  -> mask fill dominates; band it
#   champ time flat, static flat                             -> mask fill is NOT the cost; hypothesis
#                                                               dead, do not touch the mask kernel
#   static ALSO moves                                        -> INCONCLUSIVE, -c is not a clean factor
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${MC_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${MC_PORT:-9081}
REPS=${MC_REPS:-2}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champmask
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-maskcost-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS" | tee "$OUT"

# ~6,000 tokens: comfortably inside the SMALLER pool so neither arm spills or deadlocks.
python3 -c "
import random
random.seed(23)
subj=['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
verb=['is the capital of','lies north of','was founded before','trades heavily with','borders']
obj =['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
while len(' '.join(out)) < 24000:
    out.append('%s %s %s.' % (random.choice(subj), random.choice(verb), random.choice(obj)))
open('$LOGDIR/p.txt','w').write(' '.join(out))
"

run() { # $1 label  $2 mode(static|champ)  $3 n_ctx  $4 ngpub(blocks)
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local envs=(DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1) flags=()
    if [ "$2" = champ ]; then
        envs+=(DS4P_METAL_CHAMP=1)
        flags=(--kv-paged --kv-block-size 64 -ngpub "$4" -ncpub 128)
    fi
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c "$3" -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ] && { echo "NA|0"; return; }
    python3 -c "
import json
print(json.dumps({'prompt': open('$LOGDIR/p.txt').read(), 'n_predict': 8, 'temperature': 0,
                  'seed': 1, 'cache_prompt': False}))" > "$LOGDIR/$1.req"
    curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d @"$LOGDIR/$1.req" > "$LOGDIR/$1.json" 2>/dev/null
    local ms n mk
    ms=$(grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
    n=$(grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 | grep -oE "/ +[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)
    mk=0
    [ "$2" = champ ] && mk=$(grep -aci "CHAMP-PAGED ACTIVE" "$LOGDIR/$1.log")
    echo "${ms:-NA}|${n:-0}|${mk}"
}

declare -a s16 s08 c16 c08
for r in $(seq 1 "$REPS"); do
    # ⚠ pool capacity in TOKENS is held constant at 32,768 across both -c arms, so the ONLY thing
    # that changes with -c is n_kv... no: n_kv = max_blk*bs and max_blk follows the pool. Blocks are
    # therefore scaled WITH -c so the pool exactly covers the context and n_kv tracks -c. Equalising
    # capacity here would defeat the probe -- capacity IS the factor under test.
    for arm in STATIC16 CHAMP16 STATIC08 CHAMP08; do
        case "$arm" in
            STATIC16) res=$(run "$arm-r$r" static 16384 256) ;;
            CHAMP16)  res=$(run "$arm-r$r" champ  16384 256) ;;
            STATIC08) res=$(run "$arm-r$r" static  8192 128) ;;
            CHAMP08)  res=$(run "$arm-r$r" champ   8192 128) ;;
        esac
        ms=${res%%|*}; rest=${res#*|}; ntok=${rest%%|*}; mk=${rest##*|}
        printf "  %-9s r%s  prompt_eval=%-10s ms  tokens=%-6s champ_marker=%s\n" "$arm" "$r" "$ms" "$ntok" "$mk" | tee -a "$OUT"
        case "$arm" in
            STATIC16) s16+=("$ms") ;; CHAMP16) c16+=("$ms|$mk") ;;
            STATIC08) s08+=("$ms") ;; CHAMP08) c08+=("$ms|$mk") ;;
        esac
    done
done
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
# ⚠ ARM PRESENCE: a champion arm with no ACTIVE marker silently measured the ordinary paged kernel.
# That has faked a verdict twice in this lane, so it voids the run rather than footnoting it.
for v in "${c16[@]}" "${c08[@]}"; do
    [ "${v##*|}" = "0" ] && { echo "VOID: a champion arm never logged CHAMP-PAGED ACTIVE." | tee -a "$OUT"; echo "log: $OUT"; exit 2; }
done
med() { printf '%s\n' "$@" | cut -d'|' -f1 | grep -v NA | sort -n | awk '{a[NR]=$1} END{if(NR==0)print "NA"; else print a[int((NR+1)/2)]}'; }
S16=$(med "${s16[@]}"); S08=$(med "${s08[@]}"); C16=$(med "${c16[@]}"); C08=$(med "${c08[@]}")
echo "median prompt-eval ms:  STATIC c16384=$S16  c8192=$S08   CHAMP c16384=$C16  c8192=$C08" | tee -a "$OUT"
python3 - "$S16" "$S08" "$C16" "$C08" <<'PY' | tee -a "$OUT"
import sys
try: s16,s08,c16,c08 = (float(x) for x in sys.argv[1:5])
except ValueError:
    print("INCONCLUSIVE: an arm produced no timing."); raise SystemExit(2)
sr = s16/s08; cr = c16/c08
print(f"  STATIC  c16384/c8192 = {sr:.3f}x   <- control: should be ~1.00")
print(f"  CHAMP   c16384/c8192 = {cr:.3f}x   <- doubling the pool doubles the mask if the fill dominates")
print()
if sr > 1.15 or sr < 0.87:
    print("INCONCLUSIVE: static also moved with -c, so -c is not a clean one-factor knob here.")
elif cr >= 1.5:
    print("RESULT: champion prefill scales with POOL CAPACITY. The mask fill dominates.")
    print("  Fix: write only the diagonal band and classify whole blocks in blk[] (0 masked / 2 visible).")
elif cr <= 1.15:
    print("RESULT: champion prefill is FLAT in pool capacity. The mask fill is NOT the cost.")
    print("  Hypothesis dead. Do not touch the mask kernel; profile the attention body instead.")
else:
    print(f"RESULT: partial ({cr:.3f}x). The mask contributes but does not dominate; measure the")
    print("  kernels separately before committing to a rewrite.")
PY
echo "log: $OUT"
