#!/usr/bin/env bash
# MMA NOSTAGE-K PROBE -- does freeing the staged-K tile lift MMA out of its occupancy hole at D=256?
#
# THE STANDING MEASUREMENT this attacks. Paged prefill fits t = a*n + b*n^2 with:
#     linear     paged 1.3275e-3 vs static 1.3567e-3   = 0.98x, IDENTICAL
#     quadratic  paged 8.8281e-8 vs static 1.0223e-8   = 8.64x
# Identical linear + 8.6x quadratic = the cost is PER-KEY. Per-block was ruled out separately
# (--kv-block-size 8/16/32 spans 1.26x against a 1.40x error bar; per-block predicted ~4x).
#
# MMA IS ALREADY REFUTED AS THE FIX, and the condition is recorded so it is not re-tried blind:
# at D=256 enabling MMA is SLOWER (12.00s prefill / 105.70s vs scalar 7.27 / 71.62). The reason is
# occupancy, not arithmetic -- nsg collapses to 1 because the MMA tile takes 29,760 of the 32,768-byte
# threadgroup budget. See memory: "refuted needs its condition".
#
# ★ THE LEVER: DS4P_METAL_MMA_NOSTAGE_K drops the staged K tile, which the smem model
# (ggml-metal-ops.cpp:4837) says costs KT*head_dim halves -- 16 KB at KT=32/D=256. That is exactly
# the headroom between nsg=1 and nsg=2/4. V is already read straight from device, which is what makes
# destaging K viable at flat cost.
#
# ⚠ PRE-REGISTERED PREDICTION, written before the run so the result can falsify it:
#   If occupancy is the binding constraint -> nsg rises above 1 AND MMA stops losing to scalar.
#   If nsg rises and MMA is STILL slower -> occupancy was NOT the mechanism. That kills a lead I have
#   carried since the perf arc opened, and it is the more useful outcome.
#   If nsg does NOT rise -> the flag did not take effect; the run says nothing about occupancy.
#
# ⚠ PRESENCE FIRST, TWO-SIDED. Every arm's own DS4P-MMA marker is captured and the arm is marked
# UNUSABLE if it is missing. The marker reports nsg, KSTAGE and smem, so "the flag took effect" is
# read off the log rather than inferred from a timing change. DS4P_METAL_MMA is PRESENCE-checked
# (=0 still ENABLES it), so the scalar arm must UNSET it, not set it to 0.
#
# ⚠ INTERLEAVED, not blocked: arms alternate rep by rep so thermal drift hits all three equally.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${MN_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
P=${MN_PORT:-9031}
REPS=${MN_REPS:-3}
NTOK=${MN_NTOK:-6000}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/mmanostage
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/mma-nostage-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  reps=$REPS ntok=$NTOK" | tee "$OUT"

PROMPT=$(python3 -c "print(('the quick brown fox jumps over the lazy dog and then keeps running through the field . ' * $((NTOK/18+40)))[:60000])")

run_arm() { # $1 label  $2 mma(on|off)  $3 nostage(0|1)
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    local env_args=(DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1)
    [ "$2" = on ] && env_args+=(DS4P_METAL_MMA=1)
    [ "$3" = 1 ]  && env_args+=(DS4P_METAL_MMA_NOSTAGE_K=1)
    env "${env_args[@]}" nohup "$SRV" -m "$M" -ngl 99 -c 16384 -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 --kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128 \
        > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ] && { echo "NEVER_READY"; return; }
    curl -s --max-time 600 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "$(python3 -c "
import json,sys
print(json.dumps({'prompt':'''$PROMPT''','n_predict':1,'temperature':0,'seed':1,'cache_prompt':False}))")" \
      > "$LOGDIR/$1.json" 2>/dev/null
    # prompt eval ms and token count come from the server's own timing line
    grep -a "prompt eval time" "$LOGDIR/$1.log" | tail -1 \
      | grep -oE "= *[0-9.]+ ms / +[0-9]+ tokens" | grep -oE "[0-9.]+" | tr '\n' ' '
    echo
}

declare -a t_scalar t_mma t_nost
for r in $(seq 1 "$REPS"); do
    for arm in SCALAR MMA MMA_NOSTAGE; do
        case "$arm" in
            SCALAR)      v=$(run_arm "$arm-r$r" off 0) ;;
            MMA)         v=$(run_arm "$arm-r$r" on  0) ;;
            MMA_NOSTAGE) v=$(run_arm "$arm-r$r" on  1) ;;
        esac
        ms=$(echo "$v" | awk '{print $1}')
        ntk=$(echo "$v" | awk '{print $2}')
        mk=$(grep -a "DS4P-MMA ACTIVE" "$LOGDIR/$arm-r$r.log" 2>/dev/null | tail -1 \
             | grep -oiE "D=[0-9]+ bs=[0-9]+ sb=[0-9]+ KT=[0-9]+ nsg=[0-9]+ QR=[0-9]+ KSTAGE=[a-z]+ smem=[0-9]+" | head -1)
    # ⚠ -i IS LOAD-BEARING. The first version matched KSTAGE=[a-z]+ and the kernel prints
    # "KSTAGE=OFF" in caps, so every NOSTAGE arm -- the ONLY arm that can print OFF, i.e. exactly the
    # arm under test -- was scored "NO MARKER (arm UNUSABLE)". Second case-sensitivity bug of the day
    # in my own harness; the first was the multislot semantic bar. The guard did its job (it refused
    # rather than reporting a null), but a case-blind pattern would not have needed rescuing.
        [ -z "$mk" ] && mk=$(grep -a "DS4P-MMA OFF" "$LOGDIR/$arm-r$r.log" 2>/dev/null | tail -1 | grep -oE "DS4P-MMA OFF[^,]*" | head -1)
        [ -z "$mk" ] && mk="⚠ NO MARKER (arm UNUSABLE)"
        printf "  %-12s r%s  prompt_eval=%s ms / %s tok   %s\n" "$arm" "$r" "${ms:-NA}" "${ntk:-NA}" "$mk" | tee -a "$OUT"
        case "$arm" in
            SCALAR)      t_scalar+=("${ms:-NA}") ;;
            MMA)         t_mma+=("${ms:-NA}") ;;
            MMA_NOSTAGE) t_nost+=("${ms:-NA}") ;;
        esac
    done
done
pkill -f "$SRV" >/dev/null 2>&1

echo "-----" | tee -a "$OUT"
med() { printf '%s\n' "$@" | grep -v NA | sort -n | awk '{a[NR]=$1} END{if(NR==0){print "NA"}else{print a[int((NR+1)/2)]}}'; }
ms_scalar=$(med "${t_scalar[@]}"); ms_mma=$(med "${t_mma[@]}"); ms_nost=$(med "${t_nost[@]}")
echo "median prompt-eval ms:  SCALAR=$ms_scalar  MMA=$ms_mma  MMA_NOSTAGE=$ms_nost" | tee -a "$OUT"
python3 - "$ms_scalar" "$ms_mma" "$ms_nost" <<'PY' | tee -a "$OUT"
import sys
s,m,n = sys.argv[1:4]
try:
    s,m,n = float(s), float(m), float(n)
except ValueError:
    print("INCONCLUSIVE: an arm produced no timing; nothing to compare."); raise SystemExit(2)
print(f"  MMA vs scalar         : {s/m:.3f}x   ({'MMA faster' if m<s else 'MMA slower'})")
print(f"  MMA_NOSTAGE vs scalar : {s/n:.3f}x   ({'faster' if n<s else 'slower'})")
print(f"  NOSTAGE vs staged MMA : {m/n:.3f}x   ({'destaging helps' if n<m else 'destaging does not help'})")
print()
if n < s:
    print("RESULT: destaging K lifts MMA past scalar at this D. Occupancy WAS the binding constraint.")
elif n < m:
    print("RESULT: destaging K helps MMA but MMA still loses to scalar. Occupancy is A constraint,")
    print("  not THE constraint -- the per-key quadratic lives somewhere else too.")
else:
    print("RESULT: destaging K does NOT help. Occupancy is REFUTED as the mechanism, and the")
    print("  'free 16 KB and nsg rises' lead is dead. ⚠ Check the nsg column FIRST: if nsg did not")
    print("  actually rise, the flag never took effect and this says nothing about occupancy.")
PY
echo "log: $OUT"
