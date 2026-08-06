#!/usr/bin/env bash
# CHAMPION PREFILL SCALING PROBE -- fit t = a*T + b*T^2 on FOUR lengths, not two.
#
# WHY THIS EXISTS AND WHY IT REPLACES WHAT I DID. I fitted that model from TWO lengths and reported
# "linear 0.965x, quadratic 3.31x" as if it were a result. Two points fitting two parameters is an
# EXACT fit: it has no residual, no error bar, and no way to be wrong. It is a restatement of the two
# numbers in different units, not evidence that the model describes the curve. Four lengths can
# disagree with the model, which is the whole point of measuring them.
#
# ⚠ ONE SERVER PER ARM, ALL LENGTHS INSIDE IT. Each server start costs minutes and drags thermal
# state with it; issuing every length against one live server keeps the arm's conditions fixed and
# removes start-order as a confound. -cram 0 and cache_prompt false so every request really prefills.
#
# ⚠ TOKEN-EXACT. Prompts are token-id arrays from /tokenize, so T is the number fitted, not an
# estimate from characters.
#
# ⚠ n_predict = 1: this measures PREFILL. Generation would ride along and dilute it.
#
# PRE-REGISTERED:
#   model fits all 4 within a few %  -> the decomposition is real; b_champ/b_static is the target
#   4th point off the 2-point fit    -> the decomposition was an artefact of picking two lengths,
#                                       and the "3.31x per-key" number must be withdrawn
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${SC_MODEL:-$HOME/Documents/GitHub/ornith-models/Gemma4-26B-A4B-Uncensored-1M/gemma4-26b-a4b-uncensored-1M-Q4_K_M.gguf}
# ⚠ PICK A FREE PORT, DO NOT ASSUME ONE. The fixed default collided with another process already
# listening on this machine, and the failure surfaced as "static server never became healthy" -- which
# reads as a model or build problem and is neither. A gate that cannot start must say WHY.
pick_port() {
    local p
    for p in $(seq "${SC_PORT:-9091}" $(( ${SC_PORT:-9091} + 40 ))); do
        if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}
P=$(pick_port) || { echo "FAIL: no free port in range"; exit 1; }
REPS=${SC_REPS:-2}
CTX=${SC_CTX:-16384}
COUNTS=${SC_COUNTS:-"2048 4608 8192 12288"}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/champscale
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/champ-scaling-$(date +%Y%m%d-%H%M).txt
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

# ⚠ FILLER SIZED FROM THE LARGEST REQUESTED COUNT, not a constant. A fixed 70,000 chars tokenised to
# 13,508 tokens and the probe correctly refused a 131,072-token request -- but that is a gate that
# cannot reach serve-class context by construction, which is the whole point of the ladder. ~4.3
# chars/token measured on this filler, so 8x the count gives generous margin.
MAXC_PRE=$(echo "$COUNTS" | tr ' ' '\n' | sort -rn | head -1)
FILLER_CHARS=$(( MAXC_PRE * 8 ))
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  counts=$COUNTS reps=$REPS" | tee "$OUT"

python3 -c "
import random
random.seed(41)
subj=['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
verb=['is the capital of','lies north of','was founded before','trades heavily with','borders']
obj =['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
while len(' '.join(out)) < $FILLER_CHARS:
    out.append('%s %s %s.' % (random.choice(subj), random.choice(verb), random.choice(obj)))
open('$LOGDIR/f.txt','w').write(' '.join(out))
"

start() { # $1 label  $2 mode
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # ⚠ SC_SWA is a ONE-FACTOR ARM, not a tuning knob. With it at 0 the SWA layers fall back to the
    # static path and only the global layers page, which tests whether the paged SWA layers are what
    # carry the quadratic term: static gives its SWA layers a physically window-sized cache, while
    # the paged pool holds the whole context for them and relies on block skipping to stay cheap.
    local envs=(DS4P_PAGED_HYBRID=1) flags=()
    [ "${SC_SWA:-1}" = "1" ] && envs+=(DS4P_PAGED_SWA=1)
    if [ "$2" = champ ]; then
        # pool sized to the context, not to a constant: 512 blocks is 32,768 tokens at bs=64 and
        # silently deadlocks any prompt past it ("scheduler cannot make progress").
        local nblk=$(( (CTX + 63) / 64 ))
        envs+=(DS4P_METAL_CHAMP=1); flags=(--kv-paged --kv-block-size 64 -ngpub "$nblk" -ncpub 256)
    fi
    env "${envs[@]}" nohup "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 2048 -ub 2048 \
        --port $P --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    for _ in $(seq 1 900); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    # ⚠ SAY WHY IT DIED. "never became healthy" hid a port collision AND a GGML_ASSERT abort tonight,
    # two unrelated root causes behind one string, and both sent me looking at the model instead of
    # the log that was already on disk. A start failure prints the child's own last words.
    echo "  --- $1 server did not start; last lines of its log ---"
    tail -12 "$LOGDIR/$1.log" | sed 's/^/  | /'
    grep -aiE "assert|abort|couldn.t bind|out of memory|failed to allocate" "$LOGDIR/$1.log" | tail -4 | sed 's/^/  ! /'
    return 1
}
tokenize() {
    python3 -c "
import json; print(json.dumps({'content': open('$LOGDIR/f.txt').read()}))" > "$LOGDIR/t.req"
    curl -s --max-time 300 -X POST http://127.0.0.1:$P/tokenize -H 'Content-Type: application/json' \
        -d @"$LOGDIR/t.req" > "$LOGDIR/t.json"
    python3 -c "
import json; t=json.load(open('$LOGDIR/t.json'))['tokens']
open('$LOGDIR/toks.json','w').write(json.dumps(t)); print(len(t))"
}
ask() { # $1 count $2 label -> prompt-eval ms
    python3 -c "
import json; t=json.load(open('$LOGDIR/toks.json'))[:$1]
print(json.dumps({'prompt': t, 'n_predict': 1, 'temperature': 0, 'cache_prompt': False}))" > "$LOGDIR/$2.req"
    curl -s --max-time 900 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d @"$LOGDIR/$2.req" > /dev/null 2>&1
    grep -a "prompt eval time" "$LOGDIR/$3.log" | tail -1 | grep -oE "= *[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1
}

declare -A SM CM
for mode in static champ; do
    start "$mode" "$mode" || { echo "FAIL: $mode server never became healthy" | tee -a "$OUT"; exit 1; }
    NT=$(tokenize)
    MAXC=$(echo "$COUNTS" | tr ' ' '\n' | sort -rn | head -1)
    if [ "${NT:-0}" -lt "$MAXC" ]; then
        echo "INCONCLUSIVE: filler is $NT tokens, need >= $MAXC." | tee -a "$OUT"; pkill -f "$SRV"; exit 2
    fi
    for r in $(seq 1 "$REPS"); do
        for n in $COUNTS; do
            ms=$(ask "$n" "$mode-$n-r$r" "$mode")
            printf "  %-6s n=%-6s r%s  prompt_eval=%s ms\n" "$mode" "$n" "$r" "$ms" | tee -a "$OUT"
            if [ "$mode" = static ]; then SM[$n]="${SM[$n]:-} $ms"; else CM[$n]="${CM[$n]:-} $ms"; fi
        done
    done
    if [ "$mode" = champ ]; then
        MK=$(grep -ahci "CHAMP-PAGED ACTIVE" "$LOGDIR/champ.log")
        RF=$(grep -ahci "CHAMP-PAGED REFUSED" "$LOGDIR/champ.log")
    fi
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
done

echo "-----" | tee -a "$OUT"
echo "champion markers: ACTIVE=$MK REFUSED=$RF" | tee -a "$OUT"
echo "measured memory (server's own breakdown, champ arm):" | tee -a "$OUT"
grep -a "common_memory_breakdown_print" "$LOGDIR/champ.log" | tail -3 | sed 's/^.*print: //' | sed 's/^/  /' | tee -a "$OUT"
if [ "${MK:-0}" = "0" ]; then
    echo "VOID: champion arm never logged CHAMP-PAGED ACTIVE." | tee -a "$OUT"; echo "log: $OUT"; exit 2
fi
{ for n in $COUNTS; do echo "$n |${SM[$n]} |${CM[$n]}"; done; } > "$LOGDIR/fit.txt"
python3 - "$LOGDIR/fit.txt" <<'PY' | tee -a "$OUT"
import sys
rows=[]
for line in open(sys.argv[1]):
    n,s,c = line.split('|')
    s=[float(x) for x in s.split()]; c=[float(x) for x in c.split()]
    if not s or not c: continue
    s.sort(); c.sort()
    rows.append((int(n), s[len(s)//2], c[len(c)//2]))
if len(rows) < 3:
    print("INCONCLUSIVE: fewer than 3 usable lengths."); raise SystemExit(2)
print("   T        static ms    champ ms    ratio")
for T,s,c in rows: print(f"  {T:<8} {s:>10.1f} {c:>11.1f}    {c/s:.3f}x")
print()
def fit(idx):
    # least squares on t = a*T + b*T^2 across ALL points, then report the worst residual
    n=len(rows); sxx=sxy=syy=sxz=syz=0.0
    for r in rows:
        T=r[0]; t=r[idx]; x=T; y=T*T
        sxx+=x*x; sxy+=x*y; syy+=y*y; sxz+=x*t; syz+=y*t
    det = sxx*syy - sxy*sxy
    a = ( syy*sxz - sxy*syz)/det
    b = (-sxy*sxz + sxx*syz)/det
    worst = max(abs(a*r[0]+b*r[0]**2 - r[idx])/r[idx] for r in rows)
    return a,b,worst
a_s,b_s,w_s = fit(1)
a_c,b_c,w_c = fit(2)
print(f"  static  linear {a_s:.4f} ms/tok   quadratic {b_s:.4e}   worst residual {w_s*100:.1f}%")
print(f"  champ   linear {a_c:.4f} ms/tok   quadratic {b_c:.4e}   worst residual {w_c*100:.1f}%")
print(f"  -> linear ratio {a_c/a_s:.3f}x, quadratic ratio {b_c/b_s:.3f}x")
print()
# ⚠ COLLINEARITY CHECK -- the gate the residual test could not be. Over a narrow T range, T and T^2
# are nearly proportional, so a and b trade off almost freely: the CURVE fits while the COEFFICIENTS
# are undetermined. Measured on this very probe: the same model and binary gave linear 0.909x /
# quadratic 1.620x over 2k-12k and linear 1.200x / quadratic 1.151x over 32k-131k, with residuals of
# 0.0-3.5% on both. A residual gate tests fit, not identifiability, and I pre-registered one that
# could only ever confirm me.
Tmin = min(r[0] for r in rows); Tmax = max(r[0] for r in rows)
span = Tmax / Tmin
print(f"  T span {Tmin}..{Tmax} = {span:.1f}x")
if span < 12:
    print()
    print(f"COEFFICIENTS NOT IDENTIFIABLE: a {span:.1f}x span in T is too narrow to separate a*T from")
    print("  b*T^2 -- the two predictors are nearly collinear and the split between them is arbitrary.")
    print("  The RATIOS above are measured and stand. The linear/quadratic attribution DOES NOT.")
    print("  Report the ratio curve; do not target 'the quadratic term' from a fit this narrow.")
elif max(w_s,w_c) > 0.06:
    print(f"WITHDRAWN: the a*T + b*T^2 model misses a measured point by {max(w_s,w_c)*100:.1f}%.")
else:
    print("MODEL FITS and the T span is wide enough to separate the terms. Attribution usable,")
    print("  but cross-check it against a fit on a disjoint T range before targeting a term.")
PY
echo "log: $OUT"
