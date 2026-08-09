#!/usr/bin/env bash
# PAGED PARITY GATE -- is Metal paging AS FAST OR FASTER than static? (the owner's bar, 2026-08-09)
#
# ⚠ WHY THIS FILE EXISTS AT ALL. The only parity numbers this programme has -- 225k on the 9B
# (wall 0.9831, decode 1.558x) and the 512k rung -- were produced by a SCRATCH SCRIPT in the job's tmp
# directory, hardcoded to one model, deleted when the job is deleted. **The harness behind the
# headline numbers was not in the repo and could not be re-run by anyone else.** Same class as every
# undocumented instrument in this lane.
#
# ⚠ THIS IS NOT `long_context_gate`. That one is NEEDLE-GATED CORRECTNESS -- found / not found, no
# timing. This one answers the SPEED question and is needle-gated only as a VALIDITY guard:
# **a speed number from a run that got the answer wrong is not a speed number.**
#
# ⚠⚠ ARM ORDER IS A CONFOUND, MEASURED ON THIS BOX. 29% positional drift once produced 1.41x in BOTH
# directions when the truth was 1.317x. A single static-then-paged pass is ONE ordering, and this gate
# says so rather than pretending otherwise -- run it twice with PP_ORDER=paged-first and compare.
# Agreement across both orders is the claim; a single order is a data point.
#
# ⚠ NO EXPLICIT -ngpub. The pool is AUTO-FIT, which is what LLAMA_PAGED_POOL_HEADROOM governs. An
# explicit -ngpub silently disables that governor -- on 2026-08-09 a hardcoded 4096 blocks capped
# long_context_gate at 131,072 tokens and made it print "PAGED LOSES CONTEXT -- this is ours" for a
# capacity failure. Headroom defaults to 1.05 here because the 512k rung proved 1.5 REFUSES to start.
#
# usage:  PP_MODEL=<gguf> PP_CTX=262144 PP_FILL=225000 [PP_ORDER=paged-first] paged_parity_gate.sh
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
. "$(dirname "$0")/_gate_common.sh"  2>/dev/null || true

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
M=${PP_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
CTX=${PP_CTX:-262144}
FILL=${PP_FILL:-225000}
BLK=${PP_BLOCK:-64}
ORDER=${PP_ORDER:-static-first}
NPRED=${PP_NPRED:-48}
export LLAMA_PAGED_POOL_HEADROOM=${LLAMA_PAGED_POOL_HEADROOM:-1.05}

[ -f "$M" ] || { echo "missing model: $M" >&2; exit 2; }
D=${CLAUDE_JOB_DIR:-/tmp}/parity; mkdir -p "$D"
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/parity-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
NEEDLE="MAGENTA-7742"

# ⚠ ONE SERVER AT A TIME, and a lock rather than a race. Detached probes sharing a pattern-kill
# destroyed each other's servers on 2026-08-07; one survivor held a port for 16 minutes on 2026-08-09.
LOCK=$D/gpu.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
PID=""
trap 'rmdir "$LOCK" 2>/dev/null; [ -n "$PID" ] && kill $PID 2>/dev/null; scrub_abs_paths "${OUT:-}" 2>/dev/null' EXIT

pick_port() { local p; for p in $(seq 20100 20160); do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }; done; return 1; }
PORT=$(pick_port) || { echo "no free port" >&2; exit 2; }

echo "paged parity gate: $(basename "$M")" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD)  ctx=$CTX  fill~${FILL}tok  block=$BLK  order=$ORDER  headroom=$LLAMA_PAGED_POOL_HEADROOM" | tee -a "$OUT"

python3 - "$D" "$NEEDLE" "$FILL" "$NPRED" <<'PY' | tee -a "$OUT"
import json, sys
D, NEEDLE, FILL, NPRED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# ⚠ MEASURED, NOT GUESSED: 23.4 tokens per line for this filler and tokenizer. An earlier 12 made
# LC_FILL mean half of what it claimed and built prompts twice the requested size.
TOK_PER_LINE = 23.4
lines = int(FILL / TOK_PER_LINE); at = lines // 2
buf = [f"Note {i}: The secret passcode is {NEEDLE}. Remember it." if i == at
       else f"Note {i}: the quick brown fox jumps over the lazy dog near the old stone bridge."
       for i in range(lines)]
p = ("Below are many numbered notes. One of them contains a secret passcode.\n\n" + "\n".join(buf)
     + "\n\nQuestion: what is the secret passcode? Answer with the code only.\nAnswer:")
json.dump({"prompt": p, "n_predict": NPRED, "temperature": 0, "seed": 1, "cache_prompt": False},
          open(f"{D}/req.json", "w"))
print(f"  prompt: {lines} lines, needle at 50% depth, target ~{FILL} tok")
PY

arm() { # $1 = static|paged
    local flags=""; [ "$1" = paged ] && flags="--kv-paged --kv-block-size $BLK"
    : > "$D/$1.log"
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 "$SRV" -m "$M" -ngl 99 -c "$CTX" \
        -np 1 -b 512 -ub 512 --port $PORT --no-warmup -lv 4 $flags > "$D/$1.log" 2>&1 &
    PID=$!
    local i
    for i in $(seq 1 900); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
        kill -0 $PID 2>/dev/null || {
            command -v gate_cause_from_log >/dev/null 2>&1 && gate_cause_from_log "$D/$1.log" "$1 arm" | tee -a "$OUT"
            PID=""; return 1; }
        sleep 1
    done
    # ⚠ PRESENCE ASSERTED IN BOTH DIRECTIONS. static must show NO pool, paged must show one. Without
    # this, "static" can silently build a pool and the comparison is paged-vs-paged.
    local pool; pool=$(grep -ac 'n_gpu_blocks' "$D/$1.log")
    case "$1" in
      static) [ "$pool" = "0" ] || { echo "  VOID: static arm built a paged pool ($pool) -- not a static arm" | tee -a "$OUT"; kill $PID; PID=""; return 1; };;
      paged)  [ "$pool" -gt 0 ] || { echo "  VOID: paged arm shows NO pool -- not a paged arm" | tee -a "$OUT"; kill $PID; PID=""; return 1; };;
    esac
    local t0 t1
    t0=$(python3 -c 'import time;print(time.time())')
    curl -s --max-time 14400 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' --data-binary "@$D/req.json" > "$D/$1.json"
    t1=$(python3 -c 'import time;print(time.time())')
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; sleep 3
    python3 - "$D/$1.json" "$NEEDLE" "$1" "$t0" "$t1" "$D" <<'PY' | tee -a "$OUT"
import json, sys, os
f, NEEDLE, lab, t0, t1, D = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
try: d = json.load(open(f))
except Exception as e: print(f"  {lab}: UNPARSEABLE {e}"); raise SystemExit
if "error" in d: print(f"  {lab}: ERROR {str(d['error'].get('message'))[:70]}"); raise SystemExit
c = d.get("content",""); tm = d.get("timings") or {}
ok = NEEDLE in c
# ⚠ NEEDLE IS A VALIDITY GUARD, NOT THE RESULT. A speed number from a wrong answer is not a number.
print(f"  {lab:7s} needle={'PASS' if ok else 'FAIL'}  wall={t1-t0:8.1f}s  "
      f"prompt_n={tm.get('prompt_n')}  pp={tm.get('prompt_per_second',0):7.1f} tok/s  "
      f"tg={tm.get('predicted_per_second',0):6.2f} tok/s")
json.dump({"lab":lab,"ok":ok,"wall":t1-t0,"pp":tm.get('prompt_per_second',0),
           "tg":tm.get('predicted_per_second',0),"n":tm.get('prompt_n')}, open(f"{D}/{lab}.res","w"))
PY
}

# ⚠⚠ ABBA IS THE ONLY DESIGN THAT ANSWERS THIS ON THIS BOX, AND TWO PASSES DEMONSTRABLY CANNOT.
# MEASURED 2026-08-09, 35B @ 256k, four walls sorted by POSITION rather than by arm:
#     ran FIRST :  static 815.9  ·  paged 788.9
#     ran SECOND:  paged  783.2  ·  static 781.9
# **The second arm wins in BOTH passes, whichever arm it is.** static-first said paged was 4.0% FASTER;
# paged-first said paged was 0.9% SLOWER. One-and-one -> the pair is VOID.
# ⚠ And averaging the two passes is NOT a valid correction: static's wall moved 4.2% between positions
# while paged's moved 0.7% -- the confound is ~6x larger on one arm, so the means are not comparable.
# ⇒ ABBA runs static,paged,paged,static in ONE invocation and compares POSITION-MATCHED pairs:
#   position 1 vs 4 (static twice) bounds the drift; 2 vs 3 (paged twice) bounds it again; and the
#   arm comparison is made WITHIN a position, not across one.
case "$ORDER" in
  abba)        arm static; mv "$D/static.res" "$D/static1.res" 2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged1.res"  2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged2.res"  2>/dev/null
               arm static; mv "$D/static.res" "$D/static2.res" 2>/dev/null ;;
  paged-first) arm paged; arm static ;;
  *)           arm static; arm paged ;;
esac

# ⚠ THE LEGACY TWO-ARM SUMMARISER MUST NOT RUN AFTER ABBA. On 2026-08-09 the ABBA summariser printed a
# correct verdict and then this block fired anyway, VOIDing on `static.res`/`paged.res` -- the files ABBA
# RENAMES AWAY to static1/paged1/paged2/static2. Output read:
#     ⇒ UNRESOLVED: ... indistinguishable at this sample size.      <- correct
#     VOID: one or both arms produced no result -- nothing to compare. <- wrong, and printed LAST
# Harmless to the numbers, and the LAST line is the one a reader quotes. It is the stale-.res hazard
# flagged before the run, arriving as a stale-FILENAME hazard instead.
if [ "$ORDER" = abba ]; then
python3 - "$D" <<'PY2' | tee -a "$OUT"
import json, sys
D = sys.argv[1]
try:
    s1=json.load(open(f"{D}/static1.res")); p1=json.load(open(f"{D}/paged1.res"))
    p2=json.load(open(f"{D}/paged2.res")); s2=json.load(open(f"{D}/static2.res"))
except Exception:
    print("  VOID: ABBA incomplete -- one or more arms produced no result."); raise SystemExit(2)
arms=[("static",s1),("paged",p1),("paged",p2),("static",s2)]
if not all(a[1]["ok"] for a in arms):
    print("  VOID: a needle failed. A speed number from a wrong answer is not a speed number.")
    raise SystemExit(2)
print()
print("  ABBA walls by position:")
for i,(lab,a) in enumerate(arms,1):
    print(f"    pos {i}  {lab:6s}  wall={a['wall']:8.1f}s  pp={a['pp']:7.1f}  tg={a['tg']:6.2f}")
# ⚠ DRIFT FIRST. If the two same-arm walls differ by more than the arm difference, the run cannot
# resolve the question and says so instead of reporting a ratio.
drift_s=abs(s1["wall"]-s2["wall"])/max(s1["wall"],s2["wall"])
drift_p=abs(p1["wall"]-p2["wall"])/max(p1["wall"],p2["wall"])
ms=(s1["wall"]+s2["wall"])/2; mp=(p1["wall"]+p2["wall"])/2
eff=abs(mp-ms)/ms
print()
print(f"  DRIFT  static {drift_s*100:.1f}%   paged {drift_p*100:.1f}%   (same arm, positions 1&4 / 2&3)")
print(f"  EFFECT paged/static on position-balanced means = {mp/ms:.4f}  ({eff*100:.1f}%)")
print()
if eff <= max(drift_s, drift_p):
    print(f"  ⇒ **UNRESOLVED**: the arm effect ({eff*100:.1f}%) is NOT larger than the drift")
    print(f"    ({max(drift_s,drift_p)*100:.1f}%). The arms are indistinguishable at this sample size.")
    print("    Report as a TIE with a bound, never as a winner. More repeats, not a bigger claim.")
else:
    who = "FASTER" if mp < ms else "SLOWER"
    print(f"  ⇒ paged is {who} by {eff*100:.1f}%, and that exceeds the measured drift")
    print(f"    ({max(drift_s,drift_p)*100:.1f}%). ABBA cancels first-order position effects.")
print()
print("  ⚠ n=2 per arm. This bounds the drift; it does not estimate variance. A tie here means")
print("  'not distinguishable at n=2', not 'proven equal'.")
PY2
fi

if [ "$ORDER" != abba ]; then
python3 - "$D" <<'PY' | tee -a "$OUT"
import json, os, sys
D = sys.argv[1]
try:
    s = json.load(open(f"{D}/static.res")); p = json.load(open(f"{D}/paged.res"))
except Exception:
    print("  VOID: one or both arms produced no result -- nothing to compare."); raise SystemExit(2)
if not (s["ok"] and p["ok"]):
    print(f"  VOID: needle failed (static={s['ok']} paged={p['ok']}). A speed number from a wrong")
    print("  answer is not a speed number, so this pair is not readable as parity."); raise SystemExit(2)
r = p["wall"] / s["wall"]
print()
print(f"  WALL RATIO paged/static = {r:.4f}   ->  paged is {'FASTER' if r < 1 else 'SLOWER'} by {abs(1-r)*100:.1f}%")
print(f"  PREFILL  static {s['pp']:.1f} vs paged {p['pp']:.1f} tok/s   ({p['pp']/max(s['pp'],1e-9):.3f}x)")
print(f"  DECODE   static {s['tg']:.2f} vs paged {p['tg']:.2f} tok/s   ({p['tg']/max(s['tg'],1e-9):.3f}x)")
print()
print("  ⚠ ONE ORDERING. Arm order is a measured confound on this box (29% positional drift once gave")
print("  1.41x in both directions when the truth was 1.317x). Re-run with PP_ORDER=paged-first;")
print("  AGREEMENT ACROSS BOTH ORDERS is the claim. A single order is a data point.")
PY
fi
echo "log: $OUT" | tee -a "$OUT"
