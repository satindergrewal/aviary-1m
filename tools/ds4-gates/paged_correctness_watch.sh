#!/usr/bin/env bash
# PAGED CORRECTNESS WATCH -- a check that can FAIL when the paged path answers wrongly.
#
# WHY THIS EXISTS. On 2026-08-07 the default paged path produced wrong output dozens of times near
# 50k tokens, with static correct every time. It was noticed BY EYE, hours in, because nothing in the
# paged path can detect a wrong answer: a failing server's log is byte-for-byte the same shape as a
# clean one -- same warnings, same layer fallbacks, same scheduler state, then different tokens. Every
# marker, warning and assert passed while the output was wrong.
#
# So this gate does not try to find that bug. It makes the next occurrence get CAUGHT instead of
# stumbled over. Run it before any paged claim and after any paged change.
#
# ⚠ IT REPORTS A RATE, NOT A VERDICT, AND THAT IS THE POINT. The failure is intermittent: the exact
# config that failed repeatedly in one window was clean 6/6 in another. A single pass proves nothing.
# An n=1 result is precisely what invalidated an entire evening of bisection, so this takes reps per
# length and prints failures/total.
#
# ⚠ BUILT AGAINST THE SEVEN RULES THAT SESSION PAID FOR:
#   1. the factor is confirmed to vary -- the paged arm asserts its own marker, and a run with no
#      paged dispatch VOIDS rather than passes
#   2. lengths are ASSERTED from the token array, never labelled (a labelled length silently ran
#      50,473 while printing "50,480")
#   3. the token cache is keyed on the length it holds (a fixed path reused a 32k file for a 64k run)
#   4. one server at a time behind a lock, killed by PID (pattern-kill destroyed sibling servers)
#   5. the reference is STATIC -- never the paged path compared against itself (r1 == r2 scored two
#      requests agreeing on garbage as CLEAN)
#   6. a control length known good, so a run that cannot detect anything says INCONCLUSIVE
#   7. reps, because n=1 on an intermittent failure is not evidence
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${PW_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
LENS=${PW_LENS:-"8192 32768 49998"}     # last one is where the 2026-08-07 failures clustered
REPS=${PW_REPS:-3}                       # per length, per arm
CTX=${PW_CTX:-73728}
BS=${PW_BS:-16}
NPRED=${PW_NPRED:-12}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/pagedwatch
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/paged-watch-$(date +%Y%m%d-%H%M).txt
LOCK=$LOGDIR/.lock
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ANOTHER RUN HOLDS THE LOCK ($LOCK) -- refusing. Concurrent probes killed each other's"
    echo "servers on 2026-08-07 and voided two runs; one lurked for hours."
    exit 1
fi
SRVPID=""
cleanup() { [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; scrub_abs_paths "${OUT:-}" 2>/dev/null; }
trap cleanup EXIT

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
echo "lengths: $LENS   reps: $REPS   -c $CTX   --kv-block-size $BS" | tee -a "$OUT"

python3 -c "
import random
random.seed(41)
s=['Paris','Berlin','Rome','Madrid','Lisbon','Vienna','Oslo','Prague','Dublin','Warsaw']
v=['is the capital of','lies north of','was founded before','trades heavily with','borders']
o=['France','Germany','Italy','Spain','Portugal','Austria','Norway','Czechia','Ireland','Poland']
out=[]
need = max($(echo $LENS | tr ' ' ',')) * 9
while len(' '.join(out)) < need:
    out.append('%s %s %s.' % (random.choice(s),random.choice(v),random.choice(o)))
open('$LOGDIR/f.txt','w').write(' '.join(out))"

pick_port() {
    local p
    for p in $(seq "${PW_PORT:-9931}" $(( ${PW_PORT:-9931} + 60 ))); do
        if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}

# ⚠⚠ STAMP THE BINARY AT EVERY EXEC, NOT THE TIP AT LAUNCH.
#
# This gate printed `git rev-parse HEAD` once, at startup, and called that "the tip under test". It
# is not. The static arm and the paged arm exec the binary MINUTES APART, and on 2026-08-07 a run
# straddled THREE rebuilds of llama-server -- one of which had a fix deliberately disabled for a
# mutation test. Whichever binary existed at each exec is what that arm actually ran, and the header
# said 51d87602d throughout.
#
# It came back 5/6 clean, which is what makes it dangerous: a FAILING confounded run gets
# investigated, a PASSING one gets filed as evidence. So the binary's own hash is recorded per arm,
# and a mismatch between arms VOIDS the comparison instead of being averaged into it.
bin_stamp() { shasum -a 1 "$SRV" 2>/dev/null | cut -c1-12; }

start() { # $1 mode(static|paged) -> sets PORT, SRVPID; returns 1 and prints the child's log on failure
    PORT=$(pick_port) || { echo "  no free port" | tee -a "$OUT"; return 1; }
    local stamp; stamp=$(bin_stamp)
    echo "  [$1 arm binary sha1=$stamp]" | tee -a "$OUT"
    if [ -z "${BIN_STAMP_FIRST:-}" ]; then
        BIN_STAMP_FIRST="$stamp"
    elif [ "$stamp" != "$BIN_STAMP_FIRST" ]; then
        echo "VOID: the binary CHANGED between arms ($BIN_STAMP_FIRST -> $stamp)." | tee -a "$OUT"
        echo "  The two arms did not test the same code. Arms must differ in ONE thing; these differ" | tee -a "$OUT"
        echo "  in an unknown number. Rebuild finished, then re-run with nothing else touching it." | tee -a "$OUT"
        return 1
    fi
    local flags=()
    [ "$1" = paged ] && flags=(--kv-paged --kv-block-size "$BS" -ngpub $(( (CTX + BS - 1)/BS )) -ncpub 512)
    env DS4P_PAGED_HYBRID=1 "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 2048 -ub 2048 \
        --port "$PORT" --no-warmup -lv 4 -cram 0 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    SRVPID=$!
    for _ in $(seq 1 900); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    echo "  $1 server did not start; its last lines:" | tee -a "$OUT"
    tail -8 "$LOGDIR/$1.log" | sed 's/^/  | /' | tee -a "$OUT"
    grep -aiE "assert|abort|couldn.t bind|out of memory" "$LOGDIR/$1.log" | tail -3 | sed 's/^/  ! /' | tee -a "$OUT"
    return 1
}

tokenise() { # cache keyed on the length it holds
    local n=$1
    [ -s "$LOGDIR/toks-$n.json" ] && return 0
    python3 -c "
import json;print(json.dumps({'content': open('$LOGDIR/f.txt').read()}))" > "$LOGDIR/t.req"
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/tokenize" -H 'Content-Type: application/json' \
        -d @"$LOGDIR/t.req" > "$LOGDIR/t.json"
    python3 -c "
import json, sys
t = json.load(open('$LOGDIR/t.json'))['tokens']
if len(t) < $n:
    sys.exit('filler tokenises to %d, need %d -- fix the gate, not the kernel' % (len(t), $n))
p = t[:$n]
assert len(p) == $n, 'truncation did not produce the requested length'
json.dump(p, open('$LOGDIR/toks-$n.json','w'))" || return 1
}

ask() { # $1 length  $2 outfile
    python3 -c "
import json
p = json.load(open('$LOGDIR/toks-$1.json'))
assert len(p) == $1
print(json.dumps({'prompt': p, 'n_predict': $NPRED, 'temperature': 0, 'seed': 1, 'cache_prompt': False}))" > "$LOGDIR/req.json"
    curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d @"$LOGDIR/req.json" > "$2"
}

# ---------- static references ----------
start static || exit 1
declare -A REF
for n in $LENS; do
    tokenise "$n" || exit 2
    ask "$n" "$LOGDIR/static-$n.json"
    REF[$n]=$(python3 -c "
import json
try: print(json.load(open('$LOGDIR/static-$n.json')).get('content','')[:64].replace(chr(10),' '))
except Exception: print('MALFORMED')")
    printf "  static  N=%-7s ref=[%s]\n" "$n" "${REF[$n]}" | tee -a "$OUT"
done
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2

# ---------- paged arm, REPS per length ----------
declare -A FAIL
TOTAL=0; FAILS=0
start paged || exit 1
for n in $LENS; do
    FAIL[$n]=0
    for r in $(seq 1 "$REPS"); do
        ask "$n" "$LOGDIR/paged-$n-$r.json"
        got=$(python3 -c "
import json
try: print(json.load(open('$LOGDIR/paged-$n-$r.json')).get('content','')[:64].replace(chr(10),' '))
except Exception: print('MALFORMED')")
        TOTAL=$((TOTAL+1))
        if [ "$got" != "${REF[$n]}" ]; then
            FAIL[$n]=$(( ${FAIL[$n]} + 1 )); FAILS=$((FAILS+1))
            printf "  paged   N=%-7s r%s  MISMATCH out=[%s]\n" "$n" "$r" "$got" | tee -a "$OUT"
        else
            printf "  paged   N=%-7s r%s  ok\n" "$n" "$r" | tee -a "$OUT"
        fi
    done
done
MARK=$(grep -ac "paged: request registered\|DS4P-CHECKOUT" "$LOGDIR/paged.log")
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""

echo "-----" | tee -a "$OUT"
# ⚠ PRESENCE: an arm that never paged would agree with static trivially and pass a gate that only
# compares text. That exact trap cost an hour on Qwen3.6 in this lane.
echo "paged activity markers: $MARK" | tee -a "$OUT"
if [ "${MARK:-0}" -eq 0 ]; then
    echo "VOID: the paged arm shows no paged activity -- it was not exercising the path under test." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
CTRL=${LENS%% *}
if [ "${FAIL[$CTRL]:-0}" -ne 0 ]; then
    echo "INCONCLUSIVE: the control length $CTRL failed too. Something broader than the ~50k defect" | tee -a "$OUT"
    echo "  is wrong -- investigate that before reading the longer lengths." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
for n in $LENS; do
    printf "  N=%-7s failures %s/%s\n" "$n" "${FAIL[$n]}" "$REPS" | tee -a "$OUT"
done
echo | tee -a "$OUT"
if [ "$FAILS" -eq 0 ]; then
    echo "PAGED CORRECTNESS WATCH: no mismatches in $TOTAL comparisons." | tee -a "$OUT"
    echo "  ⚠ NOT a certificate. The 2026-08-07 defect was clean 6/6 in one window and failed" | tee -a "$OUT"
    echo "  repeatedly in another at the same config. This bounds the rate; it does not clear the path." | tee -a "$OUT"
    echo "log: $OUT"; exit 0
fi
echo "PAGED CORRECTNESS WATCH: FAIL -- $FAILS/$TOTAL comparisons differ from static." | tee -a "$OUT"
echo "  The paged path is answering wrongly. Static is the reference and was correct at every length." | tee -a "$OUT"
echo "  CAPTURE $LOGDIR BEFORE ANYTHING ELSE: the 2026-08-07 hunt lost every failing window because" | tee -a "$OUT"
echo "  only the output text was kept, and the failure leaves no trace in the server log." | tee -a "$OUT"
echo "log: $OUT"; exit 1
