#!/usr/bin/env bash
# ARCH SERVE GATE -- does architecture <A> actually run through the PAGED attention path, and does it
# produce the same tokens as static while doing so?
#
# WHY THIS EXISTS AS A FILE. The first four archs (ernie4-5, qwen3vl, nemotron, qwen3moe) were verified
# with ad-hoc shell typed inline. The results were right, but an inline harness cannot be re-run, cannot
# be diffed when it changes, and cannot be audited by anyone including me. With 15 archs left that is
# not a style complaint, it is 15 chances to run a subtly different check and not notice.
#
# ⚠ THE RETRACTION THIS GATE IS BUILT AROUND. On 2026-08-06 I "verified" starcoder by downloading
# StarCoder2, whose GGUF arch is `starcoder2` and whose model file is a DIFFERENT source file with no
# paged consumer in it. The run looked clean. The tell was that the STATIC arm reported ZERO
# static-path warnings -- impossible for an arch actually under test, because the static arm is where
# every layer must announce it took the static path. A zero there does not mean "no fallbacks", it
# means "nothing is wired and this gate is measuring nothing".
#
# So the negative control is load-bearing and comes FIRST:
#
#   static arm  static-path warnings MUST be > 0    <- proves the consumer exists in this arch's file
#   paged arm   static-path warnings MUST be 0      <- only then is the zero a MEASUREMENT
#   paged arm   DS4P-CHECKOUT markers MUST be > 0   <- proves the paged pool was actually exercised
#   both arms   identical output text               <- proves it is not just running, but running RIGHT
#
# Miss the first line and the last three are theatre. A run with no paged dispatch agrees with static
# trivially, and an arch with no consumer produces a perfect scorecard while testing nothing.
#
# usage:  arch_serve_gate.sh <expected-arch> <model.gguf> [extra-server-args...]
set -uo pipefail

. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

ARCH="${1:-}"
MODEL="${2:-}"
if [ -z "$ARCH" ] || [ -z "$MODEL" ]; then
    echo "usage: $0 <expected-arch> <model.gguf> [extra-server-args...]" >&2
    echo "  refusing to run without an EXPECTED arch: a gate that accepts whatever it finds cannot" >&2
    echo "  catch a wrong-vehicle download, which is the exact failure this gate was written for." >&2
    exit 2
fi
shift 2
EXTRA=("$@")

[ -f "$MODEL" ] || { echo "missing model: $(basename "$MODEL")" >&2; exit 2; }

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
CTX=${AG_CTX:-4096}
NPRED=${AG_NPRED:-8}
PROMPT=${AG_PROMPT:-"The capital of France is"}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/archgate
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/arch-$ARCH-$(date +%Y%m%d-%H%M).txt}
LOCK=$LOGDIR/.lock
mkdir -p "$LOGDIR" "$(dirname "$OUT")"

# ⚠ ONE SERVER AT A TIME, AND KILL BY PID. Detached probes sharing `pkill -f llama-server` destroyed
# each other's servers on 2026-08-07; one lurked for hours and is the leading suspect for a load
# correlation that then failed to reproduce. A pattern kill is not addressed to anyone in particular.
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ANOTHER RUN HOLDS $LOCK -- refusing rather than racing it." >&2
    exit 2
fi
SRVPID=""
cleanup() {
    [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null
    scrub_abs_paths "${OUT:-}" 2>/dev/null
}
trap cleanup EXIT

echo "arch-serve gate: expected arch=$ARCH  model=$(basename "$MODEL")" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain | wc -l | tr -d ' ')  -c $CTX" | tee -a "$OUT"

pick_port() {
    local p
    for p in $(seq "${AG_PORT:-9051}" $(( ${AG_PORT:-9051} + 60 ))); do
        lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }
    done
    return 1
}

start() { # $1 = static|paged   -> PORT, SRVPID
    PORT=$(pick_port) || { echo "  no free port" | tee -a "$OUT"; return 1; }
    local flags=()
    [ "$1" = paged ] && flags=(--kv-paged)
    # shellcheck disable=SC2086
    env ${AG_ENV:-} "$SRV" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 \
        --port "$PORT" --no-warmup -lv 4 "${flags[@]}" "${EXTRA[@]}" > "$LOGDIR/$1.log" 2>&1 &
    SRVPID=$!
    local i
    for i in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && return 0
        kill -0 "$SRVPID" 2>/dev/null || break
        sleep 1
    done
    echo "  $1 arm DID NOT SERVE; last lines:" | tee -a "$OUT"
    tail -6 "$LOGDIR/$1.log" | sed 's/^/  | /' | tee -a "$OUT"
    return 1
}

ask() {
    python3 -c "
import json
print(json.dumps({'prompt': '''$PROMPT''', 'n_predict': $NPRED, 'temperature': 0,
                  'seed': 1, 'cache_prompt': False}))" > "$LOGDIR/req.json"
    curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d @"$LOGDIR/req.json" \
      | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('content','').replace(chr(10),' '))
except Exception: print('MALFORMED')"
}

# counts, kept SEPARATE. Lumping them hides which failure mode fired: 'no paged context at all' and
# 'paged pool active but this layer is not eligible' need completely different fixes.
c_nopg()  { grep -ac "took the STATIC path -- no paged context"        "$1"; }
c_cap()   { grep -ac "fails the paged capability contract"             "$1"; }
c_hdim()  { grep -ac "does not match the paged pool's"                 "$1"; }
c_mark()  { grep -ac "DS4P-CHECKOUT"                                   "$1"; }

# ---------------- static arm ----------------
start static || exit 1
S_OUT=$(ask)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2

# ⚠ THE ARCH IS READ FROM THE LOADER, NOT THE FILENAME. A repo named "...starcoder..." can hold a
# starcoder2 model, and the model file that gets compiled is chosen by THIS string.
GOT_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/static.log" | sed 's/.*= *//' | tr -d ' \r')
S_NOPG=$(c_nopg "$LOGDIR/static.log")

printf '  STATIC  arch=%-14s out=[%s]\n' "${GOT_ARCH:-<none>}" "$S_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%s\n' \
       "$S_NOPG" "$(c_cap "$LOGDIR/static.log")" "$(c_hdim "$LOGDIR/static.log")" | tee -a "$OUT"

if [ "$GOT_ARCH" != "$ARCH" ]; then
    echo | tee -a "$OUT"
    echo "STRIKE: loader reports arch='$GOT_ARCH', gate was told to expect '$ARCH'." | tee -a "$OUT"
    echo "  This is the wrong vehicle. The model file exercised is the one selected by the loader's" | tee -a "$OUT"
    echo "  arch string, so whatever this run proves, it does not prove anything about '$ARCH'." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ⚠ NEGATIVE CONTROL, BEFORE ANY POSITIVE RESULT IS READ.
if [ "${S_NOPG:-0}" -eq 0 ]; then
    echo | tee -a "$OUT"
    echo "VOID: the STATIC arm logged ZERO static-path warnings." | tee -a "$OUT"
    echo "  Every layer of an arch that HAS the paged consumer must announce the static path when no" | tee -a "$OUT"
    echo "  paged context exists. Zero here means the consumer is absent from this arch's model file," | tee -a "$OUT"
    echo "  so the paged arm's zero would be trivially true and this gate would measure nothing." | tee -a "$OUT"
    echo "  This is exactly how the starcoder2 mis-verification passed on 2026-08-06." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ---------------- paged arm ----------------
start paged || exit 1
P_OUT=$(ask)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""

P_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/paged.log" | sed 's/.*= *//' | tr -d ' \r')
P_NOPG=$(c_nopg "$LOGDIR/paged.log"); P_CAP=$(c_cap "$LOGDIR/paged.log")
P_HDIM=$(c_hdim "$LOGDIR/paged.log"); P_MARK=$(c_mark "$LOGDIR/paged.log")

printf '  PAGED   arch=%-14s out=[%s]\n' "${P_ARCH:-<none>}" "$P_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s markers=%s\n' \
       "$P_NOPG" "$P_CAP" "$P_HDIM" "$P_MARK" | tee -a "$OUT"
echo "-----" | tee -a "$OUT"

rc=0
if [ "${P_MARK:-0}" -eq 0 ]; then
    echo "VOID: paged arm shows no DS4P-CHECKOUT activity -- the paged pool was never exercised," | tee -a "$OUT"
    echo "  so identical output would prove only that two static runs agree." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
if [ "$S_OUT" != "$P_OUT" ]; then
    echo "*** FAIL: OUTPUT DIVERGED. static and paged disagree on the same deterministic prompt. ***" | tee -a "$OUT"
    echo "    static [$S_OUT]" | tee -a "$OUT"
    echo "    paged  [$P_OUT]" | tee -a "$OUT"
    rc=1
fi
FB=$(( P_NOPG + P_CAP + P_HDIM ))
if [ "$FB" -ne 0 ]; then
    echo "*** PARTIAL: $FB layer-fallbacks under paging (nopg=$P_NOPG cap=$P_CAP headdim=$P_HDIM)." | tee -a "$OUT"
    echo "    Output may still be correct -- the fallback path is correct BY DESIGN -- but this arch" | tee -a "$OUT"
    echo "    is not fully carried by the paged path and must NOT be recorded as verified." | tee -a "$OUT"
    rc=1
fi
if [ "$rc" -eq 0 ]; then
    echo "PASS: arch=$ARCH  output identical  markers=$P_MARK  fallbacks ${S_NOPG} -> 0" | tee -a "$OUT"
    echo "  ⚠ SCOPE: short prompt at -c $CTX. This says the arch is WIRED and CORRECT at small" | tee -a "$OUT"
    echo "  context. It says nothing about long context, nothing about speed, and the ~50k" | tee -a "$OUT"
    echo "  intermittent paged defect is open underneath every row of this table." | tee -a "$OUT"
fi
echo "log: $OUT"
exit $rc
