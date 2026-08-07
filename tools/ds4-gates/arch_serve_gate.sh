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
# StarCoder2, whose GGUF arch is `starcoder2` and whose model file is a DIFFERENT source file. The run
# looked clean. Step 1 below -- loader arch == expected -- is what catches that, and it is the only
# leg that retraction ever stood on.
#
# ⚠⚠ AND THE FIRST VERSION OF THIS GATE GOT THE *MECHANISM* WRONG, WHICH IS WORSE THAN GETTING THE
# VERDICT WRONG. It required the STATIC arm to log static-path warnings and VOIDed on zero, on the
# theory that zero meant "no paged consumer in this arch's file". That is false. There are TWO paged
# consumers in this fork:
#
#   build_attn_paged_or_null -> ggml_paged_attn_banded   20 archs, warns on the failure side
#   build_attn_inp_kv_auto   -> ggml_paged_attn          11 archs, warns NEVER, in either arm
#
# So the original rule would have FALSE-VOIDed all eleven auto-path architectures as unwired while
# they were paging perfectly well. Reading "paged is live" out of a warning count is inferring a
# positive from an absence, which is the root class in this lane's scar list.
#
# ★ THE FIX IS A POSITIVE MARKER AT BOTH CONSUMERS: DS4P-CONSUME <funnel> layer <il>, emitted per
# layer per graph, requires -lv 5 (DEBUG). It also fills a real hole in the fork, not just in this
# gate: DS4P-CHECKOUT proves the block pool served a request, never that a GRAPH consumed it, and
# "correct producer, no consumer" is audit finding 5 verbatim -- paged context built, no graph reading
# it, "Ornith now runs paged" retracted. Measured on the two funnels:
#
#            ernie4_5 (banded)   starcoder2 (auto)
#   paged        540                   900
#   static         0                     0        <- the marker can be zero, so >0 is a measurement
#
# Order of checks:
#   1  loader arch == expected                     else STRIKE  (a filename is not evidence)
#   2  STATIC arm DS4P-CONSUME == 0                else VOID    (marker must discriminate)
#   3  PAGED  arm DS4P-CONSUME  > 0                else VOID    (arch-independent presence proof)
#   4  banded funnel only: static-warns > 0 static, == 0 paged  else VOID / PARTIAL
#   5  output identical                            else FAIL
#
# ⚠ STEP 4 CANNOT RUN ON THE AUTO FUNNEL. That funnel has no per-layer fallback at all, so this gate
# is strictly WEAKER for those eleven archs: it can prove paged ran and produced identical text, but
# it cannot prove every layer was carried. It says so on PASS rather than printing a clean row.
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
    # ⚠ -lv 5, NOT 4. DS4P-CONSUME is LLAMA_LOG_DEBUG and common/log.cpp:85 drops DEBUG below
    # verbosity 5. At -lv 4 the marker count read ZERO on an arch that was demonstrably paging -- the
    # gate would have VOIDed a working arch because its own probe was filtered out. Caught only
    # because the marker was verified for PRESENCE on a known-good model before being trusted.
    env ${AG_ENV:-} "$SRV" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 \
        --port "$PORT" --no-warmup -lv 5 "${flags[@]}" "${EXTRA[@]}" > "$LOGDIR/$1.log" 2>&1 &
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
c_pool()  { grep -ac "DS4P-CHECKOUT"                                   "$1"; }
# the two consumer funnels, counted apart: which one an arch uses decides which further checks the
# gate is even ENTITLED to run.
c_band()  { grep -ac "DS4P-CONSUME banded"                             "$1"; }
c_auto()  { grep -ac "DS4P-CONSUME auto"                               "$1"; }

# ---------------- static arm ----------------
start static || exit 1
S_OUT=$(ask)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2

# ⚠ THE ARCH IS READ FROM THE LOADER, NOT THE FILENAME. A repo named "...starcoder..." can hold a
# starcoder2 model, and the model file that gets compiled is chosen by THIS string.
GOT_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/static.log" | sed 's/.*= *//' | tr -d ' \r')
S_NOPG=$(c_nopg "$LOGDIR/static.log")
S_CONS=$(( $(c_band "$LOGDIR/static.log") + $(c_auto "$LOGDIR/static.log") ))

printf '  STATIC  arch=%-14s out=[%s]\n' "${GOT_ARCH:-<none>}" "$S_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s DS4P-CONSUME=%s\n' \
       "$S_NOPG" "$(c_cap "$LOGDIR/static.log")" "$(c_hdim "$LOGDIR/static.log")" "$S_CONS" | tee -a "$OUT"

if [ "$GOT_ARCH" != "$ARCH" ]; then
    echo | tee -a "$OUT"
    echo "STRIKE: loader reports arch='$GOT_ARCH', gate was told to expect '$ARCH'." | tee -a "$OUT"
    echo "  This is the wrong vehicle. The model file exercised is the one selected by the loader's" | tee -a "$OUT"
    echo "  arch string, so whatever this run proves, it does not prove anything about '$ARCH'." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ⚠ NEGATIVE CONTROL ON THE PROBE ITSELF, BEFORE ANY POSITIVE RESULT IS READ. If DS4P-CONSUME is
# non-zero WITHOUT --kv-paged, the marker does not discriminate and its count in the paged arm proves
# nothing. Measured at 0 on both funnels; this asserts it rather than trusting the measurement.
if [ "${S_CONS:-0}" -ne 0 ]; then
    echo | tee -a "$OUT"
    echo "VOID: DS4P-CONSUME fired $S_CONS times in the STATIC arm, where no paged context exists." | tee -a "$OUT"
    echo "  The presence marker is not discriminating, so its count under paging cannot be read as" | tee -a "$OUT"
    echo "  evidence of anything. Fix the marker before reading this gate." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ---------------- paged arm ----------------
start paged || exit 1
P_OUT=$(ask)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""

P_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/paged.log" | sed 's/.*= *//' | tr -d ' \r')
P_NOPG=$(c_nopg "$LOGDIR/paged.log"); P_CAP=$(c_cap "$LOGDIR/paged.log")
P_HDIM=$(c_hdim "$LOGDIR/paged.log"); P_POOL=$(c_pool "$LOGDIR/paged.log")
P_BAND=$(c_band "$LOGDIR/paged.log"); P_AUTO=$(c_auto "$LOGDIR/paged.log")
P_CONS=$(( P_BAND + P_AUTO ))

printf '  PAGED   arch=%-14s out=[%s]\n' "${P_ARCH:-<none>}" "$P_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s pool=%s\n' \
       "$P_NOPG" "$P_CAP" "$P_HDIM" "$P_POOL" | tee -a "$OUT"
printf '          DS4P-CONSUME banded=%-6s auto=%s\n' "$P_BAND" "$P_AUTO" | tee -a "$OUT"
echo "-----" | tee -a "$OUT"

rc=0
# ⚠ POOL ACTIVITY IS NOT CONSUMPTION. DS4P-CHECKOUT says the block manager handed out blocks; it is
# emitted by the scheduler and is true even if no graph reads them. That gap IS audit finding 5.
if [ "${P_CONS:-0}" -eq 0 ]; then
    echo "VOID: paged arm shows ZERO DS4P-CONSUME events -- no graph consumed the paged context." | tee -a "$OUT"
    echo "  pool checkouts = $P_POOL, so the blocks were allocated and then read by nobody. Identical" | tee -a "$OUT"
    echo "  output here proves only that two effectively-static runs agree. This is audit finding 5" | tee -a "$OUT"
    echo "  reproducing, not a passing arch." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# which funnel -- decides which further checks this gate is entitled to run
FUNNEL=mixed
[ "$P_BAND" -gt 0 ] && [ "$P_AUTO" -eq 0 ] && FUNNEL=banded
[ "$P_AUTO" -gt 0 ] && [ "$P_BAND" -eq 0 ] && FUNNEL=auto
echo "  consumer funnel: $FUNNEL" | tee -a "$OUT"

if [ "$S_OUT" != "$P_OUT" ]; then
    echo "*** FAIL: OUTPUT DIVERGED. static and paged disagree on the same deterministic prompt. ***" | tee -a "$OUT"
    echo "    static [$S_OUT]" | tee -a "$OUT"
    echo "    paged  [$P_OUT]" | tee -a "$OUT"
    rc=1
fi

if [ "$FUNNEL" = banded ] || [ "$FUNNEL" = mixed ]; then
    # ⚠ ONLY MEANINGFUL ON THIS FUNNEL, and only after the static arm proved the counter can move.
    if [ "${S_NOPG:-0}" -eq 0 ]; then
        echo "VOID: banded funnel, but the STATIC arm logged zero static-path warnings, so a zero in" | tee -a "$OUT"
        echo "  the paged arm is not a measurement -- the counter was never shown able to move." | tee -a "$OUT"
        echo "log: $OUT"; exit 2
    fi
    FB=$(( P_NOPG + P_CAP + P_HDIM ))
    if [ "$FB" -ne 0 ]; then
        echo "*** PARTIAL: $FB layer-fallbacks under paging (nopg=$P_NOPG cap=$P_CAP headdim=$P_HDIM)." | tee -a "$OUT"
        echo "    The fallback path is correct BY DESIGN, so output may still match -- but this arch is" | tee -a "$OUT"
        echo "    not fully carried by the paged path and must NOT be recorded as verified." | tee -a "$OUT"
        rc=1
    fi
fi

if [ "$rc" -eq 0 ]; then
    echo "PASS: arch=$ARCH  output identical  DS4P-CONSUME=$P_CONS ($FUNNEL funnel)" | tee -a "$OUT"
    if [ "$FUNNEL" = banded ]; then
        echo "  fallbacks ${S_NOPG} -> 0, every layer carried by the paged path." | tee -a "$OUT"
    else
        echo "  ⚠ WEAKER RESULT THAN THE BANDED FUNNEL. The auto path has no per-layer fallback and" | tee -a "$OUT"
        echo "  emits no static-path warning, so this gate CANNOT prove every layer was carried. It" | tee -a "$OUT"
        echo "  proves paged ran ($P_CONS consume events) and the text matches. Record it that way." | tee -a "$OUT"
    fi
    echo "  ⚠ SCOPE: short prompt at -c $CTX. Says the arch is WIRED and CORRECT at small context." | tee -a "$OUT"
    echo "  Says nothing about long context, nothing about speed, and the ~50k intermittent paged" | tee -a "$OUT"
    echo "  defect is open underneath every row of this table." | tee -a "$OUT"
fi
echo "log: $OUT"
exit $rc
