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
    # ⚠ A DESIGNED REFUSAL IS NOT A FAILURE, AND THE TWO MUST NOT SHARE A ROW. The hybrid and SWA
    # guards refuse --kv-paged deliberately and say so; that is the guard WORKING. Scored as
    # "did not serve" it becomes a false FAIL, and a results table that cannot separate "refused by
    # design" from "broken" is worse than no table -- the first arch it mislabels is gemma4, which has
    # SWA layers, and qwen3next/hunyuan are hybrids. Its sibling paged_multimodel_gate.sh has had this
    # branch since it was written; this gate shipped without it.
    # ⚠ A DRAFT HEAD IS NOT A MODEL. THREE of the 19 are drafters, from two separate guards:
    #   llama-context.cpp:182   Gemma4Assistant
    #   llama-context.cpp:191   EAGLE3, DFLASH  (when the GGUF carries no tok_embd/output)
    # They are speculative-decode heads and need the TARGET model's context (ctx_other, i.e. -md).
    # Reported as "did not serve" that reads like a paging defect. It is a category error in the
    # vehicle, and this gate's whole design -- compare two arms of ONE model -- does not apply to them
    # without a target attached.
    if grep -qE "requires ctx_other to be set" "$LOGDIR/$1.log" 2>/dev/null; then
        echo "  $1 arm: THIS IS A DRAFT HEAD, not a standalone model." | tee -a "$OUT"
        grep -aE "requires ctx_other to be set" "$LOGDIR/$1.log" | head -1 | cut -c1-150 | sed 's/^/  | /' | tee -a "$OUT"
        echo "  gemma4-assistant / eagle3 / dflash need the target model's context (-md). This gate" | tee -a "$OUT"
        echo "  compares two arms of ONE model and cannot answer for them in that form. Needs its own" | tee -a "$OUT"
        echo "  harness: target + draft, then the separate question of whether the DRAFT's KV is" | tee -a "$OUT"
        echo "  paged at all. One harness unblocks three of the nineteen." | tee -a "$OUT"
        return 3
    fi
    if grep -qE "not yet supported for hybrid architectures|needs DS4P_PAGED_SWA=1|requires DS4P_PAGED_HYBRID=1" \
            "$LOGDIR/$1.log" 2>/dev/null; then
        echo "  $1 arm REFUSED BY DESIGN -- this arch class is guarded:" | tee -a "$OUT"
        # ⚠ QUOTE THE LINE THAT MATCHED, not the first line mentioning kv_paged. The first version
        # grepped "kv_paged" and printed "kv_paged layer->backend map: 48 layers..." -- an unrelated
        # informational line -- as if it were the refusal. Evidence that does not come from the thing
        # being reported is not evidence, however plausible it reads.
        grep -aE "not yet supported for hybrid architectures|needs DS4P_PAGED_SWA=1|requires DS4P_PAGED_HYBRID=1" \
            "$LOGDIR/$1.log" | head -1 | cut -c1-150 | sed 's/^/  | /' | tee -a "$OUT"
        echo "  Set AG_ENV to the enabling flag and re-run, e.g." | tee -a "$OUT"
        echo "    AG_ENV=DS4P_PAGED_HYBRID=1  $0 $ARCH <model>     # hybrid archs" | tee -a "$OUT"
        echo "    AG_ENV=DS4P_PAGED_SWA=1     $0 $ARCH <model>     # SWA archs" | tee -a "$OUT"
        echo "  Until then this gate CANNOT answer for this arch. Not a pass, not a failure." | tee -a "$OUT"
        return 3
    fi
    echo "  $1 arm DID NOT SERVE" | tee -a "$OUT"
    # ⚠ SURFACE THE CAUSE, NOT THE TAIL. On an abort the last six lines are BACKTRACE FRAMES -- dyld
    # offsets and mangled symbols -- while the one line that names the fault scrolled past above it.
    # Measured: a deliberately reintroduced contiguity bug printed
    #   "GGML_ASSERT(ggml_is_contiguous(q) && "paged attn: q must be contiguous ...") failed"
    # and the gate showed six frames of `_ZN18common_init_result...` instead. A diagnostic buried
    # under its own stack trace is a diagnostic nobody reads.
    local why
    why=$(grep -aE "GGML_ASSERT|GGML_ABORT|error:|failed to |not supported|out of memory" \
          "$LOGDIR/$1.log" | grep -av "^ *[0-9]* " | tail -3)
    if [ -n "$why" ]; then
        echo "  cause:" | tee -a "$OUT"
        printf '%s\n' "$why" | cut -c1-190 | sed 's/^/  ! /' | tee -a "$OUT"
    else
        echo "  no assert/abort/error line found; last lines:" | tee -a "$OUT"
        tail -6 "$LOGDIR/$1.log" | sed 's/^/  | /' | tee -a "$OUT"
    fi
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
# ⚠⚠ COUNT ONLY WHAT HAPPENS AFTER A REQUEST EXISTS. llama-server builds the graph many times at
# startup -- memory fitting and reserve passes -- before any token is real. On gemma4 those startup
# passes take the STATIC path because the paged scheduler has no batch info yet, and the first
# version of this gate counted them as layer-fallbacks:
#
#   21 graph builds STATIC   0.311s -> 1.271s     ALL before the first request
#    8 graph builds PAGED    from 2.088s          the request arrives at 2.088s
#
# It scored gemma4 PARTIAL with "630 fallbacks" when 100% of inference-time builds were paged. The
# warnings were real, the count was real, and the verdict was false, because the work being counted
# had not happened yet. Same class as counting a log line that fires BEFORE the work it announces.
#
# So every counter slices the log at the first request. The slice marker must be present, and
# post_slice VOIDs rather than silently counting the whole file if it is missing.
post_slice() { # $1 log -> stdout: the portion after the first real request
    awk '/launch_slot_|start_loop: processing task, id = 0/{seen=1} seen' "$1"
}
has_slice()  { grep -qa "launch_slot_\|start_loop: processing task, id = 0" "$1"; }

c_nopg()  { post_slice "$1" | grep -ac "took the STATIC path -- no paged context"; }
c_cap()   { post_slice "$1" | grep -ac "fails the paged capability contract"; }
c_hdim()  { post_slice "$1" | grep -ac "does not match the paged pool's"; }
c_pool()  { post_slice "$1" | grep -ac "DS4P-CHECKOUT"; }
# the two consumer funnels, counted apart: which one an arch uses decides which further checks the
# gate is even ENTITLED to run.
c_band()  { post_slice "$1" | grep -ac "DS4P-CONSUME banded"; }
c_auto()  { post_slice "$1" | grep -ac "DS4P-CONSUME auto"; }
# startup-only counts, reported separately so the contamination stays VISIBLE rather than deleted
c_nopg_pre() { local a b; a=$(grep -ac "took the STATIC path -- no paged context" "$1"); b=$(c_nopg "$1"); echo $((a-b)); }

# ---------------- static arm ----------------
# ⚠ exit 3 is SKIPPED-BY-DESIGN, distinct from exit 1 (broken) and exit 2 (VOID/STRIKE). A caller
# that collapses them scores a working guard as a defect, or worse, buries a real failure in a
# "skipped" bucket. The three are separate exit codes so a batch runner cannot conflate them.
start static; rc_s=$?
[ "$rc_s" -eq 3 ] && { echo "log: $OUT"; exit 3; }
[ "$rc_s" -ne 0 ] && exit 1
S_OUT=$(ask)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2

# ⚠ THE ARCH IS READ FROM THE LOADER, NOT THE FILENAME. A repo named "...starcoder..." can hold a
# starcoder2 model, and the model file that gets compiled is chosen by THIS string.
GOT_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/static.log" | sed 's/.*= *//' | tr -d ' \r')
# ⚠ THE SLICE MARKER MUST EXIST IN *THIS* ARM TOO. The first version checked it only on the paged
# log. Without it here, S_NOPG and S_CONS silently read 0 and the banded branch VOIDs saying "the
# counter was never shown able to move" -- which fails closed but DIAGNOSES WRONG, sending the reader
# after a wiring fault when the real problem is a missing marker in the harness.
if ! has_slice "$LOGDIR/static.log"; then
    echo "VOID: no request marker in the STATIC log -- its counters cannot exclude startup reserve" | tee -a "$OUT"
    echo "  passes, so neither the negative control nor anything derived from it can be read." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
S_NOPG=$(c_nopg "$LOGDIR/static.log")
S_CONS=$(( $(c_band "$LOGDIR/static.log") + $(c_auto "$LOGDIR/static.log") ))

printf '  STATIC  arch=%-14s out=[%s]\n' "${GOT_ARCH:-<none>}" "$S_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s DS4P-CONSUME=%s\n' \
       "$S_NOPG" "$(c_cap "$LOGDIR/static.log")" "$(c_hdim "$LOGDIR/static.log")" "$S_CONS" | tee -a "$OUT"

# ⚠⚠ IS THE REFERENCE ITSELF SANE? This gate compares paged against static and calls a match a PASS.
# It never asked whether static was worth comparing to -- and on gemma4-12B the static arm returned
# "01111133" for the prompt "The capital of France is". Had the paged arm returned the same garbage,
# the gate would have printed PASS on two broken runs agreeing.
#
# That is the degenerate-baseline trap for the third time in this lane: an evening spent scoring a
# kernel against a reference that was itself looping, then the Q4-over-Q2 quant decision made
# precisely to avoid it, and now a gate that encoded the lesson in its COMMENTS and not in its CODE.
# A claim-vs-measure check catches lies; it does not catch absurdity. Absurdity needs its own test.
#
# The prompt is fixed and known, so the bar is specific rather than general: an answer to
# "The capital of France is" that contains almost no letters, or almost no distinct characters, is not
# an arbiter regardless of what the paged arm does.
read -r LETTERS DISTINCT <<EOF
$(printf '%s' "$S_OUT" | python3 -c "
import sys
s = sys.stdin.read()
print(sum(c.isalpha() for c in s), len(set(s.strip())))")
EOF
if [ "${LETTERS:-0}" -lt 3 ] || [ "${DISTINCT:-0}" -lt 4 ]; then
    echo | tee -a "$OUT"
    echo "VOID: the STATIC reference is degenerate -- [$S_OUT] ($LETTERS letters, $DISTINCT distinct chars)." | tee -a "$OUT"
    echo "  Static is the arbiter for this whole gate. A broken arbiter cannot judge the paged path:" | tee -a "$OUT"
    echo "  if paged returns the same garbage the gate would print PASS on two broken runs agreeing." | tee -a "$OUT"
    echo "  Fix the VEHICLE first -- wrong chat template, damaged quant, or a base model that needs a" | tee -a "$OUT"
    echo "  different prompt. This says nothing about paging either way." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

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
# The interesting case: static served fine, paged is refused by the SWA/hybrid guard. That is the
# guard doing its job, and the arch is UNANSWERED rather than failed.
start paged; rc_p=$?
if [ "$rc_p" -eq 3 ]; then
    echo "-----" | tee -a "$OUT"
    echo "SKIPPED: arch=$ARCH serves under static and is refused under --kv-paged BY DESIGN." | tee -a "$OUT"
    echo "  Record it as UNANSWERED, never as verified and never as broken." | tee -a "$OUT"
    echo "log: $OUT"; exit 3
fi
[ "$rc_p" -ne 0 ] && exit 1
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
printf '          (startup reserve passes excluded: %s static-path warnings before the first request)\n' \
       "$(c_nopg_pre "$LOGDIR/paged.log")" | tee -a "$OUT"
echo "-----" | tee -a "$OUT"

rc=0
# ⚠ the slice marker must EXIST, or every count above silently became "whole file" again
if ! has_slice "$LOGDIR/paged.log"; then
    echo "VOID: no request marker in the paged log, so the counters could not exclude startup" | tee -a "$OUT"
    echo "  reserve passes. Every number above would be contaminated by graph builds that ran" | tee -a "$OUT"
    echo "  before a token existed. Fix the marker, do not read this result." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
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
