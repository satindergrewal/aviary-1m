#!/usr/bin/env bash
# REFUSAL GATE: `--kv-paged` on an arch that cannot page must REFUSE, not ABORT, and not proceed.
#
# ⚠⚠ WHY. On 2026-08-10, `--kv-paged` on DeepSeek-V4-Flash killed the server:
#
#     E llama_paged_scheduler_init: context does not have a paged KV cache: found a
#       non-paged memory type.
#     server-context.cpp:1575: GGML_ASSERT(paged_sched && "failed to init the paged scheduler") failed
#     WARNING: Using native backtrace...
#
# `--kv-paged` is a **user-passed flag**. The scheduler had already logged the exact reason one line
# above; the assert on top of it replaced a good diagnostic with a stack trace, so the useful line
# scrolls past and the user reports a crash instead of an unsupported combination.
#
# ⚠ AND THE OBVIOUS ALTERNATIVE IS WORSE. "Warn and continue unpaged" makes the flag a no-op, and
# this lane spent 2026-08-09 proving that a paged flag which silently degrades to static is
# **indistinguishable from working**: 4.5 hours of "parity" measurements were static-vs-static.
# **A flag that does nothing is a worse outcome than a flag that refuses.**
#
# ⇒ THREE OUTCOMES, and only one is correct:
#     ABORT     GGML_ASSERT + backtrace        -- the bug this gate exists for
#     PROCEED   server starts, pages nothing   -- WORSE than the abort, and silent
#     REFUSE    named error, non-zero exit     -- correct
#
# ⚠ A FIX WITHOUT A CHECK IS A FIX THAT REGRESSES. This gate exists because the fix was written
# without one, which is the same shape as every other defect recorded in this directory.
#
# usage: RG_MODEL=<gguf-that-cannot-page> paged_refusal_gate.sh
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ AND THE SHARED LAWS -- which I FORGOT when I wrote this file, hours after driving the include
# into five other gates. `lint_common_laws.sh` caught it on the next run, and that lint's own header
# already records the same author making the same omission the same day. **Writing the lint is not
# the fix; running it is.** Used below for gate_cause_from_log on the VOID branch, where this gate
# was about to hand-roll a log diagnosis that already exists.
. "$(dirname "$0")/_gate_common.sh"  2>/dev/null || true

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
M=${RG_MODEL:-$HOME/Documents/GitHub/ornith-models/DeepSeek-V4-Flash-0731/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf}
CTX=${RG_CTX:-4096}
OUT=${OUT:-$(dirname "$0")/results/paged-refusal-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
trap 'scrub_abs_paths "${OUT:-}" 2>/dev/null' EXIT

[ -f "$M" ] || { echo "missing model: $M" >&2; exit 2; }
D=${CLAUDE_JOB_DIR:-/tmp}/refusal; mkdir -p "$D"; LOG=$D/refusal.log; : > "$LOG"

PORT=0; for p in $(seq 20300 20330); do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { PORT=$p; break; }; done
[ "$PORT" != 0 ] || { echo "no free port" >&2; exit 2; }

echo "paged refusal gate: $M" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD)  ctx=$CTX" | tee -a "$OUT"

env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 DS4P_METAL_CHAMP=1 \
    "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 --port "$PORT" --no-warmup -lv 4 \
    --kv-paged --kv-block-size 64 > "$LOG" 2>&1 &
PID=$!

# ⚠ WAIT FOR AN OUTCOME, NOT FOR A TIMEOUT. Three things can happen and the gate must distinguish
# them; a fixed sleep would score "still loading" as "refused".
served=0; died=0
for i in $(seq 1 900); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { served=1; break; }
    kill -0 $PID 2>/dev/null || { died=1; break; }
    sleep 1
done
rc=0
if [ "$served" = 1 ]; then kill $PID 2>/dev/null; wait $PID 2>/dev/null
else wait $PID 2>/dev/null; rc=$?; fi

aborted=$(grep -ac 'GGML_ASSERT\|ggml_abort\|Using native backtrace' "$LOG")
refused=$(grep -ac 'does not support paging\|does not have a paged KV cache' "$LOG")

printf '  served=%s  died=%s  exit=%s  abort_markers=%s  refusal_markers=%s\n' \
    "$served" "$died" "$rc" "$aborted" "$refused" | tee -a "$OUT"

if [ "$aborted" -gt 0 ]; then
    echo "  ⇒ **FAIL: ABORTED.** A user-passed flag crashed the server. The scheduler's own" | tee -a "$OUT"
    echo "    diagnostic is in the log above the backtrace and is what the user should have seen." | tee -a "$OUT"
    grep -m1 -oE 'context does not have a paged KV cache.*' "$LOG" | cut -c1-160 | sed 's/^/      /' | tee -a "$OUT"
    exit 1
fi
if [ "$served" = 1 ]; then
    # ⚠ THE SILENT-SUCCESS BRANCH IS THE ONE THAT LOOKS LIKE A PASS. Starting cleanly while paging
    # nothing is worse than the abort, and it is the state LAW 6 was written to catch.
    cons=$(grep -ac 'ZERO layers have consumed' "$LOG")
    echo "  ⇒ server STARTED with --kv-paged on a model that cannot page." | tee -a "$OUT"
    if [ "$cons" -gt 0 ]; then
        echo "    **FAIL: it started and paged NOTHING** (engine no-consumer alarm fired)." | tee -a "$OUT"
        echo "    This is WORSE than the abort: silent, and indistinguishable from working." | tee -a "$OUT"
        exit 1
    fi
    echo "    ⚠ INCONCLUSIVE: no no-consumer alarm either. Either this model CAN page after all" | tee -a "$OUT"
    echo "    -- in which case it is the wrong vehicle for this gate -- or the alarm never armed" | tee -a "$OUT"
    echo "    (it needs >= 8 decodes, and this gate issues none). Do not read this as a pass." | tee -a "$OUT"
    exit 2
fi
if [ "$refused" -gt 0 ]; then
    echo "  ⇒ **PASS: refused by design.** Named error, non-zero exit, no backtrace." | tee -a "$OUT"
    grep -m1 -oE '(does not support paging|does not have a paged KV cache).*' "$LOG" | cut -c1-160 | sed 's/^/      /' | tee -a "$OUT"
    exit 0
fi
echo "  ⇒ VOID: the server exited without serving and without any refusal marker. It failed for" | tee -a "$OUT"
echo "    some OTHER reason, so this run says nothing about the refusal path." | tee -a "$OUT"
# ⚠ SAY WHY, rather than pointing at a log the reader has to open. A dead arm and a refused arm
# render identically as "no result", and two harness faults in this lane were diagnosable only by
# opening a log by hand. gate_cause_from_log exists precisely for this and lives in the include
# this file was missing.
command -v gate_cause_from_log >/dev/null 2>&1 && gate_cause_from_log "$LOG" "refusal gate" | tee -a "$OUT"
exit 2
