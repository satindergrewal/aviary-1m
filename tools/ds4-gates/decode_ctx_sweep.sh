#!/usr/bin/env bash
# DECODE CONTEXT SWEEP -- at what context length does paged decode stop losing and start winning?
#
# ⚠⚠ THE QUESTION THIS EXISTS FOR. Two measurements on the SAME kernel (head_dim 256, CHAMPION,
# block_size 64), opposite signs:
#
#     9B  @   8k   decode **0.9017x**  paged 9.8% SLOWER   (pred_n=128, clean box, clears drift)
#     35B @ 256k   decode **1.3692x**  paged 36.9% FASTER  (pred_n=14,  clears drift 7.5x)
#     35B @   8k   decode   0.878x     paged ~12% slower   (pred_n=14)
#
# The 8k points agree across two models AND two window sizes, so **the variable that changes between
# 0.88 and 1.37 is CONTEXT LENGTH**, not sampling. If that is right there is a crossover, and a
# curve locates it. If instead every length below 256k sits flat near 0.90 and 256k jumps, the cause
# is something specific to the long-context path and the "fixed indirection cost vs growing attention
# cost" story is dead.
#
# ⇒ **ONE model, ONE kernel, ONE window size, ONE varying factor.** Two point estimates on different
#   models is what made this ambiguous in the first place.
#
# ⚠ A SWEEP IS NOT A LICENCE TO AVERAGE. Each rung is its own ABBA with its own drift bound. This
# prints the per-rung verdicts side by side and REFUSES to fit anything through them -- four points
# with individually-stated uncertainty is a curve you can read; a regression line through them is a
# claim none of them supports.
#
# usage: SW_MODEL=<gguf> [SW_RUNGS="8192 32768 65536 131072"] decode_ctx_sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ DELIBERATELY DOES NOT SOURCE `_gate_common.sh`, and the exemption is stated because an UNSTATED
# one is how a convention rots. This file starts no servers, issues no requests and computes no
# verdicts -- it invokes `paged_parity_gate.sh` per rung and quotes the verdicts that gate already
# printed. **The laws apply where a measurement is made, and none is made here.** If this ever grows
# a server or a comparison of its own, the include goes in with it.

M=${SW_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-35B-1M-GGUF/ornith-1.0-35b-1M-Q4_K_M.gguf}
RUNGS=${SW_RUNGS:-"8192 32768 65536 131072"}
NPRED=${SW_NPRED:-512}
OUT=${OUT:-$HERE/results/decode-sweep-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
trap 'scrub_abs_paths "${OUT:-}" 2>/dev/null' EXIT

[ -f "$M" ] || { echo "missing model: $M" >&2; exit 2; }

echo "decode context sweep: $(basename "$M")" | tee "$OUT"
echo "rungs: $RUNGS   npred=$NPRED (with ignore_eos -- a FLOOR, verified per rung)" | tee -a "$OUT"
echo | tee -a "$OUT"

LEGS=""
for ctx in $RUNGS; do
    # ⚠ FILL RATIO HELD CONSTANT ACROSS RUNGS. 200000/262144 = 0.763 is the ratio the 256k rung used;
    # varying the occupancy as well as the length would make the sweep two-factor, and a two-factor
    # sweep cannot locate a crossover in either factor.
    fill=$(( ctx * 763 / 1000 ))
    legout="$HERE/results/decode-sweep-${ctx}-$(date +%H%M%S).txt"
    echo "=== rung ctx=$ctx  fill~${fill} ===" | tee -a "$OUT"
    OUT="$legout" PP_MODEL="$M" PP_CTX="$ctx" PP_FILL="$fill" PP_NPRED="$NPRED" \
        PP_ORDER=abba PP_WARM=1 "$HERE/paged_parity_gate.sh" >/dev/null 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$legout" ]; then
        echo "  rung ctx=$ctx: FAILED (rc=$rc) -- leg VOID, and the curve has a hole at this point." | tee -a "$OUT"
        echo "  ⚠ A missing rung is not a smooth curve with fewer points. Say so and do not interpolate." | tee -a "$OUT"
        LEGS="$LEGS $ctx:VOID"
        continue
    fi
    # Pull the per-metric DECODE line and the achieved window, both verbatim.
    dec=$(grep -m1 '^  DECODE' "$legout" | sed 's/^  //')
    # ⚠ EXIT CODE IS NOT THE POST-CONDITION. paged_parity_gate ends on a `tee`, so a VOID inside its
    # python summariser does not reach the shell's exit status -- the gate can return 0 having printed
    # "VOID: the four arms do not share prompt_n". Test for the LINE THIS SWEEP NEEDS instead.
    if [ -z "$dec" ]; then
        echo "  rung ctx=$ctx: no DECODE verdict in the leg output -- the ABBA VOIDed or the summariser" | tee -a "$OUT"
        echo "  never ran. Leg VOID; see $legout. Exit code was $rc, which is why it is not the test." | tee -a "$OUT"
        LEGS="$LEGS $ctx:VOID"
        continue
    fi
    # ⚠ AN ABSENT WINDOW LINE MUST SAY SO, NOT PRINT A BLANK. Result files written before the
    # `ignore_eos` fix have no "decode window actually generated" line at all, and their tg numbers
    # were 14-token averages -- the defect that made a decode result INVERT once the window was real.
    # A blank line there would silently present an unsampled rung beside properly sampled ones.
    win=$(grep -m1 'decode window actually generated' "$legout" | sed 's/^  //')
    [ -n "$win" ] || win="⚠ no decode-window line -- this leg predates ignore_eos, so its tg is an unknown-length average and is NOT comparable to the others"
    ver=$(grep -m1 -A1 '^  DECODE' "$legout" | tail -1 | sed 's/^ *//')
    cold=$(grep -m1 -A1 'COLD-ARM CHECK' "$legout" | tail -1 | sed 's/^ *//')
    printf '  %s\n  %s\n  %s\n  cold-arm: %s\n\n' "$win" "$dec" "$ver" "$cold" | tee -a "$OUT"
    LEGS="$LEGS $ctx:$(printf '%s' "$dec" | grep -oE 'ratio [0-9.]+' | awk '{print $2}')"
done

echo "-----" | tee -a "$OUT"
echo "  decode ratio (paged/static) by context:" | tee -a "$OUT"
for l in $LEGS; do printf '    ctx=%-8s %s\n' "${l%%:*}" "${l##*:}" | tee -a "$OUT"; done
echo | tee -a "$OUT"
# ⚠ NO FIT, NO TREND LINE, NO INTERPOLATION. Each rung carries its own drift bound and its own
# UNREADABLE/CLEARS verdict, printed above. A rung whose effect did not clear its drift contributes
# NOTHING to a crossover argument no matter where its point sits, and a line drawn through four
# points hides exactly that. Read the verdicts, then the numbers.
echo "  ⚠ Read the per-rung CLEARS/UNREADABLE verdicts above before reading these numbers. A rung" | tee -a "$OUT"
echo "    whose effect did not clear its own drift contributes nothing to a crossover argument," | tee -a "$OUT"
echo "    wherever its point happens to sit. No fit is drawn here on purpose." | tee -a "$OUT"
echo | tee -a "$OUT"
echo "  Against the existing points, same kernel (head_dim 256, CHAMPION, bs=64):" | tee -a "$OUT"
echo "    9B  @   8k  0.9017  (pred_n=128)   ·   35B @ 256k  1.3692  (pred_n=14, being re-measured)" | tee -a "$OUT"
echo "log: $OUT" | tee -a "$OUT"
