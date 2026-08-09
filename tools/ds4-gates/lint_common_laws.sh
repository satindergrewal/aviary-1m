#!/usr/bin/env bash
# LINT: which gates can actually REACH the laws in _gate_common.sh?
#
# ⚠⚠ THE FINDING THIS WAS WRITTEN FOR. On 2026-08-09, `_gate_common.sh` held FIVE laws and reached
# **6 gates out of 44**. Thirty-eight result-producing gates sourced none of it.
#
# That file exists BECAUSE prose laws did not bind the next script. Its own header records three laws
# written, argued and committed on 2026-08-08, each violated by the very next gate written, and the
# stated fix: *"you cannot forget to apply an include."*
#
# ⇒ **You can, if you never add the include.** 86% of the suite never did, and nothing in the tree
#   noticed. The laws became exactly what the include was built to replace: a convention every author
#   has to remember.
#
# ⚠ AND THE AUTHOR WHO FORGOT IT WAS ME, THE SAME DAY. `arm_delta_vs_depth.sh` was written hours after
#   `_gate_common.sh` and does not source it.
#
# ⚠ THIS LINT DOES NOT FAIL THE BUILD, AND THAT IS DELIBERATE. Bulk-adding a `source` line to 38 gates
# -- most of which predate several of the laws -- would change behaviour in scripts nobody has read.
# That is the "wire all 21 architectures and gate one" shape refused earlier the same day. The value
# here is VISIBILITY: an unenforced convention that is measured stops being invisible.
#
# ⇒ Exit 0 always for the survey; use --strict to make an uncovered BAR-FEEDING gate an error, since
#   those are the ones whose verdicts reach a spend decision.
#
# usage: lint_common_laws.sh [--strict] [gates-dir]
set -uo pipefail
STRICT=0; DIR="$(dirname "$0")"
for a in "$@"; do case "$a" in --strict) STRICT=1 ;; *) DIR="$a" ;; esac; done
cd "$DIR" || { echo "no such dir: $DIR" >&2; exit 2; }
[ -f _gate_common.sh ] || { echo "VOID: no _gate_common.sh here -- nothing to measure against." >&2; exit 2; }

# The gates whose verdicts feed the 256k-1M bar. Uncovered here is worse than uncovered elsewhere.
BAR_FEEDING="long_context_gate.sh paged_parity_gate.sh arch_serve_gate.sh warm_multislot_gate.sh multislot_gate.sh"

LAWS=$(grep -cE '^gate_[a-z_]+\(\)' _gate_common.sh)
echo "lint: reach of _gate_common.sh  (${LAWS} helper laws defined)"

cov=0; unc=0; barbad=0
for g in *.sh; do
    case "$g" in _*|lint_*) continue ;; esac
    grep -q 'OUT=' "$g" 2>/dev/null || continue
    if grep -q '_gate_common.sh' "$g"; then
        cov=$((cov+1))
        case " $BAR_FEEDING " in *" $g "*) printf '  %-30s COVERED   (bar-feeding)\n' "$g" ;; esac
    else
        unc=$((unc+1))
        case " $BAR_FEEDING " in
          *" $g "*) printf '  %-30s *** UNCOVERED, AND BAR-FEEDING ***\n' "$g"; barbad=$((barbad+1)) ;;
        esac
    fi
done

# ⚠ AN INSTRUMENT THAT EXAMINED NOTHING MUST NOT REPORT A CLEAN PASS -- the failure this directory
# exists to prevent, and the reason every lint here carries this branch.
if [ $((cov+unc)) -eq 0 ]; then
    echo "VOID: matched ZERO result-producing gates. The pattern is wrong, not the tree."
    exit 2
fi

echo "-----"
printf '  reach: %d covered / %d total  (%d uncovered)\n' "$cov" "$((cov+unc))" "$unc"
if [ "$unc" -gt "$cov" ]; then
    echo "  ⚠ THE MAJORITY OF GATES CANNOT REACH THE LAWS. An include that most authors never add is"
    echo "  the same unenforced convention it was built to replace. Not a build failure -- bulk-editing"
    echo "  unread scripts is worse -- but it is the number to drive down, gate by gate, with a read."
fi
if [ "$barbad" -gt 0 ]; then
    echo "  *** $barbad BAR-FEEDING gate(s) uncovered. Those verdicts reach a spend decision. ***"
    [ "$STRICT" -eq 1 ] && exit 1
fi
echo "  ⚠ SCOPE: this measures REACH, not USE. A gate that sources the file and calls none of its"
echo "  helpers passes here. Sourcing is not calling -- that distinction has cost this lane twice."
exit 0
