#!/usr/bin/env bash
# CLAIMED-vs-ENTERED LINT -- does a gate actually enter the regime its flags claim?
#
# WHY THIS EXISTS. On 2026-08-06 a -np 2 correctness gate (hybrid_paged_gate) passed 4/4 while the
# paged path had a data-destroying multi-sequence bug. It passed because it sends its requests
# SEQUENTIALLY:
#
#     for i in 0 1 2 3; do curl ... ; done      # no &, no wait
#
# -np 2 on the command line, one sequence in flight at a time, every batch single-sequence, bug
# dormant. The gate was configured for a regime it never entered.
#
# That is why the defect survived. Grepping the suite for "-np 2" reports three gates covering
# multi-slot; reading what they DO says one, and it was a day old.
#
# ★ A GATE THAT NEVER ENTERS ITS REGIME IS INDISTINGUISHABLE FROM A GATE THAT PASSES. Flags are a
# claim; behaviour is the coverage. This lint compares the two, statically, so the gap cannot go
# unnoticed again.
#
# Exit 1 if any gate claims a regime it does not enter.
set -uo pipefail
cd "$(dirname "$0")"
fails=0

printf "%-28s %-9s %-6s %s\n" "GATE" "CLAIMS" "ENTERS" "VERDICT"
printf "%-28s %-9s %-6s %s\n" "----" "------" "------" "-------"

for f in *.sh; do
    case "$f" in _no_abs_paths.sh|lint_claimed_vs_entered.sh) continue;; esac

    np=$(grep -oE "\-np [0-9]+" "$f" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    [ -z "${np:-}" ] && continue
    [ "$np" -lt 2 ] && continue

    # The ONLY way to get two sequences into one batch is to have two requests in flight at once,
    # which in shell means a backgrounded completion call (or a helper that backgrounds them).
    conc=$(grep -cE 'curl[^|]*(completion|/v1/)[^|]*&[[:space:]]*$|^[[:space:]]*ask .*&|fire_both' "$f" 2>/dev/null)
    conc=${conc:-0}

    if [ "$conc" -gt 0 ]; then
        printf "%-28s %-9s %-6s %s\n" "$f" "-np $np" "$conc" "ok - genuinely multi-slot"
    else
        printf "%-28s %-9s %-6s %s\n" "$f" "-np $np" "0" "⚠ CLAIMS -np $np, SENDS SEQUENTIALLY - no multi-slot coverage"
        fails=$((fails+1))
    fi
done

echo
if [ "$fails" -eq 0 ]; then
    echo "CLAIMED-vs-ENTERED: PASS - every gate claiming -np>1 actually puts two requests in flight"
else
    echo "CLAIMED-vs-ENTERED: $fails gate(s) claim multi-slot and do not exercise it."
    echo "  This is not necessarily a bug in those gates -- some are perf walls where sequential is"
    echo "  correct. It IS a bug to count them as multi-slot COVERAGE, which is exactly what let a"
    echo "  data-destroying -np>1 defect survive. Either make them concurrent or drop the -np flag"
    echo "  so the claim matches the behaviour."
fi
exit $fails
