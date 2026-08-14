#!/usr/bin/env bash
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ A TRAP, NOT A TRAILING CALL. A trailing scrub is jumped over by the gate's own `exit`, while
# `grep -l scrub_abs_paths` still lists the file as a caller -- grep counts TEXT, not control flow.
# On EXIT it runs whatever path the gate takes.
# ⚠ ADDED 2026-08-09 after a full-history rewrite + force-push removed /Users/<username> from 11
# PUBLISHED commits. That cleaned the backlog; this closes the PRODUCERS. A history scrub with live
# emitters still in the tree is a fix with a regression path.
trap 'scrub_abs_paths "${OUT:-}"' EXIT
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

    # ⚠ STRIP COMMENTS FIRST (fixed 2026-08-13). After hybrid_paged_gate dropped -np 2, this lint
    # still flagged it because the explanatory comment contained the string "-np 2". A claim is
    # what the SCRIPT PASSES TO THE SERVER, not what a comment mentions.
    # ⚠ ALSO READ `${VAR:-N}` DEFAULTS (found by Grok, #9808). `-np ${BI_NP:-2}` is how the
    # best-guarded multi-slot gate (batch_offset_invariant_gate) claims 2; the old regex only
    # matched `-np [0-9]+`, so that gate was INVISIBLE -- twin of the comment-as-claim hole.
    # Same shape: warm_multislot_gate `-np ${MS_NP:-2}`.
    body=$(grep -vE '^[[:space:]]*#' "$f" 2>/dev/null)
    np=$( { echo "$body" | grep -oE '\-np [0-9]+' | grep -oE '[0-9]+'
            echo "$body" | grep -oE '\-np \$\{[A-Za-z_][A-Za-z0-9_]*:-[0-9]+\}' | grep -oE ':-[0-9]+' | grep -oE '[0-9]+'
          } | sort -rn | head -1 )
    [ -z "${np:-}" ] && continue
    [ "$np" -lt 2 ] && continue

    # The ONLY way to get two sequences into one batch is to have two requests in flight at once,
    # which in shell means a backgrounded completion call (or a helper that backgrounds them).
    # ⚠ JOIN CONTINUATION LINES FIRST (fixed 2026-08-13). grep is line-based; a curl written as
    #     curl ... /completion ... \
    #       -d '{...}' >/dev/null &
    # has `completion` and the trailing `&` on DIFFERENT lines, so the one-line regex scored it 0
    # and this lint accused the suite's best-guarded gate (batch_offset_invariant_gate, concurrent
    # curls AND a runtime n_seq>=2 presence check) of sending sequentially -- 3 of its 4 flags were
    # this false positive. A lint that mis-scores the best-guarded gate erodes trust in its real
    # finding (hybrid_paged_gate, which genuinely IS sequential). The sed joins `\`-continued lines.
    conc=$(sed -e ':a' -e '/\\$/N; s/\\\n//; ta' "$f" 2>/dev/null | grep -cE 'curl[^|]*(completion|/v1/)[^|]*&[[:space:]]*$|^[[:space:]]*ask .*&|fire_both')
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
