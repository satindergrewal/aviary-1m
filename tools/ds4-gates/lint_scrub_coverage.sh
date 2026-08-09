#!/usr/bin/env bash
# LINT: every gate that WRITES into the tracked results/ directory must scrub absolute home paths.
#
# ⚠⚠ WHY THIS EXISTS. On 2026-08-09 a full-history rewrite and force-push removed `/Users/<username>`
# from **11 already-published commits** of a PUBLIC repo. That is the privacy rule with the highest
# priority in this project and two prior history-rewrite incidents.
#
# ⚠ THE SCRUB CLEANED THE BACKLOG AND LEFT THE PRODUCERS. Immediately after the force-push, a sweep
# found FOUR gates still writing into results/ with no scrubber at all --
#     abort_paths_gate · p15_fair_ab · p15_warm_admit_gate · p28_cuda_regate
# -- and TWO more that SOURCED `_no_abs_paths.sh` and never CALLED it:
#     coverage_census · lint_claimed_vs_entered
# **Any one of them running would have put the string straight back into a tracked file.** A history
# scrub with live emitters in the tree is a fix with a regression path, and this is the check that
# closes it.
#
# ⚠ It also encodes the mirror of commit 633fb20 ("stop emitting absolute home paths into result
# files"), which fixed that day's emitters and left the already-committed backlog. Producer-only and
# backlog-only are the same mistake from opposite ends; this lint covers the producer end permanently.
#
# ⚠ A TRAP, NOT A TRAILING CALL, is what counts as covered. A trailing `scrub_abs_paths` is jumped
# over by the gate's own `exit`, while `grep -l` still lists the file as a caller -- grep counts TEXT,
# not control flow. So this lint requires the EXIT trap form specifically.
#
# usage: lint_scrub_coverage.sh [gates-dir]     exit 0 = every writer is covered
set -uo pipefail
DIR="${1:-$(dirname "$0")}"
cd "$DIR" || { echo "no such dir: $DIR" >&2; exit 2; }

bad=0 checked=0
echo "lint: gates writing to results/ must trap scrub_abs_paths on EXIT"
for g in *.sh; do
    case "$g" in _*) continue ;; esac              # libraries are sourced, not run
    # ⚠ MENTIONING results/ IS NOT WRITING TO IT. The first version flagged THIS LINT ITSELF, which
    # only names the directory in its own help text. A writer assigns OUT=; prose does not.
    grep -q 'results/' "$g" 2>/dev/null || continue
    grep -qE '^[[:space:]]*OUT=|^OUT=' "$g" 2>/dev/null || continue
    checked=$((checked+1))
    # ⚠ THE TRAP MAY BE INDIRECT, AND THE FIRST VERSION OF THIS LINT CALLED THAT A FAILURE.
    # arch_serve_gate does `trap cleanup EXIT` where cleanup() calls scrub_abs_paths -- correct, and
    # rejected by a literal `trap .*scrub_abs_paths` match. A lint that only recognises ONE spelling of
    # a correct thing reports style, not safety. Resolve the handler: if it is a function name, look
    # inside that function.
    # ⚠ READ THE WHOLE trap LINE FIRST. A quoted multi-statement handler
    #   trap 'rmdir ...; kill ...; scrub_abs_paths "${OUT:-}"' EXIT
    # was missed by a regex that tried to isolate the handler token -- paged_parity_gate.sh, which is
    # correctly covered, was reported as failing. Match the line, then fall back to a function name.
    trapline=$(grep -E "^[[:space:]]*trap[[:space:]].*EXIT" "$g" | head -1)
    covered=no
    case "$trapline" in
      *scrub_abs_paths*) covered=yes ;;
      "") : ;;
      *) fn=$(printf '%s' "$trapline" | sed -E "s/^[[:space:]]*trap[[:space:]]+//; s/[[:space:]]+EXIT.*$//" | tr -d "'\"")
         # body of `fn() { ... }` up to the closing brace at column 0
         awk -v f="$fn" '$0 ~ "^"f"\\(\\)" {inf=1} inf {print} inf && /^}/ {exit}' "$g" \
            | grep -q 'scrub_abs_paths' && covered=yes ;;
    esac
    if [ "$covered" = yes ]; then
        printf '  %-30s writes results/  trapped        OK\n' "$g"
    elif grep -q 'scrub_abs_paths' "$g"; then
        printf '  %-30s writes results/  *** CALLS BUT DOES NOT TRAP *** an early exit skips it\n' "$g"
        bad=$((bad+1))
    else
        printf '  %-30s writes results/  *** NO SCRUBBER AT ALL ***\n' "$g"
        bad=$((bad+1))
    fi
done

# ⚠ THE LINT MUST BE ABLE TO FIND ANYTHING. Zero matches means the pattern is wrong, not that the
# tree is clean -- a clean pass from an instrument that looked at nothing is the failure this whole
# directory exists to prevent.
if [ "$checked" -eq 0 ]; then
    echo "VOID: matched ZERO gates. Nothing was examined, so this exit code means nothing."
    exit 2
fi

echo "-----"
if [ "$bad" -ne 0 ]; then
    echo "FAIL: $bad of $checked results-writing gates are not covered."
    echo "  Add next to the shebang / set line:"
    echo "      . \"\$(dirname \"\$0\")/_no_abs_paths.sh\" 2>/dev/null || true"
    echo "      trap 'scrub_abs_paths \"\${OUT:-}\"' EXIT"
    exit 1
fi
echo "PASS: all $checked results-writing gates trap the scrubber on EXIT."
echo "  ⚠ This says the SCRUBBER RUNS. It does not prove the scrubber's own patterns are complete --"
echo "  that is privacy_guard.sh's job, at commit time, and the two are different checks."
