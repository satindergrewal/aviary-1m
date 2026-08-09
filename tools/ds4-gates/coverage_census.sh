#!/usr/bin/env bash
# COVERAGE CENSUS -- recompute the suite's composition instead of trusting a number in a doc.
#
# WHY: COVERAGE.md was written with "12 GATES / 2 WALLS / 11 REFUSES, unattended coverage 14 of 25".
# Four scripts were added the same night and the number silently became wrong. A count written once
# and never re-derived is a claim that decays every time the thing it describes changes -- and the
# whole point of that document is that "we have N gates" was never a meaningful number.
#
# So: derive it, print it, and let the doc quote THIS rather than a memory of it.
#
# Classification, stated so the number is reproducible rather than authoritative:
#   REFUSES  has a PLATFORM PRECONDITION guard, or ${1:?}/${2:?} required args
#   WALL     reports BEST= numbers and never prints a GATE: verdict
#   GATE     everything else
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ A TRAP, NOT A TRAILING CALL. A trailing scrub is jumped over by the gate's own `exit`, while
# `grep -l scrub_abs_paths` still lists the file as a caller -- grep counts TEXT, not control flow.
# On EXIT it runs whatever path the gate takes.
# ⚠ ADDED 2026-08-09 after a full-history rewrite + force-push removed /Users/<username> from 11
# PUBLISHED commits. That cleaned the backlog; this closes the PRODUCERS. A history scrub with live
# emitters still in the tree is a fix with a regression path.
trap 'scrub_abs_paths "${OUT:-}"' EXIT
cd "$(dirname "$0")"
g=0; w=0; r=0; tot=0
declare -a G W R
for f in *.sh; do
    case "$f" in _no_abs_paths.sh|coverage_census.sh) continue;; esac
    tot=$((tot+1))
    if grep -q "PLATFORM PRECONDITION" "$f" 2>/dev/null || grep -qE '\$\{1:\?|\$\{2:\?' "$f" 2>/dev/null; then
        r=$((r+1)); R+=("$f")
    elif grep -qE "BEST=" "$f" 2>/dev/null && ! grep -q "GATE:" "$f" 2>/dev/null; then
        w=$((w+1)); W+=("$f")
    else
        g=$((g+1)); G+=("$f")
    fi
done
echo "GATES   ($g): ${G[*]}"  | fold -w 100 -s
echo "WALLS   ($w): ${W[*]}"  | fold -w 100 -s
echo "REFUSES ($r): ${R[*]}"  | fold -w 100 -s
echo
echo "TOTAL $tot   UNATTENDED $((g+w)) of $tot   (a refusal is correct behaviour and is NOT coverage)"
