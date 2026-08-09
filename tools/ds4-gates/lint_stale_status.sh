#!/usr/bin/env bash
# LINT: does a document's HEADER claim a status its own TAIL contradicts?
#
# ⚠⚠ THE CLASS THIS EXISTS FOR BIT FOUR TIMES ON 2026-08-09/10, and one instance shipped a code
# regression:
#
#   1. `PLAN-paged-arch-support.md` -- top table said `paged wired: no` for nine archs; the state
#      section twenty lines down said 22 archs carry the consumer, listing the same files. **I read
#      the stale half, wrote "21 architectures excluded by construction", and commit cbb4c8d93 added
#      a blanket sliding-window rejection that SILENTLY DISABLED gemma4's paging.** Invisible,
#      because the fallback output is correct.
#   2. `FINDINGS-paged-cross-request.md` -- line 3 said `Status: OPEN`; line 1086 said
#      *"Root-caused, fixed, induced on demand."* I quoted the header to the foreman and the owner
#      and raised a false alarm about a defect that had been closed for half a day.
#   3. `_gate_common.sh` LAW 6 -- opened with `UNPROVEN ... THE ALARM HAS NEVER BEEN OBSERVED
#      FIRING`, five hours after I had validated it by direct control. My own file, my own law.
#   4. `paged_parity_gate.sh` cold-arm branch -- printed *"re-run with the prelude enabled"* while
#      the prelude WAS enabled, with both warm-up arms logged above it.
#
# **The header is what a reader quotes.** That is what makes the drift expensive rather than untidy:
# nobody scrolls a thousand lines to check whether the top is still true.
#
# ⇒ HEURISTIC: a resolution word in the LAST 30% of a file whose FIRST 15% carries an open-state
#   word. That is exactly shapes 1-3. It cannot see shape 4 (a stale instruction inside code), and
#   says so rather than implying coverage it does not have.
#
# ⚠ VOCABULARY, NOT MEANING. A doc that legitimately discusses both states -- a findings file with an
# open section and a closed one -- is a FALSE POSITIVE, and that is the intended trade: this prints a
# READING LIST, never a verdict. Same doctrine as lint_common_laws and lint_vacuous_pass.
#
# usage: lint_stale_status.sh [docs-dir]
set -uo pipefail
DIR="${1:-$(dirname "$0")}"
cd "$DIR" || { echo "no such dir: $DIR" >&2; exit 2; }

OPEN_RE='Status: *(OPEN|PENDING)|UNPROVEN|NOT (YET )?(VERIFIED|PROVEN|MEASURED)|⛔|TODO'
DONE_RE='CLOSED|RESOLVED|✅|VALIDATED|Root-caused|FIXED|Final status|now a measurement|SUPERSEDED'

echo "lint: header claims a status the tail contradicts"
tot=0; flag=0
for f in *.md; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f" | tr -d ' ')
    [ "$n" -ge 20 ] || continue          # too short for a header/tail split to mean anything
    tot=$((tot+1))
    head_n=$(( n * 15 / 100 )); [ "$head_n" -lt 5 ] && head_n=5
    tail_n=$(( n * 30 / 100 )); [ "$tail_n" -lt 5 ] && tail_n=5
    h=$(head -n "$head_n" "$f" | grep -cE "$OPEN_RE")
    t=$(tail -n "$tail_n" "$f" | grep -cE "$DONE_RE")
    # ⚠ A HEADER THAT CARRIES THE RESOLUTION TOO IS ALREADY ANNOTATED, NOT STALE. The first run of
    # this lint flagged `FINDINGS-paged-cross-request.md` -- the exact file whose stale header caused
    # the false alarm -- **after it had been fixed**, because the corrected version keeps the old
    # `Status: OPEN` line under a divider as history. **A lint that cannot tell "stale" from
    # "annotated with its history" trains the reader to ignore it**, which is how an unenforced
    # convention dies. If the head states the resolution as well, the reader is already served.
    hd=$(head -n "$head_n" "$f" | grep -cE "$DONE_RE")
    if [ "$h" -gt 0 ] && [ "$t" -gt 0 ] && [ "$hd" -eq 0 ]; then
        flag=$((flag+1))
        printf '  %-42s head:open×%-3s tail:resolved×%-3s  (%s lines)\n' "$f" "$h" "$t" "$n"
        head -n "$head_n" "$f" | grep -m1 -oE "$OPEN_RE" | sed 's/^/      header says: /'
        tail -n "$tail_n" "$f" | grep -m1 -oE "$DONE_RE"  | sed 's/^/      tail says:   /'
    fi
done

# ⚠ An instrument that examined nothing must not report a clean pass -- the subject of the sibling
# lint, and it would be self-refuting to get it wrong here.
if [ "$tot" -eq 0 ]; then
    echo "VOID: no .md file long enough to split. The pattern is wrong, or the directory is."
    exit 2
fi

echo "-----"
printf '  %d of %d documents flagged for a read\n' "$flag" "$tot"
echo "  ⚠ This is a READING LIST, not a verdict. A file that legitimately covers both an open and a"
echo "  closed matter is a false positive by design -- the alternative is missing the one whose"
echo "  header sent a guard into the tree and disabled a working feature."
echo "  ⚠ It also CANNOT see a stale instruction inside CODE (a gate telling you to enable a flag"
echo "  that is already enabled). That shape needs a reader, not a grep."
exit 0
