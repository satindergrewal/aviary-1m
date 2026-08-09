#!/usr/bin/env bash
# LINT: can a gate print PASS having examined NOTHING?
#
# ⚠⚠ THE FAILURE THIS MEASURES, and it has happened in this directory more than once.
#   * `privacy_guard.sh` printed **"PASS (0 added lines scanned)"** against a repo with real
#     uncommitted changes -- it defaults to `--cached`, nothing was staged, so it read an empty diff
#     and passed. `exit 0` is what a caller's `&&` chain reads. On the ONE rule here with three
#     history rewrites behind it.
#   * Two lints in this directory carry explicit `matched ZERO ... the pattern is wrong, not the
#     tree` branches for exactly this reason, written after the same shape bit.
#
# **A clean pass from an instrument that looked at nothing is worse than a failure**, because a
# failure gets investigated and a pass gets quoted.
#
# ⚠ SCOPE, STATED BECAUSE THE HEURISTIC IS CRUDE. This greps for the PRESENCE of a zero-guard
# vocabulary (`VOID`, `examined`, `checked`, `matched ZERO`, `-eq 0`). A gate that guards by some
# other spelling is a FALSE POSITIVE here, and a gate that has the words but never reaches them is a
# FALSE NEGATIVE. **This measures vocabulary, not control flow** -- the same limitation
# `lint_common_laws.sh` states about REACH vs USE, and for the same reason: the cheap check that runs
# beats the perfect check that does not exist.
#
# ⚠ AND IT DOES NOT FAIL THE BUILD, DELIBERATELY. Bulk-adding guards to gates nobody has read is the
# "wire all 21 architectures and gate one" shape this directory refuses. The value is VISIBILITY: an
# unmeasured hazard that becomes a number stops being invisible, and the number goes down gate by
# gate, with a read.
#
# usage: lint_vacuous_pass.sh [--strict] [gates-dir]
set -uo pipefail
STRICT=0; DIR="$(dirname "$0")"
for a in "$@"; do case "$a" in --strict) STRICT=1 ;; *) DIR="$a" ;; esac; done
cd "$DIR" || { echo "no such dir: $DIR" >&2; exit 2; }

# The gates whose verdicts reach a spend decision. Uncovered here is worse than uncovered elsewhere.
BAR_FEEDING="long_context_gate.sh paged_parity_gate.sh arch_serve_gate.sh warm_multislot_gate.sh multislot_gate.sh"

echo "lint: can a gate print PASS having examined nothing?"
tot=0; bad=0; barbad=0
for g in *.sh; do
    case "$g" in _*) continue ;;
    esac
    grep -qE 'PASS|CLEAN|✅' "$g" 2>/dev/null || continue
    tot=$((tot+1))
    if grep -qE 'VOID|examined|checked|matched ZERO|-eq 0' "$g"; then continue; fi
    bad=$((bad+1))
    case " $BAR_FEEDING " in
        *" $g "*) printf '  %-30s NO ZERO-GUARD  *** AND BAR-FEEDING ***\n' "$g"; barbad=$((barbad+1)) ;;
        *)        printf '  %-30s NO ZERO-GUARD\n' "$g" ;;
    esac
done

# ⚠ AN INSTRUMENT THAT EXAMINED NOTHING MUST NOT REPORT A CLEAN PASS -- which is this lint's own
# subject, so getting it wrong here would be self-refuting.
if [ "$tot" -eq 0 ]; then
    echo "VOID: matched ZERO gates that can print PASS. The pattern is wrong, not the tree."
    exit 2
fi

echo "-----"
printf '  %d of %d PASS-capable gates have no guard against examining nothing\n' "$bad" "$tot"
if [ "$barbad" -gt 0 ]; then
    echo "  *** $barbad of them are BAR-FEEDING. Those verdicts reach a spend decision. ***"
    [ "$STRICT" -eq 1 ] && exit 1
fi
echo "  ⚠ This measures VOCABULARY, not control flow: a gate guarding by another spelling is a false"
echo "  positive, and one holding the words on an unreachable path is a false negative. Read before"
echo "  editing -- and never bulk-edit gates nobody has read, which is the shape this lane refuses."
exit 0
