#!/usr/bin/env bash
# PRIVACY GUARD -- refuses, rather than reports.
#
# ⚠⚠ WHY THIS IS A FILE AND NOT A GREP I TYPE EACH TIME. On 2026-08-06 a pre-commit privacy grep
# fired, printed real hits, and the commit went through on the next line of the same invocation:
#
#     git add ... && echo "--- privacy ---"; git diff --cached | grep -nE "<home-path>|<username>"
#     git commit ...            # <- unconditional, same invocation, runs regardless
#
# (the patterns above are written as placeholders on purpose: spelling them out would make this file
#  fail its own check, which it did on the first attempt to commit it -- the guard refusing its own
#  source is the first time it was observed blocking anything real)
#
# A check that cannot STOP the thing it guards is not a check, it is a log line. That is the
# correct-producer-no-consumer class applied to the one rule in this project with ZERO tolerance and
# two prior history-rewrite incidents.
#
# ⚠ AND I REPEATED THE SHAPE ON 2026-08-09, three times in one night, in this exact form:
#
#     git diff | grep -nE "..." | sed 's/^/  LEAK? /' || echo "  privacy grep: clean"
#     git add ... && git commit ...
#
# It happened to be clean every time, so nothing leaked and nothing corrected me. **A guard that only
# works when it has nothing to catch has not been tested.** Worse, `grep | sed` makes the `||` branch
# depend on sed's exit status, not grep's -- so "clean" would print even on a hit. Two defects in one
# line, and the printed word was "clean".
#
# ⚠⚠⚠ AND ON 2026-08-10 I DID IT AGAIN, TO THIS FILE, WHILE QUOTING THIS FILE. Exact form:
#
#     git add ... && ./tools/privacy_guard.sh . 2>&1 | tail -2 && git commit ...
#                                              ^^^^^^^^^^^^^^^
# **The pipe makes `&&` read `tail`'s exit status, not the guard's.** The guard printed
# "PRIVACY GUARD: REFUSING -- 1 added line(s)" and named the offending line, `tail` exited 0, and the
# commit ran anyway. The refusal was on screen. It changed nothing. **This is the same defect as the
# `grep | sed` one four lines up: I reformatted the guard's output and destroyed its verdict.**
#
# ⇒ **WHAT ACTUALLY STOPPED IT WAS THE `--range` MODE, unpiped, on the next line of the same chain.**
#   It refused, `&&` broke, and the push never happened -- so the leak reached local history and got
#   no further. That mode was added the previous day on the argument that *"inspection is not a
#   guard"* at the public boundary, and this is the first time it caught something the staged check
#   had already waved through. **Two independent guards at two boundaries is why a broken invocation
#   of one was survivable.**
#
# ⇒ RULE, and it is about SHELL not about care: **never pipe this script.** `tail`, `head`, `sed`,
#   `grep` and `cut` all replace its exit status with their own. If you want less output, redirect to
#   a file and read that -- `>/dev/null` is safe because it does not change the exit status.
#   The durable fix is the pre-push hook in `tools/pre-commit-privacy.hook`, which cannot be piped
#   because nobody invokes it; it is offered there and deliberately not installed by the assistant.
#
# usage:  privacy_guard.sh [repo-path]     # checks STAGED changes; exit 1 = REFUSE
#         privacy_guard.sh --worktree      # checks unstaged working tree too
#         privacy_guard.sh <repo> --range origin/master..HEAD   # what a PUSH would publish
set -uo pipefail

REPO="."
MODE=staged
RANGE=""
_want_range=0
for a in "$@"; do
    if [ "$_want_range" = 1 ]; then RANGE="$a"; _want_range=0; continue; fi
    case "$a" in
        --worktree) MODE=worktree ;;
        --range)    MODE=range; _want_range=1 ;;
        *)          REPO="$a" ;;
    esac
done
cd "$REPO" || { echo "privacy_guard: no such repo: $REPO" >&2; exit 2; }

# ⚠⚠ THE MODE THIS GUARD DID NOT HAVE, AND THE ONE THAT MATTERS MOST. It could check `staged` and
# `worktree` -- both PRIVATE states -- and had no way to answer **"what am I about to PUBLISH?"**
# On 2026-08-10 that gap had a cost: I pushed five commits to a PUBLIC fork this guard has never been
# pointed at, and they were clean only because a reviewer read them. **Inspection is not a guard.**
# All three of this project's history rewrites were triggered at the public boundary, so the range
# a push would send is the diff that actually needed checking.
#   privacy_guard.sh <repo> --range origin/master..HEAD
if [ "$MODE" = worktree ]; then
    DIFF=$(git diff; git diff --cached)
elif [ "$MODE" = range ]; then
    [ -n "$RANGE" ] || { echo "privacy_guard: --range needs an argument, e.g. origin/master..HEAD" >&2; exit 2; }
    # ⚠ `git diff <range>` compares two ENDPOINTS and would miss a secret added in one commit and
    # removed in the next -- which still ships, because the blob stays reachable in the history being
    # pushed. That is precisely how the 2026-08-09 leak survived a partial scrub. `git log -p` walks
    # EVERY commit in the range instead.
    # ⚠⚠ AND THE FIRST VERSION OF THIS LINE HID A BAD REVSPEC BEHIND `2>/dev/null`. Run against the
    # fork with `origin/ds4-ports..HEAD` -- a ref that DOES NOT EXIST in that clone -- git errored,
    # stderr went to the void, DIFF came back empty, and the guard reported
    # **"the range diff is EMPTY ... Stage the change first (git add)"**: the wrong diagnosis for the
    # wrong mode, from an error I had suppressed. The empty-diff VOID caught it and stopped the
    # `&&` chain, so nothing passed -- **but a guard that says the wrong reason costs the reader the
    # time the guard was supposed to save.** Written minutes after fixing this same shape elsewhere.
    # ⇒ Validate the revspec FIRST and say so; suppress nothing.
    if ! git rev-parse --verify --quiet "${RANGE%%..*}" >/dev/null || \
       ! git rev-parse --verify --quiet "${RANGE##*..}" >/dev/null; then
        echo "privacy_guard: VOID -- '$RANGE' is not a valid revision range in $(pwd)." >&2
        echo "  Both endpoints must resolve. A missing remote-tracking ref reads as an EMPTY range," >&2
        echo "  which would otherwise look exactly like 'nothing to check'. Try: git fetch origin" >&2
        echo "  and check 'git branch -r' for the ref's real name." >&2
        exit 2
    fi
    DIFF=$(git log -p --no-merges --format='' "$RANGE")
else
    DIFF=$(git diff --cached)
fi

# Only ADDED lines can leak. A removed line containing a path is a leak being deleted.
ADDED=$(printf '%s\n' "$DIFF" | grep '^+' | grep -v '^+++')

# ⚠ Patterns are deliberately broad and are allowed to false-positive. A false positive costs one
# reading; a false negative has cost two full git history rewrites.
#   - home paths that carry a username        /Users/<name>, /home/<name>
#   - the username itself, in any casing
#   - real IPv4 outside the documentation and loopback ranges
#   - RFC1918 private addresses (LAN topology is private too, per the rule)
#   - IPv6 outside the 2001:db8:: documentation range
PAT_PATH='/Users/[a-zA-Z0-9._-]+|/home/[a-zA-Z0-9._-]+'
PAT_NAME='[Ss][Aa][Tt][Ii][Nn][Dd][Ee][Rr]'
# ⚠ OCTET-VALIDATED, NOT `[0-9]{1,3}` FOUR TIMES. The loose form matched llama.cpp's own log
# timestamps -- `0.00.321.268` is four dot-separated numbers and looks exactly like an address. It
# blocked a legitimate commit on 2026-08-09. The response to a false positive is to make the PATTERN
# right, never to wave the hit through: 321 and 268 are not octets, and a real address always is.
PAT_IP4='\b((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\b'
# ⚠⚠ THE UNCOMPRESSED-ONLY PATTERN MISSED THE MOST COMMON FORM OF IPv6, AND NOTHING HAD EVER TESTED
# IT. Controls run 2026-08-09, the first time this guard was exercised on v6 at all:
#
#     an address shaped  HHHH:HHHH:HHHH::HHHH   ** PASSED **   <- three groups, gap, one group
#     an address shaped  HHHH::HHHH:HHHH:HHHH:HHHH   refused    <- four groups after the gap
#
# (written as shapes, not literals, ON PURPOSE: spelling out a real address here makes this file
#  fail its own check -- which is exactly what happened on the first attempt to commit this fix, the
#  second time this guard has blocked its own source. The precedent set the first time stands:
#  rewrite the comment, never widen the allow-list.)
#
# `([0-9a-fA-F]{1,4}:){3,}[0-9a-fA-F]{1,4}` needs FOUR groups in a row. `::` compression collapses
# the middle, so the first shape offers only three before the gap and the pattern cannot reach the
# tail. The second matched only because it happens to carry four groups AFTER the gap -- an accident
# of the example I originally picked. **The guard worked on the shapes I had happened to try, and
# every IPv4 control passing had made me confident about a branch I had never run.**
#
# ⇒ Second alternative requires a literal `::`, which no clock time contains. Checked against C++
#   scope operators, the obvious false positive: `std::vector` (s not hex), `llama_context::decode`
#   (the boundary before `::` has no hex run starting at it), `foo::bar` (f is hex, o is not) all
#   stay clean, because the pattern needs 1-4 HEX characters immediately before the `::`.
PAT_IP6='\b([0-9a-fA-F]{1,4}:){3,}[0-9a-fA-F]{1,4}\b|\b[0-9a-fA-F]{1,4}::[0-9a-fA-F:]{0,32}'

# Allowed by the rule: documentation ranges and loopback. Everything else in those shapes is a hit.
ALLOW='127\.0\.0\.1|0\.0\.0\.0|203\.0\.113\.|198\.51\.100\.|192\.0\.2\.|255\.255\.255|2001:db8|::1\b'

HITS=$(printf '%s\n' "$ADDED" \
       | grep -nE "$PAT_PATH|$PAT_NAME|$PAT_IP4|$PAT_IP6" \
       | grep -vE "$ALLOW")
# ⚠ COUNT SEPARATELY FROM ANY PIPELINE THAT REFORMATS. The 2026-08-09 version let `sed`'s exit status
# decide the verdict. The variable is the verdict; nothing downstream may change it.
N=$(printf '%s' "$HITS" | grep -c . )

if [ "${N:-0}" -ne 0 ]; then
    echo "PRIVACY GUARD: REFUSING -- $N added line(s) match private-data patterns:" >&2
    printf '%s\n' "$HITS" | head -25 | cut -c1-200 | sed 's/^/  ! /' >&2
    echo "  Replace with the documentation values: 203.0.113.x, 10.0.x.x, 2001:db8::x," >&2
    echo "  home-server / relay.example.com, and a freshly generated fake peer ID." >&2
    exit 1
fi

# ⚠⚠ A PASS FROM AN INSTRUMENT THAT EXAMINED NOTHING IS THE FAILURE THIS PROJECT KEEPS REPEATING.
# On 2026-08-09 this guard printed **"privacy guard: PASS (0 added lines scanned)"** against a repo
# with real uncommitted changes: it defaults to `--cached`, nothing was staged yet, so it read an
# empty diff and passed. `exit 0` is what a caller's `&&` chain reads, so a guard that looked at zero
# bytes would have waved through whatever came next -- on the ONE rule in this project with zero
# tolerance and three history rewrites behind it.
#
# ⇒ Distinguish "nothing to check" from "checked and clean", and give them different exit codes:
#     empty diff        -> VOID (2). Wrong mode, wrong repo, or you forgot to `git add`.
#     diff, no + lines  -> PASS. A pure-deletion change genuinely cannot leak.
NADD=$(printf '%s\n' "$ADDED" | grep -c . )
NDIFF=$(printf '%s\n' "$DIFF" | grep -c . )
if [ "${NDIFF:-0}" -eq 0 ]; then
    echo "privacy guard: VOID -- the $MODE diff is EMPTY, so nothing was examined." >&2
    # ⚠ MODE-SPECIFIC ADVICE. The single staged-mode message was printed for a range whose refs did
    # not resolve, telling the reader to `git add` when the real problem was a nonexistent ref.
    if [ "$MODE" = range ]; then
        echo "  This is NOT a pass. The range '$RANGE' resolved but contains no commits -- nothing" >&2
        echo "  would be published by it. If you expected commits here, check the range direction." >&2
    else
        echo "  This is NOT a pass. Stage the change first (git add), or pass --worktree to include" >&2
        echo "  unstaged edits." >&2
    fi
    echo "  Exiting 2 so an && chain stops here instead of reading it as clean." >&2
    exit 2
fi
if [ "${NADD:-0}" -eq 0 ]; then
    echo "privacy guard: PASS (deletion-only change: $NDIFF diff lines, 0 added -- nothing can leak)"
    exit 0
fi
echo "privacy guard: PASS ($NADD added lines scanned, $MODE)"
exit 0
