#!/usr/bin/env bash
# Shared helper: strip the absolute home path out of a result file.
#
# WHY: gates build $OUT from $HOME and then echo it, so every result file carried
# /Users/<username>/... . A username in a file path is private data under this project's rules and
# must never appear in a non-memory file. 23 tracked result files already carry it and 18 of those
# are in pushed history -- see the privacy report. This helper stops NEW ones being created.
#
# Use at the end of a gate:  scrub_abs_paths "$OUT"
scrub_abs_paths() {
    local f="$1"
    [ -f "$f" ] || return 0
    python3 - "$f" <<'PY'
import sys, os, re
p = sys.argv[1]
s = open(p, errors='ignore').read()
home = os.path.expanduser("~")
s = s.replace(home, "$HOME")
# ⚠ THE DASH-ENCODED FORM LEAKED WHAT THE SLASH FORM COULD NOT (found by the owner, 2026-08-11):
# Claude Code session directories encode the project path with dashes, so a scratchpad path like
# /private/tmp/claude-<uid>/-Users-<name>-Documents-... carries the username in a shape the
# replace() above never matches. Scrub the whole session-tmp prefix, and the dash-encoded home
# separately in case the prefix shape changes again -- two patterns because each has failed alone.
s = re.sub(r"/private/tmp/claude-[0-9]+/[^\s'\"]*", "<SCRATCH>", s)
dash_home = "-" + home.strip("/").replace("/", "-") + "-"   # "-Users-<name>-"
s = s.replace(dash_home, "-HOME-")
open(p, 'w').write(s)
PY
}
