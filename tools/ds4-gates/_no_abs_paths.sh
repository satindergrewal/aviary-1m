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
import sys, os
p = sys.argv[1]
s = open(p, errors='ignore').read()
s = s.replace(os.path.expanduser("~"), "$HOME")
open(p, 'w').write(s)
PY
}
