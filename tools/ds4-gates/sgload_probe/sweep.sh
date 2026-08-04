#!/usr/bin/env bash
# NOTE: bash, and args passed as "$@" -- NOT an unquoted $var. zsh does not word-split
# unquoted parameter expansions, so a `for cfg in "a b c"; do ./p $cfg` loop silently runs
# the SAME config every time with only the first token applied. That produced six identical
# rows that looked like "the error is independent of n_tok" -- a fake sweep.
cd "$(dirname "$0")"
run() { printf '%-26s ' "[$*]"; ./fa_paged "$@" | grep worst | sed 's/^ *//'; }
run 64 8 3 8   32 1     # one tile, one block  -> isolates Q@K^T + P@V, no online rescale
run 64 8 3 32  32 1     # one block, full tile
run 64 8 3 32  32 4     # one block, 4 simdgroups
run 64 8 3 40  32 4     # TWO blocks -> online rescale now active
run 64 8 3 96  32 4     # three blocks
run 8  8 3 32  32 1     # tiny head dim -> half-precision error nearly vanishes
run 128 8 3 40 32 4     # larger head dim
