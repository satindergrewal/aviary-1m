#!/usr/bin/env bash
# DS4 Q@K^T single-tile contract gate. Exit status comes from the BINARY, never a pipe.
set -uo pipefail
cd "$(dirname "$0")"
[ -x ./qk_tile ] || clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal qk_tile.mm -o qk_tile
fail=0
arm() { # arm <name> <expect> <D> <n_heads> <head> <ss_pitch> <bad>
  local name="$1" expect="$2"; shift 2
  local out; out=$(./qk_tile "$@"); local rc=$?
  local got; [ $rc -eq 0 ] && got=pass || got=fail
  # NOTE: the verdict word must NOT be computed inside $( ) -- a command substitution is a
  # SUBSHELL and `fail=1` set in there is DISCARDED. An earlier revision of this file printed
  # "GATE PASS" with a MISMATCH row on screen for exactly that reason. Same family as the
  # banked `grep -c` and pipe-status incidents: the status must cross no subshell boundary.
  local mark=OK
  if [ "$got" != "$expect" ]; then mark=MISMATCH; fail=1; fi
  printf '%-38s expect=%-4s got=%-4s %s\n' "$name" "$expect" "$got" "$mark"
  echo "$out" | tail -1 | sed 's/^/      /'
}
echo "=== DS4 Q@K^T single-tile contract gate ==="
arm "D=64 h=8 head=3 (baseline)"        pass 64  8  3 8  0
arm "  same, q_pitch=D  (THE BUG)"      fail 64  8  3 8  1
arm "D=64 head=0 (first head)"          pass 64  8  0 8  0
arm "D=64 head=7 (last head)"           pass 64  8  7 8  0
arm "D=128 (larger head_dim)"           pass 128 8  5 8  0
arm "D=64 n_heads=32 (GQA-ish)"         pass 64  32 17 8 0
arm "store pitch 32 (independent)"      pass 64  8  3 32 0
arm "store pitch 40 (non-power-of-2)"   pass 64  8  3 40 0
arm "D=256 (big head)"                  pass 256 8  1 8  0
echo "=== $( [ $fail -eq 0 ] && echo 'GATE PASS -- Q@K^T tile contract confirmed' || echo 'GATE FAIL' ) ==="
exit $fail
