#!/usr/bin/env bash
# DS4 simdgroup_load pitch gate. Exit codes come from the PROBE, never from a pipe tail
# (an earlier read of this same probe printed rc=0 for a FAILING arm because $? was tail's).
set -uo pipefail
cd "$(dirname "$0")"
[ -x ./probe ] || ./build.sh >/dev/null
fail=0
arm() {  # arm <name> <expect pass|fail> <SRC_W> <LOAD> <STORE>
  local name="$1" expect="$2"; shift 2
  local out; out=$(./probe "$@"); local rc=$?
  local got; [ $rc -eq 0 ] && got=pass || got=fail
  printf '%-34s %-5s expect=%-4s got=%-4s %s\n' "$name" "[$*]" "$expect" "$got" \
         "$( [ "$got" = "$expect" ] && echo OK || { echo MISMATCH; fail=1; } )"
  echo "$out" | tail -2 | sed 's/^/      /'
}
echo "=== DS4 simdgroup_load pitch gate ==="
arm "control: pitch == row stride"  pass 8  8   8
arm "the 2*D bug (4-of-8 residual)" fail 8  16  8
arm "store pitch independent"       pass 8  8   16
arm "head_dim=64, correct pitch"    pass 64 64  8
arm "head_dim=64, 2*D bug"          fail 64 128 8
arm "hd=64, both pitches differ"    pass 64 64  32
echo "=== $( [ $fail -eq 0 ] && echo 'GATE PASS -- pitch contract confirmed' || echo 'GATE FAIL' ) ==="
exit $fail
