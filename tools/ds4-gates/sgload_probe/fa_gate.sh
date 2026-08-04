#!/usr/bin/env bash
# DS4 paged flash-attention MMA gate.
#  - bash, args via "$@": zsh does not word-split unquoted $var, which once produced six
#    identical rows that looked like a real sweep.
#  - the pass/fail flag never crosses a subshell: `fail=1` inside $( ) is discarded, which
#    once printed GATE PASS with a MISMATCH row on screen.
#  - exit status comes from the binary, never from a pipe tail.
set -uo pipefail
cd "$(dirname "$0")"
[ -x ./fa_paged ] || clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal fa_paged.mm -o fa_paged
fail=0; ran=0; skipped=0
run() {
  local out; out=$(./fa_paged "$@"); local rc=$?
  local mark
  case $rc in
    0)  mark=PASS; ran=$((ran+1)) ;;
    77) mark=SKIP-smem; skipped=$((skipped+1)) ;;
    *)  mark=FAIL; fail=1; ran=$((ran+1)) ;;
  esac
  printf '%-24s %-10s %s\n' "[$*]" "$mark" "$(echo "$out" | grep -E 'vs exact|SKIP' | sed 's/^ *//')"
}
echo "=== DS4 paged flash-attention MMA gate (D n_heads head n_tok bs nsg) ==="
run 64  8  3 8   32 1     # one tile, one block   -- Q@K^T + P@V only
run 64  8  3 32  32 1     # one block, full tile
run 64  8  3 32  32 4     # one block, 4 simdgroups
run 64  8  3 40  32 4     # two blocks   -- online rescale active
run 64  8  3 96  32 4     # three blocks -- long rescale chain
run 64  8  3 100 32 4     # ragged tail, not a multiple of QR
run 64  8  0 40  32 4     # first head
run 64  8  7 40  32 4     # last head
run 64  32 17 40 32 4     # many heads (GQA-ish Q stride)
run 8   8  3 32  32 1     # tiny head dim
run 128 8  3 40  16 2     # large head dim, tile resized to fit smem
run 64  8  3 40  16 4     # smaller paged block size
run 64  8  3 40  32 8     # 8 simdgroups
echo "=== $( [ $fail -eq 0 ] && echo "GATE PASS -- $ran ran, $skipped skipped" || echo 'GATE FAIL' ) ==="
exit $fail
