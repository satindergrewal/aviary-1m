#!/usr/bin/env bash
# Does KT trellis quantization scale with threads, or is it memory-bound?
#
# Decides whether restarting the 37h IQ2_KT run at a higher thread count is
# worth throwing away the work already done. Run each arm for a FIXED wall
# time and compare bytes processed - no need to finish either.
#
# Uses only the cores IQ2_KT is not using (it holds 16 of 32).
set -u
BIN=${BOX_MNT:?set BOX_MNT to the box storage root}/llama.cpp-k3kt/build-cuda/bin/llama-quantize
SRC=${BOX_MNT:?set BOX_MNT to the box storage root}/ablit-test/Qwen3-4B-abliterated-Q8_0.gguf
SECS=${SECS:-150}

run_arm() {
  local nthreads=$1
  local out=/tmp/scaletest_${nthreads}.gguf
  local log=/tmp/scaletest_${nthreads}.log
  rm -f "$out" "$log"
  nice -n 15 "$BIN" --allow-requantize "$SRC" "$out" IQ2_KT "$nthreads" > "$log" 2>&1 &
  local pid=$!
  sleep "$SECS"
  kill -9 $pid 2>/dev/null
  wait $pid 2>/dev/null
  # sum the source MiB of every tensor that COMPLETED
  local mib
  mib=$(grep -oE "size =[ ]+[0-9.]+ MiB" "$log" | grep -oE "[0-9.]+" | paste -sd+ - | bc 2>/dev/null)
  [ -z "$mib" ] && mib=0
  local ntensors
  ntensors=$(grep -cE "^\[ *[0-9]+/" "$log")
  echo "$nthreads $mib $ntensors"
  rm -f "$out"
}

echo "arm  threads   MiB_done   tensors   (${SECS}s each)"
A=$(run_arm 4)
echo "  A: $A"
B=$(run_arm 12)
echo "  B: $B"

python3 - "$A" "$B" <<'PYEOF'
import sys
a = sys.argv[1].split(); b = sys.argv[2].split()
t1, m1 = int(a[0]), float(a[1])
t2, m2 = int(b[0]), float(b[1])
print()
print(f"{t1:>2} threads : {m1:8.1f} MiB")
print(f"{t2:>2} threads : {m2:8.1f} MiB")
if m1 > 0:
    speedup = m2/m1
    ideal   = t2/t1
    print(f"speedup   : {speedup:.2f}x  (ideal {ideal:.2f}x)  = {speedup/ideal*100:.0f}% efficiency")
    print()
    if speedup/ideal > 0.75:
        print("=> COMPUTE-bound. More threads help. Restarting the big run at a")
        print("   higher thread count is likely worth it.")
    elif speedup/ideal < 0.45:
        print("=> MEMORY-bound. More threads buy little. Do NOT restart.")
    else:
        print("=> PARTIAL scaling. Compute the tradeoff explicitly before restarting.")
PYEOF
