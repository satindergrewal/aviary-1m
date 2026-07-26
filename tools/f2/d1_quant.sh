#!/usr/bin/env bash
# D1: three arms differing ONLY in how the MTP block (blk.64) is quantized.
#   arm1  blk.64 pinned q8_0                  = what publishers do today
#   arm2  blk.64 q4_K with NO imatrix data    = default behaviour without the F2 patch
#   arm3  blk.64 q4_K WITH imatrix data       = what the F2 patch enables
# Everything else identical: same bf16 source, same Q4_K_M target, same 20-chunk imatrix
# for all non-MTP layers.
set -u
Q=/mnt/data/dspark-test/llama.cpp-dspark/build/bin/llama-quantize
SRC=/mnt/data/dspark-test/targets/Qwen3.6-27B-bf16.gguf
OUT=/mnt/data/d1
IM_MTP=/tmp/imat_mtp20.gguf      # includes blk.64 entries
IM_NOMTP=/tmp/imat_nomtp20.gguf  # identical corpus/chunks, no blk.64 entries
LOG=$OUT/quant.log
mkdir -p "$OUT"
: > "$LOG"

run () {
  local NAME=$1; shift
  echo "=== $NAME ===" >> "$LOG"
  /usr/bin/time -f "%e s" "$@" >> "$LOG" 2>&1
  echo "--- exit $? ---" >> "$LOG"
}

# the MTP block is pushed to q2_K in arms 2 and 3 on purpose: at q4_K the two arms would
# likely be indistinguishable and the comparison would prove nothing. q2_K is where missing
# calibration data should actually show up.

# arm1: the q8_0 pin (today's practice, the acceptance ceiling)
run arm1 "$Q" --imatrix "$IM_MTP" --tensor-type "blk\.64\..*=q8_0" \
    "$SRC" "$OUT/arm1_mtp_q8.gguf" Q4_K_M 16

# arm3: q2_K WITH real MTP data (the interesting arm, so it lands first)
run arm3 "$Q" --imatrix "$IM_MTP" --tensor-type "blk\.64\..*=q2_K" \
    "$SRC" "$OUT/arm3_mtp_q2_data.gguf" Q4_K_M 16

# arm2: q2_K with NO MTP data
if [ -f "$IM_NOMTP" ]; then
  run arm2 "$Q" --imatrix "$IM_NOMTP" --tensor-type "blk\.64\..*=q2_K" \
      "$SRC" "$OUT/arm2_mtp_q2_nodata.gguf" Q4_K_M 16
else
  echo "=== arm2 SKIPPED: $IM_NOMTP not present ===" >> "$LOG"
fi

echo "=== SIZES ===" >> "$LOG"
ls -la "$OUT"/*.gguf >> "$LOG" 2>&1
echo "=== DONE ===" >> "$LOG"
