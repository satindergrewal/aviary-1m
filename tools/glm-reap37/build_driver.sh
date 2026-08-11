#!/usr/bin/env bash
# REAP37 GLM-5.2 two-quant build driver.
#
# Usage (on the box, after GLM chain finishes or a card frees):
#   ./build_driver.sh <master_safetensors_dir> <refined_map.json> <out_safetensors_dir> <gguf_out_dir>
#
# Stage order (one command each so a failure stops early, no silent partial state):
#   1. subset: master safetensors -> REAP37 safetensors (bit-exact survivors)
#   2. convert: REAP37 safetensors -> GGUF BF16 (the standard converter, GLM arch in mainline)
#   3. imatrix: compute imatrix on a calibration corpus over the BF16 master (NOT the 4-bit MLX)
#   4. quantize: IQ2_KT + IQ3_XXS, same cell zero + imatrix
#
# Each stage echoes a SHA/exit-code marker; verify before the next. No GPU needed until
# quantize, and even imatrix can be CPU-only (slow but fits).
#
# REQUIRES that refined_map.json passed the off-by-one hill-climb (refine_map.py).
set -euo pipefail

MASTER_ST="$1"
MAP="$2"
OUT_ST="$3"
GGUF_OUT="$4"
CORPUS="${5:-/tmp/glm_calib.txt}"       # calibration corpus for imatrix; default small
IQUANTS=(IQ2_KT IQ3_XXS)
CONVERTER="${CONVERTER:-<BOX>/llama.cpp/build/bin/convert_hf_to_gguf.py}"
QUANTIZER="${QUANTIZER:-<BOX>/llama.cpp-kt/build/bin/llama-quantize}"
IMATRIX_BUILDER="${IMATRIX_BUILDER:-<BOX>/llama.cpp/build/bin/llama-imatrix}"
SUBSETTER=<BOX>/glm-reap37/reap_subset_glm.py

echo "=== [1/4] subset master -> REAP37 safetensors ==="
# NOTE: subsetter currently reads a LOCAL master; the master here must be a local
# safetensors tree (zai-org/GLM-5.2 downloaded). If the source-of-truth is the GGUF
# on <BOX>/glm52-bf16, convert that to safetensors first or adjust.
python3 "$SUBSETTER" "$MASTER_ST" "$MAP" "$OUT_ST"
echo "SUBSET_DONE=$(sha256sum "$OUT_ST/config.json" | cut -d' ' -f1)"

echo "=== [2/4] safetensors -> GGUF BF16 ==="
mkdir -p "$GGUF_OUT"
python3 "$CONVERTER" "$OUT_ST" --outfile "$GGUF_OUT/glm52-reap37-bf16.gguf" --outtype bf16
echo "GGUF_DONE=$(stat -c%s "$GGUF_OUT/glm52-reap37-bf16.gguf") bytes"

echo "=== [3/4] imatrix over BF16 master ==="
# imatrix is computed from a text corpus against the model; run on CPU if no GPU window.
"$IMATRIX_BUILDER" -m "$GGUF_OUT/glm52-reap37-bf16.gguf" -f "$CORPUS" -o "$GGUF_OUT/glm52-reap37.imatrix" --show-statistics
echo "IMATRIX_DONE=$(stat -c%s "$GGUF_OUT/glm52-reap37.imatrix") bytes"

echo "=== [4/4] quantize IQ2_KT + IQ3_XXS ==="
for q in "${IQUANTS[@]}"; do
    "$QUANTIZER" "$GGUF_OUT/glm52-reap37-bf16.gguf" "$GGUF_OUT/glm52-reap37-${q}.gguf" "$q" \
        --imatrix "$GGUF_OUT/glm52-reap37.imatrix"
    echo "QUANT_${q}_DONE=$(stat -c%s "$GGUF_OUT/glm52-reap37-${q}.gguf") bytes"
done

echo "BUILD COMPLETE: $GGUF_OUT"