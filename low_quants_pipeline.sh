#!/bin/bash
# Wait for baked 1M bf16, compute imatrix, produce low-bit quants, upload each.
set -e
cd ~/Documents/GitHub/llama.cpp/models
BIN=~/Documents/GitHub/llama.cpp/build/bin
REPO=satgeze/Ornith-1.0-35B-1M-GGUF
BF16=ornith-1.0-35b-1M-bf16.gguf

until [ -f "${BF16}" ] && ! pgrep -f all_quants_pipeline >/dev/null; do sleep 120; done
echo "=== bf16 ready, computing imatrix from Q8_0 ==="
if [ ! -f calibration_data.txt ]; then
  curl -sL -o calibration_data.txt "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
fi
if [ ! -f ornith-1m.imatrix ]; then
  ${BIN}/llama-imatrix -m ornith-1.0-35b-1M-Q8_0.gguf -f calibration_data.txt \
    -o ornith-1m.imatrix -ngl 99 --chunks 96 2>&1 | tail -3
fi

for Q in IQ1_S IQ2_M IQ3_XXS Q2_K Q3_K_M IQ4_XS; do
  OUT="ornith-1.0-35b-1M-${Q}.gguf"
  if [ ! -f "${OUT}" ]; then
    echo "=== quantize ${Q} ==="
    ${BIN}/llama-quantize --imatrix ornith-1m.imatrix "${BF16}" "${OUT}" "${Q}" 2>&1 | tail -2
  fi
  echo "=== upload ${Q} ==="
  hf upload ${REPO} "${OUT}" "${OUT}" --repo-type model 2>&1 | tail -1
done
echo "ALL LOW QUANTS COMPLETE"
