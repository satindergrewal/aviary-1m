#!/bin/bash
# Ornith-1.0-9B → 1M quant ladder. Waits for the 35B pipelines to finish,
# moves 35B artifacts to ~/Documents/GitHub/ornith-models/, then builds the
# 9B ladder there: 5 official quants baked + imatrix low-bit set. Uploads all
# to satgeze/Ornith-1.0-9B-1M-GGUF.
set -e
MODELS=~/Documents/GitHub/ornith-models
OLD=~/Documents/GitHub/llama.cpp/models
BIN=~/Documents/GitHub/llama.cpp/build/bin
BAKE=~/Documents/GitHub/ornith-1m/bake_yarn.py
PY=~/Documents/GitHub/ornith-1m/.venv/bin/python
BASE=https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF/resolve/main
REPO=satgeze/Ornith-1.0-9B-1M-GGUF

# wait for both 35B pipelines
until ! pgrep -f all_quants_pipeline >/dev/null && ! pgrep -f low_quants_pipeline >/dev/null; do sleep 300; done
grep -q "ALL LOW QUANTS COMPLETE" ~/Documents/GitHub/ornith-1m/low_quants.log || echo "WARN: low-quant pipeline did not finish cleanly — check low_quants.log"

echo "=== moving 35B artifacts to ${MODELS} ==="
mkdir -p "${MODELS}"
mv -n "${OLD}"/ornith-*.gguf "${MODELS}/" 2>/dev/null || true
mv -n "${OLD}"/ornith-1m.imatrix "${MODELS}/" 2>/dev/null || true
mv -n "${OLD}"/calibration_data.txt "${MODELS}/" 2>/dev/null || true
cd "${MODELS}"

hf repo create "${REPO}" --repo-type model --private 2>/dev/null || true

for Q in Q4_K_M Q5_K_M Q6_K Q8_0 bf16; do
  SRC="ornith-1.0-9b-${Q}.gguf"
  OUT="ornith-1.0-9b-1M-${Q}.gguf"
  echo "=== 9B ${Q}: download/bake/upload ==="
  [ -f "${OUT}" ] || { curl -sL -C - -o "${SRC}" "${BASE}/${SRC}"; ${PY} ${BAKE} "${SRC}" "${OUT}" 2>&1 | tail -1; rm -f "${SRC}"; }
  hf upload ${REPO} "${OUT}" "${OUT}" --repo-type model 2>&1 | tail -1
done

echo "=== 9B imatrix ==="
[ -f ornith-9b-1m.imatrix ] || ${BIN}/llama-imatrix -m ornith-1.0-9b-1M-Q8_0.gguf \
  -f calibration_data.txt -o ornith-9b-1m.imatrix -ngl 99 --chunks 96 2>&1 | tail -2

for Q in IQ2_M IQ3_XXS Q2_K Q3_K_M IQ4_XS; do
  OUT="ornith-1.0-9b-1M-${Q}.gguf"
  echo "=== 9B ${Q}: quantize ==="
  [ -f "${OUT}" ] || ${BIN}/llama-quantize --imatrix ornith-9b-1m.imatrix \
    ornith-1.0-9b-1M-bf16.gguf "${OUT}" "${Q}" 2>&1 | tail -1
  echo "=== 9B ${Q}: smoke gate ==="
  if ~/Documents/GitHub/ornith-1m/.venv/bin/python ~/Documents/GitHub/ornith-1m/smoke_gate.py "${OUT}" "${BIN}/llama-cli" >> ~/Documents/GitHub/ornith-1m/smoke_results.log 2>&1; then
    echo "=== 9B ${Q}: PASS — uploading ==="
    hf upload ${REPO} "${OUT}" "${OUT}" --repo-type model 2>&1 | tail -1
  else
    echo "=== 9B ${Q}: FAILED smoke gate — NOT uploading (kept locally for inspection) ==="
    mv "${OUT}" "${OUT}.quarantine"
  fi
done
tail -5 ~/Documents/GitHub/ornith-1m/smoke_results.log
echo "9B LADDER COMPLETE"
