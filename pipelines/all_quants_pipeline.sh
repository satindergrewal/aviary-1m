#!/bin/bash
# Download remaining Ornith quants, bake YaRN 1M metadata, upload to HF.
set -e
cd ~/Documents/GitHub/llama.cpp/models
BAKE=~/Documents/GitHub/ornith-1m/bake_yarn.py
PY=~/Documents/GitHub/ornith-1m/.venv/bin/python
BASE=https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF/resolve/main
REPO=satgeze/Ornith-1.0-35B-1M-GGUF

for Q in Q5_K_M Q6_K bf16; do
  SRC="ornith-1.0-35b-${Q}.gguf"
  OUT="ornith-1.0-35b-1M-${Q}.gguf"
  echo "=== ${Q}: download ==="
  curl -sL -C - -o "${SRC}" "${BASE}/${SRC}"
  echo "=== ${Q}: bake ==="
  ${PY} ${BAKE} "${SRC}" "${OUT}" 2>&1 | tail -1
  echo "=== ${Q}: upload ==="
  hf upload ${REPO} "${OUT}" "${OUT}" --repo-type model 2>&1 | tail -1
  echo "=== ${Q}: done, removing source download ==="
  rm -f "${SRC}"
done
echo "ALL QUANTS COMPLETE"
