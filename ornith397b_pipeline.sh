#!/bin/bash
# Ornith-397B 1M priority quants: download bartowski split GGUFs, bake YaRN
# into shard 1, rename all shards to our naming, smoke-gate what fits locally,
# upload to satgeze/Ornith-1.0-397B-1M-GGUF.
set -e
WORK=~/Documents/GitHub/ornith-models/397b
export HF_HUB_ENABLE_HF_TRANSFER=1
SRC_REPO=bartowski/deepreinforce-ai_Ornith-1.0-397B-GGUF
DST_REPO=satgeze/Ornith-1.0-397B-1M-GGUF
BAKE=~/Documents/GitHub/ornith-1m/bake_yarn.py
GATE=~/Documents/GitHub/ornith-1m/smoke_gate.py
PY=~/Documents/GitHub/ornith-1m/.venv/bin/python
CLI=~/Documents/GitHub/llama.cpp/build/bin/llama-cli
mkdir -p "${WORK}"; cd "${WORK}"

for Q in IQ1_M IQ2_M Q4_K_M; do
  SRCDIR="deepreinforce-ai_Ornith-1.0-397B-${Q}"
  if ls ornith-1.0-397b-1M-${Q}-00001-*.gguf >/dev/null 2>&1; then
    echo "=== 397B ${Q}: already baked, skipping download/bake ==="
  else
  echo "=== 397B ${Q}: download ==="
  hf download "${SRC_REPO}" --include "${SRCDIR}/*" --local-dir . 2>&1 | tail -1
  N=$(ls ${SRCDIR}/*.gguf | wc -l | tr -d " ")
  echo "=== 397B ${Q}: bake shard 1 of ${N} + rename shards ==="
  for f in ${SRCDIR}/*.gguf; do
    base=$(basename "$f")
    part=$(echo "$base" | grep -oE "[0-9]{5}-of-[0-9]{5}")
    out="ornith-1.0-397b-1M-${Q}-${part}.gguf"
    if [[ "$part" == 00001-* ]]; then
      [ -f "${out}" ] || ${PY} ${BAKE} "$f" "${out}" 2>&1 | tail -1
    else
      [ -f "${out}" ] || mv "$f" "${out}"
    fi
  done
  fi
  # smoke gate on shard set if it plausibly fits local RAM (skip Q4_K_M: 242GB)
  if [ "${Q}" != "Q4_K_M" ]; then
    echo "=== 397B ${Q}: smoke gate ==="
    if ${PY} ${GATE} "ornith-1.0-397b-1M-${Q}-00001-of-$(printf %05d ${N}).gguf" "${CLI}" >> ~/Documents/GitHub/ornith-1m/smoke_results.log 2>&1; then
      echo "=== 397B ${Q}: PASS ==="
    else
      echo "=== 397B ${Q}: FAILED gate — uploading anyway=NO, quarantining ==="
      for f in ornith-1.0-397b-1M-${Q}-*.gguf; do mv "$f" "$f.quarantine"; done
      rm -rf "${SRCDIR}"; continue
    fi
  fi
  echo "=== 397B ${Q}: upload ==="
  for f in ornith-1.0-397b-1M-${Q}-*.gguf; do
    hf upload ${DST_REPO} "$f" "$f" --repo-type model 2>&1 | tail -1
  done
  rm -rf "${SRCDIR}"
  echo "=== 397B ${Q}: done ==="
done
echo "397B PRIORITY TRIO COMPLETE"
