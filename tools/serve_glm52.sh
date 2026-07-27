#!/usr/bin/env bash
# GLM 5.2 (KT IQ1_S): the validated serve config. See docs/GLM-5.2-SERVE-CONFIG.md
# Run ON THE BOX. Usage: serve_glm52.sh [ctx] [n_parallel]
#   default:  65536 ctx x 4 slots  (validated 2026-07-28)
#   256K try: serve_glm52.sh 262144 1   (same KV bytes, single slot, UNVERIFIED)
set -euo pipefail

CTX="${1:-65536}"
NP="${2:-4}"
MODEL=/mnt/nvme0/bigmodels/glm52-ours/GLM-5.2-ours-IQ1_S-prot.gguf
REPO=/mnt/nvme0/llama.cpp-fleet          # branch fleet-sync @ ffee9f47e
LOG=/tmp/glm_serve.log

[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 1; }

cd "$REPO"
echo "build: $(git log -1 --format='%h %s')"
echo "ctx=$CTX slots=$NP  -> log $LOG"

# -ub 256 is load-bearing: the compute buffer scales with ubatch, not context.
# Raising it costs context. Do not "optimise" it upward.
exec ./build/bin/llama-server \
  -m "$MODEL" \
  -ngl 99 \
  --tensor-split 49,51 \
  -c "$CTX" \
  -np "$NP" \
  -b 1024 -ub 256 \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  --jinja \
  --samplers 'top_k;top_p;min_p;temperature;dry;typ_p;xtc' \
  --temp 0.7 \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 1 \
  --host 0.0.0.0 --port 8090 \
  >> "$LOG" 2>&1
