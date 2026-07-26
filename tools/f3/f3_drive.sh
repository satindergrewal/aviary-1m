#!/usr/bin/env bash
# F3 real axis driver: same everything, only -c changes.
# #23658 claims -c 12032 => AR ~16%, -c 12288 => AR ~71%.
set -u
BIN=<BOX>/dspark-test/llama.cpp-dspark/build/bin/llama-server
TGT=<BOX>/dspark-test/targets/Qwen3-4B-Q8_0.gguf
DFT=<BOX>/dspark-test/head4.gguf
OUT=/tmp/f3_cvalue_results.txt
: > "$OUT"

run_one () {
  local CTX=$1 PORT=$2
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=1 nohup "$BIN" -m "$TGT" -md "$DFT" \
    --spec-type draft-dspark -ngl 99 -c "$CTX" -b 2048 -ub 512 \
    --host 127.0.0.1 --port "$PORT" > "/tmp/f3_c${CTX}.log" 2>&1 &
  # wait for readiness (max 180s)
  for i in $(seq 1 90); do
    if grep -q "listening on" "/tmp/f3_c${CTX}.log" 2>/dev/null; then break; fi
    sleep 2
  done
  if ! grep -q "listening on" "/tmp/f3_c${CTX}.log" 2>/dev/null; then
    echo "SERVER FAILED TO START at -c $CTX" >> "$OUT"
    tail -5 "/tmp/f3_c${CTX}.log" >> "$OUT"
    return 1
  fi
  sleep 2
  python3 /tmp/f3_cvalue.py "$PORT" "-c $CTX" >> "$OUT" 2>&1
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
}

run_one 12032 8660
run_one 12288 8661
echo "=== DONE ===" >> "$OUT"
