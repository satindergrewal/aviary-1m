#!/usr/bin/env bash
# F3 on the ISSUE'S OWN SPEC PATH: --spec-type draft-mtp (in-model nextn head, no -md).
# Qwen3.6-27B-Q4_K_M carries 4 nextn tensors. Only -c differs between runs.
# #23658 claims -c 12032 => AR ~16%, -c 12288 => AR ~71%.
set -u
BIN=<BOX>/dspark-test/llama.cpp-dspark/build/bin/llama-server
TGT=<BOX>/dspark-test/targets/Qwen3.6-27B-Q4_K_M.gguf
OUT=/tmp/f3_mtp_results.txt
: > "$OUT"

run_one () {
  local CTX=$1 PORT=$2
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=1 nohup "$BIN" -m "$TGT" \
    --spec-type draft-mtp -ngl 99 -c "$CTX" -b 2048 -ub 512 \
    --host 127.0.0.1 --port "$PORT" > "/tmp/f3_mtp_${CTX}.log" 2>&1 &
  for i in $(seq 1 120); do
    grep -q "listening on" "/tmp/f3_mtp_${CTX}.log" 2>/dev/null && break
    sleep 2
  done
  if ! grep -q "listening on" "/tmp/f3_mtp_${CTX}.log" 2>/dev/null; then
    echo "SERVER FAILED TO START at -c $CTX" >> "$OUT"
    tail -8 "/tmp/f3_mtp_${CTX}.log" >> "$OUT"
    return 1
  fi
  sleep 2
  python3 /tmp/f3_cvalue.py "$PORT" "MTP -c $CTX" >> "$OUT" 2>&1
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
}

run_one 12032 8670
run_one 12288 8671
echo "=== DONE ===" >> "$OUT"
