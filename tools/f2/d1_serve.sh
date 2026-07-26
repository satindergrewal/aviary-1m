#!/usr/bin/env bash
# D1 measure: serve one arm on GPU1 and capture acceptance + greedy text.
# usage: d1_serve.sh <gguf> <tag> <port> <spec-type>
set -u
BIN=/mnt/data/dspark-test/llama.cpp-dspark/build/bin/llama-server
M=$1; TAG=$2; PORT=$3; SPEC=$4
OUT=/mnt/data/d1

pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3

if [ "$SPEC" = "none" ]; then
  SPECARGS="--spec-type none"
else
  SPECARGS="--spec-type $SPEC"
fi

CUDA_VISIBLE_DEVICES=1 nohup "$BIN" -m "$M" $SPECARGS \
  -ngl 99 -c 4096 -b 2048 -ub 512 --host 127.0.0.1 --port "$PORT" \
  > "$OUT/serve_$TAG.log" 2>&1 &

for i in $(seq 1 150); do
  grep -q "listening on" "$OUT/serve_$TAG.log" 2>/dev/null && break
  sleep 2
done

if ! grep -q "listening on" "$OUT/serve_$TAG.log" 2>/dev/null; then
  echo "SERVER FAILED for $TAG" >> "$OUT/measure.log"
  tail -8 "$OUT/serve_$TAG.log" >> "$OUT/measure.log"
  exit 1
fi
sleep 2

echo "=== $TAG ($SPEC) ===" >> "$OUT/measure.log"
python3 /tmp/d1_measure.py "$PORT" "$TAG" "$OUT/res_$TAG.json" >> "$OUT/measure.log" 2>&1

pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
echo "--- $TAG complete ---" >> "$OUT/measure.log"
