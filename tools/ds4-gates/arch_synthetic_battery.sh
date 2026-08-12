#!/bin/bash
S=${S:?set S to a scratch dir containing arch-fixtures/ and arch-prompt.json}
B=${B:?set B to the llama-server binary}
G=$S/arch-fixtures
run_arm() { # gguf tag extra-args...
  local M=$1 TAG=$2; shift 2
  "$B" -m "$M" -ngl 99 -c 128 -np 1 -b 32 -ub 32 "$@" --port 21596 --no-warmup > "$G/$TAG.log" 2>&1 &
  local i; for i in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:21596/health 2>/dev/null)" = "200" ] && break; sleep 1; done
  curl -s --max-time 120 -X POST http://127.0.0.1:21596/completion -H 'Content-Type: application/json' --data-binary "@$S/arch-prompt.json" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(json.dumps(d.get('tokens',[])))
except Exception as e: print('[\"ERR\"]')"
  pkill -x llama-server 2>/dev/null; sleep 1
}
for f in "$G"/*.gguf; do
  A=$(basename "$f" .gguf)
  ST=$(run_arm "$f" "$A-static")
  P16=$(run_arm "$f" "$A-p16" --kv-paged --kv-block-size 16 -ngpub 12 -ncpub 4)
  P64=$(DS4P_METAL_CHAMP=1 DS4P_CHAMP_COUNT=1 run_arm "$f" "$A-p64" --kv-paged --kv-block-size 64 -ngpub 4 -ncpub 2)
  V16="DIFF"; [ "$ST" = "$P16" ] && V16="MATCH"
  V64="DIFF"; [ "$ST" = "$P64" ] && V64="MATCH"
  # VACUOUS-GREEN GUARD: an empty or error token array cannot certify anything.
  [ "$ST" = "[]" ] || [ "$ST" = '["ERR"]' ] && { V16="VOID"; V64="VOID"; }
  FB16=$(grep -ac 'fails the paged capability\|STATIC path' "$G/$A-p16.log" 2>/dev/null)
  FB64=$(grep -ac 'fails the paged capability\|STATIC path' "$G/$A-p64.log" 2>/dev/null)
  CH64=$(grep -ac 'DS4P-CHAMPN' "$G/$A-p64.log" 2>/dev/null)
  echo "$A | p16:$V16(fallback:$FB16) | p64+champ:$V64(fallback:$FB64,champN:$CH64) | static:$(echo $ST | head -c 40)"
done
