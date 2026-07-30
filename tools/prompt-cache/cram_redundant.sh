#!/usr/bin/env bash
# cram_redundant.sh -- is an UNCHANGED idle slot re-copied on every task launch?
#
# server-context.cpp:2393 loops over all slots on each task launch and calls prompt_save()
# for every non-processing one, with NO dirty-check. And server_prompt_cache::alloc() first
# erases any cached entry "fully contained in the current prompt" -- an identical entry
# qualifies -- and then copies the state again.
#
# Prediction if that reading is right: park a big prompt in one slot, then fire N unrelated
# requests, and the log shows "saving prompt with length <same>" N times, i.e. N redundant
# multi-hundred-MiB copies of a state that never changed.
#
# Falsifier: it appears once. Then some dirty-check or containment guard I missed is working.
set -u

BIN=~/Documents/GitHub/llama.cpp-dspark-metal/build/bin/llama-server
MODEL=~/AI/ktdev/qwen3-4b-dev-Q8_0.gguf
PORT=8794
OUT="$(dirname "$0")/redundant_results"
mkdir -p "$OUT"

python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
def blk(tag, n, start=0):
    return " ".join(f"{tag}{i} item {i} value {i*7%97}." for i in range(start, start+n))
open(f"{out}/park.txt","w").write("PARKED. " + blk("PARK", 400))       # sits idle in a slot
for k in range(4):                                                      # 4 unrelated launches
    open(f"{out}/other{k}.txt","w").write(f"OTHER{k}. " + blk(f"OTH{k}", 250, 9000+k*1000))
PY

req () {
  python3 - "$1" "$PORT" <<'PY'
import json, sys, urllib.request
pf, port = sys.argv[1], sys.argv[2]
body = json.dumps({"prompt": open(pf).read(), "n_predict": 2,
                   "temperature": 0.0, "cache_prompt": True}).encode()
r = urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/completion", body,
    {"Content-Type": "application/json"}), timeout=600)
t = json.loads(r.read())["timings"]
print(f"  {pf.split('/')[-1]:<12} prompt_n={t['prompt_n']}")
PY
}

# -np 2 so one slot can stay idle while the other works
"$BIN" -m "$MODEL" --port "$PORT" -c 16384 -np 2 -ngl 99 -cram 8192 -lv 4 --no-warmup \
  > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 1
done

echo "1) park a prompt in a slot:"
req "$OUT/park.txt"
echo "2) four unrelated requests, each a fresh task launch:"
for k in 0 1 2 3; do req "$OUT/other$k.txt"; done

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo
echo "################ VERDICT ################"
echo "--- every prompt_save in the run ---"
grep -E "saving prompt with length" "$OUT/server.log" | sed 's/^.*prompt_save/  /'
echo
PARKLEN=$(grep -oE "saving prompt with length [0-9]+" "$OUT/server.log" \
          | awk '{print $NF}' | sort | uniq -c | sort -rn | head -1)
echo "most-repeated saved length (count, length): $PARKLEN"
N=$(echo "$PARKLEN" | awk '{print $1}')
echo
if [ "${N:-0}" -gt 1 ]; then
  echo "REDUNDANT RE-SAVE: CONFIRMED -- the same unchanged state was copied $N times"
  echo "--- and were the duplicates erased first? ---"
  grep -cE "making room|erasing|removing" "$OUT/server.log" | sed 's/^/  evict-ish lines: /'
else
  echo "REDUNDANT RE-SAVE: NOT OBSERVED -- a guard I missed is working. Claim withdrawn."
fi
