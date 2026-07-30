#!/usr/bin/env bash
# cram_growing.sh -- does the prompt cache serve a GROWING prompt (the real agentic pattern)?
#
# The claim under test came from a source read only: server_prompt_cache::load uses
# get_common_prefix, so turn N+1 should reuse turn N's cached prefix and prefill only the delta.
#
# CRITICAL DESIGN POINT: every growing turn is preceded by a DIFFERENT prompt that displaces the
# slot. Without that displacement the slot itself still holds the prefix and ordinary slot-level
# n_past continuation would serve the request -- we would be measuring the wrong mechanism and
# would wrongly credit the prompt cache.
#
#   T1  A1            (5k)   cold
#   D   OTHER                displaces the slot; A1 goes to the prompt cache
#   T2  A1+d           (10k)  <- prompt cache must supply the A1 prefix
#   D   OTHER                displaces again
#   T3  A1+d+d         (15k)  <- must supply the A1+d prefix
#
# Witness: timings.prompt_n on T2/T3. If prefix reuse works it is ~the delta, not the whole prompt.
set -u

BIN=~/Documents/GitHub/llama.cpp-dspark-metal/build/bin/llama-server
MODEL=~/AI/ktdev/qwen3-4b-dev-Q8_0.gguf
PORT=8793
OUT="$(dirname "$0")/growing_results"
mkdir -p "$OUT"; rm -f "$OUT"/req.jsonl

python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
def blk(tag, n, start=0):
    return " ".join(f"{tag}{i} item {i} value {i*7%97} node {i*13%89}." for i in range(start, start+n))
seg1 = blk("ALPHA", 300)            # the stable base
seg2 = blk("BETA",  300, 1000)      # turn-2 growth
seg3 = blk("GAMMA", 300, 2000)      # turn-3 growth
other = blk("OMEGA", 900, 5000)     # the displacing prompt, divergent from token 1
open(f"{out}/t1.txt","w").write("ALPHA. " + seg1)
open(f"{out}/t2.txt","w").write("ALPHA. " + seg1 + " " + seg2)
open(f"{out}/t3.txt","w").write("ALPHA. " + seg1 + " " + seg2 + " " + seg3)
open(f"{out}/other.txt","w").write("OMEGA. " + other)
PY

req () {  # req <promptfile> <label>
  python3 - "$1" "$2" "$OUT/req.jsonl" "$PORT" <<'PY'
import json, sys, time, urllib.request, urllib.error
pf, label, log, port = sys.argv[1:5]
body = json.dumps({"prompt": open(pf).read(), "n_predict": 4,
                   "temperature": 0.0, "cache_prompt": True}).encode()
t0 = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", body,
        {"Content-Type": "application/json"}), timeout=900)
    j = json.loads(r.read())
except urllib.error.HTTPError as e:
    print(f"FAILED {label}: HTTP {e.code} :: {e.read().decode()[:300]}", file=sys.stderr)
    sys.exit(3)
t = j["timings"]
rec = {"label": label, "wall_s": round(time.time()-t0, 3),
       "prompt_n": t["prompt_n"], "prompt_ms": round(t["prompt_ms"], 1),
       "total_tokens": j.get("tokens_evaluated")}
print(json.dumps(rec)); open(log, "a").write(json.dumps(rec)+"\n")
PY
}

"$BIN" -m "$MODEL" --port "$PORT" -c 32768 -np 1 -ngl 99 -cram 8192 -lv 4 --no-warmup \
  > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 1
done

fail=0
req "$OUT/t1.txt"    "T1_base_cold"   || fail=1
req "$OUT/other.txt" "D1_displace"    || fail=1
req "$OUT/t2.txt"    "T2_grown"       || fail=1
req "$OUT/other.txt" "D2_displace"    || fail=1
req "$OUT/t3.txt"    "T3_grown_more"  || fail=1

echo "--- cache lookup decisions (every candidate, -lv 4) ---"
grep -E "looking for better prompt|prompt with length|found better prompt|making room|exceeds cache" "$OUT/server.log" | tail -30
kill $PID 2>/dev/null; wait $PID 2>/dev/null
[ "$fail" = 1 ] && echo "!!! FAILED REQUESTS -- results NOT valid"

echo
echo "################ VERDICT ################"
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]+"/req.jsonl")]
print(f"{'label':<16} {'prompt_n':>9} {'total':>7} {'prefilled%':>11} {'prompt_ms':>10} {'wall':>8}")
for r in rows:
    tot = r["total_tokens"] or 0
    pct = 100.0*r["prompt_n"]/tot if tot else float('nan')
    print(f"{r['label']:<16} {r['prompt_n']:>9} {tot:>7} {pct:>10.1f}% {r['prompt_ms']:>10} {r['wall_s']:>7}s")
g = [r for r in rows if r["label"].startswith("T") and r["label"] != "T1_base_cold"]
print()
for r in g:
    tot = r["total_tokens"]
    print(f"{r['label']}: prefilled {r['prompt_n']} of {tot} tokens "
          f"=> reused {tot-r['prompt_n']} ({100.0*(tot-r['prompt_n'])/tot:.1f}% of the prompt)")
print("\nPREFIX REUSE ON A GROWING PROMPT: "
      + ("CONFIRMED" if all(r["prompt_n"] < 0.6*r["total_tokens"] for r in g) else "NOT OBSERVED"))
PY
