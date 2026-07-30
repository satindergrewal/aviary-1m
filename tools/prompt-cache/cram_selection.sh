#!/usr/bin/env bash
# cram_selection.sh -- does the greedy `&&` in server_prompt_cache::load leave prefill on the table?
#
# server-task.cpp:1765:   if (f_keep_best < f_keep_cur && sim_best < sim_cur) { ...take it... }
# Both bests mutate during the scan, so a candidate that is BETTER on sim but WORSE on f_keep is
# skipped. I flagged this when recommending -cram 12288 (a bigger pool is what makes multiple
# entries coexist) but never tested it. Testing it before Satinder acts on the recommendation.
#
# CONSTRUCTING THE ADVERSARIAL CASE -- and note two designs that CANNOT work:
#   * two entries where one is a prefix of the other cannot coexist. alloc() refuses to save a
#     prompt already contained in an entry ("already in the cache, skipping"), and separately
#     erases entries fully contained in the new prompt. The cache keeps only the longest.
#   * so the short candidate must DIVERGE after its shared prefix, or it just vanishes.
#
# Let T be a long base text. Request R = T[:~7000].
#   B (saved FIRST, so it is the incumbent): T[:~5000] + divergent tail, len ~6000
#        -> lcp 5000, f_keep 5000/6000 = 0.83, sim 5000/7000 = 0.71
#   A (saved SECOND):                        T[:~20000], a superset of R
#        -> lcp 7000, f_keep 7000/20000 = 0.35, sim 7000/7000 = 1.00   <- covers ALL of R
#
# A is strictly better for this request (zero prefill) but loses the `&&` on f_keep.
#   PREDICT-BUG:     picks B -> prompt_n ~= 2000   (prefills 5000 -> 7000)
#   PREDICT-OPTIMAL: picks A -> prompt_n ~= 1      (whole prompt covered)
# Falsifier for my concern: prompt_n ~= 1. Then the selection is fine and I withdraw the caveat.
set -u

BIN=~/Documents/GitHub/llama.cpp-dspark-metal/build/bin/llama-server
MODEL=~/AI/ktdev/qwen3-4b-dev-Q8_0.gguf
PORT=8797
OUT="$(dirname "$0")/selection_results"
mkdir -p "$OUT"; rm -f "$OUT/req.jsonl"

python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
def blk(tag, n, start=0):
    return " ".join(f"{tag}{i} item {i} value {i*7%97} node {i*13%89}." for i in range(start, start+n))
base = blk("TEE", 1400)            # long base text T
words = base.split()
# R = a prefix of T ; A = all of T (superset of R) ; B = a shorter prefix of T that then DIVERGES
R = " ".join(words[:int(len(words)*0.35)])
A = base
B = " ".join(words[:int(len(words)*0.25)]) + " " + blk("ZED", 120, 8000)
D1 = "DEE1. " + blk("DEE1", 300, 20000)   # displacers, divergent from token 1
D2 = "DEE2. " + blk("DEE2", 300, 30000)
for n, t in (("R", R), ("A", A), ("B", B), ("D1", D1), ("D2", D2)):
    open(f"{out}/{n}.txt", "w").write(t)
PY

req () {
  python3 - "$1" "$2" "$OUT/req.jsonl" "$PORT" <<'PY'
import json, sys, urllib.request, urllib.error
pf, label, log, port = sys.argv[1:5]
body = json.dumps({"prompt": open(pf).read(), "n_predict": 2,
                   "temperature": 0.0, "cache_prompt": True}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", body,
        {"Content-Type": "application/json"}), timeout=900)
    j = json.loads(r.read())
except urllib.error.HTTPError as e:
    print(f"FAILED {label}: HTTP {e.code} :: {e.read().decode()[:300]}", file=sys.stderr); sys.exit(3)
t = j["timings"]
rec = {"label": label, "prompt_n": t["prompt_n"], "total": j.get("tokens_evaluated"),
       "prompt_ms": round(t["prompt_ms"], 1)}
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
echo "1) seed B (becomes the older/incumbent entry):"; req "$OUT/B.txt"  "seed_B"    || fail=1
echo "2) displace:";                                  req "$OUT/D1.txt" "displace1" || fail=1
echo "3) seed A (superset of R):";                    req "$OUT/A.txt"  "seed_A"    || fail=1
echo "4) displace:";                                  req "$OUT/D2.txt" "displace2" || fail=1
echo "5) THE TEST -- request R:";                      req "$OUT/R.txt"  "request_R" || fail=1

echo
echo "--- candidate evaluation for the final request (-lv 4) ---"
grep -E "looking for better prompt|prompt with length|found better prompt" "$OUT/server.log" | tail -12
kill $PID 2>/dev/null; wait $PID 2>/dev/null
[ "$fail" = 1 ] && echo "!!! FAILED REQUESTS -- NOT valid"

echo
echo "################ VERDICT ################"
python3 - "$OUT" <<'PY'
import json, sys
rows = {r["label"]: r for r in (json.loads(l) for l in open(sys.argv[1]+"/req.jsonl"))}
r = rows.get("request_R")
if not r: print("no data"); raise SystemExit
tot, pn = r["total"], r["prompt_n"]
print(f"request R: total={tot} tokens, prefilled={pn}, reused={tot-pn} ({100*(tot-pn)/tot:.1f}%)")
print()
if pn <= 4:
    print("=> picked the SUPERSET entry (A). Selection is optimal here.")
    print("   MY CAVEAT ABOUT THE && IS WITHDRAWN on this vehicle.")
else:
    print(f"=> did NOT reach full coverage: {pn} tokens prefilled that entry A could have supplied.")
    print("   Consistent with the greedy && skipping a strictly-better candidate.")
    print("   Read the candidate lines above for the deciding f_keep/sim pair.")
PY
