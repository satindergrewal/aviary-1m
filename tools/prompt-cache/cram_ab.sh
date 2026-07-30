#!/usr/bin/env bash
# cram_ab.sh -- does --cache-ram actually gate prompt-state reuse, and what does it cost?
#
# Protocol (3 requests, one slot):
#   1. long prompt A  -> slot holds A
#   2. long prompt B  -> slot must drop A; A is offered to the prompt cache HERE
#   3. long prompt A  -> restored from cache iff step 2's save was accepted
#
# Witness: timings.prompt_n on request 3.
#   cache HIT  -> prompt_n small (only the uncached tail is processed)
#   cache MISS -> prompt_n ~= full prompt length (everything re-prefilled)
# A and B are made to differ from token 1 so slot-level prefix matching cannot confound.
set -u

BIN=~/Documents/GitHub/llama.cpp-dspark-metal/build/bin/llama-server
MODEL=~/AI/ktdev/qwen3-4b-dev-Q8_0.gguf
PORT=8791
OUT="$(dirname "$0")/cram_results"
mkdir -p "$OUT"

[ -x "$BIN" ]   || { echo "FATAL: no binary $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: no model $MODEL"; exit 1; }

# ---- deterministic long prompts, divergent at token 1 -----------------------
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def mk(tag, n):
    # ~n words of deterministic, non-degenerate filler; unique leading token per arm
    parts = [f"{tag}."]
    for i in range(n):
        parts.append(f"{tag}{i} item {i} value {i*7%97} node {i*13%89} edge {i*29%83}.")
    return " ".join(parts)
for tag, name in (("ALPHA", "A"), ("BRAVO", "B")):
    open(f"{out}/prompt_{name}.txt", "w").write(mk(tag, 900))
PY
wc -w "$OUT"/prompt_A.txt "$OUT"/prompt_B.txt

req () {  # req <promptfile> <label> <logfile> ; returns non-zero on ANY http error
  local pf="$1" label="$2" log="$3"
  python3 - "$pf" "$label" "$log" "$PORT" <<'PY'
import json, sys, time, urllib.request, urllib.error
pf, label, log, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = json.dumps({
    "prompt": open(pf).read(),
    "n_predict": 8, "temperature": 0.0, "cache_prompt": True,
}).encode()
t0 = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", body,
        {"Content-Type": "application/json"}), timeout=900)
    j = json.loads(r.read())
except urllib.error.HTTPError as e:
    # LOUD: a 400 must not be mistaken for a completed arm
    print(f"REQUEST FAILED {label}: HTTP {e.code} :: {e.read().decode()[:400]}",
          file=sys.stderr)
    sys.exit(3)
wall = time.time() - t0
t = j.get("timings", {})
rec = {
    "label": label, "wall_s": round(wall, 3),
    "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
    "cache_n": j.get("tokens_evaluated"),
    "prompt_per_second": t.get("prompt_per_second"),
}
print(json.dumps(rec))
open(log, "a").write(json.dumps(rec) + "\n")
PY
}

run_arm () {  # run_arm <cram_value> <armname>
  local cram="$1" arm="$2"
  local slog="$OUT/server_$arm.log" rlog="$OUT/req_$arm.jsonl"
  rm -f "$rlog"
  echo "=========================================================="
  echo "ARM $arm : -cram $cram"
  echo "=========================================================="
  # -c 32768: prompts tokenize to ~19-20K (digit-heavy filler), measured, not assumed.
  # -lv 4: KV-size and prompt-cache lines are invisible below this. Learned the hard way.
  "$BIN" -m "$MODEL" --port "$PORT" -c 32768 -np 1 -ngl 99 \
         -cram "$cram" -lv 4 --no-warmup > "$slog" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/pid_$arm"
  # wait for health, bounded
  local ok=0
  for i in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || { echo "server died during load"; break; }
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "ARM $arm FAILED to come up; tail:"; tail -20 "$slog"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
  fi

  local failed=0
  req "$OUT/prompt_A.txt" "1_A_cold"   "$rlog" || failed=1
  req "$OUT/prompt_B.txt" "2_B_evicts" "$rlog" || failed=1
  req "$OUT/prompt_A.txt" "3_A_reuse"  "$rlog" || failed=1

  echo "--- KV geometry (the state size denominator) ---"
  grep -Ei "KV self size|kv_unified|n_ctx +=|n_ctx_per_seq" "$slog" | head -6
  echo "--- prompt-cache lines from server log ---"
  grep -Ei "prompt cache|cache size limit|making room|state (save|load|restore)|checkpoint|reusing|cache_tokens" "$slog" | tail -25

  if [ "$failed" = 1 ]; then
    echo "!!! ARM $arm HAD FAILED REQUESTS -- results are NOT valid"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 2
}

run_arm 1    "cram1"     # 1 MiB cap  -> save must be refused
run_arm 8192 "cram8192"  # shipped default -> save should be accepted

echo
echo "################ VERDICT ################"
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for arm in ("cram1", "cram8192"):
    p = f"{out}/req_{arm}.jsonl"
    if not os.path.exists(p):
        print(f"{arm}: NO DATA"); continue
    rows = [json.loads(l) for l in open(p)]
    print(f"\n--- {arm} ---")
    for r in rows:
        print(f"  {r['label']:<12} prompt_n={r['prompt_n']:<6} "
              f"prompt_ms={r['prompt_ms']:<9} wall={r['wall_s']}s")
    a1 = next((r for r in rows if r["label"] == "1_A_cold"), None)
    a3 = next((r for r in rows if r["label"] == "3_A_reuse"), None)
    if a1 and a3 and a1["prompt_n"]:
        print(f"  => reuse ratio prompt_n(3)/prompt_n(1) = "
              f"{a3['prompt_n']/a1['prompt_n']:.3f}   "
              f"(near 0 = CACHE HIT, near 1 = FULL RE-PREFILL)")
        if a1["prompt_ms"] and a3["prompt_ms"]:
            print(f"  => prefill time saved = {a1['prompt_ms']:.0f}ms -> "
                  f"{a3['prompt_ms']:.0f}ms  ({a1['prompt_ms']/max(a3['prompt_ms'],0.001):.1f}x)")
PY
