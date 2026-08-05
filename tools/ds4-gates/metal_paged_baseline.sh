#!/usr/bin/env bash
# METAL PAGED BASELINE -- the first performance number for the Apple-silicon paged path.
#
# The Metal port shipped as a deliberate rung one: a SCALAR kernel, on the reasoning that
# runs-slowly beats cannot-run. That decision was never priced. This measures the price.
#
# ONE binary, ONE model, ONE prompt, two arms differing in exactly one flag:
#   STATIC  --kv-paged absent  -> the tuned Metal flash-attention path
#   PAGED   --kv-paged present -> the scalar paged kernel
# Reported on WALL time and on prompt_ms, because on the CUDA side quoting prompt_ms alone
# is exactly how I produced a 21.6x that did not exist ([[arms-must-differ-in-one-thing]]).
#
# This gate deliberately does NOT set a pass bar. There is no prior Metal paged number to
# regress against -- inventing a threshold now would be picking the answer before measuring.
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
B=${B:-$WT/build-metal/bin/llama-server}
M=${M:-/tmp/qwen3-4b-metal-Q4_K_M.gguf}
P=${P:-8974}
NREP=${NREP:-1500}
REPS=${REPS:-3}
OUT=${OUT:-$WT/../ornith-1m/tools/ds4-gates/results/metal-paged-baseline-$(date +%Y%m%d-%H%M).txt}

mkdir -p "$(dirname "$OUT")"

python3 - "$NREP" > /tmp/metal-prompt.json <<'PY'
import json, sys
body = "alpha " * int(sys.argv[1])
print(json.dumps({"prompt": f"{body}\nRemember this: the vault code is 7741\nQuestion: repeat the fact exactly.\nAnswer:",
                  "n_predict": 32, "temperature": 0, "cache_prompt": False}))
PY

run_arm() {  # $1 = label, $2 = extra server args
    pkill -f "$B" 2>/dev/null || true
    sleep 3
    # shellcheck disable=SC2086
    nohup "$B" -m "$M" -ngl 99 -c 16384 -np 1 -b 512 -ub 512 \
        --port "$P" --no-warmup -lv 2 $2 > "/tmp/metal-$1.log" 2>&1 &
    # ⚠ READINESS MUST CHECK THE STATE, NOT THAT SOMETHING ANSWERED. /health replies 503
    # "Loading model" while the model loads, and `curl -s ... >/dev/null` exits 0 on a 503 --
    # so the old loop broke immediately and fired requests into a loading server, which is
    # exactly how the STATIC arm produced WALL=27ms with timings of -1 and no error anywhere.
    # Poll the STATUS CODE until it is 200.
    for _ in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P/health" 2>/dev/null)
        [ "$code" = "200" ] && break
        sleep 2
    done

    for r in $(seq 1 "$REPS"); do
        local t0 t1
        t0=$(python3 -c 'import time;print(int(time.time()*1000))')
        curl -s --max-time 900 -H 'Content-Type: application/json' \
             -d @/tmp/metal-prompt.json "http://127.0.0.1:$P/completion" > "/tmp/metal-resp-$1-$r.json"
        t1=$(python3 -c 'import time;print(int(time.time()*1000))')
        python3 - "/tmp/metal-resp-$1-$r.json" "$1" "$r" "$((t1 - t0))" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
tps = t.get("predicted_per_second", -1)
print(f'  {sys.argv[2]:6s} rep{sys.argv[3]}  WALL={sys.argv[4]:>6s} ms  prompt_ms={t.get("prompt_ms",-1):8.1f}  '
      f'prompt_n={t.get("prompt_n",-1):>6}  gen={t.get("predicted_ms",-1):7.1f} ms  {tps:6.2f} tok/s')
PY
    done
    pkill -f "$B" 2>/dev/null || true
}

{
    echo "=== METAL PAGED BASELINE (first perf number for the Apple-silicon paged path) ==="
    cd "$WT" && git log --oneline -1
    echo "model: $M   prompt: $NREP filler words   reps: $REPS"
    echo
    echo "--- ARM STATIC (tuned Metal FA, no --kv-paged) ---"
    run_arm static ""
    echo
    echo "--- ARM PAGED (scalar paged kernel) ---"
    run_arm paged "--kv-paged --n-gpu-blocks 2048 --n-cpu-blocks 256"
    echo
    echo "--- answers must agree (a slow kernel is still allowed to be wrong) ---"
    python3 - <<'PY'
import json, glob
def txt(p):
    return json.load(open(p))["content"]
s = txt(sorted(glob.glob("/tmp/metal-resp-static-*.json"))[0])
g = sorted(glob.glob("/tmp/metal-resp-paged-*.json"))
if not g:
    print("  PAGED ARM PRODUCED NO RESPONSE -- see /tmp/metal-paged.log")
else:
    p = txt(g[0])
    print("  static:", json.dumps(s[:70]))
    print("  paged :", json.dumps(p[:70]))
    print("  ->", "IDENTICAL" if s == p else "DIFFER (expected: different summation order)")
PY
} | tee "$OUT"

echo "witness: $OUT"
