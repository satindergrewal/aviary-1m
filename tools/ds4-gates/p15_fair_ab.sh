#!/usr/bin/env bash
# P1-5 FAIR A/B -- does a warm admit actually beat a cold prefill, with nothing else differing?
#
# WHY THIS EXISTS. The 3-request gate compared a cold FIRST request (empty cache, no
# eviction) against a warm LAST request (full cache, so its finish evicted an entry and
# SPILLED 1.6 GiB to disk). The warm arm was carrying ~1 s of write the cold arm never paid,
# and I read the difference as the restore being slow. Same class as every other trap in
# this lane: the arms differed in more than the one thing under test.
#
# THE FAIR SHAPE -- three separate server lifetimes, each request the FIRST on a fresh
# server with an EMPTY prompt cache, so no eviction and no spill can land inside a timed
# window:
#   SEED  bank on   : A then B, so A ends up on disk. Timings discarded.
#   WARM  bank on   : A -- restored from the seeded bank.
#   COLD  bank off  : A -- prefilled, nothing on disk to find.
# WARM vs COLD is then the restore, and only the restore.
set -uo pipefail

WT=${WT:-<BOX>/wt-ds4-ports}
B=${B:-$WT/build-cuda/bin/llama-server}
M=${M:-<BOX>/ktdev/qwen3-4b-dev-IQ4_KT.gguf}
P=${P:-8973}
BANK=${BANK:-<BOX>/ktdev/kvbank-fair}
NREP=${NREP:-11000}
OUT=${OUT:-/tmp/ds4gates/results/p15-fair-ab-$(date +%Y%m%d-%H%M).txt}

# ⚠ PLATFORM PRECONDITION -- REFUSE, DO NOT PRETEND. This is a box gate (CUDA / NVMe paths). Run on
# a machine without them it produces errors, no verdict, and EXITS 0 -- which a batch runner reading
# exit codes scores as a pass forever. p15_warm_admit_gate.sh did exactly that on 2026-08-06.
if [ ! -d "<BOX>/wt-ds4-ports" ] && [ ! -d "$(dirname "<BOX>/wt-ds4-ports")" ]; then
    echo "PRECONDITION FAIL: '<BOX>/wt-ds4-ports' not present -- this gate runs on the box, not here." >&2
    echo "  Refusing rather than reporting a pass." >&2
    exit 2
fi


mkdir -p "$(dirname "$OUT")" "$BANK"
rm -f "$BANK"/*.kv 2>/dev/null || true

mkprompt() {
    python3 - "$1" "$2" "$NREP" <<'PY'
import json, sys
sentinel, fact = sys.argv[1], sys.argv[2]
body = sentinel + " " + ("alpha " * int(sys.argv[3]))
print(json.dumps({"prompt": f"{body}\nRemember this: {fact}\nQuestion: repeat the fact exactly.\nAnswer:",
                  "n_predict": 24, "temperature": 0, "cache_prompt": True}))
PY
}
mkprompt one "the vault code is 7741"     > /tmp/p15f-A.json
mkprompt two "the ledger balance is 3320" > /tmp/p15f-B.json

start_server() {  # $1 = extra args
    pkill -f "$B" 2>/dev/null || true
    sleep 3
    # shellcheck disable=SC2086
    DS4P_REVALIDATE=1 nohup "$B" -m "$M" -ngl 99 -c 40960 -np 1 -b 512 -ub 512 \
        --cache-ram 2500 --cache-idle-slots --port "$P" --no-warmup -lv 4 \
        --kv-paged --n-gpu-blocks 4096 --n-cpu-blocks 256 $1 > /tmp/p15-fair.log 2>&1 &
    for _ in $(seq 1 90); do
        curl -s "http://127.0.0.1:$P/health" >/dev/null 2>&1 && break
        sleep 2
    done
}

ask() {  # $1 = json, $2 = label -> prints WALL and prompt_ms
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -s --max-time 900 -H 'Content-Type: application/json' \
         -d @"$1" "http://127.0.0.1:$P/completion" > "/tmp/p15f-resp-$2.json"
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    python3 - "/tmp/p15f-resp-$2.json" "$2" "$((t1 - t0))" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); t = d.get("timings", {})
print(f'{sys.argv[2]:9s} WALL_ms={sys.argv[3]:>6s}  prompt_ms={t.get("prompt_ms",-1):8.1f}  '
      f'prompt_n={t.get("prompt_n",-1):>6}  gen_ms={t.get("predicted_ms",-1):7.1f}')
PY
}

{
    echo "=== P1-5 FAIR A/B (each timed request is the FIRST on a fresh server) ==="
    cd "$WT" && git log --oneline -1
    echo "prompt: $NREP filler words"
    echo

    echo "--- SEED (bank on): run A then B so A lands on disk; timings discarded ---"
    start_server "--kv-bank $BANK --kv-bank-cap 8192"
    ask /tmp/p15f-A.json seed-A > /dev/null
    ask /tmp/p15f-B.json seed-B > /dev/null
    ls -la "$BANK"/*.kv 2>/dev/null | awk '{print "  banked:", $5, $9}'
    echo

    echo "--- WARM (bank on, fresh server, empty cache) ---"
    start_server "--kv-bank $BANK --kv-bank-cap 8192"
    ask /tmp/p15f-A.json warm
    grep -E "kv-bank: (admitted|DECLINING)|admitted WARM" /tmp/p15-fair.log | sed 's/^/  /' || true
    cp /tmp/p15f-resp-warm.json /tmp/p15f-keep-warm.json
    echo

    echo "--- COLD (bank OFF, fresh server, empty cache) ---"
    start_server "--no-kv-bank"
    ask /tmp/p15f-A.json cold
    echo

    echo "--- VERDICT ---"
    python3 - <<'PY'
import json
warm = json.load(open("/tmp/p15f-keep-warm.json"))
cold = json.load(open("/tmp/p15f-resp-cold.json"))
print("  output identical:", "YES" if warm["content"] == cold["content"] else "NO  *** FAIL ***")
print(f'  prompt_ms  warm {warm["timings"]["prompt_ms"]:.1f}  vs cold {cold["timings"]["prompt_ms"]:.1f}')
PY
} | tee "$OUT"

pkill -f "$B" 2>/dev/null || true
echo "witness: $OUT"
