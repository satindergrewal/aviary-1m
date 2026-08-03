#!/usr/bin/env bash
# p28_cuda_regate.sh — the LAST wall between P2-8 and CLOSED, packaged to run with zero
# thinking the moment a card frees. Pins ONE gpu, port-scoped kills only, and re-runs all
# three arms that the Mac side already passes, on CUDA:
#   ARM 1 queue-not-reject  (np2, 6 concurrent)            bar: 6/6 full-length, 0 pos-dec
#   ARM 2 recompute-preempt (40 gpu / 2 cpu blocks)        bar: 3/3, recompute>0, 0 errors
#   ARM 3 swap-preempt      (40 gpu / 128 cpu blocks)      bar: 3/3, swap-ins drained, 0 errors
#
# Usage (on the box, card free or with headroom):
#   CUDA_VISIBLE_DEVICES=0 ./p28_cuda_regate.sh [wt] [model] [port]
# Sync first (the runner does it): git -C <wt> fetch fork ds4-ports && git checkout fork/ds4-ports
#   && cmake --build build-cuda --target llama-server -j 24
set -euo pipefail
WT="${1:-/mnt/nvme0/wt-ds4-ports}"
MODEL="${2:-/mnt/nvme0/ktdev/qwen3-4b-dev-IQ4_KT.gguf}"
PORT="${3:-8953}"
BUILD="$WT/build-cuda"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/results/p28-cuda-regate-$(date +%Y%m%d-%H%M).txt"
mkdir -p "$DIR/results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
trap 'pkill -f "por[t] $PORT" 2>/dev/null || true' EXIT

echo "== sync + build ==" | tee "$OUT"
git -C "$WT" fetch fork ds4-ports >/dev/null 2>&1
git -C "$WT" checkout fork/ds4-ports >/dev/null 2>&1
git -C "$WT" log -1 --oneline | tee -a "$OUT"
cmake --build "$BUILD" --target llama-server -j 24 2>&1 | tail -1 | tee -a "$OUT"

arm() { # $1 tag  $2 server-args  $3 n_requests  $4 n_predict
    local tag="$1" args="$2" n="$3" npred="$4" log="/tmp/p28cr-$tag.log"
    pkill -f "por[t] $PORT" 2>/dev/null || true; sleep 2
    # shellcheck disable=SC2086
    nohup "$BUILD/bin/llama-server" -m "$MODEL" -ngl 99 -b 128 -ub 128 --cache-ram 0 \
        --port "$PORT" --no-warmup -lv 5 $args > "$log" 2>&1 &
    for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "$tag: FATAL no health" | tee -a "$OUT"; return 1; }
    local pids=()
    for i in $(seq 1 "$n"); do
        curl -s --max-time 900 -w "%{http_code}" -o "/tmp/p28cr-$tag-$i.json" \
            "http://127.0.0.1:$PORT/completion" \
            -d "{\"prompt\":\"Tale $i:\",\"n_predict\":$npred,\"temperature\":0,\"cache_prompt\":false}" \
            > "/tmp/p28cr-$tag-$i.code" &
        pids+=($!)
    done
    wait "${pids[@]}"
    pkill -f "por[t] $PORT" 2>/dev/null || true; sleep 1
    python3 - "$tag" "$n" "$npred" "$log" <<'PYEOF' | tee -a "$OUT"
import json, re, sys
tag, n, npred, log = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ok = 0
for i in range(1, n + 1):
    code = open(f'/tmp/p28cr-{tag}-{i}.code').read().strip()
    try:
        pn = json.load(open(f'/tmp/p28cr-{tag}-{i}.json')).get('timings', {}).get('predicted_n')
    except Exception:
        pn = None
    ok += (code == '200' and pn == npred)
d = open(log, 'rb').read().decode('utf-8', 'replace')
sw = [int(x) for x in re.findall(r'swapped=(\d+)', d)]
print(f'{tag}: complete {ok}/{n} @{npred} | max_swapped={max(sw, default=0)} '
      f'swap_outs={d.count("swapped out")} swap_ins={d.count("back in for processing")} '
      f'recompute={d.count("recomputation")} evictions={d.count("Eviction requested")} | '
      f'pos_dec={d.count("positions are decreasing")} oob={d.count("out of bounds")+d.count("block_table_id")} '
      f'deadlock={d.count("Scheduler deadlock")} decode_fail={d.count("decode failed")}')
PYEOF
}

echo "== ARM 1 queue-not-reject ==" | tee -a "$OUT"
arm qnr "-c 8192 -np 2 --kv-paged --n-gpu-blocks 1024 --n-cpu-blocks 128" 6 96
echo "== ARM 2 recompute-preempt ==" | tee -a "$OUT"
arm rec "-c 4096 -np 3 --kv-paged --n-gpu-blocks 40 --n-cpu-blocks 2" 3 400
echo "== ARM 3 swap-preempt ==" | tee -a "$OUT"
arm swp "-c 4096 -np 3 --kv-paged --n-gpu-blocks 40 --n-cpu-blocks 128" 3 400
echo "witness: $OUT"
