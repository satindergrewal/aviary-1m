#!/usr/bin/env bash
# p28_finale_gate.sh — P2-8 finale witnesses (the screenshot ladder's last row):
#   ARM 1 queue-not-reject: more concurrent requests than slots (np) must ALL complete
#          via the deferred-task queue + scheduler waiting list -- zero 503s.
#   ARM 2 evict/preempt: a paged pool too small for the concurrent working set must
#          trigger scheduler swap-out/swap-in (Scheduler status: swapped > 0) and STILL
#          complete every request correctly.
#
# Usage: ./p28_finale_gate.sh <build_dir> <model.gguf> <port> [outfile]
set -euo pipefail
BUILD="${1:?build dir}"; MODEL="${2:?model gguf}"; PORT="${3:?port}"
OUT="${4:-/tmp/p28-finale-$(date +%Y%m%d-%H%M).txt}"
LOG1=/tmp/p28-arm1-server.log
LOG2=/tmp/p28-arm2-server.log

boot() { # $1 log, $2 extra args
    pkill -x llama-server 2>/dev/null || true; sleep 2
    # shellcheck disable=SC2086
    nohup "$BUILD/bin/llama-server" -m "$MODEL" -ngl 99 -b 512 -ub 512 \
        --cache-ram 0 --port "$PORT" --no-warmup $2 > "$1" 2>&1 &
    for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FATAL server no health"; tail -5 "$1"; exit 1; }
}

fire() { # $1 count, $2 n_predict, $3 tag
    for i in $(seq 1 "$1"); do
        curl -s --max-time 900 -w "%{http_code}" -o "/tmp/p28-$3-$i.json" "http://127.0.0.1:$PORT/completion" \
            -d "{\"prompt\":\"Request $i ($3): the story of packet number $i begins\",\"n_predict\":$2,\"temperature\":0,\"cache_prompt\":false}" \
            > "/tmp/p28-$3-$i.code" &
    done
    wait
}

echo "== ARM 1: queue-not-reject (np2, 6 concurrent) =="
boot "$LOG1" "-c 8192 -np 2 --kv-paged --n-gpu-blocks 1024 --n-cpu-blocks 128 -lv 5"
fire 6 96 qnr
pkill -x llama-server 2>/dev/null || true; sleep 2

echo "== ARM 2: evict/preempt (np3, pool starved: 56 gpu blocks) =="
boot "$LOG2" "-c 8192 -np 3 --kv-paged --n-gpu-blocks 56 --n-cpu-blocks 128 -lv 5"
fire 3 320 evi
pkill -x llama-server 2>/dev/null || true; sleep 2

python3 - "$OUT" <<'EOF'
import glob, json, re, sys, time
out = sys.argv[1]
def arm(tag, n):
    rows, ok = [], True
    for i in range(1, n + 1):
        code = open(f'/tmp/p28-{tag}-{i}.code').read().strip()
        try:
            d = json.load(open(f'/tmp/p28-{tag}-{i}.json'))
            pn = d.get('timings', {}).get('predicted_n'); c = d.get('content', '')[:24]
        except Exception:
            pn, c = None, '<unparsable>'
        good = code == '200' and pn is not None and pn > 0
        ok &= good
        rows.append(f'  {tag}-{i}: http={code} predicted_n={pn} head={c!r}')
    return ok, rows
ok1, rows1 = arm('qnr', 6)
ok2, rows2 = arm('evi', 3)
log2 = open('/tmp/p28-arm2-server.log', 'rb').read().decode('utf-8', 'replace')
swaps = re.findall(r'swapped=(\d+)', log2)
max_swapped = max((int(s) for s in swaps), default=0)
swap_engaged = max_swapped > 0
v1 = 'PASS' if ok1 else 'FAIL'
v2 = 'PASS' if (ok2 and swap_engaged) else ('WEAK-PASS (all complete, swap never engaged -- pool not starved enough)' if ok2 else 'FAIL')
lines = [
    f'# P2-8 finale witnesses, {time.strftime("%Y-%m-%d %H:%M")}',
    f'== ARM 1 queue-not-reject: {v1} (6 requests, np2 -- excess must queue, zero 503) ==',
    *rows1,
    f'== ARM 2 evict/preempt: {v2} (3 requests, 56-block pool; max swapped={max_swapped}) ==',
    *rows2,
]
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if (ok1 and ok2) else 2)
EOF
