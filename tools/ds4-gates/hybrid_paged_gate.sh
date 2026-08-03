#!/usr/bin/env bash
# hybrid_paged_gate.sh — 3b HYBRID DECODE gate: does hybrid serving EXECUTE the paged
# banded attention op (the 4c-1 branch), or silently fall back to the static cache?
#
# The bar is OP-LEVEL, not prose: PASS requires >=1 PAGED_ATTN node inside a
# SERVE-TIME graph dump, plus a clean HTTP 200 with predicted_n == N_PREDICT.
# "Serving worked" alone is NOT a pass — the static cache can silently absorb the
# whole workload and produce correct-looking tokens (that is the exact failure this
# gate exists to catch; see the 2026-08-04 first run, which FAILED and exposed it).
#
# HARD-WON RECIPE FACTS ENCODED HERE (each cost a dead run to learn):
#  - kv_paged asserts n_batch == n_ubatch            => -b 512 -ub 512 is mandatory
#  - random-weight fixtures emit invalid UTF-8; the PEG content parser 500s on it
#    => constrain sampling with grammar 'root ::= [a-z ]+' (valid UTF-8 => clean 200)
#  - the DS4P construction line is libllama INFO, filtered at default verbosity,
#    and GGML_SCHED_DEBUG=2 dumps print at GGML_LOG_DEBUG   => -lv 5 for both
#  - the server log goes BINARY (model output bytes leak into it) and sched-debug
#    node lines are FRAGMENTED across writes => line-oriented grep SILENTLY fails;
#    the op scan must be a fragment-aware regex over the raw bytes (python below)
#  - load/probe/reserve graphs (t=0.00.x) legitimately contain static FLASH nodes;
#    only t>=0.01 graphs are serve-time => the verdict counts the serve window only
#
# Usage: ./hybrid_paged_gate.sh [build_dir] [port]
set -euo pipefail

BUILD="${1:-$HOME/Documents/GitHub/llama.cpp-ds4ports/build-check}"
PORT="${2:-8942}"
DIR="$(cd "$(dirname "$0")" && pwd)"
DATE="$(date +%Y%m%d)"
OUT="$DIR/results/hybrid-paged-decode-raw-$DATE.txt"
RAWLOG="$DIR/results/hybrid-paged-decode-raw-$DATE.log"
WORK="$(mktemp -d)"
N_PREDICT=8
trap 'pkill -f "inkling-moe.gguf.*--port $PORT" 2>/dev/null || true' EXIT
mkdir -p "$DIR/results"

echo "== fixture =="
"$BUILD/bin/test-llama-archs" -a inkling -s 42 --out "$WORK" >/dev/null 2>&1
FIX="$WORK/inkling-moe.gguf"
[ -f "$FIX" ] || { echo "FAIL: fixture not written"; exit 1; }

echo "== serve =="
( DS4P_PAGED_HYBRID=1 GGML_SCHED_DEBUG=2 "$BUILD/bin/llama-server" -m "$FIX" \
    -c 2048 -np 2 -b 512 -ub 512 --kv-paged --n-gpu-blocks 64 --n-cpu-blocks 16 \
    --port "$PORT" --no-warmup -lv 5 > "$RAWLOG" 2>&1 & )
for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server never healthy"; tail -20 "$RAWLOG"; exit 1; }

HTTP=$(curl -s --max-time 120 -w "%{http_code}" -o "$WORK/resp.json" \
    "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\":\"The\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"cache_prompt\":false,\"grammar\":\"root ::= [a-z ]+\"}")
pkill -f "inkling-moe.gguf.*--port $PORT" 2>/dev/null || true
sleep 1

echo "== verdict =="
python3 - "$RAWLOG" "$WORK/resp.json" "$HTTP" "$OUT" "$N_PREDICT" <<'EOF'
import json, re, sys, time
rawlog, respf, http, outf, n_predict = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
data = open(rawlog, 'rb').read().decode('utf-8', 'replace')

def ops(window):  # window: fn(ts_prefix)->bool
    from collections import Counter
    c = Counter()
    for m in re.finditer(r'(\d+\.\d+\.\d+)\.\d+ D node #\s*\d+ \(([^)]+)\):', data):
        if window(m.group(1)):
            c[m.group(2).strip()] += 1
    return dict(c.most_common())

load  = ops(lambda ts: ts.startswith('0.00.0'))
serve = ops(lambda ts: not ts.startswith('0.00.0'))

markers = {
    'pool_constructed':  'constructing the hybrid paged attention pool' in data,
    'sched_discovered':  "using the hybrid wrapper's paged attention pool" in data,
    'sched_initialized': 'paged serving: scheduler initialized' in data,
    'sched_stepped':     data.count('Scheduler status'),
    'request_registered':'paged: request registered' in data,
}
try:
    resp = json.load(open(respf))
    predicted_n = resp.get('timings', {}).get('predicted_n')
    content = resp.get('content', '')
except Exception as e:
    predicted_n, content = None, f'<unreadable: {e}>'

paged_serve = sum(v for k, v in serve.items() if 'PAGED' in k)
flash_serve = sum(v for k, v in serve.items() if 'FLASH' in k)
ok_http = (http == '200' and predicted_n == n_predict)
ok_op   = paged_serve > 0

if ok_op and ok_http:
    verdict = 'PASS — hybrid serving executed the paged banded attention op (4c-1) with a clean HTTP 200'
elif ok_http and flash_serve > 0:
    verdict = 'FAIL — serving succeeded but through STATIC banded flash: the 4c-1 paged branch never executed'
elif ok_http:
    verdict = ('FAIL — serving succeeded but through the STATIC UNFUSED path (no flash, no paged op): '
               'both use_banded_flash and the 4c-1 branch switched off; the static cache silently did all the work')
else:
    verdict = f'FAIL — request itself failed (HTTP {http}, predicted_n={predicted_n})'

lines = []
lines.append(f'# 3b HYBRID PAGED DECODE GATE (op-level), {time.strftime("%Y-%m-%d %H:%M")}')
lines.append('# recipe: DS4P_PAGED_HYBRID=1 GGML_SCHED_DEBUG=2 llama-server -b 512 -ub 512 --kv-paged, /completion + grammar, -lv 5')
lines.append('')
lines.append(f'== VERDICT: {verdict} ==')
lines.append('')
lines.append(f'http={http} predicted_n={predicted_n} content={content!r}')
lines.append(f'markers={markers}')
lines.append(f'serve-window ops: {serve}')
lines.append(f'load-window ops (context only): {load}')
lines.append(f'paged ops in serve window: {paged_serve}   flash ops in serve window: {flash_serve}')
lines.append(f'raw server log: {rawlog}')
open(outf, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if (ok_op and ok_http) else 2)
EOF
