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
# v2 adds the PARITY arm: the same 4 prompts served by the classic static path and by
# the paged engine must produce IDENTICAL text at temp 0 (token-level bar, the hybrid
# analogue of flat's server-paged-parity 8/8; bit-identical logits are NOT the bar --
# different kernels legally differ in reduction order).
#
# Usage: ./hybrid_paged_gate.sh [build_dir] [port]

# ⚠ Strip absolute home paths before this file is committed. Gates build their output path from
# $HOME and echo it, which writes /Users/<username>/... into the result. See _no_abs_paths.sh.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
# ⚠ TRAP, NOT A TRAILING CALL. The first wiring put scrub_abs_paths at the end of the file, where
# the gate's own `exit` jumped straight over it -- and a `grep -l scrub_abs_paths` still listed the
# gate as a caller, because a grep counts TEXT, not control flow. Verified end-to-end afterwards:
# the result file still carried the absolute path. On EXIT it runs whatever path the gate takes.
trap 'scrub_abs_paths "${OUT:-}"' EXIT   # ${OUT:-} : the trap fires on EARLY exits too, before OUT is assigned; set -u would abort there

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

# parity prompts against the SAME paged server (temp 0, grammar; longer decode)
PROMPTS=("The" "once upon a time" "a b c" "hello world")
for i in 0 1 2 3; do
    # a crash mid-sequence must surface as a parity FAIL, not silently kill the gate
    curl -s --max-time 120 -o "$WORK/paged-$i.json" "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"${PROMPTS[$i]}\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false,\"grammar\":\"root ::= [a-z ]+\"}" \
        || echo '{"content":null,"error":"curl failed (server died?)"}' > "$WORK/paged-$i.json"
done
pkill -f "inkling-moe.gguf.*--port $PORT" 2>/dev/null || true
sleep 1

echo "== static arm (classic serve, no --kv-paged) =="
( "$BUILD/bin/llama-server" -m "$FIX" \
    -c 2048 -np 2 -b 512 -ub 512 \
    --port "$PORT" --no-warmup > "$WORK/static.log" 2>&1 & )
for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: static server never healthy"; tail -20 "$WORK/static.log"; exit 1; }
for i in 0 1 2 3; do
    curl -s --max-time 120 -o "$WORK/static-$i.json" "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"${PROMPTS[$i]}\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false,\"grammar\":\"root ::= [a-z ]+\"}" \
        || echo '{"content":null,"error":"curl failed (server died?)"}' > "$WORK/static-$i.json"
done
pkill -f "inkling-moe.gguf.*--port $PORT" 2>/dev/null || true
sleep 1

echo "== verdict =="
python3 - "$RAWLOG" "$WORK/resp.json" "$HTTP" "$OUT" "$N_PREDICT" "$WORK" <<'EOF'
import json, re, sys, time
rawlog, respf, http, outf, n_predict, work = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
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

# parity arm: same prompts, static vs paged serve, temp 0 -> identical text required
parity, parity_rows = True, []
for i in range(4):
    try:
        p = json.load(open(f'{work}/paged-{i}.json')).get('content')
        s = json.load(open(f'{work}/static-{i}.json')).get('content')
    except Exception as e:
        p, s = f'<err {e}>', '<err>'
    same = (p == s and p is not None)
    parity &= same
    parity_rows.append(f'  prompt {i}: {"MATCH" if same else "DIFFER"}  paged={p!r}  static={s!r}')

if ok_op and ok_http and parity:
    verdict = 'PASS — 4c-1 paged op executed, clean HTTP 200, AND static-vs-paged parity 4/4 at temp 0'
elif ok_op and ok_http:
    verdict = 'FAIL — paged op executed with clean 200 but static-vs-paged PARITY BROKE (see rows)'
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
lines.append('parity (static vs paged, temp 0, grammar, n_predict 16):')
lines.extend(parity_rows)
lines.append(f'raw server log: {rawlog}')
open(outf, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if (ok_op and ok_http and parity) else 2)
EOF
