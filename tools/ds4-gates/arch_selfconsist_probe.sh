#!/usr/bin/env bash
# IS THE ARBITER SELF-CONSISTENT? -- the discriminator for the qwen3vlmoe req-1 divergence.
#
# A static-vs-paged FAIL is only a defect report if STATIC agrees with STATIC. This measures that
# directly instead of re-rolling the same comparison and hoping:
#
#   A  three identical requests inside ONE static server   -> intra-process determinism
#   B  one request in a SECOND static server                -> cross-process determinism
#   C  one request in a paged server                        -> the arm under suspicion
#
# ⚠ The observed flip pattern across 32 historical ernie4_5 rows was per-INVOCATION, not per-request,
# so A can come back perfectly stable while B still differs. Both arms are needed; measuring only A
# would clear the model on evidence that cannot see the failure mode.
set -uo pipefail
SRV=$HOME/Documents/GitHub/llama.cpp-ds4ports/build-metal/bin/llama-server
M=$HOME/Documents/GitHub/ornith-models/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf
T=${CLAUDE_JOB_DIR:-/tmp}/tmp

python3 -c "
import json
p = ('The following is a list of numbered facts about geography. ' +
     ' '.join(f'Fact {i}: city number {i} lies on a river.' for i in range(1,150)) +
     ' Question: the capital of France is')
print(json.dumps({'prompt': p, 'n_predict': 8, 'temperature': 0, 'seed': 1, 'cache_prompt': False}))
" > "$T/sc.json"

ask() { curl -s --max-time 900 -X POST "http://127.0.0.1:$1/completion" \
          -H 'Content-Type: application/json' -d @"$T/sc.json" \
        | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('content','').replace(chr(10),' '))
except Exception: print('MALFORMED')"; }

run() { # $1 port  $2 label  $3 reps  $4.. flags
    local port=$1 label=$2 reps=$3; shift 3
    "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port "$port" --no-warmup "$@" \
        > "$T/sc-$label.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = "200" ] && break
        kill -0 $pid 2>/dev/null || { echo "  $label: DID NOT SERVE"; return 1; }
        sleep 1
    done
    local r
    for r in $(seq 1 "$reps"); do printf '  %-14s rep %d  [%s]\n' "$label" "$r" "$(ask "$port")"; done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
}

echo "=== qwen3vlmoe req-1 divergence: is STATIC self-consistent? ==="
run 9301 "A static#1" 1
run 9302 "B static#2" 1
run 9303 "C paged"    3 --kv-paged
echo "=== DONE ==="
echo "READ IT LIKE THIS:"
echo "  A varies                      -> intra-process near-tie; the gate's FAIL is the instrument"
echo "  A stable, B differs from A    -> per-invocation flip; the gate's FAIL is the instrument"
echo "  A stable, B == A, C differs   -> REAL paged divergence on qwen3vlmoe. File it."
