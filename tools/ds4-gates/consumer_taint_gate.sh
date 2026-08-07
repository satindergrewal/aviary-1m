#!/usr/bin/env bash
# CONSUMER GATE v2 (logprob observable). Validates ITSELF on a model where paged is known to work, before being
# trusted on Ornith. A gate that cannot detect a WORKING consumer is worthless.
set -uo pipefail

# ⚠ PRIVACY. This gate writes into tools/ds4-gates/results/, which is TRACKED -- and it built its
# output path from $HOME, so every run committed /Users/<username>/ into the repo. the owner's rules
# name file paths containing usernames as private data at absolute highest priority, after two
# incidents that each needed a full history rewrite.
#
# ⚠ A TRAP, NOT A TRAILING CALL. paged_multimodel_gate.sh records why: a trailing scrub was jumped
# over by the gate's own `exit`, while `grep -l scrub_abs_paths` still listed the gate as a caller,
# because grep counts TEXT and not control flow. On EXIT it runs whatever path the gate takes.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT   # ${OUT:-} : fires on early exits too, before OUT is set

WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
B=$WT/build-metal/bin/llama-server
M=${M:-/tmp/qwen3-4b-metal-Q4_K_M.gguf}
P=8991
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/consumer-taint-gate-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
echo "binary: $(stat -f '%Sm' -t '%H:%M:%S' "$B")" | tee -a "$OUT"

run() { # $1 label  $2 server args  $3 env
    pkill -f "$B" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env $3 nohup "$B" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 ${EXTRA:-} $2 > "/tmp/tg-$1.log" 2>&1 &
    for _ in $(seq 1 120); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break; sleep 1
    done
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] || { echo "$1 NEVER READY" | tee -a "$OUT"; return 1; }
    # ⚠ OBSERVABLE MUST BE LOGPROBS, NOT SAMPLED TEXT. Greedy decoding is a quantised (argmax)
    # function of the activations: a high-confidence prompt yields identical TEXT even when the
    # activations genuinely changed. Hashing text is a FALSE-NEGATIVE GENERATOR for a consumer
    # probe -- it reports "no divergence" for a real perturbation. Hash the logprobs instead.
    sha=$(curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
        -d '{"prompt":"The capital of France is","n_predict":6,"temperature":0,"seed":1,"cache_prompt":false,"n_probs":10}' \
      | python3 -c '
import json,sys,hashlib
d=json.load(sys.stdin)
probs=d.get("completion_probabilities") or []
vals=[]
for tok in probs:
    for c in (tok.get("probs") or tok.get("top_logprobs") or []):
        v=c.get("prob", c.get("logprob"))
        if v is not None: vals.append(round(float(v),6))
if not vals:
    print("NO_LOGPROBS")
else:
    print(hashlib.sha256(repr(vals).encode()).hexdigest()[:12])')
    taint=$(grep -c "DS4P-PAGED-TAINT active" "/tmp/tg-$1.log" 2>/dev/null || echo 0)
    printf '%-18s sha=%s  taint_marker=%s\n' "$1" "$sha" "$taint" | tee -a "$OUT"
    echo "$sha"
}

echo "=== CONSUMER GATE on $(basename "$M") ===" | tee -a "$OUT"
a=$(run PAGED-clean  "--kv-paged --kv-block-size 32" "${SRVENV:-}" | tail -1)
b=$(run PAGED-tainted "--kv-paged --kv-block-size 32" "${SRVENV:-} DS4P_PAGED_TAINT=1" | tail -1)
pkill -f "$B" >/dev/null 2>&1
echo "---" | tee -a "$OUT"
armed=$(grep -c "DS4P-PAGED-TAINT active" /tmp/tg-PAGED-tainted.log 2>/dev/null); armed=$(printf %s "${armed:-0}" | head -1)
pagedop=$(grep -cE "DS4P-(LPK ACTIVE|MMA OFF|CHAMP-PAGED ACTIVE)" /tmp/tg-PAGED-tainted.log 2>/dev/null); pagedop=$(printf %s "${pagedop:-0}" | head -1)
echo "probe_armed=$armed  paged_op_ran=$pagedop  model=$(basename "$M")" | tee -a "$OUT"

if [ "$armed" = "0" ]; then
    echo "INCONCLUSIVE: the taint probe never armed (no paged KV write dispatched). NOT evidence of anything -- the paged pool was absent or the write path never ran." | tee -a "$OUT"
elif [ -z "$a" ] || [ -z "$b" ] || [ "$a" = "NO_LOGPROBS" ]; then
    echo "INCONCLUSIVE: no usable logprob observable." | tee -a "$OUT"
elif [ "$a" != "$b" ]; then
    echo "CONSUMED: taint CHANGED logprobs ($a -> $b) => this graph genuinely READS the paged pool" | tee -a "$OUT"
else
    echo "NOT CONSUMED: probe armed, paged writes tainted, logprobs BIT-IDENTICAL ($a) => the graph does NOT read the paged pool" | tee -a "$OUT"
fi
echo "OUT=$OUT"
