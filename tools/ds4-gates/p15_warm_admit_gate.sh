#!/usr/bin/env bash
# P1-5 WARM-ADMIT GATE (box, CUDA, paged path)
#
# THE ONLY QUESTION THIS ANSWERS: does a request whose KV came back from the disk bank
# measure FASTER than the same request prefilled from scratch, and produce the SAME answer?
# Spill without a measured warm win is a write-only museum.
#
# Shape:
#   REQ 1  prompt P  -> COLD. baseline prompt_ms.
#   REQ 2  prompt Q  -> evicts P from the RAM cache; P spills to disk.
#   REQ 3  prompt P  -> must ADMIT from the bank. warm prompt_ms + identical answer.
#
# Traps already paid for by earlier gates, encoded here:
#   - `grep -c` returns 1 on zero matches and kills a `set -e` script -> `|| true` everywhere
#   - a cache-RAM big enough to hold both prompts never exercises the DISK path at all
#   - prompt_ms from the response JSON is the authority; log lines announce work, they do
#     not measure it ([[prompt-cache-idle-slot-redundant-copies]])
set -uo pipefail

WT=${WT:-<BOX>/wt-ds4-ports}
B=${B:-$WT/build-cuda/bin/llama-server}
M=${M:-<BOX>/ktdev/qwen3-4b-dev-IQ4_KT.gguf}
P=${P:-8971}
BANK=${BANK:-<BOX>/ktdev/kvbank-warmgate}
TAG=${TAG:-2k}
OUT=${OUT:-/tmp/ds4gates/results/p15-warm-admit-$TAG-$(date +%Y%m%d-%H%M).txt}

# ⚠ PLATFORM PRECONDITION -- REFUSE, DO NOT PRETEND. This is a box/CUDA gate: it needs the NVMe KV
# bank directory. Run on the Mac it printed a JSON parse traceback, "BANK ... No such file or
# directory", "SERVER DEAD", produced NO verdict line, and EXITED 0. A gate that cannot run on this
# machine must say so and fail, not return success -- a batch runner reading exit codes scores that
# as a pass forever. Same defect paged_multimodel_gate.sh had (2026-08-06).
_bank_parent="$(dirname "${BANK:-<BOX>/ktdev/kvbank-warmgate}")"
if [ ! -d "$_bank_parent" ]; then
    echo "PRECONDITION FAIL: KV bank parent '$_bank_parent' does not exist." >&2
    echo "  This gate runs on the box (CUDA + NVMe bank), not here. Refusing rather than reporting a pass." >&2
    exit 2
fi

LOG=/tmp/p15-warm-gate.log

mkdir -p "$(dirname "$OUT")" "$BANK"
rm -f "$BANK"/*.kv 2>/dev/null || true

pkill -f "$B" 2>/dev/null || true
sleep 2

# ★ SIZING IS THE WHOLE GATE. Measured: prompt A = 355 MiB of state, prompt B = 630 MiB.
# The first attempt used --cache-ram 400, which meant B "exceeds cache size limit, skipping"
# -- B was never cached, so it never evicted A, so nothing spilled and REQ 3 was a RAM hit
# that LOOKED like a bank restore. 700 admits B (630 < 700) and then cannot hold both
# (355 + 630 = 985 > 700), so A is evicted and spilled, and REQ 3 has to come off disk.
# A gate whose limit silently excludes the evicting entry proves nothing at all.
DS4P_REVALIDATE=1 nohup "$B" -m "$M" -ngl 99 -c 40960 -np 1 -b 512 -ub 512 \
    --cache-ram ${CACHE_RAM:-700} --cache-idle-slots \
    --kv-bank "$BANK" --kv-bank-cap 4096 \
    --port "$P" --no-warmup -lv 4 \
    --kv-paged --n-gpu-blocks 4096 --n-cpu-blocks 256 > "$LOG" 2>&1 &

for _ in $(seq 1 90); do
    curl -s "http://127.0.0.1:$P/health" >/dev/null 2>&1 && break
    sleep 2
done

# two distinct ~2K prompts with ONE right answer each, so a wrong restore is visible in the
# text and not just in the timing
NREP=${NREP:-2000}
mkprompt() {  # $1 = leading sentinel (forces LCP 0), $2 = the fact
    python3 - "$1" "$2" "$NREP" <<'PY'
import json, sys
sentinel, fact = sys.argv[1], sys.argv[2]
# same filler for both prompts, same token count; only the FIRST word differs, so the two
# prompts share no common prefix at all while staying the same size
body = sentinel + " " + ("alpha " * int(sys.argv[3]))
print(json.dumps({"prompt": f"{body}\nRemember this: {fact}\nQuestion: repeat the fact exactly.\nAnswer:",
                  "n_predict": 24, "temperature": 0, "cache_prompt": True}))
PY
}

ask() {  # $1 = json file, $2 = label
    # ★ WALL TIME IS THE HONEST METRIC. The bank read happens inside prompt_load, which
    # runs BEFORE the server starts its own prompt timer -- so prompt_ms EXCLUDES the disk
    # cost of the very feature being measured. Quoting it alone flatters the result.
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -s --max-time 900 -H 'Content-Type: application/json' \
         -d @"$1" "http://127.0.0.1:$P/completion" > "/tmp/p15-resp-$2.json"
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    printf '%s' "$((t1 - t0))" > "/tmp/p15-wall-$2.txt"
    python3 - "/tmp/p15-resp-$2.json" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
wall = open(f"/tmp/p15-wall-{sys.argv[2]}.txt").read().strip()
print(f'{sys.argv[2]}\tWALL_ms={wall}\tprompt_ms={t.get("prompt_ms",-1):.1f}\t'
      f'predicted_ms={t.get("predicted_ms",-1):.1f}\tn_pred={t.get("predicted_n",-1)}\t'
      f'text={json.dumps(d.get("content",""))[:80]}')
PY
}

mkprompt one "the vault code is 7741" > /tmp/p15-A.json
mkprompt two "the ledger balance is 3320" > /tmp/p15-B.json

{
    echo "=== P1-5 WARM-ADMIT GATE ==="
    cd "$WT" && git log --oneline -1
    echo
    echo "--- REQ 1: prompt A, COLD ---"
    ask /tmp/p15-A.json A-cold
    echo "--- REQ 2: prompt B (evicts A -> spill) ---"
    ask /tmp/p15-B.json B
    echo "--- REQ 3: prompt A again, must come from the BANK ---"
    ask /tmp/p15-A.json A-warm
    echo
    # REQ 4/5 exist because the ECONOMICS GATE cannot fire on REQ 3: read bandwidth is only
    # observable by doing a read, so the first admit is always a bootstrap that admits and
    # measures. A second cycle is what proves the gate declines once both rates are known.
    echo "--- REQ 4: prompt B again (evicts A -> spill A a second time) ---"
    ask /tmp/p15-B.json B2
    echo "--- REQ 5: prompt A -- both rates now measured, gate should DECIDE ---"
    ask /tmp/p15-A.json A-decide
    echo
    echo "--- COLD vs WARM must be byte-identical ---"
    python3 - <<'PY2'
import json
a = json.load(open("/tmp/p15-resp-A-cold.json"))["content"]
b = json.load(open("/tmp/p15-resp-A-warm.json"))["content"]
print(f"cold {len(a)} chars / warm {len(b)} chars -> " + ("BYTE-IDENTICAL" if a == b else "DIFFER *** FAIL ***"))
PY2
    echo
    echo "--- BANK (must be non-empty: this is what proves the DISK path ran) ---"
    ls -la "$BANK" | tail -n +2
    echo
    echo "--- restore lines ---"
    grep -E "admitted WARM|restored .* blocks|kv-bank: (spilled|admitted)|geometry mismatch|error loading state" "$LOG" || true
    echo
    echo "--- reval refusals (want none new) ---"
    grep -c "CELL MISMATCH" "$LOG" || true
    echo
    echo "--- server alive? ---"
    (curl -s "http://127.0.0.1:$P/health" >/dev/null 2>&1 && echo "SERVER ALIVE") || echo "SERVER DEAD"
} | tee "$OUT"

pkill -f "$B" 2>/dev/null || true
echo "witness: $OUT"
