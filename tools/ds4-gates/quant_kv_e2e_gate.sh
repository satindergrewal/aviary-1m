#!/usr/bin/env bash
# QUANTISED-KV END-TO-END GATE.
#
# Retires (or refuses to retire) the startup refusal added in 652eaa05. That refusal was filed on an
# END-TO-END observable:
#     --kv-paged -ctk f16   "Paris. ..."      correct
#     --kv-paged -ctk q8_0  " thesssssss"     GARBAGE
#     static     -ctk q8_0  "Paris. ..."      correct   <- proves it was PAGED-specific
# so it can only be retired by that same observable. test-paged-vs-cpu passing is necessary and NOT
# sufficient: it exercises one op with a hand-built cache, while the garbage involved a whole
# server -- scheduler, block table, multi-layer graph, slot reuse.
#
# FOUR arms, and two of them are controls that must FAIL/agree or the gate means nothing:
#   A REFUSAL   paged+q8_0, no dev flag  -> MUST refuse at startup (guard still armed)
#   B STATIC    q8_0, no paging          -> MUST be correct (isolates paging from quantisation)
#   C PAGED f16 paged, f16               -> MUST be correct (isolates quantisation from paging)
#   D PAGED q8  paged+q8_0, dev flag     -> THE TEST
#
# ⚠ Arm D also requires a PRESENCE MARKER. The original bug was not a wrong answer from a running
# kernel -- the paged op NEVER DISPATCHED under q8_0, silently, and the server answered from the
# fallback path looking perfectly healthy. An arm D that "passes" without the marker is measuring
# the static path and would retire the refusal on a lie.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf
P=8996
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/quant-kv-e2e-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
PROMPT='The capital of France is'

{
  echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')"
  echo "model: $(basename "$M")"
} | tee "$OUT"

fails=0

# ---- Arm A: the guard must still be armed without the dev flag -------------------------------
echo "--- A REFUSAL CONTROL (paged + q8_0, no dev flag) ---" | tee -a "$OUT"
pkill -f "$SRV" >/dev/null 2>&1; sleep 2
"$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup \
       --kv-paged --kv-block-size 32 -ctk q8_0 -ctv q8_0 > /tmp/qkv-A.log 2>&1
rc=$?
if grep -q "does not support this KV cache type" /tmp/qkv-A.log && [ $rc -ne 0 ]; then
    echo "A: PASS refused at startup (rc=$rc)" | tee -a "$OUT"
else
    echo "A: FAIL guard did not fire (rc=$rc) -- a stale guard is worse than none" | tee -a "$OUT"
    fails=$((fails+1))
fi

# ---- shared runner for the answering arms -----------------------------------------------------
run() { # $1 label  $2 extra server args  $3 env assignments
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env $3 nohup "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup -lv 4 \
        $2 > "/tmp/qkv-$1.log" 2>&1 &
    for _ in $(seq 1 240); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
        echo "NEVER_READY"; return
    fi
    curl -s -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":24,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"].replace(chr(10)," ")[:160])'
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
}

echo "--- B STATIC q8_0 (quantisation without paging) ---" | tee -a "$OUT"
b=$(run B "-ctk q8_0 -ctv q8_0" "")
echo "B: $b" | tee -a "$OUT"

echo "--- C PAGED f16 (paging without quantisation) ---" | tee -a "$OUT"
c=$(run C "--kv-paged --kv-block-size 32 -ctk f16 -ctv f16" "DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1")
echo "C: $c" | tee -a "$OUT"

echo "--- D PAGED q8_0 (THE TEST) ---" | tee -a "$OUT"
d=$(run D "--kv-paged --kv-block-size 32 -ctk q8_0 -ctv q8_0" \
          "LLAMA_BANDED_QUANT_KV=1 DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1")
echo "D: $d" | tee -a "$OUT"

# ---- presence marker: did the paged op actually dispatch in arm D? ----------------------------
marker=$(grep -c "ggml_metal_op_paged_attn\|paged_attn_write_q8_0" /tmp/qkv-D.log 2>/dev/null)
marker=${marker:-0}
echo "D presence marker (paged dispatches in log): $marker" | tee -a "$OUT"
refused=$(grep -c "paged layer refused" /tmp/qkv-D.log 2>/dev/null); refused=${refused:-0}
echo "D layers refused by predicate: $refused" | tee -a "$OUT"

# ---- verdicts ---------------------------------------------------------------------------------
# ⚠ THIS BLOCK WAS `grep -ci paris` AND IT RETURNED PASS ON A FAILING RUN. Arm D answered
#       " Paris. Paris, officially known as the <ruby align=\"mtr-skin-avbrudwpdew\"
# which contains "paris" and is otherwise token soup. The original bug it is meant to catch --
# " thesssssss" -- also contains letters. A verdict weaker than the failure it exists to catch is
# not a gate, and this lane has now produced that shape more than once.
#
# The right comparison was sitting in the run the whole time: arm B is q8_0 WITHOUT paging. B vs D
# differs in exactly ONE thing, the paged path, with quantisation held constant. B and C came back
# character-identical on this model+prompt, so agreement is the calibrated expectation here, not an
# assumption -- greedy, temperature 0, fixed seed, cache_prompt off.
ok_b=$(echo "$b" | grep -ci "paris"); ok_b=${ok_b:-0}
ok_c=$(echo "$c" | grep -ci "paris"); ok_c=${ok_c:-0}

[ "$ok_b" -ge 1 ] || { echo "B: FAIL static q8_0 wrong -- quantised KV itself is broken, arm D is uninterpretable" | tee -a "$OUT"; fails=$((fails+1)); }
[ "$ok_c" -ge 1 ] || { echo "C: FAIL paged f16 wrong -- paging itself is broken, arm D is uninterpretable" | tee -a "$OUT"; fails=$((fails+1)); }

if [ "$b" != "$c" ]; then
    echo "CONTROLS DISAGREE (B != C) -- paged/static already differ at f16, so D cannot be attributed to q8_0" | tee -a "$OUT"
    fails=$((fails+1))
fi
if [ "$d" = "$b" ]; then
    echo "D: PASS paged q8_0 matches static q8_0 exactly (one-factor: paging held against it)" | tee -a "$OUT"
else
    echo "D: FAIL paged q8_0 DIVERGES from static q8_0 -- REFUSAL STAYS" | tee -a "$OUT"
    echo "    static q8_0: $b" | tee -a "$OUT"
    echo "    paged  q8_0: $d" | tee -a "$OUT"
    fails=$((fails+1))
fi
[ "$marker" -ge 1 ] || { echo "D: FAIL no presence marker -- answer came from the fallback path, not paged q8_0" | tee -a "$OUT"; fails=$((fails+1)); }

echo "-----" | tee -a "$OUT"
if [ "$fails" -eq 0 ]; then
    echo "QUANT-KV E2E GATE: PASS -- refusal may be retired" | tee -a "$OUT"
else
    echo "QUANT-KV E2E GATE: FAIL ($fails) -- refusal STAYS UP" | tee -a "$OUT"
fi
echo "log: $OUT"
exit $fails
