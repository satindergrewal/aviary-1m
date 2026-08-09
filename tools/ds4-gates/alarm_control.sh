#!/usr/bin/env bash
# CONTROL PAIR for LAW 6: does the engine's no-consumer alarm actually fire, and does DS4P-CONSUME
# actually print?
#
# ⚠⚠ WHY THIS EXISTS. `gate_assert_paged_consumed` (LAW 6 in _gate_common.sh) decides whether a paged
# arm is real by checking that the engine's WARN-level alarm stayed SILENT. On 2026-08-09 that alarm
# **should have fired and did not**: the 512k run had every attention layer refused, consumer count
# zero, and 780+ decode() calls, and its log contains not one instance of the warning. Three benign
# explanations are already dead --
#     "not in that binary"    -> the string IS in libllama.0.0.10621.dylib, the build that run used
#     "paged bypasses decode" -> server-context.cpp:3284 calls llama_decode() on the paged path
#     "n_dec never hit 8"     -> a 400k prefill at ub=512 is ~780 decode() calls
# -- leaving one: `ds4p_paged_consumer_count()` was not zero. The counter is GLOBAL with TWO call
# sites, the banded funnel (llama-graph.cpp:4742) and the AUTO funnel (:3640), so anything reaching
# the auto funnel silences the alarm no matter how completely the banded path collapses.
#
# ⇒ **A law whose instrument has never been seen firing is not a law.** This runs the two arms that
#   settle it, and it is deliberately TINY -- 8k, seconds per arm -- because the question is boolean.
#
# ⚠ ARM B IS THE NEGATIVE CONTROL FOR ARM A, and running A alone would be the same absence-based
# reasoning that caused the failure. If B shows DS4P-CONSUME > 0, the marker demonstrably prints in
# THIS binary at THIS verbosity; only then does A's silence carry information.
#
# usage: AC_MODEL=<gguf> alarm_control.sh
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
. "$(dirname "$0")/_gate_common.sh"  2>/dev/null || true

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
M=${AC_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-35B-1M-GGUF/ornith-1.0-35b-1M-Q4_K_M.gguf}
CTX=${AC_CTX:-8192}
NPRED=${AC_NPRED:-24}
[ -f "$M" ] || { echo "missing model: $M" >&2; exit 2; }

D=${CLAUDE_JOB_DIR:-/tmp}/alarmctl; mkdir -p "$D"
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/alarmctl-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"

# Same one-server lock the parity gate uses -- these arms must never run beside a timed measurement.
LOCK=${CLAUDE_JOB_DIR:-/tmp}/parity/gpu.lock; mkdir -p "$(dirname "$LOCK")"
until mkdir "$LOCK" 2>/dev/null; do echo "  waiting for the GPU lock..."; sleep 20; done
PID=""
trap 'rmdir "$LOCK" 2>/dev/null; [ -n "$PID" ] && kill $PID 2>/dev/null; scrub_abs_paths "${OUT:-}" 2>/dev/null' EXIT

pick_port() { local p; for p in $(seq 20170 20200); do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }; done; return 1; }
PORT=$(pick_port) || { echo "no free port" >&2; exit 2; }

echo "LAW 6 control pair: $(basename "$M")  ctx=$CTX  npred=$NPRED" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD)" | tee -a "$OUT"
printf '{"prompt":"Count from one to ten, then stop.","n_predict":%d,"temperature":0,"seed":1,"cache_prompt":false}\n' \
    "$NPRED" > "$D/req.json"

run_arm() { # $1 label  $2 champ  $3 blocksize  $4 verbosity
    local log="$D/$1.log"; : > "$log"
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 DS4P_METAL_CHAMP="$2" \
        "$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 --port "$PORT" --no-warmup \
        -lv "$4" --kv-paged --kv-block-size "$3" > "$log" 2>&1 &
    PID=$!
    local i ok=0
    for i in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && { ok=1; break; }
        kill -0 $PID 2>/dev/null || break
        sleep 1
    done
    [ "$ok" = 1 ] || { echo "  $1: SERVER NEVER READY -- arm VOID" | tee -a "$OUT"; PID=""; return 1; }
    curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' --data-binary "@$D/req.json" > "$D/$1.json" 2>&1
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; sleep 2
    return 0
}

# ── ARM B FIRST: prove the marker can print, before any absence is read as evidence ────────────────
echo | tee -a "$OUT"
echo "ARM B -- champion, -lv 5: can DS4P-CONSUME print at all in this binary?" | tee -a "$OUT"
if run_arm B 1 64 5; then
    b_consume=$(grep -ac 'DS4P-CONSUME' "$D/B.log")
    b_banded=$(grep -ac 'DS4P-CONSUME banded' "$D/B.log")
    b_auto=$(grep -ac 'DS4P-CONSUME auto' "$D/B.log")
    b_refuse=$(grep -ac 'fails the paged capability contract' "$D/B.log")
    b_alarm=$(grep -ac 'ZERO layers have consumed' "$D/B.log")
    printf '  DS4P-CONSUME total=%s  banded=%s  auto=%s  |  refusals=%s  alarm=%s\n' \
        "$b_consume" "$b_banded" "$b_auto" "$b_refuse" "$b_alarm" | tee -a "$OUT"
    if [ "$b_consume" -eq 0 ]; then
        echo "  ⇒ **VOID for the whole pair.** The marker does not print even on a configuration that" | tee -a "$OUT"
        echo "    demonstrably pages (refusals=$b_refuse). Arm A's silence would mean nothing." | tee -a "$OUT"
        exit 2
    fi
    echo "  ⇒ marker CONFIRMED printable. Arm A's result is now readable." | tee -a "$OUT"
    # ★ The split matters: `auto` incrementing the SAME global counter is the live hypothesis for why
    #   the alarm stayed silent in the voided run.
    [ "$b_auto" -gt 0 ] && {
        echo "  ⚠ AUTO funnel consumed $b_auto times on this arch. That is the shared-counter" | tee -a "$OUT"
        echo "    hypothesis with evidence: the alarm cannot distinguish it from banded consumption." | tee -a "$OUT"; }
fi

# ── ARM A: the configuration that refuses every layer. Does the alarm fire? ────────────────────────
echo | tee -a "$OUT"
echo "ARM A -- champion OFF, bs=64: every attention layer should refuse (64 x head_dim > 8192)." | tee -a "$OUT"
if run_arm A 0 64 5; then
    a_refuse=$(grep -ac 'fails the paged capability contract' "$D/A.log")
    a_consume=$(grep -ac 'DS4P-CONSUME' "$D/A.log")
    a_banded=$(grep -ac 'DS4P-CONSUME banded' "$D/A.log")
    a_auto=$(grep -ac 'DS4P-CONSUME auto' "$D/A.log")
    a_alarm=$(grep -ac 'ZERO layers have consumed' "$D/A.log")
    printf '  refusals=%s  |  DS4P-CONSUME total=%s banded=%s auto=%s  |  alarm=%s\n' \
        "$a_refuse" "$a_consume" "$a_banded" "$a_auto" "$a_alarm" | tee -a "$OUT"
    echo | tee -a "$OUT"
    if [ "$a_refuse" -eq 0 ]; then
        echo "  ⇒ INCONCLUSIVE: no layer was refused, so this is not the failure LAW 6 exists for." | tee -a "$OUT"
        echo "    (head_dim may be <= 128 on this model, where bs=64 is legal without the champion.)" | tee -a "$OUT"
    elif [ "$a_alarm" -gt 0 ]; then
        echo "  ✅ **LAW 6 VALIDATED.** Every layer refused AND the engine alarm fired. Its silence" | tee -a "$OUT"
        echo "    elsewhere is therefore meaningful, and the five gates that rely on it stand." | tee -a "$OUT"
    elif [ "$a_banded" -eq 0 ] && [ "$a_auto" -gt 0 ]; then
        echo "  ⛔ **LAW 6 IS DEAD, and the mechanism is confirmed.** Every banded layer refused" | tee -a "$OUT"
        echo "    (banded consume = 0) while the AUTO funnel consumed $a_auto times, keeping the" | tee -a "$OUT"
        echo "    shared global counter non-zero. The alarm measures 'some layer somewhere', not" | tee -a "$OUT"
        echo "    'the layers being timed'. Replace LAW 6 with a BANDED-ONLY count at -lv 5." | tee -a "$OUT"
    else
        echo "  ⛔ **LAW 6 IS DEAD, mechanism NOT yet explained.** Layers were refused, consumption" | tee -a "$OUT"
        echo "    was $a_consume (banded=$a_banded auto=$a_auto), and the alarm still did not fire." | tee -a "$OUT"
        echo "    Do not guess the cause from here -- read llama-context.cpp around the counter." | tee -a "$OUT"
    fi
fi
echo "log: $OUT" | tee -a "$OUT"
