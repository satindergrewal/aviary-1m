#!/usr/bin/env bash
# p17a_interleave_gate.sh — P1-7a: does a big prefill stall a live decode?
# (PLAN-DS4-PORTS P1-7 kill gate: live-slot decode degradation < 10% during a
# concurrent max-size prefill. Half (b) — no logits for prefill-only positions —
# is enforced by the chunked-prefill implementation itself; this measures half (a).)
#
# Arms against a RUNNING paged server (np >= 2):
#   SOLO: agent A decodes N tokens alone -> tpot_solo
#   MIX:  agent A starts, 0.5 s later agent B posts a LONG prompt (chunked prefill);
#         A's tpot_mix includes the prefill-overlap window.
# Report per-arm tpot + overall inflation; PASS if inflation < 10%.
#
# Usage: ./p17a_interleave_gate.sh <server-url> <long-prompt-json> [n_predict=512] [outfile]
set -euo pipefail
URL="${1:?server url}"; BIGREQ="${2:?path to long-prompt completion json}"
NPRED="${3:-512}"; OUT="${4:-/tmp/p17a-$(date +%Y%m%d-%H%M).txt}"

runA() { curl -s --max-time 600 "$URL/completion" \
    -d "{\"prompt\":\"Agent A holds a long soliloquy about networks:\",\"n_predict\":$NPRED,\"temperature\":0,\"cache_prompt\":false}" \
    -o "$1"; }

echo "== solo arm =="
runA /tmp/p17a-solo.json

echo "== mixed arm (B's big prefill lands 0.5 s into A) =="
runA /tmp/p17a-mix.json &
A_PID=$!
sleep 0.5
curl -s --max-time 600 "$URL/completion" --data @"$BIGREQ" -o /tmp/p17a-b.json
wait $A_PID

python3 - "$OUT" <<'EOF'
import json, sys, time
out = sys.argv[1]
def tpot(f):
    t = json.load(open(f)).get('timings', {})
    return t.get('predicted_ms', 0) / max(1, t.get('predicted_n', 1)), t.get('predicted_n')
solo, n_s = tpot('/tmp/p17a-solo.json')
mix,  n_m = tpot('/tmp/p17a-mix.json')
b = json.load(open('/tmp/p17a-b.json')).get('timings', {})
infl = (mix - solo) / solo * 100 if solo else float('nan')
verdict = 'PASS' if infl < 10 else 'FAIL'
lines = [
    f'# P1-7a chunked-prefill interleave gate, {time.strftime("%Y-%m-%d %H:%M")}',
    f'== VERDICT: {verdict} — live decode tpot inflation {infl:.1f}% during concurrent big prefill (bar <10%) ==',
    f'solo tpot {solo:.2f} ms/tok (n={n_s})   mixed tpot {mix:.2f} ms/tok (n={n_m})',
    f'B prefill: prompt_n={b.get("prompt_n")} prompt_ms={round(b.get("prompt_ms",0),1)}',
]
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if verdict == 'PASS' else 2)
EOF
