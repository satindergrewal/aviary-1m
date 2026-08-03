#!/usr/bin/env bash
# agent_throughput_gate.sh — Q3: N-agent continuous-batching throughput vs sequential.
#
# Measures wall-clock for N identical completion requests fired CONCURRENTLY at a
# --kv-paged server vs the SAME N requests fired ONE AT A TIME (sequential baseline,
# same server, same slots — isolates scheduler concurrency, not server restart noise).
# PASS bar: concurrent wall <= sequential wall / TARGET_X.
#
# The sequential arm runs FIRST (cold-ish cache both arms; cache_prompt=false everywhere
# so prefix reuse cannot flatter either arm). temp 0 for determinism; distinct prompts
# per agent so the scheduler cannot dedupe.
#
# Usage: ./agent_throughput_gate.sh <server-url> <n-agents> <target-x> <n-predict> [outfile]
set -euo pipefail
URL="${1:?server url}"; N="${2:-16}"; TARGETX="${3:-6}"; NPRED="${4:-128}"
OUT="${5:-/tmp/agent-throughput-$(date +%Y%m%d-%H%M).txt}"
TS() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

req() { # $1 = agent index
    curl -s --max-time 600 "$URL/completion" \
        -d "{\"prompt\":\"Agent $1 reporting. The mission status is\",\"n_predict\":$NPRED,\"temperature\":0,\"cache_prompt\":false}" \
        -o /tmp/atg-resp-$1.json
}

echo "== sequential arm: $N requests one-at-a-time =="
T0=$(TS)
for i in $(seq 1 "$N"); do req "$i"; done
T1=$(TS)
SEQ=$(python3 -c "print(f'{$T1-$T0:.2f}')")

echo "== concurrent arm: $N requests fired together =="
T2=$(TS)
for i in $(seq 1 "$N"); do req "$i" & done
wait
T3=$(TS)
CON=$(python3 -c "print(f'{$T3-$T2:.2f}')")

python3 - "$SEQ" "$CON" "$N" "$TARGETX" "$NPRED" "$OUT" <<'EOF'
import json, sys, time
seq, con, n, tx, npred, out = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
ok_all = True
toks = []
for i in range(1, n+1):
    try:
        d = json.load(open(f'/tmp/atg-resp-{i}.json'))
        pn = d.get('timings', {}).get('predicted_n')
        toks.append(pn)
        ok_all &= (pn == npred)
    except Exception:
        toks.append(None); ok_all = False
x = seq / con if con > 0 else 0
verdict = 'PASS' if (x >= tx and ok_all) else 'FAIL'
lines = [
    f'# Q3 N-AGENT THROUGHPUT GATE, {time.strftime("%Y-%m-%d %H:%M")}',
    f'== VERDICT: {verdict} — {n} agents concurrent vs sequential = {x:.2f}x (bar {tx}x, all-complete={ok_all}) ==',
    f'sequential wall: {seq:.2f} s   concurrent wall: {con:.2f} s   n_predict per agent: {npred}',
    f'predicted_n per agent: {toks}',
]
open(out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(0 if verdict == 'PASS' else 2)
EOF
