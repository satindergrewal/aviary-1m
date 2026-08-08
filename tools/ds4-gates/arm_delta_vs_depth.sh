#!/usr/bin/env bash
# DOES THE PAGED-vs-STATIC NUMERICAL DIFFERENCE GROW WITH CONTEXT DEPTH?
#
# ⚠ WHY THIS IS THE QUESTION, and why a text comparison cannot answer it. On 2026-08-09 `qwen3vlmoe`
# produced different text on the two arms at ~1.6k tokens. Graded: static's own top-2 were 0.025
# logprob apart -- a 1.03x TIE. The arms differ NUMERICALLY by ~0.1 logprob and it only became visible
# because the argmax was a coin flip. At 4k that is noise.
#
# At 512k-1M it may not be. Two things scale badly and neither is measurable by comparing strings:
#   1. the DELTA itself may grow with the number of accumulated chunks
#   2. even at a FIXED delta, the near-tie ENCOUNTER RATE grows with the number of generated positions
# ⇒ Either way, a cross-arm byte-equality gate at depth would file NUMERICS as DEFECTS. That is why
#   `gate_grade_divergence` exists, and this measures whether the clause's threshold is even the right
#   shape at depth or needs to scale.
#
# THE METRIC IS NOT "do the texts match". It is the arm-to-arm logprob delta on the SAME token:
#   pick static's top-1 token, read ITS logprob in both arms, take the difference.
# That is independent of whether a tie happened to flip, which is the confound that made the 4k result
# unreadable until it was graded.
#
# usage: arm_delta_vs_depth.sh <model.gguf> [depths...]      default depths: 1500 6000 24000
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

M="${1:-}"; shift 2>/dev/null || true
[ -f "$M" ] || { echo "usage: $0 <model.gguf> [depths...]" >&2; exit 2; }
DEPTHS=("$@"); [ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(1500 6000 24000)

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
T=${CLAUDE_JOB_DIR:-/tmp}/tmp; mkdir -p "$T"
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/armdelta-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"

echo "arm-delta vs depth: $(basename "$M")" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD)  depths: ${DEPTHS[*]}" | tee -a "$OUT"
echo "metric: logprob of STATIC's top-1 token, read in BOTH arms. Text equality is not the metric." | tee -a "$OUT"

probe() { # $1 port  $2 depth  -> json {tok: logprob, ...} for the top-5
    D="$2" python3 -c "
import json, os
# ⚠ MEASURED, NOT GUESSED. The first version used 11 tokens per filler sentence. The actual figure for
# this tokenizer is ~15.4: depth 1500 produced n_eval=2097, i.e. the 'depth' column overstated the real
# prompt by 40%, and at 6000 it overflowed -c and the PAGED arm was rejected outright:
#     queue_request: request 0 exceeds max context (9050 > 8192)
# which would have printed as '<arm failed to produce logprobs>' and read like a paged defect. The
# n_eval column is the truth; the depth label is only a target, and it must not silently exceed -c.
n = int(int(os.environ['D']) / 15.4)
p = ('The following is a list of numbered facts about geography. ' +
     ' '.join(f'Fact {i}: city number {i} lies on a river.' for i in range(1, max(n,2))) +
     ' Question: the capital of France is')
print(json.dumps({'prompt': p, 'n_predict': 1, 'temperature': 0, 'seed': 1,
                  'cache_prompt': False, 'n_probs': 5}))" > "$T/ad.json"
    curl -s --max-time 1800 -X POST "http://127.0.0.1:$1/completion" \
        -H 'Content-Type: application/json' -d @"$T/ad.json" \
      | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    e = (d.get('completion_probabilities') or [{}])[0]
    tl = e.get('top_logprobs') or []
    print(json.dumps({'n': d.get('tokens_evaluated', 0),
                      'p': {t['token']: t['logprob'] for t in tl},
                      'order': [t['token'] for t in tl]}))
except Exception:
    print('{}')"
}

# ⚠ PICK A FREE PORT, NEVER A FIXED ONE. This script used port 9500 and the first thing that happened
# after a `pkill` was that a llama-server SURVIVED the pattern kill, kept 9500, and every new server
# died with "exiting due to HTTP server error". The probe then returned {} and every row would have
# printed as an ARM FAILURE -- the same false picture as the -c overflow, from a different cause.
# arch_serve_gate has had pick_port since it was written, for exactly this. (scar: a pattern kill is
# not addressed to anyone in particular, 2026-08-07)
pick_port() {
    local p
    for p in $(seq 9500 9560); do
        lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }
    done
    return 1
}

run_arm() { # $1 arm  $2 depth -> echoes the probe json
    local flags=() port
    port=$(pick_port) || { echo '{}'; return; }
    [ "$1" = paged ] && flags=(--kv-paged)
    # ⚠ headroom for the sizing estimate being wrong AGAIN. A prompt that overflows -c is rejected by
    # the paged scheduler and produces a missing row that looks like a paging failure.
    local ctx=$(( $2 + $2 / 4 + 3072 ))
    "$SRV" -m "$M" -ngl 99 -c "$ctx" -np 1 -b 512 -ub 512 --port $port --no-warmup \
        "${flags[@]}" > "$T/ad-$1-$2.log" 2>&1 &
    local pid=$! i ok=0
    for i in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = "200" ] && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    # ⚠ SAY WHY, ON STDERR, WHEN AN ARM DIES. A bare {} becomes "<arm failed to produce logprobs>" in the
    # table, which reads as a PAGING result. Both times this fired tonight the cause was the harness --
    # a -c overflow, then a lurking server holding the port -- and both were only visible in this log.
    if [ "$ok" -ne 1 ]; then
        echo "  ARM $1 @ depth $2 DID NOT SERVE -- cause from its own log:" >&2
        grep -aE "GGML_ASSERT|error|exceeds max context|rejected|failed" "$T/ad-$1-$2.log" | tail -3 | cut -c1-160 | sed 's/^/    ! /' >&2
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
        echo '{}'; return
    fi
    probe $port "$2"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 2
}

printf '%8s %8s  %-12s %-12s %10s %10s  %s\n' depth n_eval static_top1 paged_top1 static_gap ARM_DELTA flip | tee -a "$OUT"
for d in "${DEPTHS[@]}"; do
    s=$(run_arm static "$d"); p=$(run_arm paged "$d")
    S="$s" P="$p" D="$d" python3 -c "
import json, os
s = json.loads(os.environ['S'] or '{}'); p = json.loads(os.environ['P'] or '{}')
if not s.get('order') or not p.get('order'):
    print('%8s   <arm failed to produce logprobs>' % os.environ['D']); raise SystemExit
st1 = s['order'][0]; pt1 = p['order'][0]
gap = s['p'][st1] - s['p'][s['order'][1]] if len(s['order']) > 1 else float('nan')
# ★ THE MAGNITUDE: the SAME token's logprob in both arms. Absent from paged's top-5 means the
#   difference is at least as large as paged's 5th-place margin -- reported, never silently skipped.
delta = (p['p'][st1] - s['p'][st1]) if st1 in p['p'] else float('nan')
print('%8s %8s  %-12r %-12r %10.5f %10.5f  %s'
      % (os.environ['D'], s.get('n'), st1, pt1, gap, delta, 'FLIP' if st1 != pt1 else '-'))
" | tee -a "$OUT"
done
echo "log: $OUT" | tee -a "$OUT"
echo "READ IT: ARM_DELTA flat across depth -> a fixed numerical offset; the tie threshold can stay fixed."
echo "         ARM_DELTA growing          -> the 256k-1M cross-arm gate needs a DEPTH-SCALED threshold,"
echo "                                       and paged correctness at depth is a real open question."
