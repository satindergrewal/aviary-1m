#!/usr/bin/env bash
# HOW OFTEN WOULD A CROSS-ARM BYTE-EQUALITY GATE FALSE-FAIL? A number, not an intuition.
#
# ⚠ THE PROBLEM THIS SOLVES. `arm_delta_vs_depth.sh` measures the paged-vs-static logprob delta at ONE
# position per depth. Four samples of that on 2026-08-09 spanned 0.002 to 0.168 -- a factor of ~80 --
# so the delta is POSITION-DEPENDENT, and one sample per depth cannot characterise a tail. If large-delta
# positions are ~1 in 20, four samples usually miss them and print a reassuring flat curve. A reassuring
# flat curve is exactly what would then be carried into a 256k spend decision.
#
# ⚠ WHY NOT JUST GENERATE 200 TOKENS ON BOTH ARMS AND COMPARE. Both arms are greedy and deterministic,
# so the moment one token differs the two continuations diverge and later positions are no longer
# comparable -- you would be measuring drift, not per-position disagreement. Teacher-forcing the same
# continuation into both arms needs a logits API this server does not expose.
#
# ★ THE DECOMPOSITION THAT WORKS WITH THE EXISTING API. A flip needs TWO things to meet:
#       a LARGE arm-to-arm delta   AND   a SMALL top-2 margin at that position
#   Neither alone does anything. So measure them separately and combine:
#
#       margin distribution  <- ONE static run, n_probs over MANY generated positions. Cheap, one arm.
#       delta magnitude      <- the paired first-token probes in arm_delta_vs_depth.sh
#       predicted flip rate  =  fraction of positions whose margin < the delta
#
#   That predicted rate IS the false-FAIL rate of a byte-equality gate over that many tokens, and it is
#   the quantity the 0.35 tie threshold in _gate_common.sh should be justified against.
#
# usage: margin_distribution.sh <model.gguf> [n_generate] [prompt_depth] [--kv-paged]
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

M="${1:-}"; N="${2:-200}"; DEPTH="${3:-6000}"; shift 3 2>/dev/null || true
[ -f "$M" ] || { echo "usage: $0 <model.gguf> [n_generate] [prompt_depth] [extra server args]" >&2; exit 2; }
EXTRA=("$@")

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
T=${CLAUDE_JOB_DIR:-/tmp}/tmp; mkdir -p "$T"
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/margins-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"

# ⚠ pick a free port. A survivor of a pattern kill held a hardcoded one earlier tonight and every new
# server died with "exiting due to HTTP server error", which the caller rendered as an arm failure.
pick_port() { local p; for p in $(seq 9600 9660); do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }; done; return 1; }
PORT=$(pick_port) || { echo "no free port" >&2; exit 2; }

echo "margin distribution: $(basename "$M")  n_generate=$N  prompt~${DEPTH}tok  ${EXTRA[*]}" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD)" | tee -a "$OUT"

CTX=$(( DEPTH + DEPTH / 4 + N + 3072 ))
"$SRV" -m "$M" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 --port "$PORT" --no-warmup \
    "${EXTRA[@]}" > "$T/md.log" 2>&1 &
PID=$!
for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = "200" ] && break
    kill -0 $PID 2>/dev/null || break
    sleep 1
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" != "200" ]; then
    echo "DID NOT SERVE -- cause from its own log:" | tee -a "$OUT"
    grep -aE "GGML_ASSERT|error|exceeds max context|rejected|failed" "$T/md.log" | tail -3 | cut -c1-170 | sed 's/^/  ! /' | tee -a "$OUT"
    kill $PID 2>/dev/null; exit 1
fi

D="$DEPTH" NP="$N" python3 -c "
import json, os
n = int(int(os.environ['D']) / 15.4)
p = ('The following is a list of numbered facts about geography. ' +
     ' '.join(f'Fact {i}: city number {i} lies on a river.' for i in range(1, max(n,2))) +
     ' \\n\\nEssay. The rivers of Europe have shaped its cities in ways that')
print(json.dumps({'prompt': p, 'n_predict': int(os.environ['NP']), 'temperature': 0,
                  'seed': 1, 'cache_prompt': False, 'n_probs': 2}))" > "$T/md.json"

curl -s --max-time 3600 -X POST "http://127.0.0.1:$PORT/completion" \
    -H 'Content-Type: application/json' -d @"$T/md.json" > "$T/md.out"
kill $PID 2>/dev/null; wait $PID 2>/dev/null

python3 - "$T/md.out" <<'PY' | tee -a "$OUT"
import json, sys
d = json.load(open(sys.argv[1]))
cp = d.get('completion_probabilities') or []
margins = []
for e in cp:
    tl = e.get('top_logprobs') or []
    if len(tl) >= 2:
        margins.append(tl[0]['logprob'] - tl[1]['logprob'])
if not margins:
    print("NO MARGINS -- the run produced no top_logprobs. Do not read this as 'no near-ties'.")
    raise SystemExit(2)
# ⚠⚠ REFUSE, DO NOT REPORT. The first run of this script returned ONE position -- the model emitted EOS
# immediately on a raw completion -- and the table below duly printed "0.00% flip rate" at every delta,
# from n=1. A reassuring, precise, meaningless answer that LOOKS like the result I was hoping for. A
# percentile from a handful of samples is not a distribution, and a flip rate from one position is not
# a rate. Same class as every other trap tonight: a degenerate result rendering as a clean one.
MIN_POS = 50
if len(margins) < MIN_POS:
    print(f"REFUSING: only {len(margins)} position(s) measured, need >= {MIN_POS}.")
    print(f"  stop_type={d.get('stop_type')!r} tokens_predicted={d.get('tokens_predicted')}")
    print("  A flip RATE needs a distribution. Fix the prompt so generation continues (a base model")
    print("  on a raw completion will emit EOS immediately if the prompt reads as finished), or pass")
    print("  a chat template. This says NOTHING about near-ties either way.")
    raise SystemExit(2)
# ⚠⚠ AND THE SECOND DEGENERACY, ONE LEVEL UP FROM THE FIRST. After fixing "only 1 position", the probe
# happily measured 300 -- of a ROTE LIST CONTINUATION ("Fact 391: city number 391 lies on a river."),
# 300 tokens with **20 distinct values**, distinct ratio 0.07. Margins came out 3.5-12 logprob and the
# flip rate printed 0.00% at every delta. True, and about the FILLER, not about text: a model copying a
# pattern is maximally confident, so the distribution contains no near-ties BY CONSTRUCTION.
# The qwen3vlmoe flip happened at the first genuinely uncertain position after a question -- exactly the
# kind this sample had none of. Fixing "too few samples" produced "many samples of a trivial task".
toks = [e.get('token') for e in cp]
ratio = len(set(toks)) / max(len(toks), 1)
MIN_RATIO = 0.20
if ratio < MIN_RATIO:
    print(f"REFUSING: generated text is ROTE -- {len(set(toks))} distinct of {len(toks)} tokens "
          f"(ratio {ratio:.2f} < {MIN_RATIO}).")
    print(f"  sample: {d.get('content','')[:90]!r}")
    print("  A model copying a pattern is maximally confident, so this distribution has no near-ties")
    print("  BY CONSTRUCTION and its 0% flip rate is about the PROMPT, not about the kernels. Use a")
    print("  prompt that generates varied prose.")
    raise SystemExit(2)
margins.sort()
n = len(margins)
print(f"  distinct-token ratio: {ratio:.2f} ({len(set(toks))}/{len(toks)}) -- above the {MIN_RATIO} floor")
print(f"  positions measured: {n}   prompt_tokens={d.get('tokens_evaluated')}")
print(f"  margin percentiles: p1={margins[n//100]:.5f} p5={margins[n//20]:.5f} "
      f"p25={margins[n//4]:.5f} median={margins[n//2]:.5f}")
print()
print("  PREDICTED FLIP RATE -- fraction of positions whose top-2 margin is under a given delta.")
print("  Read as: an arm-to-arm delta of D flips this share of tokens, i.e. a byte-equality gate")
print("  over N generated tokens false-fails with probability ~ 1-(1-rate)^N.")
for delta in (0.002, 0.005, 0.02, 0.05, 0.168, 0.35):
    k = sum(1 for m in margins if m < delta)
    rate = k / n
    # probability at least one flip over a 200-token and a 2000-token generation
    p200 = 1 - (1 - rate) ** 200
    p2000 = 1 - (1 - rate) ** 2000
    print(f"    delta {delta:<6} -> {k:4d}/{n} = {rate:6.2%}   P(>=1 flip in 200 tok)={p200:6.1%}"
          f"   in 2000 tok={p2000:6.1%}")
print()
print("  ⚠ SCOPE: one arch, one quant, one prompt, greedy. The margin distribution is a property of the")
print("  MODEL at these positions; the delta is a property of the KERNEL PAIR. This file measures only")
print("  the first. Combining them predicts a rate -- it does not measure one.")
PY
echo "log: $OUT" | tee -a "$OUT"

# ⚠ SCRUB ON EXIT. Both these scripts print `log: $OUT` -- an ABSOLUTE path containing the home
# directory -- into the result file they then commit. They SOURCED _no_abs_paths.sh and never CALLED
# it, so the include looked like protection and did nothing. Caught by privacy_guard.sh refusing the
# commit; the arch gate has always called it from its cleanup trap. Sourcing is not calling.
scrub_abs_paths "$OUT" 2>/dev/null || true
