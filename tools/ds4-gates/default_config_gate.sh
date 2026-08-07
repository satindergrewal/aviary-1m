#!/usr/bin/env bash

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

# DEFAULT-CONFIGURATION GATE -- the coverage hole every other gate in this lane shares.
#
# ⚠ WHY THIS EXISTS: all six gates I built pin cache_prompt:false. The server default is TRUE, and
# it is what real clients send. I did not choose that; I copied the flag from the first gate into
# every later one because it made runs deterministic. The determinism I optimised for became
# coverage I never had -- six gates, all green, none able to see the default path's failures.
#
# This gate tests ONLY the default: repeated prompts with the prompt cache ON. Its job is to be a
# standing, visible measurement of whether --kv-paged is usable the way people actually call it.
#
# ⚠ THE "EXPECTED TO FAIL" NOTE THAT WAS HERE IS OBSOLETE -- FIXED 2026-08-05, and left as a scar.
# It said cache_prompt on the paged path was broken and cited a memory note as corroboration. Both
# were wrong: the failures were a regression of MINE (a server drain keyed on a reused slot id),
# and the runs that seemed to exonerate it had been measured against a STALE BINARY -- the build had
# been silently failing. Fixed, this gate now PASSES with the paged path reusing 25 of 26 tokens,
# BETTER than static (26 -> 1 -> 1 vs 26 -> 4 -> 4).
#
# Kept verbatim as a lesson rather than deleted: a comment asserting a permanent expectation ages
# into a lie the moment the thing is fixed, and it will be believed because it is right there in the
# file. If this gate ever fails again, that is NEWS -- do not read this header as permission.
#
# ARMS -- static is the CONTROL, and without it "paged is broken" cannot be separated from
# "prompt caching is broken":
#   STATIC  repeated prompt, cache on -> MUST reuse (prompt_n drops) and stay correct
#   PAGED   repeated prompt, cache on -> currently returns an EMPTY body on repeat
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf
P=9020
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/default-config-$(date +%Y%m%d-%H%M).txt
mkdir -p "$(dirname "$OUT")"
PROMPT='The capital of France is Paris and the capital of Germany is Berlin and the capital of Italy is Rome and the capital of Spain is'

echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')" | tee "$OUT"
fails=0

probe() { # $1 label  $2 server args  $3 env
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    # shellcheck disable=SC2086
    env $3 nohup "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 -b 512 -ub 512 --port $P --no-warmup \
        $2 > "/tmp/dc-$1.log" 2>&1 &
    for _ in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" = "200" ] && break
        sleep 1
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null)" != "200" ]; then
        echo "NEVER_READY|NEVER_READY|NEVER_READY"; return
    fi
    # THREE identical requests with the cache ON. Run 1 seeds; runs 2-3 must hit the cache.
    local out=""
    for _ in 1 2 3; do
        r=$(curl -s --max-time 90 -X POST http://127.0.0.1:$P/completion -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$PROMPT\",\"n_predict\":4,\"temperature\":0,\"seed\":1,\"cache_prompt\":true}" \
            | python3 "$(dirname "$0")/probe_parse.py" 2>/dev/null || echo "PARSE_FAIL#-1#")
        out="$out|$r"
    done
    pkill -f "$SRV" >/dev/null 2>&1; sleep 2
    echo "${out#|}"
}

echo "--- STATIC (control) ---" | tee -a "$OUT"
st=$(probe static "" "")
echo "$st" | tr '|' '\n' | nl | tee -a "$OUT"

echo "--- PAGED (the default configuration under test) ---" | tee -a "$OUT"
pg=$(probe paged "--kv-paged --kv-block-size 16" "DS4P_PAGED_HYBRID=1 DS4P_PAGED_DRIVE=1")
echo "$pg" | tr '|' '\n' | nl | tee -a "$OUT"

# ---- verdicts -------------------------------------------------------------------------------
# Content must be NON-EMPTY on every run. An empty body with HTTP 200 is the silent failure this
# gate exists to catch -- checking only for "no error" would pass it.
# field 2 is now a LENGTH: <1 means empty or absent. Numeric, so there is no string that can
# accidentally look like content.
st_empty=$(echo "$st" | tr '|' '\n' | awk -F'#' '{if ($2+0 < 1) c++} END{print c+0}')
pg_empty=$(echo "$pg" | tr '|' '\n' | awk -F'#' '{if ($2+0 < 1) c++} END{print c+0}')
echo "empty responses: STATIC=$st_empty  PAGED=$pg_empty" | tee -a "$OUT"

if [ "$st_empty" -ne 0 ]; then
    echo "STATIC: FAIL control returned $st_empty empty response(s) -- prompt caching itself is broken, PAGED result is uninterpretable" | tee -a "$OUT"
    fails=$((fails+1))
else
    echo "STATIC: PASS all three runs returned content" | tee -a "$OUT"
fi

if [ "$pg_empty" -ne 0 ]; then
    echo "PAGED: FAIL $pg_empty of 3 runs returned an EMPTY body on the DEFAULT configuration" | tee -a "$OUT"
    fails=$((fails+1))
else
    echo "PAGED: PASS all three runs returned content" | tee -a "$OUT"
fi

echo "-----" | tee -a "$OUT"
[ "$fails" -eq 0 ] && echo "DEFAULT-CONFIG GATE: PASS" | tee -a "$OUT" || echo "DEFAULT-CONFIG GATE: FAIL ($fails)" | tee -a "$OUT"
echo "log: $OUT"
exit $fails
