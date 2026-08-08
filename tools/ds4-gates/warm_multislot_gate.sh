#!/usr/bin/env bash
# ⚠ PRIVACY: writes into tools/ds4-gates/results/, which is TRACKED. Scrub on EXIT, as a trap and not a
# trailing call -- a trailing scrub is jumped over by the gate's own `exit`.
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
. "$(dirname "$0")/_gate_common.sh" 2>/dev/null || true
trap 'scrub_abs_paths "${OUT:-}"' EXIT
#
# WARM MULTI-SLOT GATE -- the regime `multislot_gate.sh` and `content_diff_probe.sh` both exclude.
#
# FINDING 1c (FINDINGS-2026-08-06.md, OPEN since 2026-08-06) reproduced ONLY warm:
#     WARM (a request finished on the slot first)   7 of 12 corrupt
#     COLD (concurrent pair is the first traffic)   0 of 24 corrupt
# Both existing gates go straight to the concurrent pair, so both are COLD by construction and neither
# can see 1c. multislot_gate.sh passed 12/12 on the fixed binary tonight and that says nothing about it.
#
# THIS GATE: prime the server with ONE request and LET IT FINISH, then fire the concurrent pair on the
# SAME live server. That is the only regime that ever reproduced.
#
# ⚠ The prime is the whole experiment. If it is skipped or fails, this degrades silently into the COLD
# gate -- so the prime's completion is ASSERTED, and a failed prime aborts the rep rather than scoring it.
set -uo pipefail
WT=$HOME/Documents/GitHub/llama.cpp-ds4ports
SRV=$WT/build-metal/bin/llama-server
M=${MS_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
REPS=${MS_REPS:-6}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/warmslot; mkdir -p "$LOGDIR"
OUT=$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/warmslot-$(date +%Y%m%d-%H%M).txt
PORT=${MS_PORT:-21500}
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain|wc -l|tr -d ' ')  model=$(basename "$M")  reps=$REPS  regime=WARM" | tee "$OUT"

# ⚠ N distinct prompts with UNIQUE answers -- a swap between ANY two sequences must be detectable.
PROMPTS=(
  'Count from one to five, in words, separated by commas. Answer only.'
  'Name the first five letters of the alphabet, separated by commas. Answer only.'
  'Name the first five planets from the Sun, separated by commas. Answer only.'
  'Name the days Saturday and Sunday and the day between Monday and Wednesday, separated by commas. Answer only.'
)
EXPECT=( 'three.*four.*five' 'b, *c, *d' 'venus|mercury' 'tuesday' )
PRIME='What is two plus two? Answer with the number only.'

ask() {
  local ep body
  if [ "${MS_CHAT:-0}" = "1" ]; then
      ep="/v1/chat/completions"
      body="{\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1")}],\"max_tokens\":256,\"temperature\":0,\"seed\":1}"
  else
      ep="/completion"
      body="{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1"),\"n_predict\":256,\"temperature\":0,\"seed\":1,\"cache_prompt\":false}"
  fi
  curl -s -w '\nHTTP:%{http_code}' --max-time 300 -X POST "http://127.0.0.1:$PORT$ep" -H 'Content-Type: application/json' \
        -d "$body" \
        | python3 -c 'import json,sys
raw = sys.stdin.read()
code = raw.rsplit("HTTP:",1)[1].strip() if "HTTP:" in raw else "???"
raw = raw.rsplit("\nHTTP:",1)[0]
sys.stdin = None
try: d=json.loads(raw)
except Exception: print(f"<UNPARSEABLE http={code}>"); raise SystemExit
c = d.get("content") or (d.get("choices",[{}])[0].get("message",{}) or {}).get("content","") or ""
import re as _re
c = _re.sub(r"<think>.*?</think>", " ", c, flags=_re.S)       # answer lives AFTER the think block
c = _re.sub(r"^.*?(?:Thinking Process|<think>).*$", " ", c, flags=_re.S) if "</think>" not in d.get("content","") and "<think>" in d.get("content","") else c
out = ("<ERR:"+str(d["error"].get("message"))[:40]+">") if "error" in d else c.strip().replace(chr(10)," ")[:120]
print(out if out.strip() else f"<EMPTY http={code}>")'; }

probe() { # $1=tag $2=flags
    local tag="$1" flags="$2"
    pkill -x llama-server >/dev/null 2>&1; sleep 2
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 nohup "$SRV" -m "$M" -ngl 99 -c 8192 -np ${MS_NP:-2} -b 512 -ub 512 \
        --port $PORT --no-warmup -lv ${MS_LV:-4} ${MS_CHAT:+--jinja} $flags > "$LOGDIR/$tag.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
        kill -0 $pid 2>/dev/null || { echo "$tag: SERVER DIED" | tee -a "$OUT"; return 1; }; sleep 1; done

    # ★ CONSUMER PRECONDITION -- for the PAGED arm only. If no layer consumed the paged context, a CLEAN
    # verdict is vacuous: it measured the static path wearing paged flags. Requires MS_LV=5.
    # (Earlier tonight "Gemma4 runs zero paged layers" was asserted from a -lv 4 log that COULD NOT print
    #  the marker, then retracted. Absence is only evidence when the marker can fire.)
    local consumed=-1
    case "$flags" in *--kv-paged*)
        if [ "${MS_LV:-4}" -ge 5 ]; then consumed=0; fi ;;
    esac

    # ★ THE PRIME -- one request, run to completion, on the same slots the pair will use.
    local p; p=$(ask "$PRIME")
    case "$p" in ""|"<UNPARSEABLE>"|"<ERR:"*) echo "$tag: PRIME FAILED ($p) -- rep aborted, NOT scored (would degrade to COLD)" | tee -a "$OUT"
        # ⚠ AND SAY WHY. I first recorded this site as "log not in scope at that point" and that was
        # WRONG -- the server's log is $LOGDIR/$tag.log, written at line 70, in scope here. A reason
        # given for leaving something undone is a claim like any other and gets checked like one.
        command -v gate_cause_from_log >/dev/null 2>&1 && gate_cause_from_log "$LOGDIR/$tag.log" "$tag prime" | tee -a "$OUT"
        kill $pid 2>/dev/null; return 1;; esac

    # now the concurrent pair, WARM
    local ra rb
    local n=${MS_NP:-2} i pids=()
    for i in $(seq 1 "$n"); do
        ask "${PROMPTS[$((i-1))]}" > "$LOGDIR/$tag-S$i.txt" & pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i"; done
    local -a R=()
    for i in $(seq 1 "$n"); do R+=("$(cat "$LOGDIR/$tag-S$i.txt")"); done

    # ★ THIRD REQUEST, SEQUENTIAL, AFTER THE PAIR (Grok #7991). prime->pair only measures the pair
    # reading the PRIME's residue; tonight's defect struck the request AFTER the poisoning event, so
    # what the PAIR leaves behind is a half no run has covered. Same prompt as the prime, so its
    # correct answer is already known and a change is unambiguous.
    local p3; p3=$(ask "$PRIME")
    [ "$consumed" = "0" ] && consumed=$(grep -ac 'DS4P-CONSUME' "$LOGDIR/$tag.log" 2>/dev/null)
    local v3=OK; echo "$p3" | grep -qE '\b4\b|four' || v3=BAD
    kill $pid 2>/dev/null; sleep 1

    # A must contain counting, B must contain letters. Cross-contamination = A holding B's answer.
    local verdict=CLEAN i j
    for i in $(seq 1 "$n"); do
        # each sequence must contain ITS OWN marker...
        echo "${R[$((i-1))]}" | grep -qiE "${EXPECT[$((i-1))]}" || verdict=SUSPECT
        # ...and must NOT contain any OTHER sequence's marker. N*(N-1) ordered pairs, not one.
        for j in $(seq 1 "$n"); do
            [ "$i" = "$j" ] && continue
            echo "${R[$((i-1))]}" | grep -qiE "${EXPECT[$((j-1))]}" && verdict=CROSS-CONTAMINATED
        done
    done
    [ "$v3" = OK ] || verdict=POST-PAIR-DIRTY   # the pair left something behind
    [ "$consumed" = "0" ] && verdict=VOID-NO-CONSUMER   # paged flags set, no layer consumed -> vacuous
    printf "  %-14s np=%-2s prime=%-4s post=%-4s consume=%-6s -> %s\n" "$tag" "$n" "${p:0:4}" "${p3:0:4}" "$consumed" "$verdict" | tee -a "$OUT"
    for i in $(seq 1 "$n"); do printf "      S%-2s %s\n" "$i" "$(echo "${R[$((i-1))]}" | head -c 62)" | tee -a "$OUT"; done
    echo "$verdict"
}

declare -a V=()
for r in $(seq 1 "$REPS"); do
    if [ $((r % 2)) -eq 1 ]; then
        v=$(probe "static-r$r" "" | tail -1); V+=("static:$v")
        v=$(probe "paged-r$r"  "--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128" | tail -1); V+=("paged:$v")
    else
        v=$(probe "paged-r$r"  "--kv-paged --kv-block-size 16 -ngpub 512 -ncpub 128" | tail -1); V+=("paged:$v")
        v=$(probe "static-r$r" "" | tail -1); V+=("static:$v")
    fi
done
echo "-----" | tee -a "$OUT"
sc=0; sb=0; pc=0; pb2=0
for v in "${V[@]}"; do case "$v" in static:CLEAN) sc=$((sc+1));; static:*) sb=$((sb+1));; paged:CLEAN) pc=$((pc+1));; paged:*) pb2=$((pb2+1));; esac; done
echo "static: $sc clean / $sb bad     paged: $pc clean / $pb2 bad" | tee -a "$OUT"
echo "verdicts: ${V[*]}" | tee -a "$OUT"
if [ "$sb" -ne 0 ]; then
    echo "WARM MULTI-SLOT GATE: **VOID** -- static (no paging) scored $sb bad. The gate is measuring itself, not the code." | tee -a "$OUT"
    echo "  (a dirty static arm can never indict paging; fix the harness before reading the paged column)" | tee -a "$OUT"
elif [ "$pb2" -eq 0 ]; then
    echo "WARM MULTI-SLOT GATE: PASS -- both paths clean across $REPS warm reps (1c did not reproduce)" | tee -a "$OUT"
else
    echo "WARM MULTI-SLOT GATE: FAIL -- paged $pb2 bad, static CLEAN (1c REPRODUCES on this binary)" | tee -a "$OUT"
fi
echo "log: $OUT" | tee -a "$OUT"
