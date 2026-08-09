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

# ★★ LONG-CONTEXT MODE (added 2026-08-10) -- THE EMPTY COMPOSITION CELL.
#
# The `-np>1` corruption is FIXED and closed by measurement (c2f28a79d; 00cb274, e404116). What is
# NOT closed is the COMPOSITION: that closure ran at `-c 8192`, block 16, short prompts, while every
# parity number runs `-np 1`, block 64, 400k fill. **Neither crosses the other's constant**, so
# "multi-slot is clean" and "paging is 1.97x at 512k" have never been true of the same run.
#
# ⚠ The board said this was "warm_multislot_gate with MS_CTX raised". That understated it: `-c 8192`
# and `--kv-block-size 16` were HARDCODED, and short prompts cannot cross a block table at all.
# Three parameters, and a prompt builder.
#
# ⚠⚠ DEFAULTS REPRODUCE THE PREVIOUS BEHAVIOUR, WITH ONE NAMED EXCEPTION. MS_CTX=8192, MS_BLK=16,
# MS_FILL=0 give exactly the old gate, so the 2026-08-09 closure it produced is not retroactively
# re-scoped by an edit made today. A gate whose meaning changes under you is how a green from last
# week starts describing a different experiment.
#
# ⚠ THE EXCEPTION, STATED BECAUSE I FIRST WROTE "BYTE-IDENTICAL" AND IT STOPPED BEING TRUE ONE EDIT
# LATER: the paged pool is now DERIVED (see NGPUB below), so at the old defaults it is **579 blocks
# instead of the hardcoded 512**. The old 512 was exactly 8192/16 -- the context with ZERO headroom,
# tight by accident rather than by design. A LARGER pool cannot turn a clean run dirty, so the
# 2026-08-09 closure still holds; but "byte-identical" was a claim, it became false, and a stale
# claim in a comment is the thing this directory has the most scars about.
#
#   MS_CTX   server context           (default 8192)
#   MS_BLK   --kv-block-size          (default 16)
#   MS_FILL  filler tokens per seq    (default 0 = OFF, the original short-prompt gate)
#
# The filler is UNIQUE PER SEQUENCE, not shared. A shared prefix would let the slots hold identical
# KV, and a cross-slot read of identical bytes is INVISIBLE -- the gate would pass by construction.
# That is the same shape as the arch matrix greens that could not see past one block.
fill_prompt() { # $1 = 1-based sequence index, $2 = base prompt
    if [ "${MS_FILL:-0}" -le 0 ] 2>/dev/null; then printf '%s' "$2"; return; fi
    python3 - "$1" "$2" "${MS_FILL:-0}" <<'PY'
import sys
i, base, fill = sys.argv[1], sys.argv[2], int(sys.argv[3])
# ~12 tokens per line; unique text per sequence so no two slots can hold matching KV
n = max(1, fill // 12)
body = "".join(f"Sequence {i} archival record line {k:06d}.\n" for k in range(n))
sys.stdout.write(body + "\n" + base)
PY
}

# ⚠ N distinct prompts with UNIQUE answers -- a swap between ANY two sequences must be detectable.
PROMPTS=(
  'Count from one to five, in words, separated by commas. Answer only.'
  'Name the first five letters of the alphabet, separated by commas. Answer only.'
  'Name the first five planets from the Sun, separated by commas. Answer only.'
  'Name the days Saturday and Sunday and the day between Monday and Wednesday, separated by commas. Answer only.'
)
EXPECT=( 'three.*four.*five' 'b, *c, *d' 'venus|mercury' 'tuesday' )
PRIME='What is two plus two? Answer with the number only.'

# ⚠⚠ THE PAGED POOL WAS HARDCODED AT `-ngpub 512 -ncpub 128`, AND THAT NUMBER WAS THE OLD CELL.
#     512 blocks x block 16 = 8192 tokens = EXACTLY the `-c 8192` this gate used to run.
# So the pool silently fitted the context it was written for and nothing tied the two together.
# Measured 2026-08-10, first run of the long-context cell: block 64, two concurrent 40,006-token
# prompts needing 80,012 tokens against a pool of 512 x 64 = 32,768. Both sequences came back
#     <ERR:paged KV: the scheduler cannot make progress>
# ⇒ Not a paging defect. **A constant sized for one experiment, silently carried into another** --
#   and the engine refused by DESIGN rather than crashing, which is the behaviour this lane asked
#   for elsewhere and is worth recording as working.
# ⇒ DERIVE it from the two things it actually depends on, so it can never again be right by accident:
NGPUB=${MS_NGPUB:-$(( (${MS_CTX:-8192} / ${MS_BLK:-16}) * 110 / 100 + 16 ))}   # +10% headroom
NCPUB=${MS_NCPUB:-$(( NGPUB / 4 ))}

ask() {
  local ep body
  if [ "${MS_CHAT:-0}" = "1" ]; then
      ep="/v1/chat/completions"
      body="{\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1")}],\"max_tokens\":${MS_NPRED:-256},\"temperature\":0,\"seed\":1}"
  else
      ep="/completion"
      body="{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1"),\"n_predict\":${MS_NPRED:-256},\"temperature\":0,\"seed\":1,\"cache_prompt\":false}"
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
tp, st = d.get("tokens_predicted", -1), d.get("stop_type", "?")
out = ("<ERR:"+str(d["error"].get("message"))[:40]+">") if "error" in d else c.strip().replace(chr(10)," ")[:120]
# ⚠⚠ AN EMPTY ANSWER FROM A TRUNCATED GENERATION IS A HARNESS FAULT, NOT A WRONG ANSWER, AND
# CALLING IT SUSPECT POINTS THE READER AT THE CODE INSTEAD OF THE GATE. Measured 2026-08-10 on the
# 35B at a 40k prompt: eval time = 256 tokens against n_predict=256, raw content opening with
# "<think>" and never closing -- the reasoning model spent its entire budget thinking and the
# stripper correctly returned nothing. That is attempt-log failure #2 recurring at a larger prompt.
# ⇒ Name it, so the verdict says WHICH thing broke. "refuted needs its condition."
if not out.strip() and st == "limit":
    out = f"<TRUNCATED n={tp} stop=limit -- raise MS_NPRED>"
print(out if out.strip() else f"<EMPTY http={code} n={tp} stop={st}>")'; }

probe() { # $1=tag $2=flags
    local tag="$1" flags="$2"
    pkill -x llama-server >/dev/null 2>&1; sleep 2
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 nohup "$SRV" -m "$M" -ngl 99 -c ${MS_CTX:-8192} -np ${MS_NP:-2} -b 512 -ub 512 \
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
        ask "$(fill_prompt "$i" "${PROMPTS[$((i-1))]}")" > "$LOGDIR/$tag-S$i.txt" & pids+=($!)
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
    # ⚠⚠ AT THE DEFAULT MS_LV=4 THE LINE ABOVE NEVER RUNS, so `consumed` stays -1 and every paged
    # verdict in this gate has been UNVERIFIED rather than verified. Recording -1 instead of 0 was
    # right -- absence is only evidence when the marker can fire -- but "not checked" is where it
    # stopped, and on 2026-08-09 an unchecked paged arm produced 4.5 hours of static timings reported
    # as parity.
    # ⇒ There IS a check that works at -lv 4: the engine's own. llama-context.cpp evaluates
    #   ds4p_paged_consumer_count() == 0 after 8 decodes and warns at WARN level. Use it when the
    #   DEBUG marker is unavailable, so the default configuration is verified rather than silent.
    if [ "$consumed" = "-1" ]; then
        case "$flags" in *--kv-paged*)
            if command -v gate_assert_paged_consumed >/dev/null 2>&1; then
                if gate_assert_paged_consumed "$LOGDIR/$tag.log" "$tag" 512 >/dev/null 2>&1; then
                    consumed=1
                else
                    consumed=0
                    echo "$tag: ⚠ paged pool allocated but NO LAYER CONSUMED IT (engine alarm fired)" | tee -a "$OUT"
                    echo "      -> this arm is STATIC wearing paged flags; its verdict is vacuous." | tee -a "$OUT"
                fi
            fi ;;
        esac
    fi
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
    # ★ THE GATE MUST STATE THE SIZE OF THE QUESTION IT ANSWERED. A CLEAN here means nothing unless
    # the prompts actually crossed a block table -- the arch matrix shipped 21 greens that could not
    # see past a single block, and the convention adopted 2026-08-10 is that each green names its
    # own coverage. MS_FILL is a TARGET in tokens computed from a ~12-tok/line ESTIMATE; this reads
    # what the server actually tokenised, so the estimate is never the thing on the record.
    local ptok blocks
    ptok=$(grep -oE 'prompt eval time.*/ *[0-9]+ tokens' "$LOGDIR/$tag.log" 2>/dev/null \
           | grep -oE '/ *[0-9]+ tokens' | grep -oE '[0-9]+' | sort -rn | head -1)
    ptok=${ptok:-0}
    blocks=$(( ptok / ${MS_BLK:-16} ))
    printf "  %-14s np=%-2s prime=%-4s post=%-4s consume=%-6s ptok=%-7s blocks=%-6s -> %s\n" \
        "$tag" "$n" "${p:0:4}" "${p3:0:4}" "$consumed" "$ptok" "$blocks" "$verdict" | tee -a "$OUT"
    if [ "$blocks" -le 1 ]; then
        echo "      ⚠ COVERAGE: the longest prompt spans $blocks block(s) at --kv-block-size ${MS_BLK:-16}." | tee -a "$OUT"
        echo "        A single-block prompt cannot exercise a block TABLE, so a CLEAN verdict here is" | tee -a "$OUT"
        echo "        scoped to one block per slot. Raise MS_FILL to cross it." | tee -a "$OUT"
    fi
    for i in $(seq 1 "$n"); do printf "      S%-2s %s\n" "$i" "$(echo "${R[$((i-1))]}" | head -c 62)" | tee -a "$OUT"; done
    echo "$verdict"
}

declare -a V=()
for r in $(seq 1 "$REPS"); do
    if [ $((r % 2)) -eq 1 ]; then
        v=$(probe "static-r$r" "" | tail -1); V+=("static:$v")
        v=$(probe "paged-r$r"  "--kv-paged --kv-block-size ${MS_BLK:-16} -ngpub $NGPUB -ncpub $NCPUB" | tail -1); V+=("paged:$v")
    else
        v=$(probe "paged-r$r"  "--kv-paged --kv-block-size ${MS_BLK:-16} -ngpub $NGPUB -ncpub $NCPUB" | tail -1); V+=("paged:$v")
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
