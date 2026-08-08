#!/usr/bin/env bash
# ARCH SERVE GATE -- does architecture <A> actually run through the PAGED attention path, and does it
# produce the same tokens as static while doing so?
#
# WHY THIS EXISTS AS A FILE. The first four archs (ernie4-5, qwen3vl, nemotron, qwen3moe) were verified
# with ad-hoc shell typed inline. The results were right, but an inline harness cannot be re-run, cannot
# be diffed when it changes, and cannot be audited by anyone including me. With 15 archs left that is
# not a style complaint, it is 15 chances to run a subtly different check and not notice.
#
# ⚠ THE RETRACTION THIS GATE IS BUILT AROUND. On 2026-08-06 I "verified" starcoder by downloading
# StarCoder2, whose GGUF arch is `starcoder2` and whose model file is a DIFFERENT source file. The run
# looked clean. Step 1 below -- loader arch == expected -- is what catches that, and it is the only
# leg that retraction ever stood on.
#
# ⚠⚠ AND THE FIRST VERSION OF THIS GATE GOT THE *MECHANISM* WRONG, WHICH IS WORSE THAN GETTING THE
# VERDICT WRONG. It required the STATIC arm to log static-path warnings and VOIDed on zero, on the
# theory that zero meant "no paged consumer in this arch's file". That is false. There are TWO paged
# consumers in this fork:
#
#   build_attn_paged_or_null -> ggml_paged_attn_banded   20 archs, warns on the failure side
#   build_attn_inp_kv_auto   -> ggml_paged_attn          11 archs, warns NEVER, in either arm
#
# So the original rule would have FALSE-VOIDed all eleven auto-path architectures as unwired while
# they were paging perfectly well. Reading "paged is live" out of a warning count is inferring a
# positive from an absence, which is the root class in this lane's scar list.
#
# ★ THE FIX IS A POSITIVE MARKER AT BOTH CONSUMERS: DS4P-CONSUME <funnel> layer <il>, emitted per
# layer per graph, requires -lv 5 (DEBUG). It also fills a real hole in the fork, not just in this
# gate: DS4P-CHECKOUT proves the block pool served a request, never that a GRAPH consumed it, and
# "correct producer, no consumer" is audit finding 5 verbatim -- paged context built, no graph reading
# it, "Ornith now runs paged" retracted. Measured on the two funnels:
#
#            ernie4_5 (banded)   starcoder2 (auto)
#   paged        540                   900
#   static         0                     0        <- the marker can be zero, so >0 is a measurement
#
# Order of checks:
#   1  loader arch == expected                     else STRIKE  (a filename is not evidence)
#   2  STATIC arm DS4P-CONSUME == 0                else VOID    (marker must discriminate)
#   3  PAGED  arm DS4P-CONSUME  > 0                else VOID    (arch-independent presence proof)
#   4  banded funnel only: static-warns > 0 static, == 0 paged  else VOID / PARTIAL
#   5  output identical                            else FAIL
#
# ⚠ STEP 4 CANNOT RUN ON THE AUTO FUNNEL. That funnel has no per-layer fallback at all, so this gate
# is strictly WEAKER for those eleven archs: it can prove paged ran and produced identical text, but
# it cannot prove every layer was carried. It says so on PASS rather than printing a clean row.
#
# usage:  arch_serve_gate.sh <expected-arch> <model.gguf> [extra-server-args...]
set -uo pipefail

. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

ARCH="${1:-}"
MODEL="${2:-}"
if [ -z "$ARCH" ] || [ -z "$MODEL" ]; then
    echo "usage: $0 <expected-arch> <model.gguf> [extra-server-args...]" >&2
    echo "  refusing to run without an EXPECTED arch: a gate that accepts whatever it finds cannot" >&2
    echo "  catch a wrong-vehicle download, which is the exact failure this gate was written for." >&2
    exit 2
fi
shift 2
EXTRA=("$@")

# ⚠ DS4P_RSLOG RIDES ON THE PAGED ARM UNCONDITIONALLY. It is the only positive marker that the
# recurrent input is re-written per batch, and the defect it detects is silent in output on request 1
# by construction. Left opt-in it would be off on every future arch row, which is how the class went
# undetected across 94 result files. On the paged arm only, so the static arm's counters are untouched.
AG_ENV_PAGED="${AG_ENV_PAGED:-} DS4P_RSLOG=1"

[ -f "$MODEL" ] || { echo "missing model: $(basename "$MODEL")" >&2; exit 2; }

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
CTX=${AG_CTX:-4096}
NPRED=${AG_NPRED:-8}
PROMPT=${AG_PROMPT:-"The capital of France is"}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/archgate
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/arch-$ARCH-$(date +%Y%m%d-%H%M).txt}
LOCK=$LOGDIR/.lock
mkdir -p "$LOGDIR" "$(dirname "$OUT")"

# ⚠ ONE SERVER AT A TIME, AND KILL BY PID. Detached probes sharing `pkill -f llama-server` destroyed
# each other's servers on 2026-08-07; one lurked for hours and is the leading suspect for a load
# correlation that then failed to reproduce. A pattern kill is not addressed to anyone in particular.
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ANOTHER RUN HOLDS $LOCK -- refusing rather than racing it." >&2
    exit 2
fi
SRVPID=""
cleanup() {
    [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null
    scrub_abs_paths "${OUT:-}" 2>/dev/null
}
trap cleanup EXIT

echo "arch-serve gate: expected arch=$ARCH  model=$(basename "$MODEL")" | tee "$OUT"
echo "tip: $(cd "$WT" && git rev-parse --short HEAD) dirty=$(cd "$WT" && git status --porcelain | wc -l | tr -d ' ')  -c $CTX" | tee -a "$OUT"

pick_port() {
    local p
    for p in $(seq "${AG_PORT:-9051}" $(( ${AG_PORT:-9051} + 60 ))); do
        lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }
    done
    return 1
}

start() { # $1 = static|paged   -> PORT, SRVPID
    PORT=$(pick_port) || { echo "  no free port" | tee -a "$OUT"; return 1; }
    local flags=()
    [ "$1" = paged ] && flags=(--kv-paged)
    # shellcheck disable=SC2086
    # ⚠ -lv 5, NOT 4. DS4P-CONSUME is LLAMA_LOG_DEBUG and common/log.cpp:85 drops DEBUG below
    # verbosity 5. At -lv 4 the marker count read ZERO on an arch that was demonstrably paging -- the
    # gate would have VOIDed a working arch because its own probe was filtered out. Caught only
    # because the marker was verified for PRESENCE on a known-good model before being trusted.
    [ -n "${AG_MMPROJ:-}" ] && flags+=(--mmproj "$AG_MMPROJ" --jinja)
    # ⚠ AG_ENV_PAGED IS APPLIED TO THE PAGED ARM ONLY, AND THAT ASYMMETRY IS THE POINT.
    # The sensitivity control below poisons `s_copy`, which lives in llm_graph_input_mem_hybrid::
    # set_input -- a function a hybrid model runs in BOTH arms. Poisoned through AG_ENV it would
    # corrupt static and paged identically, they would AGREE, and the control would report the leg as
    # INSENSITIVE while actually proving it works. A control that fails closed by accident is worse
    # than no control. DS4P_RSLOG rides here too so the static arm's counters stay untouched.
    local armenv="${AG_ENV:-}"
    [ "$1" = paged ] && armenv="$armenv ${AG_ENV_PAGED:-}"
    # shellcheck disable=SC2086
    env $armenv "$SRV" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 \
        --port "$PORT" --no-warmup -lv 5 "${flags[@]}" "${EXTRA[@]}" > "$LOGDIR/$1.log" 2>&1 &
    SRVPID=$!
    local i
    for i in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && return 0
        kill -0 "$SRVPID" 2>/dev/null || break
        sleep 1
    done
    # ⚠ A DESIGNED REFUSAL IS NOT A FAILURE, AND THE TWO MUST NOT SHARE A ROW. The hybrid and SWA
    # guards refuse --kv-paged deliberately and say so; that is the guard WORKING. Scored as
    # "did not serve" it becomes a false FAIL, and a results table that cannot separate "refused by
    # design" from "broken" is worse than no table -- the first arch it mislabels is gemma4, which has
    # SWA layers, and qwen3next/hunyuan are hybrids. Its sibling paged_multimodel_gate.sh has had this
    # branch since it was written; this gate shipped without it.
    # ⚠ A DRAFT HEAD IS NOT A MODEL. THREE of the 19 are drafters, from two separate guards:
    #   llama-context.cpp:182   Gemma4Assistant
    #   llama-context.cpp:191   EAGLE3, DFLASH  (when the GGUF carries no tok_embd/output)
    # They are speculative-decode heads and need the TARGET model's context (ctx_other, i.e. -md).
    # Reported as "did not serve" that reads like a paging defect. It is a category error in the
    # vehicle, and this gate's whole design -- compare two arms of ONE model -- does not apply to them
    # without a target attached.
    if grep -qE "requires ctx_other to be set" "$LOGDIR/$1.log" 2>/dev/null; then
        echo "  $1 arm: THIS IS A DRAFT HEAD, not a standalone model." | tee -a "$OUT"
        grep -aE "requires ctx_other to be set" "$LOGDIR/$1.log" | head -1 | cut -c1-150 | sed 's/^/  | /' | tee -a "$OUT"
        echo "  gemma4-assistant / eagle3 / dflash need the target model's context (-md). This gate" | tee -a "$OUT"
        echo "  compares two arms of ONE model and cannot answer for them in that form. Needs its own" | tee -a "$OUT"
        echo "  harness: target + draft, then the separate question of whether the DRAFT's KV is" | tee -a "$OUT"
        echo "  paged at all. One harness unblocks three of the nineteen." | tee -a "$OUT"
        return 3
    fi
    if grep -qiE "not yet supported for hybrid architectures|not yet supported for KV-sharing|needs DS4P_PAGED_SWA=1|requires DS4P_PAGED_HYBRID=1" \
            "$LOGDIR/$1.log" 2>/dev/null; then
        echo "  $1 arm REFUSED BY DESIGN -- this arch class is guarded:" | tee -a "$OUT"
        # ⚠ QUOTE THE LINE THAT MATCHED, not the first line mentioning kv_paged. The first version
        # grepped "kv_paged" and printed "kv_paged layer->backend map: 48 layers..." -- an unrelated
        # informational line -- as if it were the refusal. Evidence that does not come from the thing
        # being reported is not evidence, however plausible it reads.
        grep -aiE "not yet supported for hybrid architectures|not yet supported for KV-sharing|needs DS4P_PAGED_SWA=1|requires DS4P_PAGED_HYBRID=1" \
            "$LOGDIR/$1.log" | head -1 | cut -c1-150 | sed 's/^/  | /' | tee -a "$OUT"
        # ⚠ NOT EVERY DESIGNED REFUSAL HAS AN ENABLING FLAG, AND SUGGESTING ONE SENDS THE READER IN
        # CIRCLES. Printed the hybrid/SWA hint unconditionally until 2026-08-09, when `gemma4` was
        # re-run with DS4P_PAGED_SWA=1 and refused again -- by a DIFFERENT guard. Its blocker is
        # KV-SHARING (`n_layer_kv_from_start`), which has no env var because there is no
        # implementation behind one: the layers that reuse an earlier layer's cache would read the
        # STATIC cache, which paging never fills, and attend over empty KV. The guard is the only
        # thing standing between that arch and a silently wrong answer.
        if grep -qi "not yet supported for KV-sharing" "$LOGDIR/$1.log" 2>/dev/null; then
            echo "  ⚠ NO ENABLING FLAG EXISTS FOR THIS ONE. The blocker is KV-SHARING, not SWA and not" | tee -a "$OUT"
            echo "  hybrid: some layers reuse an earlier layer's cache, and paging never fills the" | tee -a "$OUT"
            echo "  static cache they read. Enabling it would produce a silently wrong answer, which" | tee -a "$OUT"
            echo "  is strictly worse than refusing. See FINDINGS-paged-kv-sharing-splitbrain.md --" | tee -a "$OUT"
            echo "  diagnosed, unfixed, and blocked on implementation rather than on a vehicle." | tee -a "$OUT"
        else
            echo "  Set AG_ENV to the enabling flag and re-run, e.g." | tee -a "$OUT"
            echo "    AG_ENV=DS4P_PAGED_HYBRID=1  $0 $ARCH <model>     # hybrid archs" | tee -a "$OUT"
            echo "    AG_ENV=DS4P_PAGED_SWA=1     $0 $ARCH <model>     # SWA archs" | tee -a "$OUT"
        fi
        echo "  Until then this gate CANNOT answer for this arch. Not a pass, not a failure." | tee -a "$OUT"
        return 3
    fi
    echo "  $1 arm DID NOT SERVE" | tee -a "$OUT"
    # ⚠ SURFACE THE CAUSE, NOT THE TAIL. On an abort the last six lines are BACKTRACE FRAMES -- dyld
    # offsets and mangled symbols -- while the one line that names the fault scrolled past above it.
    # Measured: a deliberately reintroduced contiguity bug printed
    #   "GGML_ASSERT(ggml_is_contiguous(q) && "paged attn: q must be contiguous ...") failed"
    # and the gate showed six frames of `_ZN18common_init_result...` instead. A diagnostic buried
    # under its own stack trace is a diagnostic nobody reads.
    local why
    why=$(grep -aE "GGML_ASSERT|GGML_ABORT|error:|failed to |not supported|out of memory" \
          "$LOGDIR/$1.log" | grep -av "^ *[0-9]* " | tail -3)
    if [ -n "$why" ]; then
        echo "  cause:" | tee -a "$OUT"
        printf '%s\n' "$why" | cut -c1-190 | sed 's/^/  ! /' | tee -a "$OUT"
    else
        echo "  no assert/abort/error line found; last lines:" | tee -a "$OUT"
        tail -6 "$LOGDIR/$1.log" | sed 's/^/  | /' | tee -a "$OUT"
    fi
    return 1
}

# ★ IMAGE MODE. Some architectures have no usable TEXT vehicle at all. hunyuan_vl's only published
# GGUF is HunyuanOCR, an image model: on text-only prompts it returns '' or ' $ $ $ $', so the static
# reference is degenerate BY CONSTRUCTION and the gate correctly VOIDs. That is not a paging result,
# it is the wrong question asked of the model.
#
# With AG_MMPROJ + AG_IMAGE the gate asks the question the model can answer -- an OCR prompt over a
# deterministic locally-generated PNG -- and the reference becomes sane. Verified before wiring:
# the same model answers 'PARIS' to an image containing the word PARIS.
#
# The degenerate-reference guard still applies unchanged, so a vehicle that cannot read its own test
# image still VOIDs rather than being compared to nothing.
ask() {
    if [ -n "${AG_IMAGE:-}" ]; then
        python3 -c "
import base64, json
b = base64.b64encode(open('$AG_IMAGE','rb').read()).decode()
print(json.dumps({'messages': [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b}},
    {'type': 'text', 'text': '''${AG_VPROMPT:-What word is written in this image? Answer with the word only.}'''}]}],
    'max_tokens': $NPRED, 'temperature': 0, 'seed': 1}))" > "$LOGDIR/req.json"
        curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$LOGDIR/req.json" \
          | python3 -c "
import json,sys
try: print(json.load(sys.stdin)['choices'][0]['message']['content'].replace(chr(10),' '))
except Exception: print('MALFORMED')"
        return
    fi
    ask_one "$PROMPT" "$NPRED"
}

# ★ ask_one -- one raw completion, prompt and length passed in. Split out of ask() so the sequence
# leg can send DIFFERENT prompts without duplicating the request/parse code. `cache_prompt: False` is
# kept: the leg tests carried STATE, not the prompt cache, and a cache hit would skip the prefill
# whose chunk-2 handoff is the thing under test.
ask_one() { # $1 prompt  $2 n_predict
    AO_P="$1" AO_N="$2" python3 -c "
import json, os
print(json.dumps({'prompt': os.environ['AO_P'], 'n_predict': int(os.environ['AO_N']),
                  'temperature': 0, 'seed': 1, 'cache_prompt': False}))" > "$LOGDIR/req.json"
    curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d @"$LOGDIR/req.json" \
      | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('content','').replace(chr(10),' '))
except Exception: print('MALFORMED')"
}

# ★★ THE SEQUENCE LEG -- three requests to ONE server, LONG, SHORT, LONG.
#
# WHY THIS EXISTS. Every one of the 94 result files this gate has produced is a ONE-REQUEST verdict
# against a FRESH server, and the two defects this lane actually found are both invisible in that
# shape:
#
#   cross-request corruption  request 1 is CLEAN (the recurrent buffer is zero on fresh allocation);
#                             it first bites at request 2 / chunk 2, the first batch needing carried
#                             state
#   prompt_clear crash        needs `long, short, long` -- thirteen gates missed it because none ever
#                             sent a SHORT prompt after a LONG one
#
# So a matrix of single-request greens cannot see either, on any arch, and re-running it on a fixed
# binary would produce fresh greens that still could not see them. The leg is the fix for the harness,
# not for the code.
#
# ⚠ THE PROMPTS MUST DIFFER IN LENGTH AND IN CONTENT. Sending one prompt three times re-tests the
# prompt cache; sending three prompts of equal length never builds the short-after-long ledger state.
# Both mistakes have been made in this lane, the second of them three hours after writing down the law.
#
# ⚠ AND THEY MUST FIT IN -c $CTX WITH ROOM FOR n_predict. Sized at ~1.8k and ~2.6k tokens against the
# 4096 default: several 512-token chunks each, so the chunk-2 handoff is exercised, with headroom that
# does not depend on a tokenizer this gate must work across 15 architectures. A prompt that overflows
# the context returns a server error, and an error compared against an error MATCHES.
SEQ_LONG1="${AG_SEQ_LONG1:-$(python3 -c "print(('The following is a list of numbered facts about geography. ' + ' '.join(f'Fact {i}: city number {i} lies on a river.' for i in range(1, 150))) + ' Question: the capital of France is')")}"
SEQ_SHORT="${AG_SEQ_SHORT:-The capital of Japan is}"
# ⚠ THE FILLER'S SUBJECT MATTER IS LOAD-BEARING, WHICH I FOUND BY BREAKING IT. The first LONG2 filled
# with laboratory inventory ("Item 12: beaker 12 holds 36 millilitres.") and both `nemotron` and
# `qwen3vl` answered the trailing question with a bare "?" -- degenerate, VOIDing two otherwise fine
# rows. LONG1's geography filler, same grammar and the same trailing question form, worked on both.
# So the leg's arbiter can be destroyed by the padding rather than by anything under test. Keep the
# filler in the same domain as the question being asked.
SEQ_LONG2="${AG_SEQ_LONG2:-$(python3 -c "print(('The following is a list of numbered facts about rivers and towns. ' + ' '.join(f'Note {i}: river number {i} flows past town {i}.' for i in range(1, 190))) + ' Question: the capital of Italy is')")}"

ask_seq() { # echoes exactly 3 lines, one per request, in order
    ask_one "$SEQ_LONG1" "$NPRED"
    ask_one "$SEQ_SHORT" "$NPRED"
    ask_one "$SEQ_LONG2" "$NPRED"
}

# counts, kept SEPARATE. Lumping them hides which failure mode fired: 'no paged context at all' and
# 'paged pool active but this layer is not eligible' need completely different fixes.
# ⚠⚠ COUNT ONLY WHAT HAPPENS AFTER A REQUEST EXISTS. llama-server builds the graph many times at
# startup -- memory fitting and reserve passes -- before any token is real. On gemma4 those startup
# passes take the STATIC path because the paged scheduler has no batch info yet, and the first
# version of this gate counted them as layer-fallbacks:
#
#   21 graph builds STATIC   0.311s -> 1.271s     ALL before the first request
#    8 graph builds PAGED    from 2.088s          the request arrives at 2.088s
#
# It scored gemma4 PARTIAL with "630 fallbacks" when 100% of inference-time builds were paged. The
# warnings were real, the count was real, and the verdict was false, because the work being counted
# had not happened yet. Same class as counting a log line that fires BEFORE the work it announces.
#
# So every counter slices the log at the first request. The slice marker must be present, and
# post_slice VOIDs rather than silently counting the whole file if it is missing.
post_slice() { # $1 log -> stdout: the portion after the first real request
    awk '/launch_slot_|start_loop: processing task, id = 0/{seen=1} seen' "$1"
}
has_slice()  { grep -qa "launch_slot_\|start_loop: processing task, id = 0" "$1"; }

c_nopg()  { post_slice "$1" | grep -ac "took the STATIC path -- no paged context"; }
c_cap()   { post_slice "$1" | grep -ac "fails the paged capability contract"; }
c_hdim()  { post_slice "$1" | grep -ac "does not match the paged pool's"; }
c_pool()  { post_slice "$1" | grep -ac "DS4P-CHECKOUT"; }
# the two consumer funnels, counted apart: which one an arch uses decides which further checks the
# gate is even ENTITLED to run.
c_band()  { post_slice "$1" | grep -ac "DS4P-CONSUME banded"; }
c_auto()  { post_slice "$1" | grep -ac "DS4P-CONSUME auto"; }
# ★ THE REGRESSION ASSERTION FOR THE 2026-08-08 CORRUPTION FIX, and the hybrid detector in one count.
# `DS4P-RS who=hybrid` is emitted once per call of llm_graph_input_mem_hybrid::set_input, i.e. once per
# serving batch on a hybrid arch. The defect made that function return before the recurrent write, so
# the write executed EXACTLY ONCE -- at graph construction, on the 2-token warmup batch -- for the whole
# server lifetime. The probe sits OUTSIDE the `if (s_copy)` guard on purpose, so `s_copy=<null>` (tensor
# absent) and silence (never called) are different lines rather than the same nothing.
#   0 post-slice  -> this arch is not hybrid, or the hybrid path is not taken
#   1 post-slice  -> ⚠ THE DEFECT IS BACK: one write, then frozen indices for every later batch
#   >1            -> the recurrent input is refreshed per batch, which is the fixed behaviour
c_rs()    { post_slice "$1" | grep -ac "DS4P-RS who=hybrid"; }
# startup-only counts, reported separately so the contamination stays VISIBLE rather than deleted
c_nopg_pre() { local a b; a=$(grep -ac "took the STATIC path -- no paged context" "$1"); b=$(c_nopg "$1"); echo $((a-b)); }

# ---------------- static arm ----------------
# ⚠ exit 3 is SKIPPED-BY-DESIGN, distinct from exit 1 (broken) and exit 2 (VOID/STRIKE). A caller
# that collapses them scores a working guard as a defect, or worse, buries a real failure in a
# "skipped" bucket. The three are separate exit codes so a batch runner cannot conflate them.
start static; rc_s=$?
[ "$rc_s" -eq 3 ] && { echo "log: $OUT"; exit 3; }
[ "$rc_s" -ne 0 ] && exit 1
S_OUT=$(ask)
S_SEQ=$(ask_seq)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2

# ⚠ THE ARCH IS READ FROM THE LOADER, NOT THE FILENAME. A repo named "...starcoder..." can hold a
# starcoder2 model, and the model file that gets compiled is chosen by THIS string.
GOT_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/static.log" | sed 's/.*= *//' | tr -d ' \r')
# ⚠ THE SLICE MARKER MUST EXIST IN *THIS* ARM TOO. The first version checked it only on the paged
# log. Without it here, S_NOPG and S_CONS silently read 0 and the banded branch VOIDs saying "the
# counter was never shown able to move" -- which fails closed but DIAGNOSES WRONG, sending the reader
# after a wiring fault when the real problem is a missing marker in the harness.
if ! has_slice "$LOGDIR/static.log"; then
    echo "VOID: no request marker in the STATIC log -- its counters cannot exclude startup reserve" | tee -a "$OUT"
    echo "  passes, so neither the negative control nor anything derived from it can be read." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
S_NOPG=$(c_nopg "$LOGDIR/static.log")
S_CONS=$(( $(c_band "$LOGDIR/static.log") + $(c_auto "$LOGDIR/static.log") ))

printf '  STATIC  arch=%-14s out=[%s]\n' "${GOT_ARCH:-<none>}" "$S_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s DS4P-CONSUME=%s\n' \
       "$S_NOPG" "$(c_cap "$LOGDIR/static.log")" "$(c_hdim "$LOGDIR/static.log")" "$S_CONS" | tee -a "$OUT"

# ⚠⚠ IS THE REFERENCE ITSELF SANE? This gate compares paged against static and calls a match a PASS.
# It never asked whether static was worth comparing to -- and on gemma4-12B the static arm returned
# "01111133" for the prompt "The capital of France is". Had the paged arm returned the same garbage,
# the gate would have printed PASS on two broken runs agreeing.
#
# That is the degenerate-baseline trap for the third time in this lane: an evening spent scoring a
# kernel against a reference that was itself looping, then the Q4-over-Q2 quant decision made
# precisely to avoid it, and now a gate that encoded the lesson in its COMMENTS and not in its CODE.
# A claim-vs-measure check catches lies; it does not catch absurdity. Absurdity needs its own test.
#
# The prompt is fixed and known, so the bar is specific rather than general: an answer to
# "The capital of France is" that contains almost no letters, or almost no distinct characters, is not
# an arbiter regardless of what the paged arm does.
read -r LETTERS DISTINCT <<EOF
$(printf '%s' "$S_OUT" | python3 -c "
import sys
s = sys.stdin.read()
print(sum(c.isalpha() for c in s), len(set(s.strip())))")
EOF
if [ "${LETTERS:-0}" -lt 3 ] || [ "${DISTINCT:-0}" -lt 4 ]; then
    echo | tee -a "$OUT"
    echo "VOID: the STATIC reference is degenerate -- [$S_OUT] ($LETTERS letters, $DISTINCT distinct chars)." | tee -a "$OUT"
    echo "  Static is the arbiter for this whole gate. A broken arbiter cannot judge the paged path:" | tee -a "$OUT"
    echo "  if paged returns the same garbage the gate would print PASS on two broken runs agreeing." | tee -a "$OUT"
    echo "  Fix the VEHICLE first -- wrong chat template, damaged quant, or a base model that needs a" | tee -a "$OUT"
    echo "  different prompt. This says nothing about paging either way." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

if [ "$GOT_ARCH" != "$ARCH" ]; then
    echo | tee -a "$OUT"
    echo "STRIKE: loader reports arch='$GOT_ARCH', gate was told to expect '$ARCH'." | tee -a "$OUT"
    echo "  This is the wrong vehicle. The model file exercised is the one selected by the loader's" | tee -a "$OUT"
    echo "  arch string, so whatever this run proves, it does not prove anything about '$ARCH'." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ⚠ NEGATIVE CONTROL ON THE PROBE ITSELF, BEFORE ANY POSITIVE RESULT IS READ. If DS4P-CONSUME is
# non-zero WITHOUT --kv-paged, the marker does not discriminate and its count in the paged arm proves
# nothing. Measured at 0 on both funnels; this asserts it rather than trusting the measurement.
if [ "${S_CONS:-0}" -ne 0 ]; then
    echo | tee -a "$OUT"
    echo "VOID: DS4P-CONSUME fired $S_CONS times in the STATIC arm, where no paged context exists." | tee -a "$OUT"
    echo "  The presence marker is not discriminating, so its count under paging cannot be read as" | tee -a "$OUT"
    echo "  evidence of anything. Fix the marker before reading this gate." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ---------------- paged arm ----------------
# The interesting case: static served fine, paged is refused by the SWA/hybrid guard. That is the
# guard doing its job, and the arch is UNANSWERED rather than failed.
start paged; rc_p=$?
if [ "$rc_p" -eq 3 ]; then
    echo "-----" | tee -a "$OUT"
    echo "SKIPPED: arch=$ARCH serves under static and is refused under --kv-paged BY DESIGN." | tee -a "$OUT"
    echo "  Record it as UNANSWERED, never as verified and never as broken." | tee -a "$OUT"
    echo "log: $OUT"; exit 3
fi
[ "$rc_p" -ne 0 ] && exit 1
P_OUT=$(ask)
P_SEQ=$(ask_seq)
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""

P_ARCH=$(grep -m1 "print_info: arch" "$LOGDIR/paged.log" | sed 's/.*= *//' | tr -d ' \r')
P_NOPG=$(c_nopg "$LOGDIR/paged.log"); P_CAP=$(c_cap "$LOGDIR/paged.log")
P_HDIM=$(c_hdim "$LOGDIR/paged.log"); P_POOL=$(c_pool "$LOGDIR/paged.log")
P_BAND=$(c_band "$LOGDIR/paged.log"); P_AUTO=$(c_auto "$LOGDIR/paged.log")
P_CONS=$(( P_BAND + P_AUTO ))

printf '  PAGED   arch=%-14s out=[%s]\n' "${P_ARCH:-<none>}" "$P_OUT" | tee -a "$OUT"
printf '          static-path warns=%-6s cap-fails=%-5s headdim-fails=%-5s pool=%s\n' \
       "$P_NOPG" "$P_CAP" "$P_HDIM" "$P_POOL" | tee -a "$OUT"
printf '          DS4P-CONSUME banded=%-6s auto=%s\n' "$P_BAND" "$P_AUTO" | tee -a "$OUT"
printf '          (startup reserve passes excluded: %s static-path warnings before the first request)\n' \
       "$(c_nopg_pre "$LOGDIR/paged.log")" | tee -a "$OUT"
echo "-----" | tee -a "$OUT"

rc=0
# ⚠ the slice marker must EXIST, or every count above silently became "whole file" again
if ! has_slice "$LOGDIR/paged.log"; then
    echo "VOID: no request marker in the paged log, so the counters could not exclude startup" | tee -a "$OUT"
    echo "  reserve passes. Every number above would be contaminated by graph builds that ran" | tee -a "$OUT"
    echo "  before a token existed. Fix the marker, do not read this result." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
# ⚠ POOL ACTIVITY IS NOT CONSUMPTION. DS4P-CHECKOUT says the block manager handed out blocks; it is
# emitted by the scheduler and is true even if no graph reads them. That gap IS audit finding 5.
if [ "${P_CONS:-0}" -eq 0 ]; then
    echo "VOID: paged arm shows ZERO DS4P-CONSUME events -- no graph consumed the paged context." | tee -a "$OUT"
    echo "  pool checkouts = $P_POOL, so the blocks were allocated and then read by nobody. Identical" | tee -a "$OUT"
    echo "  output here proves only that two effectively-static runs agree. This is audit finding 5" | tee -a "$OUT"
    echo "  reproducing, not a passing arch." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# which funnel -- decides which further checks this gate is entitled to run
FUNNEL=mixed
[ "$P_BAND" -gt 0 ] && [ "$P_AUTO" -eq 0 ] && FUNNEL=banded
[ "$P_AUTO" -gt 0 ] && [ "$P_BAND" -eq 0 ] && FUNNEL=auto
echo "  consumer funnel: $FUNNEL" | tee -a "$OUT"

if [ "$S_OUT" != "$P_OUT" ]; then
    echo "*** FAIL: OUTPUT DIVERGED. static and paged disagree on the same deterministic prompt. ***" | tee -a "$OUT"
    echo "    static [$S_OUT]" | tee -a "$OUT"
    echo "    paged  [$P_OUT]" | tee -a "$OUT"
    rc=1
fi

# ---------------- sequence leg: long, short, long, on ONE server ----------------
P_RS=$(c_rs "$LOGDIR/paged.log")
HYBRID=no; [ "${P_RS:-0}" -gt 0 ] && HYBRID=yes
echo "  multi-request leg: hybrid=$HYBRID  recurrent writes after first request=$P_RS" | tee -a "$OUT"

# ⚠ THE REGRESSION ASSERTION GOES BEFORE THE OUTPUT COMPARISON, because it fires on runs whose output
# still matches. That is the whole character of the defect: request 1 is clean, so a gate that only
# reads text calls it a pass. Exactly one write means the function ran once and returned early ever
# after -- the 2026-08-08 defect verbatim.
if [ "$HYBRID" = yes ] && [ "$P_RS" -eq 1 ]; then
    echo "*** FAIL: the recurrent input was written ONCE across every serving batch. ***" | tee -a "$OUT"
    echo "    That is the 2026-08-08 corruption regressing: llm_graph_input_mem_hybrid::set_input is" | tee -a "$OUT"
    echo "    returning before its recurrent block, so s_copy holds whatever the warmup batch left" | tee -a "$OUT"
    echo "    there. Output may still match -- request 1 is clean because the buffer starts zeroed." | tee -a "$OUT"
    rc=1
fi

seq_line() { printf '%s\n' "$2" | sed -n "$1p"; }

# ⚠⚠ THE STATIC SEQUENCE IS THE ARBITER HERE TOO, AND IT CAN BE DEGENERATE FOR REASONS THAT HAVE
# NOTHING TO DO WITH PAGING. These prompts are ~2k tokens; overflow the context, hit a template
# problem, or trip a refusal and the server returns an error or an empty string -- IN BOTH ARMS. Two
# errors compare EQUAL and the leg prints three green rows. Same trap the single-request reference
# already guards, one function further down, which is how it would have been missed.
for i in 1 2 3; do
    read -r L D <<EOF
$(seq_line "$i" "$S_SEQ" | python3 -c "
import sys
s = sys.stdin.read()
print(sum(c.isalpha() for c in s), len(set(s.strip())))")
EOF
    if [ "${L:-0}" -lt 3 ] || [ "${D:-0}" -lt 4 ]; then
        echo "VOID: static sequence request $i is degenerate -- [$(seq_line "$i" "$S_SEQ")]" | tee -a "$OUT"
        echo "  ($L letters, $D distinct chars). The multi-request leg has no arbiter, and a paged arm" | tee -a "$OUT"
        echo "  returning the same nothing would compare EQUAL. Fix the vehicle or shrink the prompts" | tee -a "$OUT"
        echo "  (AG_SEQ_LONG1/AG_SEQ_LONG2); this says nothing about paging either way." | tee -a "$OUT"
        echo "log: $OUT"; exit 2
    fi
done

SEQ_DIVERGED=0
for i in 1 2 3; do
    a=$(seq_line "$i" "$S_SEQ"); b=$(seq_line "$i" "$P_SEQ")
    lbl=$([ "$i" = 2 ] && echo "short" || echo "long")
    if [ "$a" != "$b" ]; then
        SEQ_DIVERGED=1
        echo "*** FAIL: request $i of 3 ($lbl) DIVERGED between static and paged. ***" | tee -a "$OUT"
        echo "    static [$a]" | tee -a "$OUT"
        echo "    paged  [$b]" | tee -a "$OUT"
        rc=1
    else
        printf '    req %d/3 %-5s match  [%s]\n' "$i" "$lbl" "$a" | tee -a "$OUT"
    fi
done

# ★★ SENSITIVITY CONTROL. Without it a green here is unreadable: it means either "state is carried
# correctly" or "this leg cannot see state at all", and those are the same picture. DS4P_RSPOISON
# overwrites s_copy AFTER the normal write, so a leg that can see recurrent state MUST diverge.
#
# ⚠ PAGED ARM ONLY. The poison lives in a function a hybrid model runs in BOTH arms; applied to both,
# the two would corrupt identically, AGREE, and the control would report itself insensitive while
# actually working. Asymmetry is what makes it a control.
#
# ⚠ Runs only on hybrid archs -- there is no recurrent state to poison anywhere else, so on a
# non-hybrid arch the leg's power is UNMEASURED and the verdict says so rather than implying it.
if [ "$HYBRID" = yes ] && [ -z "${AG_NO_CONTROL:-}" ] && [ "$SEQ_DIVERGED" -eq 0 ]; then
    echo "  sensitivity control: re-running the PAGED arm with DS4P_RSPOISON=0xDEADBEEF ..." | tee -a "$OUT"
    # ⚠ the control reuses start(), which writes $LOGDIR/paged.log. Keep the real arm's log, or every
    # count printed above becomes unreproducible from the file that is still sitting there afterwards.
    cp "$LOGDIR/paged.log" "$LOGDIR/paged-main.log" 2>/dev/null
    AG_ENV_PAGED="$AG_ENV_PAGED DS4P_RSPOISON=0xDEADBEEF"
    if start paged; then
        C_OUT=$(ask); C_SEQ=$(ask_seq)
        kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""
        CTRL_FIRED=0
        [ "$C_OUT" != "$S_OUT" ] && CTRL_FIRED=1
        for i in 1 2 3; do
            [ "$(seq_line "$i" "$C_SEQ")" != "$(seq_line "$i" "$S_SEQ")" ] && CTRL_FIRED=1
        done
        if [ "$CTRL_FIRED" -eq 1 ]; then
            echo "    CONTROL FIRED -- poisoned recurrent state produced different text, so the leg" | tee -a "$OUT"
            echo "    above is sensitive to recurrent state and its match is a measurement." | tee -a "$OUT"
        else
            echo "VOID: the poison control did NOT fire. Corrupting s_copy on the paged arm changed" | tee -a "$OUT"
            echo "  nothing in the output, so this leg cannot see recurrent state and its green says" | tee -a "$OUT"
            echo "  nothing about carried state. Do not record this arch as multi-request verified." | tee -a "$OUT"
            echo "    static  [$S_OUT]" | tee -a "$OUT"
            echo "    poisoned[$C_OUT]" | tee -a "$OUT"
            rc=2
        fi
    else
        echo "  ⚠ control arm did not serve; sensitivity UNPROVEN for this row." | tee -a "$OUT"
    fi
fi

if [ "$FUNNEL" = banded ] || [ "$FUNNEL" = mixed ]; then
    # ⚠ ONLY MEANINGFUL ON THIS FUNNEL, and only after the static arm proved the counter can move.
    if [ "${S_NOPG:-0}" -eq 0 ]; then
        echo "VOID: banded funnel, but the STATIC arm logged zero static-path warnings, so a zero in" | tee -a "$OUT"
        echo "  the paged arm is not a measurement -- the counter was never shown able to move." | tee -a "$OUT"
        echo "log: $OUT"; exit 2
    fi
    FB=$(( P_NOPG + P_CAP + P_HDIM ))
    if [ "$FB" -ne 0 ]; then
        echo "*** PARTIAL: $FB layer-fallbacks under paging (nopg=$P_NOPG cap=$P_CAP headdim=$P_HDIM)." | tee -a "$OUT"
        echo "    The fallback path is correct BY DESIGN, so output may still match -- but this arch is" | tee -a "$OUT"
        echo "    not fully carried by the paged path and must NOT be recorded as verified." | tee -a "$OUT"
        rc=1
    fi
fi

if [ "$rc" -eq 0 ]; then
    echo "PASS: arch=$ARCH  output identical  DS4P-CONSUME=$P_CONS ($FUNNEL funnel)" | tee -a "$OUT"
    if [ "$FUNNEL" = banded ]; then
        echo "  fallbacks ${S_NOPG} -> 0, every layer carried by the paged path." | tee -a "$OUT"
    else
        echo "  ⚠ WEAKER RESULT THAN THE BANDED FUNNEL. The auto path has no per-layer fallback and" | tee -a "$OUT"
        echo "  emits no static-path warning, so this gate CANNOT prove every layer was carried. It" | tee -a "$OUT"
        echo "  proves paged ran ($P_CONS consume events) and the text matches. Record it that way." | tee -a "$OUT"
    fi
    if [ "$HYBRID" = yes ]; then
        echo "  multi-request: 3 requests (long, short, long) on ONE server all match, recurrent input" | tee -a "$OUT"
        echo "  re-written $P_RS times after the first request, and a poisoned control proved the leg" | tee -a "$OUT"
        echo "  can see recurrent state. This row is NOT a single-request verdict." | tee -a "$OUT"
    else
        echo "  multi-request: 3 requests (long, short, long) on ONE server all match. ⚠ NON-HYBRID, so" | tee -a "$OUT"
        echo "  there is no recurrent state to poison and the leg's POWER IS UNMEASURED here: it covers" | tee -a "$OUT"
        echo "  the short-after-long ledger shape, and nothing proves it would catch a subtler carry" | tee -a "$OUT"
        echo "  defect on this arch. Weaker than the hybrid rows, on purpose, and recorded as such." | tee -a "$OUT"
    fi
    echo "  ⚠ SCOPE: short prompt at -c $CTX. Says the arch is WIRED and CORRECT at small context." | tee -a "$OUT"
    echo "  Says nothing about long context, nothing about speed, and the ~50k intermittent paged" | tee -a "$OUT"
    echo "  defect is open underneath every row of this table." | tee -a "$OUT"
fi
echo "log: $OUT"
exit $rc
