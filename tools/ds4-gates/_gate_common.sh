# _gate_common.sh -- ONE place for the gate laws this lane keeps re-deriving wrong.
#
# ⚠ WHY THIS FILE EXISTS. Three laws were written, argued and COMMITTED on 2026-08-08, and each was then
# violated by the next script written -- twice within three hours for the n_predict one:
#
#   vary-length : "every gate that sends N requests should send them at TWO prompt lengths" -- committed
#                 3a29fe1, then the next gate sent one prompt three times.
#   n_predict   : "size n_predict to the model's ANSWER LATENCY or the gate measures the harness" --
#                 committed 29461f0 after a 16-token cutoff voided a run, then a new gate was written
#                 with n_predict=24 and voided the same way on a reasoning model.
#   warm regime : FINDINGS-2026-08-06 records 1c as WARM-only (7/12 warm vs 0/24 cold) and warns that
#                 content_diff_probe's 12/12 CLEAN is vacuous because it never enters warm -- and a
#                 second gate was then run with the same blind spot.
#
# **Prose laws do not bind the next script.** `_no_abs_paths.sh` already proved the fix in this same
# directory: the privacy scrub became an include precisely because remembering to call it failed. This
# generalises that. You cannot forget to apply an include.

# ---- N_PREDICT: sized to ANSWER LATENCY, not to answer length -------------------------------------
# A reasoning model spends its first tokens opening a <think> block. 16 and 24 both voided runs.
GATE_N_PREDICT_DEFAULT=${GATE_N_PREDICT_DEFAULT:-256}
gate_n_predict() { echo "${GATE_N_PREDICT:-$GATE_N_PREDICT_DEFAULT}"; }

# Strip a think block before scoring. Answer lives AFTER it; scoring the raw head scores the harness.
gate_strip_think() { python3 -c '
import sys, re
t = sys.stdin.read()
t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S)
if "<think>" in t and "</think>" not in t:   # truncated mid-think: no answer was reached
    print("<TRUNCATED-IN-THINK>"); raise SystemExit
print(" ".join(t.split()))'; }

# ---- VARY-LENGTH: distinct is not enough, LENGTHS must differ meaningfully ------------------------
# ⚠ 7936 vs 7935 satisfies DISTINCTNESS and NOT length coverage: a one-token delta shares the whole
# prefix, so the runs correlate (~1.1 confirmations, not 2). Boundary-class defects need real spread.
gate_length_pair() { local base="${1:-8000}"; echo "$base $(( base / 2 ))"; }

# ---- WARM REGIME: prime, ASSERT the prime finished, then the concurrent pair, then a THIRD ---------
# ⚠ A warm gate whose prime silently fails degrades into the cold gate it exists to replace AND STILL
#   PRINTS CLEAN. Abort the rep instead of scoring it.
# ⚠ prime->pair measures the pair reading the prime's residue. It does NOT measure what the PAIR leaves
#   behind -- and tonight's defect struck the request AFTER the poisoning event. A third sequential
#   request is required to cover that half. (credit: Grok, #7991)
gate_warm_ok() { case "$1" in ""|"<UNPARSEABLE>"|"<TRUNCATED-IN-THINK>"|"<ERR:"*) return 1;; *) return 0;; esac; }

# ---- STATIC CONTROL: a dirty control VOIDS the run, it never indicts the code ----------------------
# Static runs no paging. If static is dirty the gate is measuring itself, and the paged column cannot
# be read at all. This caught a false "1c REPRODUCES" before it left the box.
gate_verdict() { # $1=static_bad $2=paged_bad $3=label
    if [ "$1" -ne 0 ]; then
        echo "$3: **VOID** -- static (no paging) scored $1 bad. The gate is measuring itself, not the code."
        return 2
    elif [ "$2" -ne 0 ]; then
        echo "$3: FAIL -- paged $2 bad, static CLEAN"; return 1
    else
        echo "$3: PASS -- both paths clean"; return 0
    fi
}

# ---- PRE-LAUNCH: SMOKE, NOT `bash -n` ------------------------------------------------------------
# ⚠ `bash -n` is a PARSER pass. An unbound variable under `set -u` is a RUNTIME fault and sails straight
#   through it -- verified on this box:
#       printf 'set -u\necho "$undefined"' > t.sh ; bash -n t.sh  -> PASSES SILENTLY
#                                                    bash    t.sh  -> unbound variable
#   Claiming "bash -n would have caught it" was FALSE and was repeated in the next message after being
#   said once. An instrument claim gets the same verification as a code claim. (correction: Grok #8033)
#
# ⚠ `shellcheck` (SC2154) would catch it statically and is NOT INSTALLED on this box. Do not write a
#   checklist step that cannot be executed here.
#
# ⇒ THE STEP THAT WORKS: a ONE-REP SMOKE before the full grid. It executes every line the grid executes,
#   costs ~40 s, and would have caught all three of tonight's runtime faults -- the unbound variable, the
#   n_predict truncation, and the empty 4th sequence.
#       MS_REPS=1 ./<gate>.sh    # read the per-sequence dump, THEN run the grid
gate_smoke_note() { echo "pre-launch: MS_REPS=1 smoke (bash -n does NOT catch runtime faults; shellcheck absent here)"; }

# ---- MAGNITUDE: a systematic difference is NOT graded by byte comparison --------------------------
# ⚠⚠ THE LAW THIS ENCODES, learned by breaking it on 2026-08-09. `qwen3vlmoe` paged text differed from
#   static, reproducibly. A three-arm probe showed BOTH arms deterministic across processes, so I filed
#   it as OPEN DEFECT and reported "the fifth real defect". The top-2 logprobs:
#
#       static   ' on' p=0.16328    ' located' p=0.15927     <- 0.025 logprob apart. A 1.03x TIE.
#       paged    ' located' 0.17962 ' on' 0.16509
#
#   A ~0.1 logprob numerical difference flipped an argmax static itself won by 0.025. Not a wrong
#   answer. The file was retracted the same hour, by me, before anyone asked.
#
# ⇒ THE REASONING ERROR, stated so the next gate cannot repeat it:
#       determinism separates RANDOM from SYSTEMATIC.
#       it CANNOT separate LARGE-systematic from TINY-systematic.
#       byte comparison is blind to MAGNITUDE. "Reproducible" NEVER upgrades a finding's severity.
#   I wrote "a near-tie would have made an arm vary; neither did, so the tie is refuted." False -- a tie
#   is resolved DETERMINISTICALLY AND DIFFERENTLY by two implementations. (sharpened: Grok #8200)
#
# ⚠ AND IT SCALES THE WRONG WAY. At 4k this cost one filed-then-retracted finding. At 512k-1M the
#   near-tie ENCOUNTER RATE is not one position, so a cross-arm byte-equality gate WILL false-fail
#   stochastically at depth -- i.e. it would file NUMERICS as DEFECTS at exactly the owner's bar. Any
#   long-context cross-arm gate must carry this clause. It lives here, not in prose, because prose laws
#   in this directory have been violated by the very next script three times.
#
# The field is `top_logprobs`. ⚠ NOT `top_probs` and NOT `probs` -- the first parser looked for those,
# printed nothing, and "no data" read exactly like "no divergence". Third instrument that night to read
# silent from being aimed at the wrong key.
GATE_TIE_LOGPROB="${GATE_TIE_LOGPROB:-0.35}"   # measured tie: 0.025. A real disagreement is >> this.

gate_top2_gap() { # $1=port  $2=prompt  -> "<top1_token>|<top2_token>|<logprob_gap>"; "" on failure
    local port="$1" prompt="$2"
    GT_P="$prompt" python3 -c "
import json, os
print(json.dumps({'prompt': os.environ['GT_P'], 'n_predict': 1, 'temperature': 0,
                  'seed': 1, 'cache_prompt': False, 'n_probs': 5}))" > "${TMPDIR:-/tmp}/gate_top2.json"
    curl -s --max-time 900 -X POST "http://127.0.0.1:$port/completion" \
        -H 'Content-Type: application/json' -d @"${TMPDIR:-/tmp}/gate_top2.json" \
      | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    tl = (d.get('completion_probabilities') or [{}])[0].get('top_logprobs') or []
    if len(tl) < 2: raise ValueError('fewer than 2 candidates')
    print('%s|%s|%.5f' % (tl[0]['token'], tl[1]['token'], tl[0]['logprob'] - tl[1]['logprob']))
except Exception:
    print('')"
}

# Grade a cross-arm TEXT difference. Call this BEFORE printing the word 'defect'.
#   exit 0 = TIE      -> a divergence NOTE. Not a defect. Do not file one.
#   exit 1 = REAL     -> the arms disagree where the model is confident.
#   exit 2 = UNGRADED -> the gap could not be measured; say UNGRADED, never 'defect'.
gate_grade_divergence() { # $1=static_gap_triplet (from gate_top2_gap on the STATIC arm)
    local trip="$1" gap tok1 tok2
    [ -z "$trip" ] && { echo "  UNGRADED: no top-2 gap measured. A cross-arm text difference without a"
                        echo "  magnitude is not gradeable -- report it as a DIVERGENCE, never a defect."; return 2; }
    tok1=${trip%%|*}; gap=${trip##*|}; tok2=${trip#*|}; tok2=${tok2%%|*}
    if python3 -c "import sys; sys.exit(0 if abs(float('$gap')) < float('$GATE_TIE_LOGPROB') else 1)"; then
        echo "  TIE, NOT A DEFECT: static's own top-2 are ['$tok1'] vs ['$tok2'], only $gap logprob apart"
        echo "  (threshold $GATE_TIE_LOGPROB). A small numerical difference between the paths flipped an"
        echo "  argmax static itself barely won. Report a divergence NOTE. Neither arm is ground truth."
        return 0
    fi
    echo "  SUBSTANTIVE: static separates ['$tok1'] from ['$tok2'] by $gap logprob (>= $GATE_TIE_LOGPROB),"
    echo "  so the arms disagree where the model is CONFIDENT. This one is gradeable as a defect."
    return 1
}

# ---- DEAD ARM: SAY WHY, DO NOT JUST SAY DEAD ------------------------------------------------------
# ⚠ THE GAP THE 2026-08-09 AUDIT FOUND, present in four gates at once. Every one of them DETECTS a
# server that never came up and aborts rather than scoring it -- which is the important half and was
# already right. None of them prints the server's OWN CAUSE.
#
# That half matters because a dead arm and a degraded arm render IDENTICALLY as "the paged column is
# missing". Twice in one night a harness fault produced a picture of paging collapsing at length:
#
#   -c overflow      "queue_request: request 0 exceeds max context (9050 > 8192)"  -> paged arm refused
#   port collision   "exiting due to HTTP server error"                            -> every arm died
#
# Both were deterministic, both looked like data, both pointed the same alarming way, and NEITHER was
# catchable by re-running. Each was diagnosed only by opening a log by hand and reading three lines.
# This puts those three lines in front of the reader instead.
gate_cause_from_log() { # $1 = server log  $2 = label
    local log="${1:-}" label="${2:-arm}"
    if [ ! -s "$log" ]; then
        echo "  $label DID NOT SERVE -- and its log is missing or empty, so the cause is UNKNOWN."
        echo "  Do not read this as a paging result; it is not a result at all."
        return
    fi
    echo "  $label DID NOT SERVE -- cause, from its own log:"
    # ⚠ SURFACE THE FAULT, NOT THE STACK. On an abort the last lines are BACKTRACE FRAMES while the one
    # line naming the fault scrolled past above them. Filtering out frame lines is what makes this
    # readable; a diagnostic buried under its own stack trace is a diagnostic nobody reads.
    # ⚠⚠ THE FIRST MATCH, NOT THE LAST. I wrote "surface the fault, not the stack" above and then
    # implemented `tail -3`, which shows the LAST errors -- and the last errors are usually
    # CONSEQUENCES. Caught by running this against a real failure log from earlier tonight: the fault
    # was `queue_request: request 0 exceeds max context (9050 > 8192)`, and tail-3 reported
    # "failed to find a memory slot" and "speculative decoding not supported" instead -- both true,
    # both downstream, neither the cause. First two matches (the fault) plus the last (the fatal
    # consequence, if different) is the shape that reads correctly.
    local first last
    first=$(grep -aE "GGML_ASSERT|GGML_ABORT|exceeds max context|rejected the request|HTTP server error|error:|failed to |not supported|out of memory|needs DS4P_" "$log" \
          | grep -av "^ *[0-9]* " | head -2)
    last=$(grep -aE "GGML_ASSERT|GGML_ABORT|exceeds max context|rejected the request|HTTP server error|error:|failed to |not supported|out of memory|needs DS4P_" "$log" \
          | grep -av "^ *[0-9]* " | tail -1)
    if [ -n "$first" ]; then
        printf '%s\n' "$first" | cut -c1-190 | sed 's/^/  ! FIRST: /'
        [ "$last" != "$(printf '%s' "$first" | tail -1)" ] && printf '%s\n' "$last" | cut -c1-190 | sed 's/^/  ! LAST:  /'
    else
        echo "  no assert/abort/error line matched; last lines of the log:"
        tail -4 "$log" | cut -c1-170 | sed 's/^/  | /'
    fi
}
