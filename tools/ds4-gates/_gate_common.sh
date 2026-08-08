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
