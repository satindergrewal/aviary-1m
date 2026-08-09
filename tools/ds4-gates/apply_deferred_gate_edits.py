#!/usr/bin/env python3
"""Apply the edits deferred while `paged_parity_gate.sh` was EXECUTING.

⚠⚠ WHY A STAGED APPLIER AND NOT `Edit` AT THE TIME. **bash reads a running script incrementally,
by byte offset.** Editing the file under a live interpreter shifts every offset after the edit
point, and the shell resumes mid-token in whatever now occupies that position. The failure is not a
clean error: it is a partially-executed script that can kill a server mid-arm and leave a plausible
`.res` behind. A 3.5-hour arm was in flight, so the edits waited and the file waited with them.

⚠ EVERY ANCHOR IS ASSERTED TO OCCUR EXACTLY ONCE before anything is written, and the file is only
rewritten if ALL edits resolve. A partial apply on a gate script is the same hazard as a partial
write_slots table: worse than not starting. Refuses if the gate is still running.

usage:  apply_deferred_gate_edits.py [--dry-run]
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "paged_parity_gate.sh")

EDITS = [
    # ---------------------------------------------------------------- A
    ("A: per-position ARM LOGS -- pos4 was overwriting pos1's log",
     '''  abba)        arm static; mv "$D/static.res" "$D/static1.res" 2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged1.res"  2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged2.res"  2>/dev/null
               arm static; mv "$D/static.res" "$D/static2.res" 2>/dev/null ;;''',
     '''  # ⚠⚠ THE .res FILES WERE PER-POSITION AND THE .log FILES WERE NOT, SO pos4 DESTROYED pos1's LOG.
  # Found on 2026-08-10 by trying to use them: `prefill_curve.py` differences the per-`-b` progress
  # lines into a within-run prefill rate curve, and the one comparison that separates a COLD-ARM
  # penalty from genuine DRIFT is pos1's curve against pos4's -- same arm type, different position.
  # That comparison was impossible, because `arm()` truncates "$D/$1.log" on entry and both static
  # arms write the same path. **Every within-arm diagnostic this lane has ever run therefore covered
  # only the LAST arm of each type, and nothing said so.**
  # ⇒ cp, not mv: "$D/static.log" keeps working for everything that already reads it.
  abba)        arm static; mv "$D/static.res" "$D/static1.res" 2>/dev/null
               cp "$D/static.log" "$D/static1.log" 2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged1.res"  2>/dev/null
               cp "$D/paged.log"  "$D/paged1.log"  2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged2.res"  2>/dev/null
               cp "$D/paged.log"  "$D/paged2.log"  2>/dev/null
               arm static; mv "$D/static.res" "$D/static2.res" 2>/dev/null
               cp "$D/static.log" "$D/static2.log" 2>/dev/null
               echo "  per-position logs kept: static1/paged1/paged2/static2.log" | tee -a "$OUT"
               echo "    ⇒ cold-vs-drift split:  prefill_curve.py \\$D/static1.log \\$D/static2.log" | tee -a "$OUT" ;;'''),

    # ---------------------------------------------------------------- B
    ("B: gate_tip_stamp -- a clean sha on a dirty tree names the wrong binary",
     '''echo "tip: $(cd "$WT" && git rev-parse --short HEAD)  ctx=$CTX''',
     '''# ⚠ `git rev-parse --short HEAD` prints a CLEAN sha even when the worktree that built the binary
# has uncommitted edits, so an artifact can name a commit the measured binary is not. LAW 7's
# gate_tip_stamp appends +dirty(N) -- the COUNT, because one edit and thirty are different situations.
echo "tip: $(gate_tip_stamp "$WT")  ctx=$CTX'''),

    # ---------------------------------------------------------------- C
    ("C: prelude failure-vocabulary VOID -- a failed warm-up printed a duration and moved on",
     '''        printf '  warm-up %-6s discarded  (%.0fs)\\n' "$a" "$(python3 -c "print($t1-$t0)")" | tee -a "$OUT"''',
     '''        # ⚠⚠ A WARM-UP THAT FAILED STILL PRINTED A DURATION AND THE RUN CONTINUED. The only check
        # here was "did the server come up"; everything after that was timed and discarded without
        # ever being READ. So a prelude that loaded the model, refused every paged layer and
        # returned an error produced `warm-up paged discarded (6s)` -- indistinguishable from a
        # healthy one, on the arm whose whole purpose is to remove the cold-arm confound from the
        # NEXT measurement. Exit-0-did-nothing, in the prelude.
        # ⇒ Read the prelude's own log for the failure vocabulary the measured arms already VOID on.
        local wfail; wfail=$(grep -acE 'failed to load model|error loading model|fails the paged capability contract|ZERO layers have consumed|failed to init the paged scheduler' "$D/warm-$a.log")
        if [ "${wfail:-0}" -gt 0 ]; then
            echo "  ⚠⚠ VOID: warm-up ($a) logged $wfail failure line(s) -- the prelude did NOT warm anything." | tee -a "$OUT"
            grep -m2 -oE 'failed to load model.*|error loading model.*|fails the paged capability contract.*|ZERO layers have consumed.*|failed to init the paged scheduler.*' "$D/warm-$a.log" | sed 's/^/      /' | tee -a "$OUT"
            echo "     Refusing to continue: the measured arms would inherit a cold-arm confound this" | tee -a "$OUT"
            echo "     gate would then report as an EFFECT. Fix the prelude or run with PP_WARM=0 and" | tee -a "$OUT"
            echo "     accept the cold first arm EXPLICITLY." | tee -a "$OUT"
            return 1
        fi
        printf '  warm-up %-6s discarded  (%.0fs)\\n' "$a" "$(python3 -c "print($t1-$t0)")" | tee -a "$OUT"'''),

    # ---------------------------------------------------------------- D
    ("D: PER-RUN $D -- every invocation clobbered the previous run's logs",
     '''D=${CLAUDE_JOB_DIR:-/tmp}/parity; mkdir -p "$D"''',
     '''# ⚠⚠ $D WAS A FIXED PATH, SO EVERY INVOCATION DESTROYED THE PREVIOUS ONE'S EVIDENCE -- and the
# sweep that drives this gate invokes it FOUR TIMES IN A ROW. `decode_ctx_sweep.sh` runs rungs
# 8k/32k/64k/128k through here back to back; each rung overwrote the last rung's logs, and the 512k
# run overwrote all of them. The per-rung `.txt` verdicts survived because they carry the ctx in
# their filename; **the logs, which hold the only within-run data this lane has, did not.**
#
# Found on 2026-08-10 when the prefill-curve analysis wanted to test its plateau at a second rung
# and there was nothing left to test it against. Same class as pos4 overwriting pos1, one level up:
# **the artifacts that get a unique name survive and the ones that do not are silently destroyed.**
#
# ⇒ Per-run directory keyed by ctx and time, plus a `latest` symlink so anything reaching for the
#   old fixed path still finds the most recent run.
#
# ⚠⚠⚠ AND THIS EDIT NEARLY DISABLED THE ONE-SERVER-AT-A-TIME LOCK. The lock path is built from $D
# at :82, THIRTEEN LINES BELOW THIS ONE. Making $D unique per invocation would give two concurrent runs
# their OWN lock directory each, so both would acquire it and both would start a server on the same
# GPU -- mutual exclusion gone, silently, and the symptom would be two slow arms nobody could
# attribute. **Guard-for-A-disables-B: an edit for evidence preservation switching off contention
# control, in the same file, in the same change.** That is a ★★★ class in this lane with three
# prior instances, and it was caught by reading the neighbourhood of the anchor rather than the
# anchor. ⇒ DPARENT is the FIXED path and the lock stays on it (edit E); only the run dir moves.
DPARENT=${CLAUDE_JOB_DIR:-/tmp}/parity; mkdir -p "$DPARENT"
D=$DPARENT/c${CTX}-$(date +%Y%m%d-%H%M%S); mkdir -p "$D"
ln -sfn "$D" "$DPARENT/latest" 2>/dev/null
echo "  run dir: .../parity/$(basename "$D")   (previous runs are no longer overwritten)"'''),

    # ---------------------------------------------------------------- E
    ("E: pin gpu.lock to the FIXED parent -- D is now per-run (pairs with D)",
     '''LOCK=$D/gpu.lock''',
     '''# ⚠⚠ THE LOCK MUST NOT LIVE UNDER THE PER-RUN $D. It is what makes concurrent invocations
# exclude each other, so it has to be at a path they SHARE. When $D became per-run (edit D) this
# line would have handed every invocation its own lock and let them all run at once.
LOCK=$DPARENT/gpu.lock'''),
]


def running():
    try:
        out = subprocess.run(["pgrep", "-f", "paged_parity_gate"],
                             capture_output=True, text=True).stdout.strip()
        return bool(out)
    except OSError:
        return False


def main():
    dry = "--dry-run" in sys.argv
    # ⚠ THE WHOLE REASON THIS FILE EXISTS. Refuse while the interpreter holds the file open.
    if running() and not dry:
        print("REFUSING: paged_parity_gate.sh is still running (pgrep matched).")
        print("  bash reads a running script by byte offset; editing it now can resume mid-token.")
        print("  Re-run this when the gate has exited.  --dry-run checks anchors safely.")
        return 1

    src = open(GATE).read()
    plan, bad = [], 0
    for name, old, new in EDITS:
        n = src.count(old)
        status = "OK" if n == 1 else ("ANCHOR MISSING" if n == 0 else f"AMBIGUOUS x{n}")
        print(f"  [{status:^14}] {name}")
        if n != 1:
            bad += 1
        else:
            plan.append((old, new))
    if bad:
        # ⚠ ALL-OR-NOTHING. A partially applied gate script is worse than an unapplied one: it
        # still runs, and it reports.
        print(f"\nREFUSING: {bad} of {len(EDITS)} anchors did not resolve exactly once. Nothing written.")
        print("  The gate has been edited since these were staged -- re-read it and re-stage.")
        return 2
    if dry:
        # ⚠⚠ A DRY-RUN THAT CHECKS THE PRISTINE FILE CANNOT SEE A COLLISION THE EDITS CREATE. Replay
        # the edits against a COPY so the same per-step assertion that guards the real apply also
        # guards the preview. The first version checked all five anchors against the original, said
        # "all anchors resolve", and then produced a broken file -- a green from a check that
        # examined a state the run never reaches.
        probe = src
        for i, (old, new) in enumerate(plan):
            if probe.count(old) != 1:
                print(f"\nDRY-RUN FAILS at step {i+1}: anchor occurs {probe.count(old)}x after the "
                      f"preceding edits (it was unique in the original).")
                return 2
            probe = probe.replace(old, new, 1)
        print("\ndry-run: all anchors resolve, INCLUDING against the evolving file. Nothing written.")
        return 0
    for i, (old, new) in enumerate(plan):
        # ⚠⚠⚠ UNIQUENESS MUST HOLD AGAINST THE EVOLVING TEXT, NOT THE ORIGINAL -- AND THAT COST A
        # BROKEN GATE SCRIPT ON 2026-08-10.
        #
        # Edit D's COMMENT documents the lock line, so its replacement text contains the literal
        # string that is edit E's ANCHOR. Measured:
        #     occurrences of edit E's anchor in the original file: 1
        #     after edit D is applied:                             2
        # so `replace(anchor, new, 1)` for edit E hit **edit D's comment**, not the real line at :82.
        # That spliced a replacement into the middle of a backtick pair, bash stopped treating the
        # following `#` lines as comments, and `bash -n` failed 136 lines later on a COMMENT
        # containing `ds4p_paged_consumer_count() == 0`.
        #
        # ⇒ **And the damage was exactly what edit E exists to prevent: the real lock line stayed
        #   `$D`-relative, so mutual exclusion would still have been silently disabled.** The fix
        #   for a coupling introduced a fresh instance of that coupling, THROUGH ITS OWN
        #   DOCUMENTATION OF THE COUPLING -- the same shape as `privacy_guard.sh` refusing its own
        #   source for spelling out the patterns it forbids.
        #
        # ⇒ Two fixes, because either alone is fragile: this per-step assertion (the general one),
        #   and edit D's comment no longer spelling the anchor literally (the specific one).
        if src.count(old) != 1:
            print(f"REFUSING mid-apply at step {i+1}: anchor now occurs {src.count(old)}x. "
                  f"An earlier edit's replacement text contains it. Nothing further written; "
                  f"`git checkout paged_parity_gate.sh` to restore.")
            return 3
        src = src.replace(old, new, 1)
    open(GATE, "w").write(src)
    rc = subprocess.run(["bash", "-n", GATE]).returncode
    print(f"\napplied {len(plan)} edits; bash -n -> {rc}")
    # ⚠ Report the POST-CONDITION, not the exit code of the write.
    print("  verify:  grep -c 'per-position logs kept' " + GATE)
    return rc


if __name__ == "__main__":
    sys.exit(main())
