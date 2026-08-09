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
        print("\ndry-run: all anchors resolve. Nothing written.")
        return 0
    for old, new in plan:
        src = src.replace(old, new, 1)
    open(GATE, "w").write(src)
    rc = subprocess.run(["bash", "-n", GATE]).returncode
    print(f"\napplied {len(plan)} edits; bash -n -> {rc}")
    # ⚠ Report the POST-CONDITION, not the exit code of the write.
    print("  verify:  grep -c 'per-position logs kept' " + GATE)
    return rc


if __name__ == "__main__":
    sys.exit(main())
