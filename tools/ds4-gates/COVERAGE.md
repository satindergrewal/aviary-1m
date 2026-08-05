# ds4-gates: what actually runs

Written 2026-08-06 after running every script in this directory against fork `53ed79e2` on the Mac.

**"We have 25 gates" was never a meaningful number.** Six had ever been run against this binary.
Running the rest surfaced seven defects — and only one was in the code under test. The other six were
the measuring apparatus reporting that things were fine.

## The three categories

| Category | Count | Runs unattended? | Scoring |
|---|---|---|---|
| **GATE** — produces a verdict | 12 | yes | PASS / FAIL is meaningful |
| **WALL** — produces numbers | 2 | yes | no verdict *by design*; do not score as vacuous |
| **REFUSES** — needs args (7) or the box (4) | 11 | no, by design | exit 2 / exit 1 is CORRECT, and is **not coverage** |

So unattended coverage is **14 of 25**, not 25. Eleven scripts need a target or a machine they are not
being given.

## Scoring rule (two of my scorers were wrong in one night — use this one)

```
exit 0 + verdict line       ran and concluded
exit 2 + PRECONDITION       correctly refused — not a pass, not a failure, NOT coverage
exit != 0, no verdict line  died before concluding; investigate the GATE, not the code
exit 0, NO verdict line     ran, concluded nothing, reported success  ← the dangerous row
                            (unless it is a WALL, which legitimately has no verdict)
```

⚠ **The verdict vocabulary is part of the scorer.** Mine was
`PASS|FAIL|GATE:|MATCH|DIVERG|PRECONDITION` and it scored `metal_paged_baseline` as vacuous when it
had passed, printing `-> IDENTICAL`. A scorer needs a positive control like any other marker: assert
it recognises a known-passing gate before trusting it on the rest.

## Defects this sweep found

| # | Where | What |
|---|---|---|
| 1 | `state_serdes_gate` | had been FAILING at 1236x short, reporting success, unread → **real code defect** |
| 2 | `paged_multimodel_gate` | with no args, fell through the loop and exited 0 having tested nothing |
| 3 | `paged_multimodel_gate` | printed a failure and exited 0 anyway |
| 4 | `p15_warm_admit` + 3 box gates | exited 0 on a machine they cannot run on |
| 5 | the batch runner | reported 4 passes from 0 executions (`timeout` absent on macOS; `$?` after a pipe reads `tail`) |
| 6 | the pre-commit privacy grep | fired and did not block the commit |
| 7 | `scrub_abs_paths` | written, wired to nothing; then wired *after* an `exit` that jumped it |

**Six of seven were in the apparatus.** A gate that exists but does not execute looks exactly like a
gate that passes — every gap here was absence-shaped.

## Conventions to keep

- A gate that cannot run **refuses loudly and exits non-zero**. `abort_paths_gate` had this right
  (`URL="${1:?server url}"`); `paged_multimodel_gate` had the identical requirement and fabricated a
  pass. Same problem, opposite behaviour.
- Result files must not carry absolute home paths — gates build `$OUT` from `$HOME` and echo it.
  `_no_abs_paths.sh` provides `scrub_abs_paths`; wire it as `trap 'scrub_abs_paths "${OUT:-}"' EXIT`.
  The `${OUT:-}` matters: the trap fires on early exits too, before `OUT` exists, and `set -u` aborts
  there.
- Every marker needs to be **two-sided** and read at the right moment. Startup-emitted markers may be
  read after startup; **work-emitted markers may not** — they do not exist until work has run.
