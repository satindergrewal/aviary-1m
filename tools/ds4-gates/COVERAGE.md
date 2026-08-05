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

## ⚠ Flags are a CLAIM. Behaviour is the coverage.

`hybrid_paged_gate.sh` runs `-np 2`, is a correctness gate, and **passed 4/4** while the paged path
had a data-destroying multi-sequence bug. It passed because it sends requests sequentially:

```bash
for i in 0 1 2 3; do curl ... ; done      # no &, no wait
```

`-np 2` on the command line, one sequence in flight at a time, every batch single-sequence, bug
dormant. **The gate was configured for a regime it never entered — and that is why the defect
survived.**

Grepping the suite for `-np 2` reports three gates covering multi-slot. Reading what they *do* says
one, and it was written the day the bug was found:

| gate | claims | enters | |
|---|---|---|---|
| `hybrid_paged_gate` | `-np 2` | no | sequential |
| `p28_cuda_regate` | `-np 3` | no | sequential (also box-only) |
| `p28_finale_gate` | `-np 3` | no | sequential (also box-only) |
| `multislot_gate` | `-np 2` | **yes** | genuinely concurrent — and it FAILS |

`./lint_claimed_vs_entered.sh` checks this statically and exits non-zero on a mismatch. It carries a
positive control (it must find `hybrid_paged_gate`) because two of the audit tools written the same
night were blind to exactly what they audited — a scanner whose pattern could not cross a pipe, and
a verdict scorer that did not know the word `IDENTICAL`.

The lint deliberately says what it is **not** claiming: a sequential `-np` gate may be a perf wall
where sequential is correct. The bug is counting it as multi-slot *coverage*. Fix is narrow — make
it concurrent, or drop the `-np` flag so the claim matches the behaviour.
