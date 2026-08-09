# ds4-gates: what actually runs

Written 2026-08-06 after running every script in this directory against fork `53ed79e2` on the Mac.

**"We have 25 gates" was never a meaningful number.** Six had ever been run against this binary.
Running the rest surfaced seven defects — and only one was in the code under test. The other six were
the measuring apparatus reporting that things were fine.

## The three categories

| Category | Runs unattended? | Scoring |
|---|---|---|
| **GATE** — produces a verdict | yes | PASS / FAIL is meaningful |
| **WALL** — produces numbers | yes | no verdict *by design*; do not score as vacuous |
| **REFUSES** — needs args or the box | no, by design | exit 2 / exit 1 is CORRECT, and is **not coverage** |

⚠ **The counts are NOT written here — run `./coverage_census.sh`.** This section originally said
"12 GATES / 2 WALLS / 11 REFUSES, unattended coverage 14 of 25". Four scripts were added the same
night and that number silently became wrong (it is now 18 of 29). A count written once and never
re-derived decays every time the thing it describes changes — which is the same failure this whole
document is about. The census script derives it, so quoting it cannot go stale.

Refusals still are not coverage: those scripts need a target or a machine they are not being given.

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
| `multislot_gate` | `-np 2` | **yes** | genuinely concurrent — and it FAILS *(true when written 2026-08-06 01:43; see the note below)* |

⚠ **"and it FAILS" was TRUE WHEN WRITTEN and is NOT true now — annotated rather than deleted,
because it records a real observation.** At 01:43 on 2026-08-06 the `-np > 1` absolute-offset
corruption was live; the fix landed **ten hours later** the same day (`c2f28a79d`, 11:23) and the
defect was closed by measurement on 2026-08-09 (`00cb274`, `e404116` — four-way warm clean on two
model families). Deleting the row would destroy the evidence that the gate **detected** a real
defect, which is the strongest thing anyone can say about a gate.

`./lint_claimed_vs_entered.sh` checks this statically and exits non-zero on a mismatch. It carries a
positive control (it must find `hybrid_paged_gate`) because two of the audit tools written the same
night were blind to exactly what they audited — a scanner whose pattern could not cross a pipe, and
a verdict scorer that did not know the word `IDENTICAL`.

The lint deliberately says what it is **not** claiming: a sequential `-np` gate may be a perf wall
where sequential is correct. The bug is counting it as multi-slot *coverage*. Fix is narrow — make
it concurrent, or drop the `-np` flag so the claim matches the behaviour.

## Gates added 2026-08-06 (and what they do when the regime is ABSENT)

That last column is the point. A gate's behaviour when it cannot measure anything is what decides
whether its silence means something.

| gate | tests | regime absent → |
|---|---|---|
| `multislot_gate.sh` | two concurrent sequences produce their own answers | FAILs on no-swap/no-pool markers |
| `batch_offset_invariant_gate.sh` | `sum(batch_lens) == n_tokens` per dispatch | **INCONCLUSIVE, exit 2 — never PASS** |
| `lint_claimed_vs_entered.sh` | a gate's `-np` flag matches its behaviour | n/a (static) |
| `long_context_gate.sh` | needle retrieval at 25.7K vs static control | UNINTERPRETABLE if the static control fails |
| `rset_leak_probe.sh` | Metal residency-set leak, 5 one-factor arms | arm marked UNUSABLE if its marker is missing |
| `prefill_scaling_probe.sh` | prefill cost curve, static vs paged | point DISCARDED if it exceeds the context |
| `slot_stagger_probe.sh` | does the defect need a shared prefill batch | arm UNUSABLE if the stagger did not happen |

`batch_offset_invariant_gate` is verified in both directions: FAIL (exit 1, 600/600) on the broken
tree, INCONCLUSIVE (exit 2) at `-np 1` where no multi-sequence dispatch can occur. A PASS in the
second case would reproduce exactly the failure that hid the defect it was written for.

## Instruments added 2026-08-09, and the category the three did not have

⚠ **The same staleness this file's own header warns about, recurring against the file that warns about
it.** The header says the COUNTS are not written here because four scripts were added in one night and
a hardcoded number silently went wrong. **Four more were added on 2026-08-09.** The counts were fine —
`coverage_census.sh` derives them (**47 total, 36 unattended**, up from 29/18). **The PROSE was not**,
and none of the four appeared here until this entry. ⇒ *The earlier fix solved the numeric half and left
the descriptive half exactly as fragile.*

| script | category | what it produces | needs a model? |
|---|---|---|---|
| `arch_selfconsist_probe.sh` | **WALL** | 3 arms — static ×N in one process, static in a SECOND process, paged — to separate a **random** flip from a **systematic** one. Read-off is written IN the file, before any run. | yes |
| `arm_delta_vs_depth.sh` | **WALL** | the paged-vs-static logprob delta **on the same token** across context depths. Deliberately not text equality. | yes |
| `margin_distribution.sh` | **WALL** | the model's top-2 margin distribution over many generated positions, and the **predicted false-fail rate** of a byte-equality gate. **REFUSES** on <50 positions or a distinct-token ratio <0.20. | yes |
| `lint_paged_consumers.sh` | ⚠ **SOURCE LINT — a fourth category** | whether every paged consumer announces itself, so `arch_serve_gate` can see it. exit 0/1/2 = PASS/FAIL/VOID. | **NO — no model, no GPU, runs anywhere** |

⚠ **`lint_paged_consumers.sh` does not fit the three categories and I am not squeezing it in to keep the
table tidy.** GATE / WALL / REFUSES all describe things that *serve a model and observe it*. This reads
**source**. It is the only instrument here that can run with no GPU, no weights and no server, which
makes it the only one that could ever go in CI. `coverage_census.sh` classifies it a GATE — defensible,
since it emits a real PASS/FAIL/VOID — but the distinction that matters to a reader is **what it looks
at**, not what it returns.

⚠ **The three WALLs above are WALLs on purpose.** None emits PASS/FAIL, because none should: a delta, a
margin distribution and a self-consistency triple are **magnitudes**, and the whole lesson of
2026-08-09 is that a magnitude scored as a verdict is how a 1.03× tie got filed as a defect. **Scoring
them would recreate the defect they exist to prevent.**
