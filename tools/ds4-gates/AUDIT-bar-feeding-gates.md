# AUDIT: can the bar-feeding gates name their own failure mode?

**2026-08-09 · READ pass, no fixes applied · scored by reading verdict logic, not by grep.**

## The law being tested

> **A gate that cannot name WHICH of these four it is looking at may not say "defect".**

All four render identically as *"the paged column is wrong or missing"*, and all four occurred in one
session, with the **code fine in three of them**:

| # | rendering | the discriminator it needs |
|---|---|---|
| 1 | a real correctness defect | — |
| 2 | a **coin-flip tie** two kernels resolve differently | magnitude (top-2 gap) — **only bites text-comparison gates** |
| 3 | a **dead** arm — ctx overflow, port collision, never served | detect it **and print the server's own cause** |
| 4 | an **instrument** on the wrong key / log level / class | prove the marker **can fire**, both directions |

Plus the standing rule: a **dirty static control voids the run** rather than indicting the code.

## Result

| gate | 2 tie | 3 dead arm | 4 marker | static control | verdict |
|---|---|---|---|---|---|
| `long_context_gate` | **N/A — immune** | PARTIAL | **Y, both directions** | **Y** | **sound** |
| `warm_multislot_gate` | **N/A — immune** | PARTIAL | **Y** | **Y** | **sound** |
| `multislot_gate` | **N/A — immune** | not read | not read | **Y** | **sound on what was read** |
| `arch_serve_gate` | **Y** (as of tonight) | **Y** | **Y** | **Y** | sound, and only after 6 h of work |
| `agent_throughput_gate` | **N/A — timing only** | **N** | **N** | **N** | ⚠ **weakest, and least load-bearing** |

## ★ THE FINDING, AND IT CUTS AGAINST THE RANKING I ARGUED FOR

**The gate that feeds the 256k/1M bar is immune to the failure I measured tonight.**
`long_context_gate` scores `NEEDLE="MAGENTA-7742"` **retrieved**, not text equality. A near-tie flipping
one token elsewhere does not remove a distinctive retrieved string. **The 67%-per-200-tokens false-fail
rate cannot reach it**, and neither can it reach the 225k parity numbers, for the same reason.

⇒ Same for `warm_multislot` (semantic content markers: "A must contain counting, B must contain
letters") and `multislot_gate`, which explicitly keeps **byte-equality as a secondary observation,
never as the verdict** — and goes further, printing `⚠ BAR FAULT ... this is NOT a code defect` when
outputs are byte-identical yet score bad.

⇒ **Only `arch_serve_gate` ever used text equality as its verdict**, which is exactly the gate that
produced the one divergence in 100+ runs, and it is now the one with the magnitude clause.

## What the audit actually found, stated plainly

**I predicted most of the six would fail at least one column. They do not.** The prediction was wrong
in the direction that flattered tonight's work, which is the direction to distrust.

**Two things are genuinely weaker than the rest:**

1. **Column 3 is PARTIAL almost everywhere.** Gates detect a dead arm (`NEVER_READY`, `PRIME FAILED`,
   `evaluated only N prompt tokens`) and **abort rather than score it** — which is the important half.
   None of them prints the **server's own cause**. Tonight two harness faults (a `-c` overflow and a
   port collision) were diagnosable only by opening a log by hand.
2. **`agent_throughput_gate` fails three columns** — no marker check, no dead-arm cause, no static
   control. It is 62 lines and measures concurrency wall-clock only. **Least load-bearing of the six,
   and it should not be quoted for anything but a ratio.**

## ⇒ REVISED PRIORITY (supersedes the "audit above the spend" call of 05:15)

The argument for putting the audit above the spend was *"a 256k run gated by an instrument that cannot
name its own failure mode returns a verdict-shaped maybe."* **That premise is false for the gate that
feeds the bar.**

⇒ **The audit drops BELOW the spend. The 256k/1M run is NOT blocked on it.**
⇒ What remains is a **small, bounded** job, not a suite audit: add **cause-on-failure printing** to the
four gates that abort without it. That is the one real gap, it is the same fix already applied to
`arm_delta_vs_depth` and `margin_distribution`, and it is worth doing whenever, not first.

⚠ **SCOPE OF THIS AUDIT.** `multislot_gate`'s columns 3 and 4 were **not read** — recorded as unread
rather than assumed from the two that were. A partial audit that reports its gaps is usable; one that
infers the rest is the thing this document exists to prevent.
