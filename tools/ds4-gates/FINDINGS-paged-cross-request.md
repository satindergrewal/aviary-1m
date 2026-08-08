# Paged KV: an INTERMITTENT correctness failure near ~50k tokens

**Status: OPEN. Real, reproduced dozens of times, NOT champion-specific, predates the
2026-08-06/07 kernel work. Rate unmeasured. Every length-based conclusion previously in this
document is WITHDRAWN — see "What was withdrawn and why".**

Config where it was observed: default paged path with `DS4P_METAL_CHAMP` **unset**,
`--kv-block-size 16 -ngpub 4608 -ncpub 512`, Ornith-1.0-9B-1M IQ2_M,
`-c 73728 -np 1 -b 2048 -ub 2048 -cram 0`, `cache_prompt: false`, `temperature 0`, `seed 1`,
prompt `toks[:N]` with N asserted in the harness.

## What the failure looks like

Static is the reference and was **coherent in every single measurement**, so this is not the model or
its YaRN extension.

- Paged output ranges from *fluent but different from static* to outright degenerate
  (`' H and 和Q of/路 ） NOTE –'`, `'. - - - - - '`, `' is the 1 - - - -'`).
- In one stretch it also showed a **first-request-correct, all-later-requests-wrong** pattern that
  survived an intervening unrelated prompt.

## ⚠ It is intermittent across windows, not across configurations

The single most important fact, and the one that invalidates most of the original analysis:

| | |
|---|---|
| Earlier in the session | the config above failed repeatedly, ~30 observations, always the same wrong text |
| Later, same binary, same flags, same prompt, same N | **0 failures in 6 independent server starts** |

Same everything. It failed in one window and will not fail in another.

**Strongest correlate, filed as a lead and not a finding:** the failing stretch overlapped hours when
this box ran concurrent heavy GPU work (a lurking probe doing repeated 50k inferences, plus other
jobs). The clean stretch had one server at a time under a lock. That is a correlation across two
windows — weak evidence. Memory pressure or allocator interaction would fit everything observed:
intermittent, config-independent, content-independent, and invisible to both the poison test and the
slot-coverage invariant because none of the paged bookkeeping is wrong.

## The load hypothesis is dead, and the failure is UNINSTRUMENTED

| condition | failures |
|---|---|
| quiet box, one server under a lock | **0 / 6** independent server starts |
| second server hammering 49k prefills throughout | **0 / 5** |

Deliberately recreating the concurrent-GPU-work condition that correlated with the failing window
produced nothing. The load correlation was two windows and it does not survive a controlled test.

**And the failing runs left no trace.** A failing server's log against a clean one at the same config:

| | failing (`ND-paged`) | clean (`RT-1`) |
|---|---|---|
| `build_attn_paged_or_null` static-path warnings | 88 | 88 |
| distinct layers taking the static path | 8 | 8 |
| `fails the paged capability contract` | 0 | 0 |
| warnings or errors present in one and not the other | **none** | |

Same warnings, same layer fallbacks, same scheduler state — then different tokens.

⇒ **Every existing marker, warning and assert on the paged path passes while the output is wrong.**
So the first job is not to find the bug; it is to add a check that can *fail* when it happens, because
none of the current ones can. This defect is not merely intermittent, it is uninstrumented.

*(Side note for anyone reading the speed numbers: 8 layers take the static path in every run, clean or
broken. That is this model's shape, not a defect — but the paged path is not carrying all the
attention even when it is on.)*

## What survives — eliminations that were repeated, not sampled

Each varied the factor, confirmed it varied, and held across multiple arms.

| Candidate | Test | Result |
|---|---|---|
| Block content / stale KV | `DS4P_KV_POISON=1` fills the pool with `0xFF` (every fp16 a NaN) | request 1 still correct; failure unchanged |
| Write/read slot mapping | `DS4P_SLOT_COVER=1` asserts `slots[t] == btab[pos/bs]*bs + pos%bs` for every token | 0 mismatches over ~50,000 tokens, both requests |
| Block accounting | checkout log | identical across requests: `n=3155 free_before=4608`, three releases for three requests |
| Block identity / free-list order | `DS4P_FREELIST_FIFO=1`, with `DS4P_METADUMP` confirming the IDs moved | req1 blocks `0..3154`, req2 blocks `3238+` — **disjoint** — still failed identically |
| The champion | reproduces with `DS4P_METAL_CHAMP` unset at bs=16 | not champion-specific |
| Decode `nwg` path | `DS4P_CHAMP_VEC_NWG=1` | unchanged |
| The model | static reference at every length measured | static coherent everywhere |

**Not ruled out — control was vacuous:** graph reuse.
`llm_graph_input_attn_kv_paged::can_reuse()` returns false unconditionally, so
`LLAMA_GRAPH_REUSE_DISABLE=1` could not vary anything on this path.

**Could not be tested:** the ubatch. `llama-paged-scheduler.cpp:82` asserts
`n_batch == ctx->n_ubatch()`, so `-b 2048 -ub 512` aborts at startup; varying `-ub` requires moving
`-b` with it, which makes the arm two-factor.

## What was withdrawn and why

All of the following were measured **n=1 per point** on a failure now known to be intermittent. They
are not evidence.

- A single-token onset at 49,997 (clean) → 49,998 (broken).
- Its "confirmation" on a second, unrelated prompt breaking at the identical token.
- The block-size dependence (`bs=64` clean through 95,000).
- The `-c` dependence.
- The earlier "two distinct defects" split (already retracted once, for a different reason).

The n=1 design came from observing ~30 agreeing failures **all at one length inside one window** and
generalising determinism from that. The caveat was written into this document and committed — and
then three further experiments were built on the number it warned about instead of re-testing it.
**Publishing a limitation is not respecting it.**

## Probes

- `DS4P_METADUMP=N` — first N ubatches: `n_tokens`, `n_seq`, `ctx_lens[0]`, `offs[0]`, `lens[0]`,
  `slots[0..3]`, `btab[0..3]`. Prints the cap it is using.
- `DS4P_SLOT_COVER=1` — the write/read invariant, as an assertion rather than a dump.
- `DS4P_KV_POISON=1` — pre-existing; fills the pool with `0xFF`.

## Next step, and it is the only one worth taking first

~~Measure the rate under load.~~ **Done: 0/5 under deliberate load, 0/6 quiet. The load hypothesis is
dead and the failure is not currently reproducible on demand.**

**Add instrumentation that can fail.** The log comparison above shows every existing check passing on
a run that produced wrong output. Candidates, cheapest first:

1. A per-request output hash against a static reference, run automatically at several lengths — so the
   next occurrence is *caught* rather than noticed by eye.
2. An assertion on the attention output itself (finite, bounded, non-degenerate) rather than on the
   bookkeeping around it, since the bookkeeping is provably clean.
3. Keep `DS4P_SLOT_COVER` on in any long-running paged server: it costs one comparison per token and
   is the only invariant here that could have caught a mapping fault.

Until something can fail on demand, no bisect is worth running.

## Harness rules this hunt paid for

1. **Confirm the independent variable changed before interpreting the dependent one.** Three controls
   this session could not vary their factor and were read as refutations.
2. **Assert lengths, do not label them.** `toks[:50480]` on a 50,473-token list ran 50,473 while the
   output said `rem=0`.
3. **Key caches on their inputs.** A fixed `LOGDIR` reused a 32k token file for a 64k run.
4. **One probe at a time, kill by PID.** Detached scripts sharing `pkill -f llama-server` destroyed
   each other's servers; one lurked for hours — and is the leading suspect for the load correlation
   above.
5. **A comparator needs an external reference.** `r1 == r2` reported CLEAN for two requests agreeing
   on garbage.
6. **Include a control arm that reproduces the KNOWN result.** The `-c` sweep's baseline came back
   clean, which is the only reason the intermittency was caught at all.
7. **Never build on an n=1 result you have already flagged as n=1.**

---

# 2026-08-08 — reproduced at 225k, with the champion ACTIVE, and a first rate estimate

This session hit the same defect independently and re-derived most of this document from scratch
before reading it. What follows is what it adds.

## Range extended: ~50-73k → 224,992 tokens

| context | config | runs | result |
|---|---|---|---|
| ~50-73k | `DS4P_METAL_CHAMP` **unset**, bs=16 | ~30 obs | documented above |
| **224,992** | **CHAMPION ACTIVE**, bs=64, `--no-kv-unified` | **2 of 6** | **FAIL** |

⇒ **Not champion-specific in the other direction either.** This document recorded it with the champion
*unset*; it also fires with the champion *active* at bs=64. The kernel is not the variable.

## First rate estimate — this document says "rate unmeasured"

Paged + champion at 225k, all inside the model's native 250k:

| | runs |
|---|---|
| pass | `parity_fair/paged_opt`, `parity_rep/rep_paged_first`, `parity_ab/B_champ_nolazy`, `notrace/resp` |
| **FAIL** | `parity_bins/paged_first`, `ladder/p225` |

**4 pass / 2 fail ≈ 33%.** Pooling is legitimate *only because* the failures do not track any config
variable (below).

## Three suspects blamed and retracted this session — all config-independent, as this file predicted

| blamed | evidence | why it died |
|---|---|---|
| paging in general | 1 failing run | static clean under identical conditions, then the same config passed 4x |
| `DS4P_PP_TRACE` (an instrument added that day) | 1 failing traced run, 4 untraced passes | the next untraced run reproduced it |
| `LLAMA_PAGED_POOL_HEADROOM=1.05` | 1 failing run at 1.05 | **the other failure ran at the default 1.5** |

**The two failures differ in both variables that were blamed.** This file's "config-independent" line
already said so; it was re-derived the expensive way.

⚠ **Graph reuse is still the only unexcluded candidate — and it cannot be the cause of *these*
failures.** Measured on the ladder arms: `s225` (clean, static) `graphs reused = 14`; `p225`
(corrupted, paged) `graphs reused = 0`. Reuse is structurally disabled on the paged path, so it is
**constant across passing and failing paged runs** and explains neither. "Control is vacuous" means
**untested**, not guilty. Building `can_reuse` remains worth doing to *close* the suspect.

## The 88-vs-88 diff reproduces at 4.5x the length

| | failing (p225) | passing (notrace) |
|---|---|---|
| `initializing paged KV cache` | 1 | 1 |
| `took the STATIC path` | 88 | 88 |
| `CHAMP-PAGED ACTIVE` | 1 | 1 |
| `n_gpu_blocks` | 2 | 2 |

Identical on every logged axis, then different tokens — exactly as recorded at ~50k. **`prompt_n` is
224992 on both**, so the full prompt is ingested; it counts tokens *submitted*, not tokens *attended*,
and cannot distinguish corruption from silent truncation.

## Failure output is context-dependent, not a fixed string
`' 123456789012345'` at 225k · `'\n187878787878787'` at 501k. Degenerate digit sequences in both
cases, but **not the same string** — so "byte-identical, therefore deterministic state" holds *within*
a context size and breaks *across* one.

## ⚠ 512k is PENDING, not a third failure
`ladder/p512` (501,733 tokens) returned garbage — but that is **2x the model's native 250k**, and
`memory/yarn-context-honesty.md` records that extrapolation beyond native is expected to be unstable.
Without its paired static arm this cannot separate a paging defect from YaRN degradation. **Not
counted in the rate above.**

## Measured prefill cost (holds regardless of output correctness)
225k ≈ 215 tok/s · 512k ≈ 112 tok/s — roughly half the throughput for 2.2x the context, the expected
O(n²) term.

⇒ Extrapolating (n=2, labelled as such): **1M prefill ≈ 4-5 h per arm.** At a ~33% failure rate, a
paired 1M run has ~1-in-3 odds of an uninterpretable paged arm after half a day. **A short-context
detector is therefore the precondition for measuring the top of the range at all**, not merely a
convenience.

---

# ★★ 2026-08-08 — REPRODUCED DETERMINISTICALLY AT 8k. It is cross-request, not long-context.

Ten sequential requests, one paged server, ~7,979-token prompt (needle at 50%), ~15 s each:

```
run  1: FOUND ' MAGENTA-7742\n\n<think>\nThe user wa'
run  2: MISS  ' 1234567890\n\nNote 34'
run  3..10: MISS, byte-identical
FAILURES: 9 / 10
```

**Request #1 correct. Every subsequent request wrong, byte-identical.** At a prompt length **6x below**
the range this document previously recorded (~50-73k).

⇒ **Context length was never the variable.** Every failure observed this session was at 225k or 512k
because that is where the looking happened. The trigger is a **second request on a live paged server**.

This is the pattern this file already named — *"first-request-correct, all-later-requests-wrong ...
survived an intervening unrelated prompt"* — now reproduced deterministically and cheaply.

## The "~33% intermittent rate" was an artefact of the harnesses

| harness | sends a warmup request? | 225k paged outcome |
|---|---|---|
| `parity_bins.sh` | **YES** | **FAIL** |
| `ladder.sh` | **YES** | **FAIL** |
| `parity_fair.sh` | no | PASS |
| `parity_ab.sh` | no | PASS |
| `parity256k.sh` | no | PASS |
| `parity_rep.sh` | YES | PASS ← **real exception, not explained** |

Both failures came from harnesses whose warmup made the measured request **#2**. All three no-warmup
harnesses passed. **The variable was in the test scripts, not in the paged path.** A deterministic bug
was treated as stochastic for hours, and a pooled failure rate plus an interleaved A/B were built on
top of that phantom.

⚠ `parity_rep` sends a warmup and passed. 5 of 6 is a strong signal, not a law. The canary is the
cleaner evidence.

⚠ **The check that established this nearly returned the opposite.** The first version grepped for
`warm`, which also matches the `--no-warmup` *server flag*, and reported YES for all six harnesses —
i.e. "warmup does not explain it". One character class between a correct finding and a confidently
wrong one, in the check meant to settle the question.

## The detector this unlocks
**Two ~15-second 8k requests.** Request 1 clean + request 2 wrong ⇒ the paged path is broken. No long
context, no 4-5 hour prefill, deterministic. Previously the cheapest known reproduction was a
40-minute 225k needle run.

## Ladder result (all four arms, needle-gated)
| arm | ctx | kernel | wall | needle |
|---|---|---|---|---|
| p225 | 262144 | CHAMPION | 1020 s | **FAIL** |
| s225 | 262144 | STATIC | 1039 s | pass |
| p512 | 524288 | CHAMPION | 4469 s | **FAIL** |
| s512 | 524288 | STATIC | 4597 s | pass |

**Paged failed both, static passed both.** `s512` retrieving a needle at **501,733 tokens — 2x the
model's native 250k** — also removes the YaRN-extrapolation caveat: `p512`'s failure was this defect,
not context degradation. Prefill throughput was within 1% across arms at both sizes.

---

# THE VARIABLE: the FINAL PARTIAL PREFILL CHUNK (2026-08-08)

Not the ubatch size. **The size of the last, partial prefill batch.**

## How the ubatch framing died
A sweep over ubatch values came back **non-monotonic**, which no threshold can produce:

| ub | result |
|---|---|
| 384 (6 blocks) | CLEAN |
| 400 | CORRUPTS |
| 416 | CORRUPTS |
| **432** | **CLEAN** ← clean *between* two corrupting values |
| 448 (7 blocks) | CORRUPTS |

The earlier "threshold at 7 blocks" read was an artefact of sampling only block multiples. Sampling
between them is what killed it.

## What the same data says once you compute the last chunk
Prompt = 7427 tokens. `last = N mod ub`:

| ub | last batch | result |
|---|---|---|
| 256 | 3 | CLEAN |
| 320 | 67 | CLEAN |
| 384 | 131 | CLEAN |
| 432 | 83 | CLEAN |
| 400 | 227 | **CORRUPTS** |
| 416 | 355 | **CORRUPTS** |
| 448 | 259 | **CORRUPTS** |
| 512 | 259 | **CORRUPTS** |

**8 for 8.** Small final chunk clean, large final chunk corrupt — including the 432 anomaly, which is
clean *because* 432 happens to divide 7427 into a remainder of 83.

## The decisive test: hold ub, move the remainder
An 8/8 split over one prompt is a correlation. The falsifiable version changes the remainder **without
touching the ubatch**, by varying prompt length instead.

`-ub 512` held fixed; `N = 15*512 + r`; three requests per point on a fresh server.

**`-ub 512`, remainder 3 → CLEAN (3/3 needle FOUND).** The same ubatch that corrupted at remainder 259.
Same binary, same flags, champion live. Only the prompt length differs, and the outcomes are opposite.

⇒ **The ubatch cannot be the cause.** The final partial chunk can.

⚠ Prompt lengths are **constructed as tokens**, not estimated from text: `/tokenize` once, prepend
filler tokens to hit `N` exactly, and assert `prompt_n == N` on every response. A text prompt cannot
hit a token count, and a 3-token error silently relabels the point being measured.

## ⚠ Consequence: the workaround line changes
Remainders live in `[0, ub)`. So `-ub 256` is **not** safe in general — a differently-sized prompt can
leave a 255-token final chunk, inside the corrupting band. The earlier "-ub 256 is clean, n=5" was true
for a 7427-token prompt (remainder 3) and is **prompt-dependent luck, not margin**.

**Guaranteed-clean setting pending refinement: `-ub 128`** (max remainder 127, below the bracket floor).

## Where it points
The last-chunk case is exactly the `is_prefill && !mid_prefill` predicate in the paged scheduler
(`src/llama-paged-scheduler-impl.cpp`). A sufficiently large final chunk writes different KV on a
repeat request — consistent with the measured checksum divergence (req1 4.40e11 vs req2 4.59e11,
identical at N=512, diverging by N=1024).

⚠ Resonance, recorded with its refutation attached: the fleet-era **partial-last-block** lead was never
refuted on the merits — the test that was supposed to settle it was shown to be vacuous. Tonight's
partial-*chunk* variable is the same family one level up (batch rather than block). Recorded as a
possible early sighting, **not** asserted as the same defect.
