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

## ⚠ REFUTED SAME NIGHT: the remainder is not the variable either

The decisive test held `-ub 512` and walked the remainder by varying prompt length. Pre-registered
prediction was "remainder <= 131 CLEAN". **It failed at 131.**

| N | remainder | paged | static |
|---|---|---|---|
| 7683 | 3 | CLEAN | — |
| 7747 | 67 | CLEAN | — |
| 7811 | 131 | **CORRUPTS** | — |
| 7840 | 160 | **CORRUPTS** | — |
| 7872 | 192 | **CORRUPTS** | — |
| 7907 | 227 | **CORRUPTS** | — |
| 7939 | 259 | **CORRUPTS** | — |
| 8035 | 355 | **CORRUPTS** | — |
| 8128 | 448 | **CORRUPTS** | **CLEAN** |

Last chunk 131 is CLEAN at `ub=384` (N=7427) and CORRUPT at `ub=512` (N=7811). Same last-chunk size,
opposite outcome. So both single-variable theories now have a killing counterexample:

- **pure ubatch** — `ub=512` is clean at remainder 3 and corrupt at remainder 259.
- **pure remainder** — remainder 131 is clean at `ub=384` and corrupt at `ub=512`.

The 8/8 split stands as data and dies as an explanation: one prompt length cannot distinguish the
remainder from anything else that moves with it.

### Premises checked before asserting the refutation
- `-b == -ub` in **both** sweeps, read from the server banners (`n_batch = n_ubatch = 384/400/432/512`).
  Had `-b` stayed 512 while `-ub` varied, the whole x-axis would have been fiction.
- Env identical across sweeps: `DS4P_PAGED_HYBRID/SWA/METAL_CHAMP=1`, `n_ctx=32768`.
- Block allocation confirms the lengths reached the server: 121/122/123 blocks for N=7683/7747/7811.
- **Accidental control:** N was stepped by exactly 64, so all three points leave the *same* partial-block
  fill (61 unused tokens in the last block). Two clean, one corrupt ⇒ **partial last BLOCK fill is not
  the variable**, ruled out by construction rather than by argument.

### ⚠ Order confound in the sweep harness
`start_srv` is called **once**, before the loop, so those 8 points are 24 sequential requests on **one**
server. Request "1" of remainder 131 is really request #7. For a defect whose signature is *first request
clean, later requests wrong*, a monotone corrupt tail is equally explained by a **sticky state that
tripped at request #7**. The ascending sweep cannot separate the two.

Resolution arms, fresh server each: **A** remainder 131 alone (sticky ⇒ req1 clean; geometry ⇒ 3/3
corrupt) · **B** descending, 448 then 3 · **C** 355 then 67 (is the state recoverable).

### What survives the confound
`-ub 512` is **CLEAN at N=7683** and **CORRUPT at N=7427**, each as the first prompt on its own fresh
server, each with identical repeats. Same binary, same flags, same ubatch, opposite outcome.
**Prompt geometry governs; the ubatch setting does not.**

The static arm retrieves the needle at N=8128 where paged misses 3/3, so the corrupting prompts are
fine and every paged failure above is a real paged failure.

---

# RESOLVED SHAPE: a trigger + a sticky state (2026-08-08, fresh-server data)

Two facts that had been fused into one:

1. **The trigger is a property of the prompt.** Some prompts poison a paged server, others never do.
2. **The damage is sticky.** A triggering prompt answers request 1 correctly and is wrong from request 2
   onward; after that **every** request is wrong regardless of which prompt is sent. Corrupt output is
   degenerate (` 1234567890` in every observed case). Shrinking the prompt does not recover it.

This explains every symptom in this file at once: *first request correct, later requests wrong* ·
*survived an intervening unrelated prompt* · warmup-sending harnesses failing while no-warmup harnesses
passed · the phantom "~33% intermittent" rate.

## Order confound found and removed
The ascending sweep ran 8 points as 24 sequential requests on **one** server, so its monotone corrupt
tail was equally explained by a sticky state tripping at request #7. Re-run on **fresh servers**:
remainder 131, which "corrupted" in the ascending sweep, is **clean 3/3 alone**. Every number below is
fresh-server, ≤3 requests, one model load.

## Length is ruled out in both directions
| N | final chunk | verdict |
|---|---|---|
| 8195 | 3 | CLEAN |
| 8451 | 259 | **TRIGGERS** |

Both are *longer* than the 8128 trigger. A long prompt with a small final chunk is clean; a longer one
with a large final chunk poisons. Absolute prompt length moves the wrong way.

## The chunk sequence was measured, not computed
`created llama_batch: ... n_tokens=` at `-lv 5`:
- `ub=400, N=7427` → 18×400 + **227**
- `ub=512, N=7907` → 15×512 + **227**

Same final chunk size, opposite verdicts. So *final chunk size alone* is insufficient — and the hand
arithmetic was right, which is why the counterexample is real.

## The rule that survives all 20 points
`ub=512` is 8 blocks, so its chunks always start block-aligned; `ub=400` is 6.25 blocks, so its final
chunk starts at **offset 32 inside a block**.

> **Corruption iff `(block_offset + final_chunk_size) >= 256`** — the span the final prefill chunk covers
> measured from its block-aligned start.

| case | offset | final chunk | span | verdict |
|---|---|---|---|---|
| `ub=400 N=7427` | 32 | 227 | 259 | **TRIGGERS** |
| `ub=512 N=7936` | 0 | 256 | 256 | **TRIGGERS** |
| `ub=512 N=7937` | 0 | 257 | 257 | **TRIGGERS** |
| `ub=512 N=7907` | 0 | 227 | 227 | CLEAN |
| `ub=432 N=7427` | 48 | 83 | 131 | CLEAN |

The two counterexamples that killed the simpler rules are what force this one: *final-chunk-size < 256*
fails on `ub=400`; *touches ≤ 4 blocks* fails on `N=7936`. The span rule absorbs both.

⚠ 20/20 is a **fit** — both rules were built from the same points. Open tests: the one-token razor
(N=7935, span 255, predicted CLEAN) and whether the constant is **256 entries** or **4 blocks**
(separable only by changing `--kv-block-size`; note bs=16 disables the champion kernel, so that arm
needs a same-kernel control).

## Workaround
Bound the final chunk below the threshold for **every** prompt: **`-ub 128`**. `-ub 256` permits a
255-token final chunk and, at a non-block-multiple ubatch, a span above 256 — it is prompt-dependent,
not safe.

---

# The constant: 256 entries, NOT the block size — and NOT the champion kernel

`--kv-block-size 16` (which **refuses** the champion: `CHAMP-PAGED REFUSED (bs!=64) D=256 bs=16`, so the
run uses the scalar paged kernel):

| span | verdict |
|---|---|
| 32 | CLEAN |
| 128 | CLEAN |
| 255 | CLEAN |
| 256 | **TRIGGERS** ← positive control |
| 256 (2nd length) | **TRIGGERS** |

The positive control fires, so the clean points are evidence rather than absence. Two conclusions:

1. **The threshold is 256 irrespective of block size.** "4 blocks" predicts 64 at bs=16; a span of 128
   sailed through. Dead.
2. **The bug is not in the champion kernel.** Two different attention kernels, same 256 boundary, same
   sticky signature ⇒ whatever is sized for 256 sits **upstream of the kernel choice**.

## Boundary, consolidated
| CLEAN | TRIGGERS |
|---|---|
| span 255 N=7935 bs64 champ | span 256 N=7936 bs64 champ |
| span 255 N=8447 bs64 champ | span 256 N=8448 bs64 champ |
| span 255 bs16 scalar | span 256 N=7936 bs16 scalar |
| span 128 bs16 scalar | span 256 N=8448 bs16 scalar |
| span 32 bs16 scalar | span 257 N=7937 bs64 |
| span 227 N=7907 bs64 | span 511 N=7679 bs64 |
| spans 3 / 67 / 131 / 160 / 192 | span 259 `ub=400` (offset 32) bs64 |

25 fresh-server points, zero violations, boundary sharp to **one token**.

## ⚠ Three second-model attempts, all VOID — and the admission bar that came out of them
`256 entries` and `head_dim = 256` are the same number on Ornith and cannot be separated on it.

| model | head_dim | outcome |
|---|---|---|
| Qwen3.5-4B-DFlash | 128 | **not a model** — a draft head (`dflash requires ctx_other`, 6 blocks, `target_layers`) |
| Qwen3-4B-Q8_0 | 128 | garbage (`妫妫妫…`) at request 1 — **and identical garbage on the STATIC path**, so the model, not paging |
| Nemotron-Mini-4B | 128 | bare `\n` at request 1 on every span — wants a chat template |
| Gemma4-12B | 512 | clean at 255/256/511/512 — **and 48 of 48 layers took the STATIC path** |

**Gemma ran zero paged layers.** The pool was allocated, the champion was loaded, and the no-consumer
guard stayed silent — all three are *producer-side* signals. Ornith for comparison: 8 of 32 layers
static (the SWA layers), 24 running paged.

⇒ **Admission bar for any model in this investigation:** distinct layers logging `took the STATIC path`
must be **strictly less than `n_layer`**, plus request 1 coherent and correct. One grep, applied before
any span is scored.

⇒ ⚠ **Separate, larger finding for the arch-support file:** the paged path silently declines an entire
arch while reporting a healthy startup. Gemma4 is on the 19-list and this file's own notes record it as
PASSING under `DS4P_PAGED_SWA=1`; on this checkpoint it runs fully static with the pool allocated and
the champion loaded. Whatever that PASS measured, it was not paged attention.

**So the constant is still `256 entries` vs `head_dim=256`, unresolved**, and resolving it needs a model
that clears the admission bar.

---

# Localisation: the damage is at the first token of the NEXT request's second chunk

## The coordinate
KV read back through the block table, request 1 vs request 2, bisected on N:

| arm | first diverging N | earliest bad position |
|---|---|---|
| span 256 (TRIGGERS), ub=512 | 513 | **512** |
| span 255 (CLEAN), ub=512 | 7935 | tail only — the benign per-request tail difference |
| span 259 (TRIGGERS), ub=400 | 401 | **400** |
| span 83 (CLEAN), ub=432 | 7427 | tail only |

**The damage lands on the first token of the second prefill chunk, wherever that chunk starts.** Move the
ubatch, the damage moves with it, one token exact, twice.

⚠ **Not the block grid.** At `ub=400` the block boundaries are 384 and 448; both are clean and 400 is not.

⚠ **A clean run also diverges at the tail.** "The caches differ" was never the signal — *how far back the
difference reaches* is. Running only the trigger arm would have produced a finding true of clean runs too.

⚠ **Request 1 answers correctly out of a cache that is already wrong.** Every output-based gate in this
file is therefore weaker than it looks: they cannot see damage that has not surfaced yet.

## Ruled out, each with a firing control
| suspect | how it died |
|---|---|
| block allocation / freelist order | same 7427 prompt: `ub=400` (triggers) and `ub=432` (clean) both `CHECKOUT n=117 / RELEASE n=117` twice — **byte-identical** |
| `max_blocks` / consumer stride | chunk ledger: `stride=125` in both requests |
| `n_past`, resolved write slots | chunk ledger: `n_past`, `slot_first`, `slot_last`, `bt_len` **identical** for chunks 1-3 and the last chunk |
| graph reuse | `DS4P_NO_GRAPH_REUSE=1` → still TRIGGERS (control: reuse ON still triggers; clean prompt stays clean) |
| compute-buffer sizing | the `does not match expectation` warning is identical between clean and corrupt at every ubatch — it tracks the ubatch, not the defect |

⚠ **Bookkeeping trap:** `request_id` is **0 for both requests** (one slot, id reused). Grouping the chunk
ledger by rid compares a request against itself and looks completely normal. Split on `n_past` resetting
to 0.

## ★ What the model actually is
`DS4P_KVSUM_LAYERS=32` (the probe previously printed **one** layer, so every earlier positional claim was
a claim about layer 3):

- **8 layers** have a paged tensor — 3, 7, 11, 15, 19, 23, 27, 31 — and **all 8 first diverge at N=513**.
- **24 layers** return `-1`: no paged tensor at all.
- `arch = qwen35`, `n_swa = 0`, `llama_memory_recurrent: 50.25 MiB, 32 layers`.

**Ornith-9B is 8 full-attention layers and 24 recurrent layers** (Qwen3-Next shape: three linear-attention
layers, then one attention layer). The non-paged layers are **not** SWA layers — `n_swa = 0`.

⇒ The 24 recurrent layers carry state that is **not in the paged cache and has never been checksummed**.
Layers 0-2 are recurrent and feed layer 3, the first paged layer and the first to show damage. Recurrent
state entering chunk 2 wrong on request 2 would corrupt layer 3's K/V at position 512 and cascade — which
is a route from a trigger at the end of request 1 to damage at the start of request 2 without any
scheduler value changing.

⚠ Connects to the existing scar *recurrent state is never rolled back* (found for speculation, guarded by
refusing paged speculation at `n_rs_seq > 0`; the rollback itself was never fixed). Second implication of
the same subsystem, first one in plain serving. **Not** asserted as the same bug — the speculation case
needed rejected drafts and this one has none.

⚠ The 88 `took the STATIC path` warnings all fire **before** the first request (line 209 vs 356) — graph
construction, not per-batch fallback.

---

# ⚠ CORRECTION: the poison does NOT affect single-chunk prompts

| arm | result |
|---|---|
| fresh server, 221-token prompt (**one** prefill chunk) | FOUND, FOUND |
| span-256 poison: req1 / req2 | FOUND / **MISS** — server confirmed poisoned |
| **221-token prompt on the poisoned server** | **FOUND, FOUND** |

Every "sticky" prompt tested earlier was 7000+ tokens, i.e. 15 chunks. The earlier claim *"every request
after it is garbage, whatever prompt you send"* is **wrong**.

⇒ **Corrected:** after the trigger, every subsequent **multi-chunk** prefill is corrupted from its second
chunk onward; single-chunk prefills are untouched.

⇒ That is sharper, not weaker: **there is no persistent corrupted data.** The damage is manufactured
fresh at each chunk-to-chunk handoff. A poisoned server has nothing wrong sitting in its cache — it has
something wrong in how it carries state between prefill chunks, and a request with no handoff never meets
it. This is the strongest support for the recurrent-state route: those 24 layers are exactly the state
that must be carried across a chunk boundary, and a single-chunk request carries nothing.

# ★ Second defect, deterministic: a stale position ledger takes the server down

Sequence **long → long → short → short → long** ends in HTTP 500 and process exit:

```
slot launch_slot_: task 94 | paged: request registered (7936 tokens)
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
  - the last position stored in the memory module of the context (the KV cache) for seq 0 is X = 7950
  - the tokens for seq 0 in the input batch have a starting position of Y = 0
decode: failed to initialize batch -> llama_decode ret = -1 -> "paged decode failed"
```

7950 is the **previous long request's** last position (7936 + generation). The short requests in between
never reset the context's memory-module ledger, so the next request is refused by the batch validator.

**Two-ledgers class, sixth instance in the lane, third in this file** — the paged manager's bookkeeping
and the context's memory module are separate, and only one is maintained. The two fixed earlier were on
the speculation path; this one is plain serving with no drafts involved.

⚠ The abort that follows (`ggml-metal-device.m:657: GGML_ASSERT([rsets->data count] == 0)`) is a
**shutdown-path** check — its own comment says *"most likely you haven't deallocated all Metal resources
before exiting"*. The decode failure is the defect; the assert is teardown. Reporting the assert as the
crash would send someone into the Metal backend for a position-bookkeeping bug.

⇒ Any mixed-length workload hits this, which is every real chat server.

⇒ It may not be separate from the corruption: a stale position ledger surviving across requests is exactly
what would make the next request's chunk handoff compute the wrong carry. Not merged — but the fix for the
loud one gets tested against the silent one immediately.

---

# ★ FIXED: the crash. `launch_slot` cleared the mirror but not the context ledger

## The minimal reproducer is three ordinary requests
| arm | sequence | result |
|---|---|---|
| A | long, long, short, short, long | **HTTP 500** on step 5 |
| B | long, short, long | **HTTP 500** on step 3 |
| C | short, long | 200, but the long answers **wrong on its first request** |
| D | **clean** long, short, **clean** long | **HTTP 500** — independent of the corruption trigger |
| E | trig, short, trig, trig | 500 on step 3, **process ABORT** on step 4 |
| F | short, short, long | all 200, all correct — a long request **is** needed first |

Arm D is the important one: both long slots use the span-255 prompt, which is clean forever on its own.
**The crash has nothing to do with the corruption.**

```
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
  - the last position stored in the memory module of the context for sequence 0 is X = 7949
  - the tokens for sequence 0 in the input batch have a starting position of Y = 0
decode: failed to initialize batch -> llama_decode ret = -1 -> HTTP 500 "paged decode failed"
```
Then on the next request: `GGML_ASSERT(remaining_prompt > 0 && "prefill candidate with no prompt
remainder")` at `llama-paged-scheduler-impl.cpp:769` — the group believes its prompt is fully prefilled,
so the chunker gets a candidate with nothing to do and aborts the process.

## The line
```cpp
if (params_base.kv_paged) {
    slot.prompt.clear();      // mirror only  -> slot.prompt_clear();  // mem.seq_rm + mirror
}
```
`prompt_clear()` does `mem.seq_rm(id, -1, -1)` **first**, then clears the mirror. **Two-ledgers class,
sixth in the lane** — and this instance was introduced by the earlier mirror fix (`bc8274a80`) in this
same file. Closing one ledger left the other open behind it.

## Verified — fresh server per arm (fork `4c3b18144`, local only)
| arm | before | after |
|---|---|---|
| long, short, long | 500 | **200, correct** |
| long, short, long, long | ABORT | **survives** |
| span 255 × 3 | CLEAN | CLEAN — no regression |
| short × 3 | CLEAN | CLEAN — no regression |
| **span 256 × 3 (corruption razor)** | CORRUPT | **CORRUPT — unchanged** |

## ⚠ Do not ship this alone
Before the fix, `long, short, long` failed **loudly** with a 500. After it, that request returns **200
with confident garbage** on exactly the sequence where the corruption also fires. The change is correct —
the ledger must be cleared and the crash hits perfectly ordinary prompts — but on its own it converts a
visible failure into an invisible one. It ships with the corruption fixed, or behind the same guard.

## ⚠ And it weakens the leading corruption suspect
`seq_rm(id, -1, -1)` resets the **recurrent** memory on this hybrid context. The fix runs it on every
paged launch and the corruption is byte-identical. Stale recurrent state across the chunk handoff — the
leading suspect since the all-layer probe — survives only if it is not what `seq_rm` resets.

## Suspects eliminated, each with a firing control
ubatch size · prompt length · block size · block count · partial-block fill · champion kernel · `n_ctx`
padding · block allocation and freelist order · `max_blocks`/stride · `n_past` · resolved write slots ·
graph reuse · compute-buffer sizing · the context memory ledger.

**Still standing:** the final prefill chunk's span from its block-aligned start, at exactly 256.

---

# ⚠ "Request 1 is clean" is a property of a FRESH SERVER, not of the defect

Pre-fix, `short, long` corrupted the long on its **first** request, which looked like a third route with
no span-256 trigger in its history. Re-run under the fix, with a **pure** arm added (no triggering prompt
anywhere):

| arm | sequence | result |
|---|---|---|
| control | longCLEAN × 2 | FOUND, FOUND |
| **pure** | short, longCLEAN, longCLEAN | **FOUND, FOUND, FOUND** |
| orig | short, longTRIG | FOUND, **MISS** |

⚠ **The pure arm is not evidence the fix closed anything.** Pre-fix arm F was `short, short, longCLEAN` →
all FOUND: a short in front of a clean long was *already* clean. The pure arm has no pre-fix baseline of
its own and confirms something that was already true. Crediting the patch with this would have been a
"no baseline, credited anyway" error.

⇒ **Arm C was never a short-route.** It was the razor bug firing on the trigger prompt's *first* request
because the server was no longer fresh. The short removed the fresh-server condition; it poisoned nothing.

| history | prompt | outcome |
|---|---|---|
| fresh server | trigger | request 1 CLEAN, request 2+ wrong |
| **any** prior request | trigger | request 1 **already** wrong |
| any prior request | clean | clean, forever |

⇒ **Two defects, cleanly separated — now tested rather than assumed.** There is no third route.

⇒ ★ And it sharpens the open item: the razor's damage **survives a full `seq_rm` between requests**, so
its carrier is **not** anything the context memory module owns. That excludes the ledger outright and
constrains the recurrent-state theory to the case where what the chunk handoff reads is not what `seq_rm`
resets.

⚠ Unpaid: *"`seq_rm` clears the recurrent state"* is read from source, **not measured**. Every conclusion
above about what the fix rules out depends on it. The instrument is a direct read of the recurrent tensors
— same shape as `debug_seq_kv_checksum`, different memory module.

---

# ⚠ Open thread: the recurrent input is not being set during paged serving

The intra-request recurrent carry is the one handoff a between-request `seq_rm` can never test, and it
happens exactly where the damage lands. `s_copy` names the cell each sequence's recurrent state is taken
**from**, so it *is* that handoff.

**Measured (`DS4P_RSLOG`, both a clean and a triggering run, two requests each):**

| | requests | `DS4P-RS` lines | who |
|---|---|---|---|
| span 256 trigger | 2 | **1** | `hybrid` |
| span 255 clean | 2 | **1** | `hybrid` |

**One line, at graph build, on a 2-token batch. Zero across 32 prefill chunks and 32 decode steps.**
The probe is *outside* the `if (s_copy)` guard, so this is "the function is not entered", not "the tensor
was null". `llm_graph_input_rs::set_input` is instrumented in the same build and never fires either.

Three facts that cannot all be true:
1. `qwen35.cpp:155` calls `build_inp_mem_hybrid()`, so the input is constructed.
2. `llm_graph_result::set_inputs()` iterates **every** input unconditionally.
3. The probe fires once, at build.

⇒ Either the paged graph does not carry this input, or it does not reach the `set_inputs` call site at
`llama-context.cpp:1488`. **If the recurrent input is never refreshed per batch during paged serving,
that is a stale-input defect of exactly this bug's shape** — but the two readings point at different
files and a line count cannot separate them. Next: enumerate the 10 `build_rs` sites and instrument the
one the paged graph actually takes.

## ⚠ Four instrument failures on this one question
Each would have produced a confident, clean-looking negative:
1. the KV checksum printed **one layer of eight** — every positional claim was scoped to layer 3;
2. the recurrent probe went into **`llm_graph_input_rs`**, a class this model never calls — it fired zero
   times and the silence read as "identical";
3. it then sat **inside the `if (s_copy)` guard**, where "no tensor" and "never called" are
   indistinguishable;
4. the analyser script broke on a `sed` edit and printed a Python traceback — while still printing
   `=== DONE ===`. The counts above are read from the logs with `grep`, not from that script.

**Every one was caught by checking whether the instrument fired, before reading what it said.**

Probe committed default-off as fork `e4a115df7`.

---

# ★★★ ROOT CAUSE AND FIX: an attention-only guard was skipping the RECURRENT input every batch

```cpp
void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    if (inp_attn->self_k_idxs == nullptr || inp_attn->self_k_idxs->buffer == nullptr) {
        ... "static attn inputs unconsumed (all attention layers took the paged path)"
        return;                       // <-- and everything below it
    }
    ...
    if (inp_rs->s_copy) {             // <-- THE RECURRENT HANDOFF. Never reached under paging.
        for (i < n_rs) data[i] = mctx->get_recr()->s_copy(i);
    }
}
```

Under paging every attention layer takes the paged branch, so the static attention inputs are unconsumed
and the allocator leaves them without buffers. The guard is **correct about attention**. The `return` also
skipped the recurrent block, which has nothing to do with attention.

⇒ **`s_copy` was written once, at graph construction, on the 2-token warmup batch, and never refreshed
for any real batch.** On this arch that is **24 of 32 layers** carrying recurrent state through indices
nobody updates, for the life of the server.

## Measured before the fix was written
| | |
|---|---|
| `llm_graph_result::set_inputs` calls during one request | **31** (31 batches — the call site *is* reached) |
| `llm_graph_input_mem_hybrid` present in the input list | **all 31** (`typeid` on every input) |
| recurrent write executions | **1** (graph build only) |

Naming the inputs is what cracked it: counting said "the hybrid input is absent", and it was there all
along — the function was entered and returned early.

## Fix and verification
Scope the guard: skip only the attention writes, let the recurrent block run. Original intent preserved.

| arm | before | after |
|---|---|---|
| `ub=512` span 256 × 3 | req1 FOUND / req2+ MISS | **FOUND, FOUND, FOUND** |
| `ub=400` span 259 × 3 | req1 FOUND / req2+ MISS | **FOUND, FOUND, FOUND** |
| `ub=512` span 255 × 3 | CLEAN | CLEAN — no regression |
| long, short, long | (ledger-fix arm) | all 200, all correct |

`DS4P_RSLOG` asserts the recurrent write now fires **once per batch** — 93 serving writes for 3 requests,
was **0**. Without that assertion the four greens would not be readable. Two different trigger geometries,
different ubatches, different chunk starts. Fork `6391c5e63`, local only.

## ⇒ The span-256 rule was a SYMPTOM, not the disease
It describes exactly *when* the stale indices became visible in the output — 25 points, one token wide,
all still true — but it was never the cause. That is why fifteen suspects came back clean, why the carrier
survived `seq_rm`, why it ignored the block grid, and why it first bit at chunk 2: **chunk 2 is the first
batch that needs carried state.**

## ★ Three root causes tonight, one class
**A guard written for A silently disables B.**
1. the prompt mirror — cleared, but not the context ledger;
2. the context ledger — cleared at launch, but only the mirror half;
3. the attention guard — skipped attention, and took the recurrent input with it.

Three separate defects, one file, one night. Worth a review rule rather than three patches: **an early
return inside a function that serves two subsystems.**

## Still open
- All three fixes are **local and unpushed**.
- `-ub 128` should no longer be needed; the full 25-point grid is re-running against this build.
- Gemma4-12B running **48 of 48 layers static** under paging is unexplained and unrelated to this bug.

---

# ✅ VERIFIED: the original ubatch sweep, re-run against the fix

The grid the investigation **started** from — same 7427 prompt, same eight ubatch values, 4 requests per
point, fresh server each, protocol matched to the original. It predates every theory formed during this
session, so it cannot have been fitted to the fix.

| ub | final-chunk span | before | after |
|---|---|---|---|
| 256 | 3 | CLEAN | CLEAN |
| 320 | 67 | CLEAN | CLEAN |
| 384 | 131 | CLEAN | CLEAN |
| 432 | 131 | CLEAN | CLEAN |
| 400 | 259 | **CORRUPT** | **CLEAN** |
| 416 | 387 | **CORRUPT** | **CLEAN** |
| 448 | 259 | **CORRUPT** | **CLEAN** |
| 512 | 259 | **CORRUPT** | **CLEAN** |

**8/8 clean, 4/4 FOUND at every point.** The four that corrupted on every build now pass; the four that
were always clean are unchanged. Different ubatches, different chunk counts, different chunk-2 start
positions (400 and 416 are not block-aligned), different spans.

⚠ `ub=416` (span 387, the largest in the grid and the only one past six blocks) was named **before** the
run as the point most likely to survive the fix if it only handled the common case. It came back clean 4/4.

⚠ `ub=432` is the point whose non-monotonic result — clean *between* 416 and 448 corrupting — killed the
first theory and set the night's direction. It is still clean, for a different reason than the one
originally proposed.

Plus the `sepbracket` grid, also predating the fix: `rem 3 @ 8195` CLEAN, **`rem 259 @ 8451` CLEAN (was
TRIGGERS)**, `rem 160/192/227` CLEAN.

## Final state
| defect | status |
|---|---|
| silent cross-request corruption | **FIXED** — `6391c5e63`, root-caused, 8/8 + 5/5 pre-existing grids green |
| `long, short, long` → HTTP 500 + abort | **FIXED** — `4c3b18144`, root-caused, regressions green |
| prompt mirror outliving its pool | **FIXED** — `bc8274a80` (earlier) |

`-ub 128` is no longer required. All three fixes are **local and unpushed**, and tested on Ornith-9B on
Metal only — `llm_graph_input_mem_hybrid` is generic to every hybrid arch, so the fix is reasoned correct
beyond that and *measured* only there.

**Still open and unrelated:** Gemma4-12B runs **48 of 48 layers static** under paging while the pool and
champion markers look healthy.

---

# ⚠ RETRACTION: "Gemma4 ran zero paged layers" was an artefact of the log level

Earlier this file claimed Gemma4-12B ran **48 of 48 layers static**, that the arm was therefore 100%
vacuous, and derived an admission bar from it. Re-run at `-lv 5`:

| | |
|---|---|
| `DS4P-CONSUME` | **320** — layers *did* consume the paged context |
| `DS4P-SET` | 1008 |
| distinct static-path layers | 48 |
| capability-contract refusals | **64** |
| head dims | `n_embd_head_k = 512`, `n_embd_head_k_swa = 256` |

**`DS4P-CONSUME` requires `-lv 5`. The earlier run was `-lv 4`, so those events were invisible, not
absent** — and that invisibility was read as "zero paged layers". Same class as reading 0 from a filtered
log line (three prior incidents on file).

⇒ Two things collapse:
- **"Gemma ran zero paged layers" — WRONG.** It runs a mix.
- **The admission bar "distinct static-path layers < `n_layer`" — INVALID as stated.** A layer can log the
  static path in one batch and consume paged in another, so `48 == n_layer` does not mean nothing paged.
  A valid bar must count `DS4P-CONSUME` at `-lv 5`, not infer from a marker's absence at a log level that
  cannot print it.
- The four CLEAN verdicts from that arm are **unexplained**, not vacuous.

## And the model is broken independently of paging
| arm | "The capital of France is" | "2 + 2 =" |
|---|---|---|
| **paged** | `'01111111'` | — |
| **static** (pool asserted absent: 0 lines) | `'01111111'` | `' 1111111'` |

**Byte-identical garbage with paging off.** So this checkpoint mangles short raw completions on its own —
it answered the long needle prompt correctly in the earlier arm, which points at a missing chat template
rather than a paged defect.

⇒ **No paged defect is demonstrated for Gemma4.** The earlier line *"the paged path silently declines an
entire arch while reporting a healthy startup"* is **withdrawn** — it was built on a consume count that
could not have appeared, and the output failure it was paired with reproduces on the static path.

⇒ What *is* real: the shape contract fires **64** times on this arch (two head dims, 512 global vs 256
SWA, against a pool allocated once at 512) and degrades to the static path rather than corrupting. That is
a loud, principled refusal working as designed — the opposite of a silent decline.

---

# ★★★★ MECHANISM DERIVED: the buffer held an out-of-range row index, and only on the trigger geometry

`DS4P_RSPRE` reads `s_copy` **immediately before** `set_input` writes it. Both geometries, two requests
each, 63 probe lines apiece:

| arm | non-zero pre-write values |
|---|---|
| span 256 (**TRIGGER**) | **one**, at probe line 33: `ntok=512 n_rs=1 pre= 978561024` |
| span 255 (CLEAN) | **none** — all 63 are 0 |

Probe line 33 of 63 = 1 build line + 31 batches of request 1 + **the first batch of request 2** — exactly
where the defect bites, and the clean prompt never produces it.

## The complete chain
1. `set_input` returned early under paging ⇒ `s_copy` was **never written** for any serving batch.
2. Request 1's buffer holds 0 (fresh allocation) ⇒ **request 1 is correct**.
3. At request 2's first batch the buffer holds **978,561,024** — allocator garbage — **and only when the
   previous request's chunk geometry left it that way. That is the span-256 trigger.**
4. `s_copy` is a **row index** consumed by `get_state_rows` into a **1-row** state tensor. 978 million is
   not a row ⇒ garbage recurrent state ⇒ corrupted output from request 2 onward, permanently.
5. The fix writes 0 on every batch, so the garbage never reaches the graph.

## It accounts for every property that survived fifteen eliminated suspects
- **request 1 clean, request 2+ wrong** — the buffer is zero on first allocation and dirty afterwards.
- **sticky for the server's life** — a wrong state row poisons the carry, and nothing restores it.
- **survives `seq_rm`** — nothing re-reads that buffer; the ledger clear cannot touch it.
- **ignores the block grid** (384/448 clean, 400 not) — `s_copy` is not block-addressed.
- **first bites at chunk 2, at position == ubatch** — chunk 1 starts from a legitimately zeroed state, so
  a bad row index is harmless there; only chunk 2 onward depends on the carried row.
- **the span-256 threshold** — whether that buffer is dirty at request 2's first batch is a function of the
  previous request's allocation pattern, which is a function of its chunk geometry.

⚠ The falsifier was registered in advance: *"pre-write zero everywhere ⇒ my story is wrong and the fix
works for a reason I have not identified."* It came back **non-zero, trigger arm only, at the predicted
batch**. The written value is provably constant (`0` in all 94 recorded batches, `n_rs=1`, `head=0`), so
the fix cannot work by correcting the index — it works by overwriting garbage.

⇒ Status: **root-caused, mechanism derived, verified at 8k (8/8 + 5/5 pre-existing grids) and at 225k
(430 chunks, needle PASS), and faster than static on all three measures.**
