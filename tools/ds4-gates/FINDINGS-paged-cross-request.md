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
