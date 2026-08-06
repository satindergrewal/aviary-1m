# Paged KV: one graded correctness defect with a single-token onset

**Status: OPEN, blocking, NOT champion-specific, predates the 2026-08-06/07 kernel work.**

Reproduced on the **default paged path with `DS4P_METAL_CHAMP` unset**, at `--kv-block-size 16`,
Ornith-1.0-9B-1M IQ2_M, `-c 73728 -np 1 -b 2048 -ub 2048 -cram 0`, `cache_prompt: false`,
`temperature 0`, `seed 1`. Each length is `toks[:N]` from one token list with the length **asserted**
in the harness, not labelled.

## The onset is a single token

Bisected with a **static reference measured in the same run** at every point, lengths asserted in the
harness, `toks[:N]` from one token list:

| N | paged vs static | paged output |
|---|---|---|
| 49,996 | matches, char for char | ` Dublin is the capital of Spain. Prague is t` |
| **49,997** | **matches** | ` is the capital of Germany. Prague is the ca` |
| **49,998** | **differs** | ` H and 和Q of/路 ） NOTE –` |
| 49,999 | differs | ` before Germany. Berlin was founded before G` (fluent) |
| 50,000 | differs | ` Germany was the - - - - ` |
| 50,008 | differs | `. - - - - - ` |

**Clean at 49,997, broken at 49,998.** Static is coherent on both sides.

### Confirmed as a LENGTH, on an unrelated prompt

The onset reproduces at the **identical token** on a second stimulus with a different seed *and* a
different vocabulary (people/verbs/CS-objects — no cities, no countries, no shared prefix):

| N | static | paged | verdict |
|---|---|---|---|
| 49,997 | ` formalised a type system. Barbara formalise` | same | **matches** |
| 49,998 | `ised a type system. Barbara formalised a typ` | ` /f ,  s, -在家之间的` | **differs** |

Two unrelated prompts, one onset. This is a property of the code, not of a prompt.

This test was pre-registered to be able to demote the finding — a clean alt result would have made
the edge "the onset *for this prompt*" and required a second correction here. It broke at the same
token instead.

The edge sits on nothing structural. 49,997 is odd — not a multiple of the block size (16), the
ubatch (2,048), or any power of two — and the path handles it perfectly. Both 49,997 and 49,998
require the same 3,125 blocks at bs=16, so it is not the cache geometry. `grep` for a matching
constant in `src/` and `tools/server/` returns nothing but an unrelated RoPE θ and a comment.

Something counts tokens and stops being right one token before 50,000.

Past the edge the corruption is **graded but not monotonic**: 49,998 is severe garbage, 49,999 is
fluent and merely wrong, 50,000 degenerates again. Severity past the onset appears data-dependent.

### Correction to an earlier reading

This document previously claimed **two** distinct defects — "A degenerate, including request 1" and
"B cross-request, request 1 correct". That split came from a comparator that only asked `r1 == r2`,
before a static column existed at every point. 49,999 is fluent-and-wrong **on request 1**, which
breaks the dividing line. One graded defect fits all the data; two defects fit only the subset
available when the split was written.

The cross-request pattern at 50,473 is most likely the same fault far enough past the onset that
request 1 still survives and later ones do not.

## The failures as originally observed

Static is the reference and is **fluent and coherent at every length below**, so neither failure is
the model or its YaRN extension.

| N | static | paged | class |
|---|---|---|---|
| 49,920 | `. Dublin is the capital of Spain. Prague is` | r1 = r2 = matches static | clean |
| 50,048 | ` is the capital of France. Dublin is the cap` | r1 = r2 = ` is the 1 - - - -` | **A: degenerate** |
| 50,176 | `. Prague is the capital of France. Prague wa` | r1 = `. - - - - - ` | **A: degenerate** |
| 50,473 | (64k gate: 4 identical static reps) | r1 matches, r2+ fluent-different | **B: cross-request** |
| 50,480 | ` was founded before Italy. Prague was founde` | r1 matches static, r2 = ` was founded before Germany. Berlin was` | **B: cross-request** |

- **A — degenerate output, first request included.** Bracketed to 128 tokens: clean at 49,920,
  degenerate at 50,048. Needs no cross-request state to explain, so it is the cheaper target.
- **B — cross-request corruption.** The first request after server start matches static exactly;
  every later one is fluent and different, identically so across repeats, regardless of intervening
  content (an intervening reversed-token prompt does not clear it).

Both are **non-monotonic in length**: clean at 65,536 (32 ubatches, 4,096 blocks) with both failures
present below it.

## Ruled out, with the test that ruled it out

Each of these varied the factor and confirmed it varied before the outcome was read.

| Candidate | Test | Result |
|---|---|---|
| Block content / stale KV | `DS4P_KV_POISON=1` fills the pool with `0xFF` (every fp16 a NaN) | request 1 still correct; failure unchanged |
| Write/read slot mapping | `DS4P_SLOT_COVER=1` asserts `slots[t] == btab[pos/bs]*bs + pos%bs` for every token | 0 mismatches, both requests |
| Block accounting | checkout log | identical across requests: `n=3155 free_before=4608`, three releases for three requests |
| Block identity / free-list order | `DS4P_FREELIST_FIFO=1`, with `DS4P_METADUMP` confirming the block IDs moved | req1 blocks `0..3154`, req2 blocks `3238+` — **disjoint** — and it still fails identically |
| Partial final block | N = 50,480 with the length asserted (remainder 0) | fails |
| Partial final ubatch | N = 49,152 (24 ubatches exact) vs 49,160 (partial), same 3,072 blocks | both clean |
| Block-count band | N = 49,152 = 3,072 blocks, inside the supposed band | clean |
| The champion | reproduces with `DS4P_METAL_CHAMP` unset at bs=16 | not champion-specific |
| Decode `nwg` path | `DS4P_CHAMP_VEC_NWG=1` | unchanged |
| The model | static reference at 49,920 / 50,048 / 50,176 / 50,480 | static fluent at all four |

**Not ruled out — the earlier control was vacuous:** graph reuse.
`llm_graph_input_attn_kv_paged::can_reuse()` returns false unconditionally, so
`LLAMA_GRAPH_REUSE_DISABLE=1` could not vary anything on this path. That test proved nothing and the
candidate is untested, not refuted.

## Probes added for this

- `DS4P_METADUMP=N` — first N ubatches: `n_tokens`, `n_seq`, `ctx_lens[0]`, `offs[0]`, `lens[0]`,
  `slots[0..3]`, `btab[0..3]`. Prints the cap it is using.
- `DS4P_SLOT_COVER=1` — the write/read invariant above, as an assertion rather than a dump.
- `DS4P_KV_POISON=1` — pre-existing; fills the pool with `0xFF`.

## Harness rules this hunt paid for

1. **Confirm the independent variable changed before interpreting the dependent one.** Two vacuous
   controls were read as refutations.
2. **Assert lengths, do not label them.** `toks[:50480]` on a 50,473-token list silently ran 50,473
   while the output said `rem=0`.
3. **Key caches on their inputs.** A fixed `LOGDIR` reused a 32k token file for a 64k run.
4. **One probe at a time, kill by PID.** Detached scripts sharing `pkill -f llama-server` destroyed
   each other's servers; one lurked for hours.
5. **A comparator needs an external reference.** `r1 == r2` reported CLEAN for two requests agreeing
   on garbage. One static column re-interpreted an entire sweep — and then overturned the
   two-defect reading above.
6. **Bisection beat reasoning twice in one session, on the same lane.** Ten mechanisms proposed and
   refuted over hours; six bisect steps in ~90 minutes produced a single-token edge. The one earlier
   diagnosis that worked — the decode kernel — also fell out of a split (prefill passes, incremental
   fails), not out of theory.

## Stated limits

- Every bisect point is **n=1 per length**. The defect was deterministic across ~30 arms (identical
  wrong text under poison, under FIFO, across disjoint block ranges), so n=1 is defensible — but the
  exact edge is where that assumption bites hardest, and it is the first thing to re-test.
- ~~The edge was found on one token list.~~ **Tested and resolved:** a second, unrelated stimulus
  breaks at the identical token. The onset is a length.
- The onset was measured at `--kv-block-size 16` on one config. Whether it moves with block size,
  `-ub`, or `-c` is untested, and *that* is now the highest-value remaining experiment: if 49,998 is
  invariant to all three, the count that matters is not any of them.
