# Paged KV: two correctness defects past ~50k tokens

**Status: OPEN, blocking, NOT champion-specific, predates the 2026-08-06/07 kernel work.**

Reproduced on the **default paged path with `DS4P_METAL_CHAMP` unset**, at `--kv-block-size 16`,
Ornith-1.0-9B-1M IQ2_M, `-c 73728 -np 1 -b 2048 -ub 2048 -cram 0`, `cache_prompt: false`,
`temperature 0`, `seed 1`. Each length is `toks[:N]` from one token list with the length **asserted**
in the harness, not labelled.

## The two failures

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
   on garbage. One static column re-interpreted an entire sweep.
