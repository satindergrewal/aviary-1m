# M7 — tiled paged prefill (the ~20× gap)

**Status:** design, 2026-08-04. Ledger entry M7 (marginal-ledger memory). Tips at writing:
ds4-ports `cb3246f9`.

## The measured gap

The paged decode kernel handles prefill by putting the query axis on `grid.z`
(`pagedattn.cu:383`): one block per (head, seq, query-token), each block scanning that
query's whole visible context. Correct, and 25–45× faster than the reference kernel it
replaced (`db3afe58`), but the work is **O(n²) dot products with no data reuse**: at 8K
prefill it measures ~0.68 ms/token against ~0.025 ms/token for a real flash-attention
tile pipeline — the ~20× number in the ledger.

The cost is not the arithmetic, it is the **memory traffic**: every query block re-reads
the entire K/V span from HBM. A tile pipeline reads each KV tile **once per query tile**
instead of once per query token.

## Two candidate implementations

### A. Two-level tiling inside `ggml_paged_attn` (native)

Grid becomes `(head, seq, query_tile)` with `Q_TILE` queries per block (32 or 64). Each
block:

1. loads its `Q_TILE × head_dim` queries into registers/shared once;
2. walks the block table in `KV_TILE` chunks (`KV_TILE` = `block_size` or a multiple),
   staging each tile in shared memory **once**;
3. computes the `Q_TILE × KV_TILE` score tile, applies the analytic band
   (`visibility_window`) and the banded rel-bias, and folds it into per-query online
   softmax state (m, l, acc) held in registers;
4. writes each query's output when its walk ends.

Traffic drops by `Q_TILE×`. Keeps the paged block table as the only KV addressing scheme,
so nothing else in the engine changes.
**Cost:** a real attention kernel — tile shapes, register pressure, the band edge cases,
GQA head mapping, and a correctness gate against the existing kernel.

### B. Gather-and-delegate to the banded-FA path

At prefill, gather the sequence's blocks into a contiguous KV view and call the existing
`FLASH_ATTN_EXT_BANDED` kernel (already CUDA-complete, arcs 1–3 closed, `8f5ffd8d` +
`0ec96d43`), then continue decode on the paged kernel.

**Cost:** a gather (copy) of the whole prefix per prefill — O(n) bytes, but it buys a
kernel that is already tuned and already gated. For chunked prefill the gather is
incremental (only the new chunk's blocks are ever written; the prefix is already
contiguous if we keep the gathered buffer alive for the request's prefill phase).
**Risk:** a second copy of the prefix KV during prefill (memory), and the two paths must
agree bit-for-bit at the handoff.

## Decision rule (measure, don't argue)

The deciding number is **prefill share of end-to-end time at the target working point**
(agent cold-start at 100K+ on a big model). If prefill is <10% of the request, neither
option is worth the risk and M7 stays parked with this doc as the record. Measure first:

```
# prefill-share probe (box, one card):
#   time a 100K-token prefill + 256-token decode, paged vs static-banded serve,
#   report prompt_ms / (prompt_ms + predicted_ms) for both.
```

If prefill share is large, **prefer B first**: it reuses a kernel that is already
correctness-gated on both backends, and its failure mode (extra memory) is visible and
bounded, where A's failure mode (a subtly wrong band edge in a new kernel) is the class
of bug this lane spent the night proving is expensive to find. Ship A only if B's gather
cost measures worse than the ~20× it removes.

## Gate (either option)

1. `test-paged-banded` + `test-paged-kv` green.
2. Prefill-equivalence: same prompt, tiled vs current kernel, **bit-identical logits** at
   the first sampled position (the fork-residual work built the instrument for this:
   `llama_paged_debug_seq_kv_checksum`, `a84acc8c`).
3. Throughput: ≥5× prefill tokens/s at 8K, no decode regression, measured on the box.
4. The P2-8 arm suite (`p28_cuda_regate.sh`) still 3/3 — prefill changes touch the same
   scheduler batches that preemption rides on.


## Route correction after reading the wiring (2026-08-04 08:1x, tip `483ffb05`)

Phase 1 (split-K decode) shipped and measured 6.1× (220.7 → 36.2 ms/token at 22K,
`results/m7-prefill-share-probe-20260804.txt`). For the prefill half, **option B costs more
than it looked**:

- The paged KV tensor is `[head_dim, block_size, 2*n_heads_kv, n_blocks]` — K heads and V
  heads interleaved on axis 2, blocks on axis 3 (`llama-kv-cache-paged.cpp:81`).
- `ggml_flash_attn_ext_banded` wants ordinary `k`/`v` tensors — separate, contiguous over
  the KV axis (`ggml.h:2462`).
- So the gather is not a view: it is a real de-interleaving copy, and it has to happen
  **inside the graph** (the paged branch in `inkling.cpp` builds one fused node). That
  means a new ggml op (`paged_gather_kv`) plus a graph-level branch that picks
  gather+banded for prefill ubatches and the paged op for decode ubatches — op, CPU
  reference, CUDA kernel, backend dispatch, and the graph surgery in every model that
  gains a paged branch.

**Corrected route: option A, tiled inside `ggml_paged_attn`.** It stays entirely within one
file (`pagedattn.cu`), needs no new op and no graph change, and the tile loop also removes
the residual 4.6× on decode (each split-K slice still walks one KV token at a time). Shape:
grid `(head, seq, query_tile)`, `Q_TILE` = 32 queries per block, KV walked in `block_size`
tiles staged once in shared memory, per-query online softmax in registers, the analytic band
and rel-bias applied on the score tile. Split-K composes with it: a long context can split
across the grid and combine with the same log-sum-exp kernel already written.

Gate unchanged (bit-identical/gated max_abs vs the current kernel, ≥5× prefill, P2-8 3/3).
