# DS4 paged-attention Metal MMA path: design

Status as of ornith `0455983`: the **contract is measured** (gates 6/6 and 9/9), the **code
does not exist yet**. `kernel_paged_attn_f32` (ggml-metal.metal:2990) contains zero
`simdgroup_load` — it is scalar dot-product + `simd_sum` with Q-block K/V tile staging,
2,940 ms wall, 2.60x off static.

This file is the design, written down so it needs no re-derivation.

**Credit:** the architecture is read off ggml's Metal flash-attention kernel
(`ggml/src/ggml-metal/ggml-metal.metal`, `kernel_flash_attn_ext`, lines ~6900–7200) by
Georgi Gerganov and the ggml contributors. Adapted here to paged KV (block tables, per-seq
context lengths, visibility windows, relative-position bias).

## The measured contract (do not re-derive — gates in `tools/ds4-gates/sgload_probe/`)

`simdgroup_load(frag, src, pitch)` fills `frag[r][c] = src[r*pitch + c]`. Therefore:

| tensor | source layout | pitch to pass | transpose |
|---|---|---|---|
| Q | `q[tok*n_heads*D + head*D + d]` — **head-interleaved** | `n_heads * D` | no |
| K | staged tile `tk[t*D + d]` | `D` | **yes** (need `d x kt`) |
| V | staged tile `tv[t*D + d]` | `D` | no |
| O / P | threadgroup scratch | its **own** stride | no |

Load and store pitches are **independent** (measured). Passing `D` as the Q pitch is the
bug that cost fifteen arms: fragment row `f` would read query token `f*(D/(n_heads*D))`,
collapsing rows. See `invented-constraint` in memory.

## The architectural key

**The O accumulator lives in threadgroup memory, not in fragments.** This is what makes
online softmax work with MMA at all:

- Per-row rescale `O *= ms` is done by **scalar threads** on threadgroup memory. There is no
  per-row scale operation on a `simdgroup_matrix`, and none is needed.
- Fragments are **transient per iteration**: load `lo[]` from `so`, accumulate `P@V`, store
  back to `so`.
- Same for the score tile: `mqk` is stored to `ss` in threadgroup memory, softmaxed there by
  scalar threads (where masking and rel-bias are trivial), then re-loaded as the `P`
  fragment for `P@V`.

Champion reference for the rescale: `so4[j*PV4 + i] *= ms;` inside the online-softmax loop.

## Layout

Per threadgroup: `nsg` simdgroups, each owning **8 query rows** ⇒ `QR = 8*nsg` rows.
Grid x = `ceil(n_tokens_total / QR)`. Key tokens processed `C` at a time, `C = block_size`
(assert `bs % 8 == 0`).

Threadgroup memory:

| buffer | size | purpose |
|---|---|---|
| `tk` | `bs*D` half | staged K (already exists) |
| `tv` | `bs*D` half | staged V (already exists) |
| `ss` | `QR*SH` float | score tile, then P. `SH = bs` (+pad to avoid bank conflicts) |
| `so` | `QR*PV` float | O accumulator. `PV = D` (+pad) |
| `M`, `S` | `QR` float each | running row max and row sum |

For `bs=32, D=64, nsg=4` ⇒ `QR=32`: tk+tv 8 KB, ss 4 KB, so 8 KB, M/S 256 B ≈ **20.3 KB**,
inside the 32 KB limit. **Size the dispatch allocation and the kernel layout in the SAME
change** — a mismatch there already caused one crash and one invalid "ALL PASSED" (the
fallback silently ran when smem exceeded 32 KB).

## Per-block loop

1. `threadgroup_barrier` (previous iteration's readers), stage `tk`/`tv` for physical block
   `block_table[seq*max_blocks + bi]`, `threadgroup_barrier`.
2. **Q@K^T**: for each 8-key sub-tile `cc` of the block, `mqk = 0`; for `i` over `D/8`:
   `simdgroup_load(mq, q + (qbase*n_heads + head)*D + 8*i, n_heads*D)`,
   `simdgroup_load(mk, tk + cc*8*D + 8*i, D, 0, true)`,
   `simdgroup_multiply_accumulate(mqk, mq, mk, mqk)`.
   Wrap every load group in `simdgroup_barrier(mem_flags::mem_none)` before and after — the
   champion does; the current kernel does not.
   Store `mqk` to `ss + row*SH + cc*8` (own stride).
3. **Online softmax on `ss`** (scalar, per query row `j`, lanes over the `C` columns):
   apply `args.scale`; add rel bias when `0 <= rd < rel_extent`; mask `-INF` outside
   `[lo, q_pos]`; `M_new = simd_max(...)`; `ms = exp(M - M_new)`; `vs = exp(s - M_new)`;
   `S = S*ms + simd_sum(vs)`; write `vs` back into `ss` (this is `P`); rescale
   `so[j*PV + i] *= ms` for all `i`.
4. `threadgroup_barrier`.
5. **O += P@V**: load `lo[]` from `so`, for each `cc`: `simdgroup_load(vs, ss + 8*cc, SH)`,
   `simdgroup_load(mv, tv + cc*8*D + ..., D)`, `multiply_accumulate`. Store `lo[]` back.
6. After all blocks: `dst = so / (S + 1e-6)`.

## Non-negotiables carried from the scalar path

- **Tail rows CLAMP, never early-return.** A non-uniform return leaves the threadgroup's
  barriers with missing threads (deadlock/UB). Invalid rows compute a duplicate and never
  write. The MMA path has *more* barriers, so this matters more, not less.
- **Cross-seq packs** (rows in one pack belonging to different sequences) must keep using
  the unshared per-row walk — a shared staged tile would be wrong.
- Per-row trip counts may diverge inside a staged block **only** because no barrier lives in
  the token loop. The MMA path puts barriers between phases; every barrier must be
  threadgroup-uniform, derived from `tgpig` alone.

## Landing discipline

Add the MMA path as a **separate branch** behind a condition, leaving the scalar path intact
as fallback. A broken MMA path must not regress the working 12/12 / 2,940 ms kernel.

Order: wire → single-tile dump == ref **in the kernel** → **12/12 with a presence marker**
(a log line proving the MMA branch actually ran — a float arm once "ALL PASSED" while the
fallback silently ran) → WALL vs **2,945** → target **≤1,134**. No "SHIPPED" until the gap
is ≤1.0x.
