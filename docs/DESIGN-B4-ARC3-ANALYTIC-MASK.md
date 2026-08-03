# DESIGN — B4 arc-3: analytic causal/SWA band (drop the redundant kq_mask)

**Status: DESIGN, source-confirmed, correctness resolved. Implementation NOT started.**
Author: Fable-DS4 | Date: 2026-08-03 | Lane: ds4-ports. This is the durable form of the
arc-3 finding banked in memory `ds4-ports-lane.md`; a fresh session implements from here.

## What this closes
B4's banded regime still pays a **~4 B per (query-token × key-token) floor** = the F16
`kq_mask` input plane + its `ggml_cont` copy. Measured: that is **~15.3 GiB at ub2048×2M**
(the #1694/#1696 regime ladder + Grok's corrected arithmetic). Arcs 1+2 removed the
*fallback* (dense-vs-banded, multi-seq, q8); this removes the *banded floor itself*.

## The finding (source-confirmed, not inferred)
`fill_mask` (src/llama-graph.cpp:412-443) builds the F16 kq_mask as a **pure function of
positions** — a cell (p1 query, p0 key) is masked iff:
- `s0 != s1` (different sequence), OR
- `cparams.causal_attn && p0 > p1` (future), OR
- `is_masked_swa(n_swa, swa_type, p0, p1)` (outside sliding window);
otherwise the value is `use_alibi ? -|p0-p1| : 0`.

There is **no data or empty-cell dependence.** In the per-stream banded path (arc-1):
- each stream is a single sequence → `s0 == s1` always (the seq term never fires),
- positions are the contiguous monotonic tail, so
  `rel_dist = iq + (n_kv - n_q) - ik == (p1 - p0)` **exactly**,
- the kernel **already computes `rel_dist`** to gate the rel_logits lookup
  (ggml-cuda/fattn-banded.cu:120 ; ggml-cpu/ops.cpp:8681-8687).

⇒ The kq_mask re-encodes exactly the band membership the kernel already derives for free.
For inkling `use_alibi == false` (rel_logits IS the positional bias), so visible cells add
`0` — dropping the mask and adding nothing is exact.

## The one subtlety (verified, do NOT skip)
`n_swa` (LLM_KV_ATTENTION_SLIDING_WINDOW, gates `is_masked_swa`) and
`inkling_rel_extent`/`_swa` (INKLING_REL_EXTENT keys, gate rel_logits) are **separate keys
with different values** (inkling.cpp:20 vs :29-30). So the visibility band is NOT `rel_dist <
rel_extent`. The analytic mask needs its own **visibility window**:
- base/global layer: pure causal → mask iff `rel_dist < 0` (window = +inf / n_kv),
- SWA layer: mask iff `rel_dist < 0 || rel_dist >= n_swa` (window = n_swa).
`rel_logits` keeps its **own independent** `0 <= rel_dist < rel_extent` gate, unchanged.

## Implementation steps (flag-gated for a clean A/B)
1. **Op**: add a `visibility_window` (int64) to `ggml_flash_attn_ext_banded` op_params, and an
   "analytic band" mode = (mask == nullptr && window set). ggml/include/ggml.h + the op
   constructor + `ggml_flash_attn_ext_banded_supported`.
2. **Kernels** (CPU ops.cpp + CUDA fattn-banded.cu): when mask is absent and window is set,
   apply `score = -INF` where `rel_dist < 0 || rel_dist >= window`. This is a 2-line extension
   of the existing rel_dist branch. Keep the explicit-mask path intact.
3. **Caller** (src/models/inkling.cpp banded branch): behind `DS4P_ANALYTIC_BAND` (default
   OFF), when the layer's mask is pure causal/SWA (always true here), pass
   `window = is_swa ? n_swa : INT64_MAX` and **drop the `ggml_view_4d(mask) + ggml_cont`**
   entirely; else keep today's mask path.
4. **Gates** (all pre-built patterns exist):
   - test-backend-ops: add analytic-band cases that must equal the explicit-mask cases
     (same q/k/v/rel, one passes a mask tensor, one passes window+null — outputs must match).
   - banded_equiv_gate.py on the synthetic under DS4P_ANALYTIC_BAND=1 (bit-identical vs off).
   - compute-buffer A/B on the synthetic: the ~4 B/pair mask term must VANISH
     (269→~5 MiB-class at the c=16384 point; the whole 15.3 GiB floor at scale).
   - CUDA op verdict in a window (bundle with the next box window).

## Concrete op_params layout (ggml.c:5533 `ggml_flash_attn_ext_banded`, verified)
Current encoding:
- `float params[] = {scale, 0, 0}` → `op_params` bytes [0,12) (scale, unused max_bias, unused softcap)
- `memcpy(op_params + 16, &rel_extent, sizeof(int64_t))` → bytes [16,24)
- bytes [24,64) FREE (op_params is 16×int32 = 64 B).

**Arc-3 step-1 patch (mechanical):**
- constructor: add `int64_t visibility_window` param; `memcpy(result->op_params + 24,
  &visibility_window, sizeof(int64_t))`. `mask==nullptr && visibility_window>0` ⇒ analytic mode.
- kernels: read it exactly like rel_extent — `int64_t window; memcpy(&window, dst->op_params + 24,
  sizeof(window));` (CUDA launcher passes it as a kernel arg alongside rel_extent; CPU reads
  inline). Apply `-INF` where `rel_dist < 0 || rel_dist >= window` when mask absent.
- callers: the existing `ggml_flash_attn_ext_banded(...)` sites pass `visibility_window = 0`
  (mask mode, unchanged); only the inkling analytic path passes a real window.

**⚠ GROUNDING NUANCE (verified ggml-cuda/fattn-banded.cu:254) — do not miss:** the CUDA path
derives `rel_extent` from `rel->ne[0]` (the TENSOR SHAPE), NOT from op_params — even though the
constructor also writes it to op_params+16. `visibility_window` has NO backing tensor, so it
MUST travel via op_params+24 AND be added to the launcher signature (fattn-banded.cu:56 takes
rel_extent as a kernel arg; add window the same way, reading it from `dst->op_params+24` in
`ggml_cuda_flash_attn_ext_banded` at :233-254). The CPU compute fn (ggml-cpu/ops.cpp, the
`GGML_OP_FLASH_ATTN_EXT_BANDED` case — find its exact name at impl time) reads op_params
inline. **The 5 files: ggml.h (decl) · ggml.c (constructor) · fattn-banded.cu (launcher arg +
kernel band) · ops.cpp (CPU kernel band) · 2 call sites (inkling.cpp real window,
test-backend-ops pass 0).** This is ONE coherent unit — implement in a single focused pass with
fresh context, compile-check, then the 4 gates. Do NOT land it half-applied.

## Provenance
Finding: control-flow read (fill_mask, banded kernels, inkling hparams) — file:line above.
Subtlety: verified (separate KV keys). Convergence: this is K-2 finding #4
(docs/K2-METAL-FA-DIFF.md — MLX Steel does causal analytically, zero mask traffic); two
independent investigations land on the same mechanism.
