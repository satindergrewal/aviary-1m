# K-2 — MLX Steel flash-SDPA vs llama.cpp Metal FA: technique diff

Author: Fable-DS4 (analysis by a dedicated read-only agent, curated) | Date: 2026-08-03
Sources: `kernels-community/metal-flash-sdpa` @ main (= MLX "Steel" SDPA, Apache-2.0, 2102
lines, scratch copy read in full) vs `ggml/src/ggml-metal/ggml-metal.metal` @ ds4-ports
`4bea1cdc` (`kernel_flash_attn_ext_impl` prefill :6415-7023, `_vec` decode :7280-7710,
`_pad` :6242, `_blk` :6314, `_vec_reduce` :7849; host dispatch ggml-metal-ops.cpp:2578-3130).
Provenance: control-flow read with line refs both sides; NO performance numbers measured —
impact estimates are bandwidth arithmetic, not benchmarks.

## Verdict in one paragraph

llama.cpp's Metal FA deficits vs MLX Steel are ALL in the dense-f16 **prefill hot path**:
small Q tiles and threadgroup-memory round-trips for the O and P matrices, plus mask-tensor
bandwidth for causal attention. The two big ones plausibly move M3 Max long-context prefill
by **tens of percent** (bandwidth arithmetic, unmeasured). Everything llama.cpp lacks is
orthogonal to what it does BETTER than MLX's extract — quantized-KV attention in-kernel,
split-KV decode, arbitrary-mask FLOP skipping, MLA head shapes — so the techniques are
graftable without giving anything up. But the #1 item is a smem/register budget redesign of
`kernel_flash_attn_ext_impl`, not a patch: exactly the shape of work for a future dedicated
kernel-only session, not a side quest.

## Ranked: what Steel has that our Metal FA lacks (prefill, M3 Max class)

1. **32-row Q tiles → 4× KV-bandwidth amortization.** Steel processes BQ=32 queries per pass
   over K/V (instantiations :2085-2092); ours hardwires Q=8 (`OP_FLASH_ATTN_EXT_NQPSG 8`,
   ggml-metal-impl.h:109) — we stream the ENTIRE K and V from device 4× more often per query
   row. Long-context prefill FA is KV-bandwidth-bound on ~400 GB/s UMA ⇒ the single largest
   structural lever. Cost: ~4× the smem for Q staging + register pressure = the redesign.
2. **Register-resident O accumulator across the KV loop.** Steel's O tile never leaves
   registers between KV blocks (:1734, :2008, stored once :2048); ours round-trips O through
   threadgroup memory EVERY 64-token block (rescale in smem :6811-6821, load :6836, store
   :6896) — at 1M ctx ≈ 16K full-DV smem round-trips per threadgroup plus 2 barriers/block.
3. **Register-resident P feeding the PV MMA.** Steel uses the score fragments directly as the
   MMA operand (:2033) with `simd_shuffle_xor` row reductions (:926); ours touches smem three
   extra times per block for S/P (:6709 store, :6786 scalar read, :6809 write, :6854 reload).
4. **Analytic causal masking via function constant — zero mask traffic.** Steel's `do_causal`
   FC clamps the KV loop analytically (:1787-1811) + in-register diagonal masking (:1868);
   ours has no causal FC — causality is a materialized half-precision mask tensor that gets
   pre-scanned (`_blk` kernel) and re-read per block. The blk machinery recovers the FLOPs
   but not the bandwidth: at 1M KV × 512-row ubatch ≈ **~1 GiB of mask reads per layer-op**
   that Steel simply doesn't do, plus 2 extra dispatches (`_pad`+`_blk`) per FA node.
5. *(minor)* log2-domain softmax with scale folded into Q at load (:1706 vs our per-score
   multiply :6786) + `[[max_total_threads_per_threadgroup]]` occupancy attribute (absent from
   our FA kernels) — single-digit percent at best.

## Where ours is equal or BETTER (do not regress these in any graft)

- **Quantized-KV attention in-kernel** (dequant fn-pointer templates :6404, :6714-6774) —
  Steel has nothing; for our workloads (q8/q4 KV at depth) this is decisive.
- **Split-KV decode + two-pass LSE reduce** (:7396, `_vec_reduce` :7849) — 32 workgroups
  stride a 1M KV; Steel's extract has no split-K at all.
- **KV-tail pad pre-pass** (:6242) keeps the hot loop branch-free (cleaner than Steel's
  predicated tail loads).
- **Mask-block skip generalizes beyond causal** (SWA, unified-cache per-seq masks); Steel's
  analytic path covers causal only.
- **Robustness under fast-math**: `precise::tanh` softcap, -FLT_MAX/2 sentinel, S==0 guard —
  Steel's -inf init + plain tanh can NaN on fully-masked rows.
- **Head-size coverage** to dk576/dv512 incl. asymmetric MLA — Steel caps at 256, DK==DV.

## Disposition

- Feeds the **dedicated kernel-only sessions** lane (his call when to open it): the concrete
  work item is "re-tile `kernel_flash_attn_ext_impl` to BQ≥16/32 with register-resident O/P,
  gated by same-boot ABBA prefill curves at 32K/128K/512K on M3 Max + byte-identical
  (fast-math-tolerant) output checks". Items 4-5 are independent smaller patches.
- Beneficiaries: every Mac model — Ornith-35B daily driver first, V4-on-Mac revival (his
  20+ t/s bar) second.
- Credit if grafted: MLX project (Apple, MIT/Apache) via kernels-community/metal-flash-sdpa;
  per [[credit-ported-code]] the technique credit goes in doc + commit.
- NOT actioned now: Phase 1-2 of the approved lane sequence owns the box lanes; this is
  Mac-lane raw material, recorded and parked with its trigger.
