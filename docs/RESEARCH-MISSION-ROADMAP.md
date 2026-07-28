# Mission roadmap: fit big models on the box, context to the limit (2026-07-28)

Synthesis of six research sweeps (RESEARCH-SPEC-DECODE-LONGCTX, -KV-QUANT,
-KV-EVICTION-SPARSE-ATTN, -DSA-INDEXER, -LLAMACPP-KV-INVENTORY, -K3-SQUEEZE). Mission
(Satinder, verbatim): "make big models run in expected state on my resources with as much big
context window they can have to their limits." Priority: FIT first, then relentless speed.

## The three walls (what actually blocks us, in order)

1. **Weights wall** (K3-class): 1TB+ can never statically fit 320GB fast memory. Breaks only
   with per-expert NVMe paging. THE endgame problem.
2. **Compute-buffer wall** (GLM-class at long ctx): the worst-case attention graph, not KV,
   caps context. Scales n_ctx x n_ubatch. Named for the first time by the inventory sweep;
   our own 8K->64K ub-lever win two nights ago was this wall. Only `-amb` (ik) caps it directly.
3. **KV wall** (dense/GQA models at long ctx): matters for Qwen3.6-27B-class; largely absent on
   MLA (GLM) and hybrid-SSM (K3) where KV is already tiny.

## Ranked lanes (fit-value x reachability; each maps to an existing capability)

### TIER 1 - reachable now, highest mission-fit

**L1. Flip production KV to K=q8_0 (ship-today, zero code).**
Every GLM serve this week ran q4/q4 KV. K errors re-rank attention pre-softmax (KLD 5.5 vs
0.005 for K=q8+V=q4). MLA KV is so cheap (~25-47MiB/1K) that q8 costs almost nothing. Gate:
NIAH + loop-rate before "production-safe." Grok enforcing as foreman. STATUS: recommended,
Satinder's word to flip a live serve.

**L2. DSA indexer harvest on GLM (the compounding corner).**
GLM ships a top-2048 selector in hardware. llama.cpp merged the correctness path (#25407,
Jul 24 - our fleet build has it) but it runs MASK-shaped = slower than dense. ik_llama.cpp
already ships the FUSED kernels (-dsa/-dsatk/-fidx, fused top-k CUDA, #2103/#2109) ~2 weeks
ahead. Prize: 7.7-31x KV traffic / 13-31x FLOP at 128K -> order-of-magnitude context.
Inverted risk: dense-above-2048 is off-spec (model trained on top-2048). PATH: (a) A/B our
build's #25407 path vs dense at 64K on QUALITY (should be parity-or-better); (b) port ik's
fused kernel OR track mainline #25917 for the O(k) decode harvest. Opus's KV/TurboQuant lane.

**L3. `-amb` (attention-max-batch) port from ik.**
The only flag anywhere that caps the compute-buffer wall directly. Our ub-lever was a partial
version. Small port from ik, big context headroom on GLM-class. Opus's lane.

### TIER 2 - the K3 endgame (build order)

**L4. DeepSeek-V4-Flash (284B) as the first giant - MERGED upstream, proven on our hardware.**
loFT ran it at 35 t/s fully-in-VRAM on 2x Pro 6000. Zero new code. The warmup giant between
GLM and K3. Then V4-Pro (1.6T, ~400GB@2bpw) = light-paging stepping stone.

**L5. Per-expert paged serving (the critical path for K3).**
VRAM<-RAM<-NVMe, async io_uring (7.4x faster than mmap, measured), LFRU hub-protection,
per-layer slot pools, fixed addresses for CUDA-graph safety. Three open PoCs to synthesize
(llama.cpp #23324, vLLM #38256, issue #20757). The `--n-cpu-moe` evolution. Binding constraint:
our 128GB RAM (second Gen5 NVMe slot = ~28GB/s bandwidth doubler if RAM stays fixed).

**L6. DSpark/MTP as the expert-prefetch ORACLE (our lane becomes fit-critical).**
MoE-SpeQ: draft predicts the big model's experts at 90.9%, speculative lookahead -> 96-99%
cache hit. While verifying token t, draft tokens t+1..t+k say which experts to prefetch. K3
ships its own EAGLE-3-style MTP draft. DSpark stops being PHASE-2 speed; it's the I/O scheduler
that makes paging viable. Directly extends our published DSpark work.

**L7. REAP 50% expert prune + FloE two-tier experts.**
REAP: one-shot router-weighted prune, 50% near-lossless, PROVEN on Kimi-K2 itself (code exists).
FloE: resident ultra-low-bit expert copy computes if the MXFP4 copy hasn't arrived from NVMe =
never stall decode. Merges with our one-file/two-profile quant-frontier lane. Quality gate:
REAP x sub-4-bit double-quant is unmeasured; loop-rate eval mandatory.

### TIER 3 - banked, trigger-gated

- LongSpec/EAGLE-3.1/BudgetDraft depth-robust drafter training (our pipeline) - when a head
  needs long-ctx acceptance; v0.4+ material.
- SuffixDecoding/SAM ngram upgrade - agentic workloads; unclaimed in llama.cpp-land.
- FP8-E4M3 KV ggml type (Blackwell-native), KVTuner per-layer pairs - GLM-now + K3-scale.
- KVzip/KVzap query-agnostic eviction - only if KV genuinely can't fit sparsified (multi-turn
  safe, unlike SnapKV).

## PHASE-2 SPEED (park until fit solved, per Satinder)
FlashKDA fused-kernel port; MoonEP zero-copy/static-shape 2-GPU EP; PEARL async draft/verify
overlap (our ~18ms/tok sync-stall, paper-shaped); fused MXFP4 grouped-GEMM; sparse-verification
(Dustin-class) for 1M.

## The convergence (why nothing we built is a detour)
- DSpark (published) -> the expert-prefetch oracle for K3 paging (L6).
- Two-profile quant lane -> FloE resident-fallback experts (L7).
- MTP port work -> the giants' draft path AND the paging oracle.
- DSA harvest (L2) -> GLM-class becomes the solved warmup.
- Every lane points at the K3 endgame. The field built the components; the integration is ours.
