# Research sweep: KV-cache quantization + structural KV reduction (2026-07-28)

Companion to RESEARCH-SPEC-DECODE-LONGCTX.md. Produced under the mission mandate
(fit-first; PHASE-2 SPEED items banked not actioned).

All research complete. Compiling the full report.

---

# KV-Cache Quantization and Structural KV Reduction: Deep Survey (July 2026)

Framing used throughout: llama.cpp stores K per-token (each token's key vector is contiguous; q4_0/q8_0 blocks of 32 run along the channel dim *within* one token), applies RoPE before writing K to cache, and has no outlier handling. KIVI-class methods quantize K **per-channel** (groups run along the *token* axis for a fixed channel), which is the orthogonal axis. This one layout fact drives most of the porting notes below.

## A. KV QUANTIZATION METHODS

### 1. KIVI (ICML 2024)
1. **Mechanism:** Tuning-free asymmetric 2-bit: K quantized per-channel (grouped along token axis, group 32), V per-token; a residual window of the most recent ~128 tokens kept in fp16 for both K and V; quantize-on-window-fill streaming.
2. **Measured:** ~2.6x peak memory reduction (incl. weights), 2.35-3.47x throughput, up to 4x batch. Near-lossless LongBench/GSM8K on Llama-2/Mistral/Falcon at 2-bit; their ablation shows per-token K at 2-bit explodes PPL (this is the central K-orientation result).
3. **Inference-only:** yes, no calibration.
4. **Code:** https://github.com/jy-yuan/KIVI (custom CUDA kernels, GQA + Llama-3 support added 2025).
5. **Paper:** https://arxiv.org/abs/2402.02750
6. **llama.cpp port:** conflicts with ggml row-major per-token K blocks; needs K stored/grouped along the token dimension (effectively a transposed K cache or a new block type whose quant groups span tokens), a mixed-precision recent-window ring buffer, and FA kernel dequant changes. Medium-heavy surgery in llama-kv-cache + FA kernels.

### 2. KVQuant (NeurIPS 2024)
1. **Mechanism:** Per-channel K quantization done **pre-RoPE** (RoPE applied on-the-fly post-dequant), sensitivity-weighted k-means non-uniform datatypes (nuqX), per-vector dense-and-sparse with ~1% outliers kept fp16 in a sparse structure, first-token (attention sink) kept fp16.
2. **Measured:** <0.1 PPL degradation at 3-bit (Wikitext-2/C4, Llama/Llama-2/Llama-3/Mistral); enables Llama-7B at 1M ctx on one A100-80GB, 10M on 8 GPUs; ~1.7x kernel speedup vs fp16 matvec.
3. **Inference-only** but needs offline calibration (per-layer datatypes, outlier thresholds).
4. **Code:** https://github.com/SqueezeAILab/KVQuant
5. **Paper:** https://arxiv.org/abs/2401.18079
6. **Port:** pre-RoPE K caching is the big conflict for standard-attention models (llama.cpp caches post-RoPE K); ironically llama.cpp's **MLA path already caches a no-RoPE latent + separate RoPE part**, so the pre-RoPE idea is partially structural there. nuqX lookup tables and the sparse outlier path would be new machinery in FA kernels. Heavy.

### 3. QAQ (arXiv 2024, ICCVW 2025)
1. **Mechanism:** Quality-adaptive per-token bit allocation driven by attention scores; separate non-uniform strategies for K and V (theoretical K/V sensitivity split); dedicated outlier handling.
2. **Measured:** up to ~10x compression with "negligible" impact; evaluation thinner than peers, and its reliance on attention scores makes it incompatible with FlashAttention-style kernels (scores never materialized).
3. **Inference-only.**
4. **Code:** https://github.com/ClubieDong/KVCacheQuantization
5. **Paper:** https://arxiv.org/abs/2403.04643
6. **Port:** needing materialized attention scores disqualifies it for llama.cpp's FA path. Skip.

### 4. ZipCache (NeurIPS 2024)
1. **Mechanism:** Channel-separable token-wise quantization (decouples channel outliers from token grouping without huge per-group metadata) + saliency-adaptive bit-width per token using normalized attention scores, with a probe approximation to stay FlashAttention-compatible.
2. **Measured:** 4.98x compression with 0.38% accuracy drop (Mistral-7B GSM8K); 37.3% prefill and 56.9% decode latency reduction, 19.8% GPU memory reduction (LLaMA-3-8B @4096).
3. **Inference-only.**
4. **Code:** official repo is ThisisBillhe/ZipCache on GitHub (listed in Awesome-KV-Cache-Compression).
5. **Paper:** https://arxiv.org/abs/2405.14256
6. **Port:** mixed per-token bit-widths inside one cache tensor plus saliency probes; ggml has no variable-bit-per-row storage. Heavy; the channel-separable normalization idea alone could inform a better ggml K type.

### 5. GEAR (2024)
1. **Mechanism:** Quantize ~98% of entries to 4-bit uniform, then approximate the quantization residual with a low-rank matrix (one-step power iteration) plus a small sparse outlier matrix (~2% fp16).
2. **Measured:** near-FP16 quality at ~4-bit average; up to 2.38x throughput, 2.29x peak memory reduction vs prior alternatives.
3. **Inference-only.**
4. **Code:** https://github.com/opengear-project/GEAR
5. **Paper:** https://arxiv.org/abs/2403.05527
6. **Port:** the low-rank+sparse residual has to be rebuilt/applied every decode step (extra GEMMs outside the FA kernel); doable as graph ops but ugly, and the win over plain good 4-bit is modest for the added complexity.

### 6. CQ Coupled Quantization (NeurIPS 2024)
1. **Mechanism:** Contiguous channel groups are jointly vector-quantized with one shared codebook index (k-means/Fisher-derived centroids), exploiting inter-channel dependency; goes to 1 bit/channel.
2. **Measured:** at 1-bit preserves model quality far better than scalar methods; at 2-bit competitive-or-better vs KIVI/KVQuant on PPL and downstream tasks. Needs calibration data.
3. **Inference-only** + offline codebook calibration.
4. **Code:** no widely-adopted official repo surfaced (NeurIPS supplementary).
5. **Paper:** https://arxiv.org/abs/2405.03917
6. **Port:** codebook lookup inside attention = new dequant path; centroid tables per layer are small, but VQ dequant in FA CUDA kernels is real work. The successor CommVQ (below) is the better VQ target.

### 7. 2025-2026 successors (the current frontier)
- **RotateKV** (IJCAI 2025) - outlier-aware Walsh-Hadamard rotations + channel reordering before 2-bit quant: <0.3 PPL delta at 2-bit (Llama-2-13B), <1.7% GSM8K drop, 3.97x peak memory, 2.32x decode speedup. Inference-only w/ calibration. Paper: https://arxiv.org/abs/2501.16383
- **CommVQ** (Apple, ICML 2025) - additive vector quantization with a **RoPE-commutative codebook** so decoding fuses codebook into attention; 87.5% KV reduction (2-bit equiv) and a viable 1-bit regime. Paper: https://arxiv.org/abs/2506.18879 ; Apple page: https://machinelearning.apple.com/research/commutative-vector-quantization
- **XQuant (rematerialization, Berkeley)** - cache quantized layer-input activations X instead of K/V and rematerialize K,V on the fly (2x savings before quantization since one tensor replaces two); XQuant-CL adds cross-layer delta compression for ultra-low bits at near-FP16 accuracy. Trades FLOPs for memory. Paper: https://arxiv.org/abs/2508.10395 (distinct from the cross-layer "XQuant" at https://arxiv.org/pdf/2510.11236)
- **KVSink** - predicts attention-sink tokens beyond position 0 (via extreme-activation-outlier evolution) and preserves them; beats Preserve-First-N, improves KVQuant PPL. Paper: https://arxiv.org/abs/2508.04257
- **Kitty** (Nov 2025) - 2-bit with dynamic channel-wise precision boost + residual cache. Paper: https://arxiv.org/pdf/2511.18643
- **TurboQuant** - randomized Hadamard transform + Lloyd-Max scalar quant at 3-3.25/4-4.25 bpv; the one with a **living llama.cpp community thread** (below).
- 2026 wave (titles/IDs for the watchlist, numbers not yet verified by me): RateQuant rate-distortion mixed precision (arXiv 2605.06675), TurboAngle angle quantization (2603.27467), OScaR (2605.19660), OSCAR spectral rotation 2-bit (2605.17757), VecInfer outlier-suppressed VQ (2510.06175), AnTKV anchor-token sub-bit VQ (2506.19505), InnerQ (2602.23200), RoPE-aware bit allocation (2606.24033), CSR 1-bit sparse representation (2412.11741), MixKVQ query-aware mixed precision for long reasoning (2512.19206).

### 8. Mixed per-layer/per-head precision
- **KVTuner** (ICML 2025): offline multi-objective search over hardware-friendly per-layer (K-bits,V-bits) pairs; nearly lossless at **3.25-bit average** for Llama-3.1-8B-Instruct and **4.0-bit** for the more sensitive Qwen2.5-7B on math reasoning; +21.25% throughput vs KIVI-KV8. Explicitly built for coarse-grained, kernel-friendly quant, i.e. philosophically compatible with ggml cache types. Paper: https://arxiv.org/abs/2502.04420 ; code linked from the paper (KVTuner repo on GitHub).
- **More for Keys, Less for Values**: theory result - key projection matrices have systematically larger spectral/Frobenius norms, so K carries higher information density; K4/V2 retains up to 98.3% of K4/V4 accuracy. https://arxiv.org/abs/2502.15075
- **PM-KVQ** (2025): progressive mixed-precision for long-CoT reasoning models; documents that static low-bit KV specifically breaks long reasoning chains. https://arxiv.org/pdf/2505.18610
- **MoQAE** (ACL 2025): mixture-of-experts style bit allocation. https://aclanthology.org/2025.acl-long.531.pdf
- **llama.cpp per-head evidence** (issue #21385): bottom ~2% entropy "sink heads" dominate quant error; skipping quantization on 3 of 144 heads beat optimal bit redistribution across all others; q4_0 KV was **lossless (BLEU 1.000) on hybrid-attention models** but lossy on standard ones. Closed as not planned upstream. https://github.com/ggml-org/llama.cpp/issues/21385
- **Porting:** per-layer cache types is small plumbing in llama.cpp (per-layer tensors already exist; type is currently a global); per-head types would need KV layout changes. Per-layer K/V pairs via a KVTuner-style offline calibration is one of the cheapest real upgrades available to a fork.

### 9. FP8 and NVFP4 KV practice (engines, 2026 state)
- **vLLM (blog, Apr 2026):** E4M3 throughout, uncalibrated scale=1.0 as worst case. Reasoning tasks: 0.7-2 pt degradation (Qwen3.5-27B recovers 99% on AIME25). Long context: Llama-3.3-70B recovers 97-98% of AUC@128K; Qwen3.5-27B fully recovers AUC@1M. **MLA models (Kimi-K2.5 Flash-MLA) show small but systematic degradation uncalibrated; calibration recommended.** Throughput +14.9% (Llama-3.1-8B). https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- **TensorRT-LLM:** historically E5M2 default vs vLLM E4M3; FP8 context FMHA via --use_fp8_context_fmha. Comparison: https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079
- **NVFP4 KV (NVIDIA blog, TensorRT Model Optimizer, Blackwell-only):** FP4 values + FP8 block scales, dequant to FP8 before attention math. Qwen3-Coder-480B / Llama-3.3-70B: <1% loss overall; **RULER-64K 94.6 vs 95.5-95.6 (FP8/FP16)**, MMLU-PRO -0.7pt; 50% memory vs FP8 (4x vs BF16), up to 3x TTFT at long context. NVIDIA's own framing: FP8 KV safe everywhere; NVFP4 validate-per-task, especially reasoning. https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- **Port note:** ggml has no FP8 KV cache type today; E4M3 KV = new ggml type + FA vec/mma dequant paths. RTX Pro 6000 is Blackwell, so both FP8 and NVFP4 have hardware support waiting.

### 10. Sink-aware quantization and residual windows (cross-cutting)
Attention sinks are massive-activation outlier tokens that most heads attend to with huge weight. Two failure couplings: (a) a sink inside a quant group inflates the group's range and destroys resolution for every other element; (b) tiny relative error on a sink's K flips softmax mass allocation model-wide. Hence: KVQuant keeps token 0 fp16 (clear PPL win at 2-3 bit); KVSink/AnTKV detect sinks at arbitrary positions; KIVI's fp16 residual window protects recent tokens (whose K statistics are still forming) and doubles as the accumulation buffer for per-channel group stats. llama.cpp's "sink head" finding (#21385) is the per-head shadow of the same phenomenon. Cheap, high-yield rule: never quantize the first N tokens and the last W tokens, whatever the cache type.

## B. STRUCTURAL / ARCHITECTURAL KV REDUCTION

### 11. Palu (ICLR 2025)
1. **Mechanism:** SVD-decompose KV projection weights offline (group-head medium granularity), cache the low-rank latent, reconstruct K/V on the fly; rank search per module; matrix fusion; optional quantization of the latent.
2. **Measured:** >91.25% KV compression when combined with 3-bit quant, with up to 1.19 *lower* PPL than quant-only baselines at similar rates; RoPE forces reconstruct-then-rope for K (extra compute).
3. **Inference-only** (post-training; Fisher-based calibration; optional light finetune helps).
4. **Code:** https://github.com/shadowpa0327/Palu
5. **Paper:** https://arxiv.org/abs/2407.21118
6. **Port:** requires converted checkpoints (decomposed weights) + new attention graph; essentially a mini-MLA conversion of the GGUF + llama.cpp graph. Heavy; and see the caveat under item 15.

### 12. xKV (2025)
1. **Mechanism:** cross-layer SVD at **prefill**: group adjacent layers' KV, factor into a shared token basis + per-layer reconstruction matrices (dominant singular vectors align across layers).
2. **Measured:** up to 8.03x KV compression on Llama-3.1-8B with RULER avg 88.5% (43% better than single-layer SVD); up to 6.8x higher compression than prior inter-layer methods at +2.7% accuracy; works on MLA models (DeepSeek-Coder-V2-Lite) per the repo.
3. **Inference-only**, prefill-phase (decode-generated tokens are not compressed the same way).
4. **Code:** https://github.com/abdelfattah-lab/xKV
5. **Paper:** https://arxiv.org/abs/2503.18893 ; site: https://abdelfattah-lab.github.io/xKV/
6. **Port:** needs runtime batched SVD at prefill + reconstruction in attention. Research-grade; poor fit for llama.cpp's incremental server workloads.

### 13. MiniCache (NeurIPS 2024)
1. **Mechanism:** merge adjacent-layer KV in middle-to-deep layers: disentangle magnitude/direction, SLERP-interpolate directions, retain highly distinct token pairs unmerged.
2. **Measured:** ~1.53x from merging alone (LLaMA-2-7B ShareGPT); 5.02x combined with 4-bit quant, ~5x throughput, -41% memory, "near-lossless" on their evals. Later cross-layer papers (xKV) report merging methods degrade more on strict long-context suites.
3. **Inference-only.**
4. **Code:** linked from paper (AminJun/MiniCache lineage); primary artifact is the paper.
5. **Paper:** https://arxiv.org/abs/2405.14366
6. **Port:** cross-layer shared tensors break llama.cpp's per-layer cache assumption; retention masks add scatter/gather in attention. Medium-heavy, and quality margin at 100K+ unproven.

### 14. EigenAttention (EMNLP 2024 Findings)
1. **Mechanism:** project Q/K/V into a calibration-derived low-rank eigenbasis; attention computed in the low-rank space.
2. **Measured:** up to 40% KV reduction and up to 60% attention-latency reduction with "minimal" drop; degrades past that.
3. **Inference-only** w/ calibration.
4. **Code:** https://github.com/UtkarshSaxena1/EigenAttn
5. **Paper:** https://arxiv.org/abs/2408.05646
6. **Port:** modified projection weights + rank-reduced attention shapes; moderate, but see next item.

### 15. The 2026 verdict on low-rank vs quantization
**"Quantization Dominates Rank Reduction for KV-Cache Compression" (arXiv 2604.11501):** at matched storage budgets across 5 models (124M-14B, MHA+GQA), quantization beats rank reduction by 4-364 PPL; INT4 matches FP16 on LAMBADA (+0.23 PPL Mistral-7B) where rank-32 at identical storage collapses to 0.4% accuracy; joint K+V INT4 = 75% reduction at +0.18 PPL. Explanation: removing a dimension can discretely flip which token wins attention; quantization noise is bounded and preserves score ordering (projection damage exceeds quantization damage by 3x2^(2b) per direction under their softmax Fisher metric); basis choice is irrelevant (<0.4 PPL spread). **Practical read: post-hoc low-rank (Palu/xKV/EigenAttention) is dominated at equal bytes; low-rank earns its keep only when trained/distilled into the model (MLA-style) or fused for speed.** https://arxiv.org/abs/2604.11501

### 16. CLA - Cross-Layer Attention (NeurIPS 2024)
KV projections computed only in a subset of layers; others reuse the previous layer's KV. CLA2+MQA: 2x KV reduction with minimal PPL change on 1B/3B models, Pareto-better than plain MQA/GQA. **Design-time: needs pretraining from scratch** (no retrofit recipe). Paper: https://arxiv.org/abs/2405.12981. Not portable as software; only relevant to future model choices.

### 17. MLKV (NAACL Findings 2025)
KV heads shared across layers as well as within (goes below MQA's floor, down to 6x smaller than MQA). Demonstrated by **uptraining** Pythia-160M variants; quality degrades gracefully until extreme sharing. Training needed. Paper: https://arxiv.org/abs/2406.09297 ; code: github.com/zaydzuhri/pythia-mlkv (from paper). Small-scale evidence only.

### 18. YOCO - You Only Cache Once (MSR, NeurIPS 2024)
Decoder-decoder: self-decoder (efficient/linear attention) builds a single global KV cache; top-half cross-decoder attends to it. Inference memory improved by "orders of magnitude" at long context; near-perfect 1M-token needle retrieval; prefill can early-exit (dramatic TTFT gains). **Pretrain-from-scratch architecture** (YOCO++ 2026 adds KV residual connections: https://arxiv.org/pdf/2604.13556). Paper: https://arxiv.org/abs/2405.05254. Model-selection relevance only.

### 19. GQA re-grouping / uptraining (baseline reference)
Original recipe: convert MHA→GQA by mean-pooling KV heads per group, then uptrain ~5% of original pretraining compute to recover quality. Same recipe applies to re-grouping an existing GQA model to fewer KV heads. Training needed, cheap-ish but not free; quality recovery well-attested at scale. Paper: https://arxiv.org/abs/2305.13245

### 20. TransMLA (2025)
1. **Mechanism:** converts any GQA model to MLA with full DeepSeek-stack compatibility (orthogonal decomposition + decoupled RoPE handling).
2. **Measured:** 93% KV compression on LLaMA-2-7B, 10.6x inference speedup @8K; **~6B tokens** of finetuning restores benchmark parity; unlocks DeepSeek-specific optimizations (vLLM/SGLang MLA kernels, FP8, MTP).
3. **Training needed** (6B tokens - days on a small cluster, not feasible on a 2-GPU box for 35B+).
4. **Code:** https://github.com/fxmeng/TransMLA (converter forks exist, e.g. bet0x/transmla-converter).
5. **Paper:** https://arxiv.org/abs/2502.07864
6. **Port:** output is just an MLA model; llama.cpp already runs MLA. The cost is the finetune, not the runtime.

### 21. MHA2MLA (ACL 2025)
1. **Mechanism:** partial-RoPE removal (drop RoPE from low-contribution dims) + joint SVD of KV projections.
2. **Measured:** needs only **0.3-0.6% of pretraining tokens**; Llama2-7B: 92.19% KV reduction with 0.5% LongBench drop; composes with 4-bit KV quant.
3. **Training needed** (but 10x less data than TransMLA).
4. **Code:** https://github.com/JT-Ushio/MHA2MLA
5. **Paper:** https://arxiv.org/abs/2502.14837
6. **Port:** same as TransMLA - result is an MLA checkpoint.

### 22. X-EcoMLA (AMD, 2025)
1. **Mechanism:** SVD-initialize MLA from existing attention weights, then post-training **distillation** (optionally from a bigger teacher).
2. **Measured:** Llama3.2-1B: 6.4x KV compression retaining 100% of average score using an 8B teacher, **3.4B tokens, ~9h on 8x MI300X**.
3. **Training needed** (lightest of the three conversions).
4. **Code:** https://github.com/AMD-AIG-AIMA/AMD-Hybrid-Models ; checkpoints: https://huggingface.co/amd/X-EcoMLA-1B8B-fixed-kv64-DPO
5. **Paper:** https://arxiv.org/abs/2503.11132
6. **Port:** as above; conversion cost scales with model size - 1B-scale evidence only so far.

### 23. NVIDIA DMC - Dynamic Memory Compression (ICML 2024)
1. **Mechanism:** per-head, per-layer learned decision each step: append new KV or weighted-merge into the last entry; learned compression ratios vary by head/layer.
2. **Measured:** up to 4x KV compression preserving downstream performance on Llama-2 7B/13B/70B; beats uptrained GQA and eviction (H2O, TOVA); up to ~7x H100 throughput.
3. **Training needed:** continued pretraining on a "negligible" fraction of original data (still a real multi-GPU retrofit).
4. **Code:** no public official release found.
5. **Paper:** https://arxiv.org/abs/2403.09636
6. **Port:** would need a variable-length merge-capable cache; no artifact to port anyway.

### 24. NVIDIA DMS - Dynamic Memory Sparsification (NeurIPS 2025)
1. **Mechanism:** teach the model (in ~1K training steps) to emit keep/evict signals with **delayed eviction** (evicted tokens linger in a sliding window, implicitly merging info before removal); "inference-time hyper-scaling" = spend the freed memory on longer/more parallel reasoning.
2. **Measured:** 8x KV compression at ~1K steps; at matched memory reads, Qwen-R1 32B gains +12.0 AIME24, +8.6 GPQA, +9.7 LiveCodeBench; retrofit "within hours on a single DGX H100" for 8B.
3. **Training needed** (light, but training nonetheless).
4. **Code:** no public release found at time of writing.
5. **Paper:** https://arxiv.org/abs/2506.05345
6. **Port:** eviction decisions live in the model; llama.cpp side would need cache-slot removal mid-sequence (its cache supports token removal via the KV cell API, so runtime is less of a blocker than the missing checkpoints/training).

## C. SPECIAL QUESTIONS

### Q1. How low can KV quant go before long-context quality collapses? K vs V? 100K+ behavior?
- **The floor by method class (short/mid context):** 8-bit (int or FP8-E4M3): lossless to ~0.5pt across essentially all published evals. 4-bit with sane grouping: near-lossless (KVQuant nuq4, KIVI-4, +0.18-0.23 PPL INT4 in the 2026 dominance study). 3-bit: fine only with per-channel K + non-uniform + outlier isolation (<0.1 PPL, KVQuant), naive 3-bit fails ("severe loss at 3-bit" across methods in the DeepSeek quant study, arXiv 2505.02390). 2-bit: viable only with the full toolkit (per-channel K, rotations, sinks, residual window: KIVI, RotateKV <0.3 PPL, <1.7% GSM8K). 1-bit: only VQ/coupled-codebook methods (CQ, CommVQ, AnTKV, CSR) and losses become task-visible; treat 1-bit as research.
- **K > V, unanimously.** Evidence: KIVI's distribution analysis (K has fixed massive channel outliers, V none); llama.cpp's own PR #7412 ("K cache seems to be much more sensitive than V"); More-for-Keys theory (K projections carry larger spectral/Frobenius norms); KVTuner's importance analysis. Mechanism: K error perturbs pre-softmax logits, so one bad value can discretely re-rank which tokens get attention (then autoregression compounds it); V error enters post-softmax as a bounded, attention-weighted average. That is also *why* per-channel works for K (outliers live in fixed channels; grouping along tokens confines them) while per-token is fine for V.
- **100K+ contexts:** published direct evidence is mostly 8-bit/float: FP8 recovers 97-98% AUC@128K (Llama-3.3-70B) and 100% AUC@1M (Qwen3.5-27B) in vLLM; NVFP4 costs ~1pt at RULER-64K; KVQuant demonstrates *functionality* (passkey) at 1M-10M with 3-bit but not full RULER suites. The error-accumulation literature (MixKVQ, PM-KVQ, HQMQ) agrees on the shape: cached K/V error is write-once (static), but its effect accumulates through autoregressive feedback - "token flipping" cascades - and shows up first in math/code/long-CoT, while PPL and short retrieval look clean. PM-KVQ exists precisely because static low-bit KV breaks long reasoning chains. Direct relevance to your mission: this is the paper-world twin of your measured loop-rate-vs-bit-width axis - PPL misses it, task-level greedy reproducibility catches it. Extra caution stacking KV quant on YaRN-extrapolated context (already off-distribution); no paper covers that combination - your NIAH harness is the arbiter.

### Q2. llama.cpp q4_0/q8_0 KV vs KIVI-style, and community findings
No formal paper benchmarks llama.cpp's cache types against KIVI head-to-head. But the picture assembles cleanly:
- **llama.cpp q4_0 on BOTH K and V collapses; K is the culprit.** Discussion #23470 (Qwen2.5-7B, KLD): q8/q8 KLD 0.0018 (98.0% top-token match); K=q8+V=q4 KLD 0.0048 (96.7%); **q4/q4 KLD 5.509 (11.6% top-token match) - unusable**. Community optimum reported: K=q8_0 + V=q5_0 at 43.8% of bf16 cache size. https://github.com/ggml-org/llama.cpp/discussions/23470
- **Independent confirmation:** the "q4_0 cliff" benchmark (Qwen2.5-Coder-7B @8K: q8_0 output 81.6% similar to f16, q4_0 only 8.3% similar, silently, at full speed) https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark ; JohannesGaessler's original PR #7412 measurements (K much more sensitive than V; V=q4_0 costs less than going q6_K on weights) https://github.com/ggml-org/llama.cpp/pull/7412 ; per-head issue #21385 (+8% quality from entropy-based allocation; 3 sink heads dominate; q4_0 lossless on hybrid-attention models) https://github.com/ggml-org/llama.cpp/issues/21385
- **Why the gap vs KIVI at the same nominal bits:** llama.cpp q4_0 = round-to-nearest, symmetric-ish, blocks of 32 **within one token's channels** for K - exactly the orientation KIVI's ablation shows exploding at low bits (K channel outliers land inside every group). At 4-bit the orientation penalty is partially absorbed for many models (hence Q4 KV PPL deltas of -0.7% Gemma-3 / +2.8% Llama-3.1 / +3.0% DeepSeek reported in the per-head issue's references); at 2-3 bit it is fatal. KIVI/KVQuant at 2-3 bit beat llama.cpp q4_0's *quality* while using fewer bits, because grouping axis + outliers + sink/window handling dominate at low bit-width.
- **TurboQuant thread = the live porting playbook** (discussion #20969): rotation+Lloyd-Max at 3-3.25 bpv, community CUDA (RTX 5090, FA support, 98.8% of q8_0 prefill), Metal, and Vulkan forks; PPL within +/-2% of baseline through 32K on **Qwen3.5-35B** (your model family); ggml-specific traps documented (block 32 beats paper's 128 for FA; column-major can silently transpose rotation matrices; per-token dequant overhead at 110K ctx). https://github.com/ggml-org/llama.cpp/discussions/20969

### Q3. MLA models: does quantizing the latent hurt more?
Yes, moderately - the evidence is consistent:
- The MLA latent is already a learned dense compression (~4-14x vs equivalent MHA); it has **less redundancy to absorb rounding error**, and the concatenated structure (compressed no-RoPE part + RoPE part; asymmetric K=192/V=128 head dims in DeepSeek) is numerically heterogeneous.
- **vLLM production data:** GQA models are fine with uncalibrated FP8 KV; **Kimi-K2.5 (MLA) showed a consistent systematic downward shift** until calibrated. SnapMLA (https://arxiv.org/abs/2602.10718) keeps the **RoPE part in high precision** and uses per-token FP8 for the rest to reach near-parity (1.91x throughput).
- **Community-scale data:** DeepSeek +3.0% PPL at Q4 KV vs Gemma-3 -0.7% / Llama-3.1 +2.8% (per-head issue references); the 3-bit wall is sharper for DeepSeek-family (2505.02390).
- **llama.cpp specifics:** quantized cache on the newest MLA-family arch was outright broken until PR #25202 (Hadamard rotation not applied to compressed caches → garbage with -ctk/-ctv q4_1; merged 2026-07-07) - if you test quantized KV on DSA/MLA models, be on a build after that. https://github.com/ggml-org/llama.cpp/pull/25202
- **Practical rule:** GQA (Ornith/Qwen3.5): K=q8_0 is safe, push V lower. MLA (GLM/DeepSeek fleet): q8_0 both, keep the RoPE sub-tensor effectively high-precision, and demand long-context + loop-rate evidence before anything below 8-bit - the byte *savings* are also smaller in absolute terms because the latent is tiny (~576-656 dims/token/layer).

### Q4. KV offload systems (short)
- **LMCache:** open-source KV layer for vLLM/SGLang/Dynamo: prefix-hash-indexed KV blocks tiered across GPU→pinned CPU RAM→local disk→remote (Redis/Mooncake/InfiniStore/GDS), "prefill once, reuse everywhere" incl. cross-instance; ~10x TTFT improvements reported with GDS/NIXL paths. https://github.com/LMCache/LMCache ; https://docs.lmcache.ai/
- **Mooncake (Moonshot/Kimi):** KVCache-centric *disaggregated* serving: separate prefill and decode clusters plus a global KV pool built from underutilized DRAM/SSD/RDMA across the fleet; KV-aware scheduler trades storage for compute; +75% requests in production (up to +115% on A800); FAST'25 best paper. https://arxiv.org/abs/2407.00079 ; https://kvcache-ai.github.io/Mooncake/
- **NVIDIA Dynamo KVBM:** a runtime block manager tracking KV blocks across G1 (HBM) / G2 (pinned host) / G3 (local NVMe) / G4 (remote object store), with NIXL as the universal transfer layer and KV-aware routing on top; storage vendors (VAST etc.) plug in as G4. https://docs.nvidia.com/dynamo/design-docs/component-design/kvbm-design
- **vLLM CPU offload:** v0.11+ OffloadingConnector: block-granular async GPU↔pinned-CPU offload with an OffloadingManager doing bookkeeping/eviction, effectively extending prefix cache into DRAM. https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html
- **Mapping onto llama.cpp:** the concept that transfers is *tiering + hash-indexed reuse*, and llama.cpp already has the primitive endpoints: `--slot-save-path` + `/slots/{id}?action=save|restore` (whole-slot KV to disk = a manual G3), `--cache-reuse` prefix reuse (an in-VRAM G1 prefix cache), and host-buffer KV (`--no-kv-offload`, sync and slow = a degenerate G2). What's missing is block-granular async paging and a cross-request hash index; a sidecar daemon (PERCH-adjacent) orchestrating slot save/restore by prompt-prefix hash would get you a poor-man's LMCache without touching llama.cpp internals, but true G2 paging during decode would require surgery in the unified KV-cache + backend scheduler.

## D. TOP-5 BEST-FIT PICKS (fixed VRAM, llama.cpp fork, no-degradation bar, inference-only first)

1. **Ship today, zero code: K=q8_0 + V=q5_1 (or q5_0/q4_0 after your own NIAH pass)** - maintainer- and community-validated (KLD 0.005 vs 5.5 for q4/q4); ~45-55% KV VRAM back vs f16 with near-baseline fidelity; never put K below q8_0.
2. **Port TurboQuant-style rotated 3-3.25 bpv cache types into your CUDA fork** - working community CUDA/Metal/Vulkan forks to crib from, validated on Qwen3.5-35B to 32K at 98%+ of q8_0 speed; best quality-per-bit reachable inside ggml's per-token block layout (rotation fixes the K-outlier problem without transposing the cache).
3. **KVTuner-style per-layer (K,V) precision pairs + sink protection (first-N and sink-head/token fp16, KIVI-style recent fp16 window)** - offline calibration per model, small plumbing (per-layer cache types), directly targets the no-loops/no-recall-collapse bar; literature says this is the cheapest path to ~3.25-4 bit average "nearly lossless".
4. **FP8 E4M3 KV type for ggml (Blackwell-native)** - the only sub-8-bit-cost option with *production-grade 128K-1M evidence* (vLLM); halves KV vs f16 at effectively zero quality risk on GQA models; NVFP4 later as the 4x option at ~1pt long-context cost.
5. **For the MLA fleet (GLM/DeepSeek): stay q8_0 on the latent, high-precision RoPE part, build after PR #25202** - the latent is already compressed; below 8-bit is where MLA-specific degradation starts, and the absolute VRAM win is small anyway. (Structural conversions - TransMLA/MHA2MLA/X-EcoMLA - and DMS are the strongest *training-required* options if a finetune budget ever appears; MHA2MLA's 0.3-0.6% tokens is the cheapest of those.)

Sources: key URLs are inline per entry. Primary community threads: [llama.cpp #23470](https://github.com/ggml-org/llama.cpp/discussions/23470), [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969), [#21385](https://github.com/ggml-org/llama.cpp/issues/21385), [PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412), [PR #25202](https://github.com/ggml-org/llama.cpp/pull/25202); engine practice: [vLLM FP8 KV blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html), [NVIDIA NVFP4 KV blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/), [SqueezeBits vLLM-vs-TRT-LLM KV quant](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079), [q4_0 cliff benchmark](https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark); papers: [KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079), [QAQ](https://arxiv.org/abs/2403.04643), [ZipCache](https://arxiv.org/abs/2405.14256), [GEAR](https://arxiv.org/abs/2403.05527), [CQ](https://arxiv.org/abs/2405.03917), [RotateKV](https://arxiv.org/abs/2501.16383), [CommVQ](https://arxiv.org/abs/2506.18879), [XQuant](https://arxiv.org/abs/2508.10395), [KVSink](https://arxiv.org/abs/2508.04257), [Kitty](https://arxiv.org/pdf/2511.18643), [KVTuner](https://arxiv.org/abs/2502.04420), [More-for-Keys](https://arxiv.org/abs/2502.15075), [PM-KVQ](https://arxiv.org/pdf/2505.18610), [quant-dominates-rank](https://arxiv.org/abs/2604.11501), [Palu](https://arxiv.org/abs/2407.21118), [xKV](https://arxiv.org/abs/2503.18893), [MiniCache](https://arxiv.org/abs/2405.14366), [EigenAttention](https://arxiv.org/abs/2408.05646), [CLA](https://arxiv.org/abs/2405.12981), [MLKV](https://arxiv.org/abs/2406.09297), [YOCO](https://arxiv.org/abs/2405.05254), [GQA](https://arxiv.org/abs/2305.13245), [TransMLA](https://arxiv.org/abs/2502.07864), [MHA2MLA](https://arxiv.org/abs/2502.14837), [X-EcoMLA](https://arxiv.org/abs/2503.11132), [DMC](https://arxiv.org/abs/2403.09636), [DMS](https://arxiv.org/abs/2506.05345), [SnapMLA](https://arxiv.org/abs/2602.10718), [Mooncake](https://arxiv.org/abs/2407.00079), [reasoning-quant study](https://arxiv.org/abs/2504.04823), [DeepSeek quant drop](https://arxiv.org/pdf/2505.02390); offload: [LMCache](https://github.com/LMCache/LMCache), [Dynamo KVBM](https://docs.nvidia.com/dynamo/design-docs/component-design/kvbm-design), [vLLM offloading connector](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html).
---

# Addendum (2026-07-30): the FA/KV-quant support guard, and why `-fa on` disables it

Source-verified in `llama.cpp-idxfilter` (= `fleet`). Resolves the quarantined claim
"`GGML_CUDA_FA_ALL_QUANTS` off → 25× CPU fallback": **refuted for the default path**, real but
narrow for ours.

## What our build actually compiles

`build/CMakeCache.txt:656` → `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF` (upstream default,
`ggml/CMakeLists.txt:208`). With it off, `ggml_cuda_fattn_kv_type_supported()`
(`ggml/src/ggml-cuda/fattn.cu:338`) accepts only:

| accepted | rejected when OFF |
|---|---|
| F32, F16, BF16, **Q4_0**, **Q8_0** | **Q4_1, Q5_0, Q5_1** |

Plus a second restriction at `fattn.cu:450` — **`K->type` must equal `V->type`**:

```c
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) { return BEST_FATTN_KERNEL_NONE; }
#endif
```

Watch the switch fallthrough at `:344-352`: Q4_1/Q5_0/Q5_1 `return false` only because of the
`#ifndef`; with FA_ALL_QUANTS **on** they fall through to `return true`. Easy to misread.

## Why unsupported would be catastrophic — and why it normally isn't

`BEST_FATTN_KERNEL_NONE` → `ggml_cuda_flash_attn_ext_supported()` (`fattn.cu:596`) →
`ggml_backend_cuda_supports_op()` (`ggml-cuda.cu:5227`) returns **false**. The scheduler then
has to place that attention node somewhere else, i.e. **CPU**. That is the cliff the claim
describes.

It normally does not happen, because `resolve_fused_ops()`
(`src/llama-context.cpp:604-657`) pre-empts it. The probe is **sound**: it calls
`graph_reserve()` to build the *real* graph with the *real* KV cache types, then for every
fused FA node compares the device the scheduler actually assigned
(`ggml_backend_sched_get_tensor_backend`) against the device that layer's weights live on
(`model.dev_layer(node.il)`). On mismatch it logs

```
layer N is assigned to device X but Flash Attention is assigned to device Y
        (usually due to missing support)
```

and sets FA **disabled** — degrading to non-FA GPU attention, not to CPU.

## The catch: the guard only runs under `-fa auto`

```c
if (cparams.auto_fa) { resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn); ... }
```

`cparams.auto_fa` is true only for `LLAMA_FLASH_ATTN_TYPE_AUTO` (`llama-context.cpp:282`).
**`-fa on` sets ENABLED, so the probe never runs and the guard is gone.** Counter-intuitive:
explicitly enabling FA *removes* a safety net that `auto` provides. `--split-mode tensor`
also forces AUTO→ENABLED (`:3651-3654`), skipping the probe the same way.

The guards at `:3661-3686` do **not** cover this — they check block-size divisibility
(`n_embd_head_k % blck_size`) and "V quantization requires flash_attn", never kernel
availability.

Upstream flags its own blind spot in a `TODO` at `:3631`: `model.dev_layer()` is "still wrong
for cases like `--no-kv-offload`".

## Where this leaves our config

`tools/serve_glm52.sh` uses **`-fa on`** with **`-ctk q4_0 -ctv q4_0`**.

- **No current problem.** Q4_0 is in the compiled set and K type == V type, so both
  restrictions pass. Nothing is falling back to CPU today.
- **But the net is off.** Any future KV-type experiment under `-fa on` — `q5_0`, or a mixed
  `-ctk q8_0 -ctv q4_0` — gets no probe and no warning. Under `-fa auto` the same mistake
  produces a loud warning and clean degradation.
- Practical rule for KV-quant work: **experiment under `-fa auto`**, pin `-fa on` only once a
  combination is known-good. Cheap insurance for exactly the context-ceiling sweeps this doc
  is about.
- Note also that GLM_DSA's main attention is `GGML_OP_FLASH_ATTN_EXT_DSA`, which **aborts
  loudly** on an unsupported configuration (`ggml-cuda.cu:2374-2377`) rather than falling back
  silently — a different and safer failure mode than plain FA.

---

## Runtime witness (2026-07-30): `-fa on` turns a loud startup failure into a running server

The section above was **source-only**. Now measured. **Vehicle: Qwen3-4B Q8_0, Mac Metal,
`llama.cpp-dspark-metal` @ b874d7414 — not CUDA, not GLM.** The probe code
(`src/llama-context.cpp`) is backend-agnostic; the *set* of supported KV types is not.

Harness `tools/prompt-cache/fa_probe_ab.sh`, six arms. Witness: presence/absence of
`resolve_fused_ops` Flash-Attention output, plus whether the server came up.

| arm | KV types | `-fa` | came up | FA resolution line |
|---|---|---|---|---|
| auto_f16 | f16/f16 | auto | yes | `Flash Attention enabled` |
| on_f16 | f16/f16 | on | yes | **none — probe never ran** |
| auto_q80kv | q8_0/q8_0 | auto | yes | `Flash Attention enabled` |
| on_q80kv | q8_0/q8_0 | on | yes | **none** |
| **auto_mixed_kv** | **q8_0/q4_0** | **auto** | **NO** | `Flash Attention not supported, set to disabled` |
| **on_mixed_kv** | **q8_0/q4_0** | **on** | **yes** | **none** |

### 1. The probe runs only under `auto` — confirmed

Every `auto` arm logs a Flash-Attention resolution. Every `on` arm has **no Flash-Attention line
at all**, going straight from `flash_attn = enabled` to the Gated Delta Net probes. Exactly what
`if (cparams.auto_fa)` predicts.

### 2. The CPU-offload mechanism is real, and the probe names it

```
W resolve_fused_ops: layer 0 is assigned to device MTL0 but Flash Attention is assigned
                     to device CPU (usually due to missing support)
W resolve_fused_ops: Flash Attention not supported, set to disabled
```

**"Flash Attention is assigned to device CPU"** — the scheduler really did place the attention
node on the CPU backend, and the probe caught it. The reconstruction in the section above was
correct.

### 3. ★ The footgun is worse than "the guard is skipped"

Under `-fa auto` with the bad combo, the probe disables FA, and then the
`V cache quantization requires flash_attn` guard fires:

```
E llama_init_from_model: failed to initialize the context:
      quantized V cache was requested, but this requires Flash Attention
```

**The server refuses to start. Loud and safe.**

Under `-fa on` with the *same* combo, the server **comes up** — no probe, FA forced on, in a
configuration `auto` would have rejected outright.

So `-fa on` does not merely remove a warning. **It converts a startup failure into a silently
running server.** That is the practical case for sweeping KV-quant types under `-fa auto`.

### Boundary of this result

- Proven: the probe is `auto`-only; the CPU-assignment detection works; `on` starts a server that
  `auto` refuses.
- **Not** proven: which combos **CUDA** compiled. `q8_0/q4_0` is unsupported *on Metal* here;
  on CUDA with `GGML_CUDA_FA_ALL_QUANTS=OFF` the `K->type != V->type` rule at `fattn.cu:450`
  predicts the same outcome, but that is an expectation, not a measurement.
- Our live `-ctk q4_0 -ctv q4_0` (K == V, both in the compiled set) remains unaffected.
