# Research sweep: KV eviction / selection / decode-time sparse attention (2026-07-28)

Third sweep artifact. Companion to RESEARCH-SPEC-DECODE-LONGCTX.md and RESEARCH-KV-QUANT.md.
Mission mandate: fit-first, quality-first, no-retraining-preferred; PHASE-2 SPEED banked.

**CONVERGENCE NOTE:** the top-2 picks (Quest, DuoAttention) are the compute-sparsity-with-
full-KV family. GLM 5.2's built-in DSA lightning indexer IS this family in hardware, and our
Sweep-2 finding is that mainline serves it MASK-shaped (pays indexer cost, harvests no O(k)
compute). Building the gather-attention path = getting Quest's win for free on the model that
ships the selector. This is the highest-mission-fit corner found across all sweeps.

All research complete. Compiling the full report now.

---

# KV-Cache Eviction / Selection / Decode-Time Sparse Attention: Deep Research Report (July 2026)

Verified against arXiv abstracts, official repos, and project pages this session. Grouped by family: (A) decode-time sparsity that RETAINS full KV (no permanent information loss), (B) permanent eviction/compression, (C) retrieval-head-aware hybrids, (D) trained methods, (E) critique/benchmark papers, (F) surveys and resources, (G) notable 2025-2026 additions, then the honest-degradation synthesis and the top-5 picks.

Constraint legend used in "Port" notes: (a) fused FA kernels expose no per-token attention scores; (b) page-level metadata / fixed positional policies are easy; (c) contiguous per-sequence ring buffer, not paged.

---

## A. Decode-time sparsity, full KV retained (no permanent loss)

**1. StreamingLLM (attention sinks)**
- Mechanism: Keep the first ~4 tokens ("attention sinks") plus a rolling recency window; discard middle KV. Positions are re-indexed inside the cache so RoPE stays in-distribution. Note: this one does permanently drop middle KV, but it is a fixed positional policy, listed here because it is the base primitive for streaming heads.
- Measured: Stable perplexity out to 4M+ tokens on Llama-2/MPT/Falcon/Pythia; up to 22.2x per-token speedup vs sliding-window-with-recomputation.
- Quality: Zero recall of anything outside sinks+window, by design. NIAH outside window = 0. It is a fluency method, not a long-context method.
- Training: Inference-only.
- Code: github.com/mit-han-lab/streaming-llm
- Paper: https://arxiv.org/abs/2309.17453 (ICLR 2024)
- Port: Trivial; llama.cpp effectively already has this (context shift / --keep, and sink support). Useful only as a guardrail or for streaming heads inside other methods.

**2. Quest (query-aware KV page selection)** - priority
- Mechanism: KV stored in fixed 16-token pages; per page keep elementwise min/max of keys. At each decode step, upper-bound each page's max possible attention score using q against the min/max metadata, select Top-K pages, run attention only on those. First two layers stay dense. Full KV kept in memory; sparsity is compute/bandwidth-only.
- Measured: Up to 7.03x self-attention speedup, 2.23x end-to-end inference latency reduction at 32K with token budget ~4096 (K=256 pages x 16).
- Quality: Near-lossless on PG19 ppl, passkey, LongBench at 4K budget in-paper. Caveats: MHA-era eval; with GQA, page scores must be aggregated across the query-head group (repo has GQA support; slight accuracy cost). The Sparse Frontier confirms token/page selection at decode is the most sparsity-tolerant family.
- Training: Inference-only.
- Code: github.com/mit-han-lab/Quest (CUDA kernels, FlashInfer-based)
- Paper: https://arxiv.org/abs/2406.10774 (ICML 2024)
- Port: Moderate and the best-matched to constraint (b): metadata is page min/max only, no attention scores needed. Contiguous ring buffer can be treated as fixed-stride logical pages. Work = small scoring kernel + top-k + a block-gather/indexed variant of the FA decode kernel.

**3. InfLLM**
- Mechanism: Training-free block memory: sliding window + initial tokens always attended; distant context stored as blocks, each summarized by a few "representative" keys; per step retrieve top-k relevant blocks (optionally CPU-offloaded, loaded on demand).
- Measured: Mistral/Llama (trained at 32K or less) extrapolate to 1,024K with long-distance dependencies intact (near-100% passkey at 1M in-paper); comparable to continually-trained long-context baselines on InfiniteBench.
- Quality: Good on retrieval-style tasks; block granularity can miss fine-grained multi-needle; slower than Quest-style due to lookup overhead.
- Training: Inference-only.
- Code: github.com/thunlp/InfLLM
- Paper: https://arxiv.org/abs/2402.04617 (NeurIPS 2024)
- Port: Moderate (representative-key metadata is constraint-(b)-friendly); the CPU-offload half is bigger engine surgery.

**4. RetrievalAttention**
- Mechanism: Offload most KV to CPU RAM; build an attention-aware ANN index over keys (handles query/key OOD); at decode retrieve the 1-3% most relevant KV per step, plus a small static GPU-resident set (sinks/recent); attention computed on the retrieved subset.
- Measured: 8B model, 128K context on a single RTX 4090 (24GB) at 0.188 s/token; accesses only 1-3% of KV; near full-attention accuracy.
- Quality: Near-lossless reported across RULER-style evals; index build adds prefill-side latency; per-context index construction is the hidden cost.
- Training: Inference-only.
- Code: github.com/microsoft/RetrievalAttention
- Paper: https://arxiv.org/abs/2409.10516
- Port: Hard: C++ ANN index per context, CPU attention path, PCIe scheduling. Conceptually compatible with llama.cpp CPU offload, but it is a subsystem, not a patch.

**5. MagicPIG (LSH sampling)** - priority
- Mechanism: Treats attention as an expectation and SAMPLES keys with LSH (SimHash) proportional to attention weight, with importance-sampling correction, instead of biased top-k. Hash tables and the sampled attention run on CPU; sinks+recent computed exactly on GPU.
- Measured: Decode throughput 1.76-4.99x vs GPU-only attention baselines across hardware; typical config K=10, L=150 touches on the order of ~2% of KV. ICLR 2025 Spotlight.
- Quality: Key selling point: on aggregation-style tasks where attention is NOT concentrated (where Quest/top-k underestimate), sampling keeps estimation error low; near-full accuracy on RULER variants at a few percent budget.
- Training: Inference-only.
- Code: github.com/Infini-AI-Lab/MagicPIG (note: CPU path currently Intel-only, AVX-512-tuned)
- Paper: https://arxiv.org/abs/2410.16179
- Port: Moderate-hard. LSH build/probe is plain C++ (llama.cpp-friendly), but ~150 hash tables per layer add memory and the design assumes CPU has spare bandwidth. On your 2x96GB GPUs, a GPU-resident top-k (Quest) likely beats CPU sampling; MagicPIG matters most if you push KV to host RAM.

**6. ShadowKV** - priority
- Mechanism: Pre-RoPE keys are strongly low-rank: store an SVD low-rank key cache on GPU, offload full values to CPU; at decode, reconstruct approximate keys, score chunk landmarks, select ~1.56% of KV (plus 0.2-0.3% static outlier chunks on GPU), gather only the needed values from CPU, exact attention on the selected set.
- Measured: >6x GPU KV memory reduction; up to 6x larger batches; throughput +3.04x (Llama-3.1-8B, 122K), +2.95x (Llama-3-8B-1M, 244K), +2.56x (GLM-4-9B-1M, 60K) on A100; accuracy maintained on RULER, LongBench, NIAH at the 1.56% sparse budget.
- Quality: Accuracy preserved in-paper because nothing is deleted (values all live on CPU); risk is the approximation in the low-rank landmark scoring, which they bound with outlier retention.
- Training: Inference-only (per-context SVD at prefill time).
- Code: github.com/ByteDance-Seed/ShadowKV
- Paper: https://arxiv.org/abs/2410.21465 (ICML 2025)
- Port: Hard: prefill-time SVD per layer, CPU value store, async CUDA gather pipeline. Big payoff for capacity (it is one of the few "no-loss" methods that actually frees VRAM), but it is a serving-system-scale change to llama.cpp's ring buffer world (constraint c).

**7. TidalDecode (position-persistent sparse decode)**
- Mechanism: Two designated "token selection" layers run full attention to find the current top-k token positions; all other layers reuse that same sparse position set (cross-layer spatial coherence); periodic full re-selection prevents drift.
- Measured: Up to 2.1x decode latency reduction at matched quality; budget ~4096 tokens.
- Quality: Closely matches full attention on NIAH/LongBench in-paper; relies on cross-layer overlap assumption, which degrades on some architectures.
- Training: Inference-only.
- Code: github.com/DerrickYLJ/TidalDecode
- Paper: https://arxiv.org/abs/2410.05076 (ICLR 2025)
- Port: Moderate: needs attention scores only in 2 layers, which can be computed with a non-fused or score-emitting kernel just for those layers (bounded violation of constraint (a)); all other layers use an indexed FA kernel.

**8. SparQ Attention**
- Mechanism: Bandwidth reducer: (1) fetch only the top-r components (by |q|) of K to approximate scores, (2) top-k select keys, (3) fetch full K,V for selected, (4) compensate the dropped mass with a running mean value vector.
- Measured: Up to 8x reduction in attention data transfer with minimal accuracy loss (Llama-2/3, Mistral, Gemma, Pythia across QA/summarization/LM tasks).
- Quality: One of the more honestly evaluated bandwidth methods (they test repetition and hard tasks); GQA weakens it (shared KV heads blur per-head |q| structure).
- Training: Inference-only.
- Code: github.com/graphcore-research/llm-inference-research (research code; see also graphcore-research.github.io/posts/sparq/)
- Paper: https://arxiv.org/abs/2312.04985 (ICML 2024)
- Port: Moderate-hard: wants K stored twice (row- and dimension-sliced) for efficient partial reads; on 96GB GPUs where KV sits in HBM the win is smaller than for CPU-offload scenarios.

**9. Loki (low-rank keys)**
- Mechanism: Offline PCA on keys (calibration data); at decode compute approximate attention scores in the low-d PCA space (effective key rank ~80 of 128 captures ~90% variance), top-k select (12.5-25% of tokens), then exact attention on selected.
- Measured: Attention-op speedups reported up to ~40-45% (config-dependent; abstract does not headline a single number); maintains model quality better than H2O/exact-top-k baselines at equal budget.
- Quality: Degradation small at 25% budget on LM-harness tasks; PCA transforms generalize across calibration sets.
- Training: Inference-only (one-time PCA calibration artifact per model).
- Code: github.com/hpcgroup/loki
- Paper: https://arxiv.org/abs/2406.02542 (NeurIPS 2024)
- Port: Moderate: store extra projected-key tensor, small GEMM + top-k + gather; no attention-score exposure needed (bypasses constraint (a)).

**10. LServe (unified sparse attention serving)** - priority
- Mechanism: The integration blueprint: (i) static sparsity: convert half of all attention heads to DuoAttention-style streaming heads (sinks+recent), applied in BOTH prefill and decode, fused block-wise into the attention kernel; (ii) dynamic sparsity: Quest-style query-centric top-k KV page selection for the remaining heads at decode, with hierarchical paging (physical pages summarized into logical pages) and a reusable page-selection cache across steps. Claims a constant token budget suffices irrespective of context length.
- Measured: Prefill up to 2.9x and decode 1.3-2.1x faster than vLLM on average, with long-context accuracy maintained (LongBench/RULER/NIAH); composes with W4A8KV4 quantization (QServe).
- Quality: Accuracy preserved at ~4K page budget in-paper; static+dynamic gains are multiplicative.
- Training: Head classification reuses DuoAttention's lightweight offline optimization; otherwise inference-only.
- Code: github.com/mit-han-lab/omniserve (MLSys 2025)
- Paper: https://arxiv.org/abs/2502.14866
- Port: The whole system is hard, but it is the single best C++/CUDA reference design for your engine: every component is constraint-(b) friendly (page metadata, static per-head policies, block-sparse FA kernels), none needs per-token score readback.

**11. XAttention (prefill block sparsity)**
- Mechanism: Training-free block-sparse PREFILL: score each attention block by the sum of its strided antidiagonal elements (cheap proxy that catches both vertical and slash patterns), threshold-select blocks, run block-sparse FA.
- Measured: Up to 13.5x attention speedup in prefill; accuracy on par with full attention on RULER, LongBench, VideoMME, VBench.
- Quality: Near-parity in-paper; it does not address decode.
- Training: Inference-only.
- Code: github.com/mit-han-lab/x-attention
- Paper: https://arxiv.org/abs/2503.16428 (ICML 2025)
- Port: Moderate: a strided-sampling score pass + block-sparse prefill FA kernel. Directly attacks your 1M-token prefill wall without touching decode quality or KV contents.

**12. SeerAttention**
- Mechanism: Small learnable per-head gates (pooled Q,K through linear layers) predict block-level sparsity; gates are trained by self-distillation against the frozen model's own pooled attention maps (model weights untouched).
- Measured: 90% sparsity at 32K with minimal perplexity loss; 5.67x kernel speedup vs FlashAttention-2.
- Quality: Near-lossless at high sparsity in-paper; per-model gate artifact required.
- Training: Lightweight gate training per model (not inference-only in the strict sense, but weights frozen; recipe and pre-trained gates for popular models in repo).
- Code: github.com/microsoft/SeerAttention
- Paper: https://arxiv.org/abs/2410.13276
- Port: Moderate: gates are tiny MLPs (could ship as extra GGUF tensors); needs block-sparse FA kernels. The per-model training step is a workflow cost against your no-training preference.

**13. SeerAttention-R (decode/reasoning variant)**
- Mechanism: Same self-distilled gating adapted to autoregressive decode: no query pooling, gate shared across the GQA group selects KV blocks per decoded token.
- Measured: Gates trained on only 0.4B tokens; near-lossless AIME accuracy at a 4K token budget with block sizes 64/128; TileLang kernel up to 9x faster than FlashAttention-3 dense decode at 90% sparsity (H100).
- Quality: Near-lossless on AIME/MATH500/GPQA for Qwen3-class models in-paper.
- Training: Lightweight gate distillation per model.
- Code: github.com/microsoft/SeerAttention (same repo)
- Paper: https://arxiv.org/abs/2506.08889
- Port: Same as SeerAttention; the most credible learned-gate decode sparsity for long generations (your loop-prone regime).

**14. InfiniteHiP (2025 addition, whole-package)**
- Mechanism: Training-free modular hierarchical token pruning (multi-stage, parallelizable) + KV offload to host + out-of-length RoPE generalization adjustments; drop-in for pretrained transformers.
- Measured: 3M tokens on a single L40s 48GB; 18.95x attention decode speedup at 1M; 7.25x end-to-end decode at 3M.
- Quality: Maintains strong long-context scores in-paper while extending beyond trained length (directly relevant to your YaRN-1M regime); independent replications thinner than for Quest/ShadowKV.
- Training: Inference-only.
- Code: github.com/DeepAuto-AI/hip-attention (Triton; SGLang integration exists)
- Paper: https://arxiv.org/abs/2502.08910
- Port: Hard (Triton to CUDA/C++ rewrite of a multi-stage system), but the closest published thing to your whole mission in one package; worth mining for its OOL-generalization tricks even if you do not port the system.

---

## B. Permanent eviction / compression (information is deleted)

**15. H2O (heavy-hitter oracle)**
- Mechanism: During decode keep a fixed budget of "heavy hitters" (tokens with highest accumulated attention scores) plus recent tokens; greedy eviction each step (dynamic submodular formulation).
- Measured: With 20% budget: up to 29x throughput vs DeepSpeed Zero-Inference/HF Accelerate/FlexGen (OPT-6.7B/30B; mostly a bigger-batch effect) and up to 1.9x lower latency; near-parity on HELM-era short-context tasks.
- Quality: Honest read in 2026: evaluated in the OPT/2K-4K era. On long-context retrieval it collapses (Quest and others show failed passkey; SCBench shows multi-turn failure; accumulated-score bias favors early tokens).
- Training: Inference-only.
- Code: github.com/FMInference/H2O
- Paper: https://arxiv.org/abs/2306.14048 (NeurIPS 2023)
- Port: Hard: needs per-step per-token attention scores = fused-FA kernel modification (constraint (a)), plus per-step eviction against the ring buffer. Low ROI; historically important only.

**16. Scissorhands**
- Mechanism: "Persistence of importance": tokens pivotal once remain pivotal; keep tokens whose attention exceeds a threshold within a history window, budgeted eviction at test time.
- Measured: Up to 5x KV memory reduction without quality loss (OPT-era evals); up to 20x combined with 4-bit quantization.
- Quality: Same era-caveat as H2O; long-context retrieval untested and expected to fail similarly.
- Training: Inference-only.
- Code: No official release (third-party reimplementations only).
- Paper: https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- Port: Hard (attention scores needed). Skip.

**17. TOVA (Token Omission Via Attention)**
- Mechanism: Frames decoder LLMs as bounded multi-state RNNs: at each step drop the token with the lowest attention score from the current query, per layer. No accumulation, extremely simple.
- Measured: Near-parity with full model using 1/8 of the cache; up to 4.8x throughput via bigger batches; up to 88% KV memory reduction.
- Quality: Better than H2O-style on their evals, but same class of failure on exact-recall long-context tasks and multi-turn (it is in the Pitfalls paper's degraded set).
- Training: Inference-only.
- Code: github.com/schwartz-lab-NLP/TOVA
- Paper: https://arxiv.org/abs/2401.06104
- Port: Hard (per-step scores; constraint (a)).

**18. SnapKV** - priority
- Mechanism: One-shot compression at end of prefill: attention from an "observation window" (last ~16-64 prompt tokens) onto the whole prompt scores prompt tokens per head; keep top-k with 1D pooling (keeps clustered neighborhoods) plus the window itself; decode then appends normally.
- Measured: 3.6x faster generation and 8.2x memory efficiency at 16K input; enables 380K-token prompts on one A100; negligible retrieval degradation at 1024-token budget in its own needle eval; consistent LongBench parity at 1-4K budgets.
- Quality: THE caveat (matters most for you): the compression is fitted to the query present at prefill end. SCBench: such methods "perform well only on the first query" and fail on follow-ups over shared context; Pitfalls (ACL 2026) shows whole instructions get ignored under multi-instruction prompts. Exact-recall (RULER multi-key) degrades well before LongBench does.
- Training: Inference-only.
- Code: github.com/FasterDecoding/SnapKV
- Paper: https://arxiv.org/abs/2404.14469 (NeurIPS 2024)
- Port: Moderate: no decode-time scores needed; one extra scoring pass (W queries x all keys per head) after standard fused prefill, then cache compaction. The ring buffer makes compaction awkward but doable (constraint (c)). Do not deploy for multi-turn agentic use without KVzip-style query-agnostic scoring.

**19. PyramidKV**
- Mechanism: SnapKV-style scoring with per-LAYER budget shaping: lower layers (attention spread wide) get big budgets, higher layers (attention funneled onto few tokens) get small ones ("pyramidal information funneling").
- Measured: Matches full-cache LongBench with 12% of KV (Llama-3-8B/70B, Mistral); in extreme 0.7% setting beats other compression methods by up to +20.5 absolute on TREC.
- Quality: The "12% = full" claim is LongBench-specific; LongBench needs few exact tokens, so it flatters eviction. Same first-query-only limitation as SnapKV.
- Training: Inference-only.
- Code: github.com/Zefan-Cai/KVCache-Factory
- Paper: https://arxiv.org/abs/2406.02069
- Port: Moderate (as SnapKV; per-layer budgets are easy since llama.cpp KV is per-layer anyway).

**20. Ada-KV**
- Mechanism: Head-wise adaptive budget allocation with a proven L1 eviction-loss bound: reallocate budget from attention-concentrated heads to dispersed heads; plugs into SnapKV/PyramidKV (Ada-SnapKV, Ada-Pyramid).
- Measured: Consistent gains over uniform allocation at equal total budget, largest on Needle-type retrieval at small budgets; adopted in NVIDIA kvpress (AdaKVPress), where the leaderboard's best combo is AdaKV + ExpectedAttention.
- Quality: Improves the frontier but does not fix the query-dependence problem of the underlying scorer.
- Training: Inference-only.
- Code: github.com/FFY0/AdaKV (NeurIPS 2025)
- Paper: https://arxiv.org/abs/2407.11550
- Port: Harder than SnapKV: ragged per-head cache sizes break the uniform per-sequence layout (constraint (c)); needs indexed/paged KV or per-head offsets plus a nonuniform-length FA kernel.

**21. NACL**
- Mechanism: One-shot eviction during encoding combining proxy-token scoring (attention from a designated proxy set, debiasing the sink bias) with a diversified RANDOM eviction component for robustness.
- Measured: Up to 5x KV reduction with >95% performance maintained; +80% (short-task) / +76% (long-task) improvement over prior eviction baselines.
- Quality: The random component is a real robustness insight (later echoed by sampling methods); still a permanent-eviction method with the same multi-turn caveat.
- Training: Inference-only.
- Code: github.com/PaddlePaddle/Research (NLP/ACL2024-NACL; PaddlePaddle-based)
- Paper: https://arxiv.org/abs/2408.03675 (ACL 2024)
- Port: Moderate; scoring pass akin to SnapKV. PaddlePaddle code means reimplementation, not porting.

**22. KVzip (query-agnostic eviction)** - priority for the multi-turn problem
- Mechanism: After prefill, run the model to RECONSTRUCT the context from its own cache (a "repeat the context" pass); importance of each KV pair = max attention it receives during reconstruction; evict lowest. Scoring is query-independent, so one compressed cache serves all future queries. Also offers a context-independent per-head variant (DuoAttention-style, no per-context overhead).
- Measured: 3-4x KV reduction and ~2x FlashAttention decode latency reduction with negligible loss across QA, retrieval, reasoning, and code tasks; models 3B-70B (Llama-3.1-8B, Qwen2.5-14B, Gemma3-12B), contexts to 170K. Compression overhead is roughly one extra forward pass over the context (chunked).
- Quality: This is the direct answer to the SCBench/second-query critique: evaluated multi-query, retains performance where SnapKV-style drops. Below ~20-25% retention, degradation appears (their curves are honest).
- Training: Inference-only.
- Code: github.com/snu-mllab/KVzip (NeurIPS 2025; also integrated in NVIDIA kvpress)
- Paper: https://arxiv.org/abs/2505.23416
- Port: Moderate-hard: the reconstruction pass needs attention-score capture (constraint (a)), but only during a one-time, per-context compression step, where you can afford an unfused/chunked scoring kernel. See KVzap below for the cheap approximation.

**23. R-KV (reasoning-decode compression)**
- Mechanism: Decode-time compression aimed at chain-of-thought bloat: rank generated-token KV by attention importance PLUS semantic redundancy (similarity between reasoning steps), evict redundant steps on the fly.
- Measured: ~100% of full-cache accuracy using only 10% of the KV cache on math reasoning (R1-distill class models), sometimes exceeding full cache; ~90% memory saving enabling large batch/throughput gains.
- Quality: Targets generated KV, complementary to prompt-side methods. Interesting to you for a second reason: it operationalizes redundancy detection of looping/repetitive generation.
- Training: Inference-only.
- Code: github.com/Zefan-Cai/R-KV (NeurIPS 2025)
- Paper: https://arxiv.org/abs/2505.24133
- Port: Hard: needs decode-time attention scores + hidden-state similarity tracking (constraint (a)).

**24. KeyDiff (attention-score-free eviction)**
- Mechanism: Evict by key GEOMETRY only: geometrically distinctive keys (low cosine similarity to an anchor/mean of keys) empirically receive high attention; evict the least distinctive. No attention scores at all, so fully compatible with fused FA.
- Measured: <0.04% LongBench gap at 8K budget (~23% KV reduction) for Llama-3.1-8B / 3.2-3B; near-baseline Math500 for R1-Distill-Llama-8B; up to 30% end-to-end latency reduction vs other eviction methods (it avoids the score-materialization they need).
- Quality: Safe zone is modest (~20-25% trim); it is an approximation whose failure modes at aggressive budgets are less mapped.
- Training: Inference-only.
- Code: No official repo found (Qualcomm AI Research authors).
- Paper: https://arxiv.org/abs/2504.15364
- Port: EASY, the single most constraint-(a)-friendly eviction method: keys are already in the cache, anchor+cosine is a tiny kernel, plus ring-buffer compaction. Good first eviction experiment for llama.cpp.

**25. RocketKV (NVIDIA, two-stage)**
- Mechanism: Stage 1: coarse permanent eviction via SnapKV. Stage 2: dynamic fine-grain top-k "hybrid sparse attention" over the survivors, approximating scores with head- and sequence-dimension reductions.
- Measured: Effective compression up to 400x, end-to-end decode speedup up to 3.7x, peak memory reduction up to 32.6% (A100), negligible loss on their long-context suite.
- Quality: Inherits SnapKV's first-query dependence in stage 1; the eye-catching 400x is the product of both stages under favorable tasks.
- Training: Inference-only.
- Code: github.com/NVlabs/RocketKV (ICML 2025)
- Paper: https://arxiv.org/abs/2502.14051
- Port: Moderate-hard (custom approximation kernels; NVlabs CUDA-adjacent code is a usable reference).

**26. NVIDIA kvpress lane: Expected Attention + KVzap (2025-2026)**
- Expected Attention (arXiv https://arxiv.org/abs/2510.00636): training-free scoring that computes expected attention to each KV pair in CLOSED FORM from the distributional statistics of future queries (activation Gaussianity), needing no attention scores, works for both prefill and decode. Constraint-(a) friendly by construction.
- KVzap (arXiv https://arxiv.org/abs/2601.07891, Jan 2026, Jegou & Jeblick): fast input-adaptive approximation of KVzip's reconstruction scoring that works in prefill and decode; 2-4x KV compression with negligible accuracy loss on Qwen3-8B/32B and Llama-3.1-8B across RULER, LongBench, AIME25; state of the art on the kvpress leaderboard.
- Code for both: github.com/NVIDIA/kvpress (20+ press implementations, HF-transformers-based, plus the public RULER-vs-compression leaderboard; best combo currently AdaKV + ExpectedAttention, with KVzap the SOTA single method).
- Port: Expected Attention is moderate (needs running query statistics per head, then closed-form scoring); KVzap moderate. kvpress itself is the best place to validate policies in Python before committing CUDA work.

---

## C. Retrieval-head-aware methods (per-head policies)

**27. Retrieval Head paper (Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality")**
- Mechanism (finding, not a method): <5% of attention heads are sparse, universal, causal "retrieval heads" implementing copy-paste from context. Masking the top-20 retrieval heads of Llama-2-7B-80K destroys NIAH (model hallucinates fluently); masking 20 random heads keeps 94.7%. Also degrades CoT that refers back to context.
- Measured: Head-detection scripts + published head lists for Llama/Mistral/Yi/Qwen.
- Training: Analysis only.
- Code: github.com/nightdessert/Retrieval_Head
- Paper: https://arxiv.org/abs/2404.15574 (ICLR 2025)
- Port: Not a method; its head lists directly seed DuoAttention/RazorAttention-style per-head policies in your engine.

**28. DuoAttention** - priority
- Mechanism: Classify each KV head as retrieval (keeps FULL KV) or streaming (keeps only sinks+recent, constant size), via a lightweight offline optimization on synthetic passkey data that learns per-head gate values (model weights frozen; hours, not training runs). Inference uses two cache pools.
- Measured: KV memory cut 2.55x (MHA) / 1.67x (GQA); decode speedup up to 2.18x / 1.50x; prefill 1.73x / 1.63x; with 8-bit quantization, Llama-3-8B reaches 3.3M tokens on one A100. Minimal degradation on LongBench/NIAH because retrieval heads are untouched.
- Quality: The honest framing: for GQA models (your fleet) the ceiling is ~1.67x memory, not 2.55x. Robust across queries (nothing query-specific is deleted from retrieval heads), so it does not share SnapKV's multi-turn failure; SCBench still ranks full-KV dynamic sparsity higher on hard multi-turn suites.
- Training: One-time per-model head identification (configs shipped for popular models; you can run their recipe for your fleet on your own GPUs).
- Code: github.com/mit-han-lab/duo-attention (ICLR 2025)
- Paper: https://arxiv.org/abs/2410.10819
- Port: Moderate: no scores needed at inference (constraint (a) fine); the work is allocating two KV pools per layer keyed by KV-head class and running two attention calls (or one kernel with per-head window masks; llama.cpp's SWA plumbing is a good starting point). Constraint (c) is the main friction.

**29. RazorAttention**
- Mechanism: Training-free per-head policy: retrieval heads (found via echo/induction probing, no optimization) keep full cache; all other heads keep sinks+recent AND fold every dropped token into a per-head "compensation token" (running mean of dropped K/V), recovering some aggregate information.
- Measured: >70% KV cache reduction with no noticeable performance impact claimed (Llama/Qwen family evals); FlashAttention-compatible.
- Quality: Later independent evals (SCBench class) place it below DuoAttention on hard retrieval; the heuristic head ID is less reliable than Duo's optimization.
- Training: Inference-only.
- Code: No official release.
- Paper: https://arxiv.org/abs/2407.15891 (ICLR 2025)
- Port: Moderate: same two-tier layout as DuoAttention, plus a trivial compensation-token accumulator; head lists can come from the Retrieval_Head repo. A cheap way to prototype the two-tier cache before committing to Duo's optimization.
- Related one-liner: HeadKV (https://arxiv.org/abs/2410.19258, ICLR 2025) allocates per-head budgets by retrieval-AND-reasoning head importance; reports strong retention at very low budgets (1.5% claims on contextual QA); same ragged-layout port cost as Ada-KV.

---

## D. Trained methods (for completeness; rank low for your no-retraining mission)

**30. Activation Beacon**
- Mechanism: Special beacon tokens progressively compress preceding chunk activations (K/V at every layer) at sampled ratios; model finetuned with compression-based auto-regression to use condensed activations.
- Measured: 2x inference acceleration and 8x KV memory reduction; trained at up to 20K length yet works at 128K; comparable quality to uncompressed baseline in their evals.
- Training: REQUIRED (that is the method).
- Code: github.com/FlagOpen/FlagEmbedding (Activation-Beacon)
- Paper: https://arxiv.org/abs/2401.03462
- Port: Not portable without finetuning your fleet. Skip.

**31. MoBA (Moonshot)**
- Mechanism: Mixture of Block Attention: context in 2048-token blocks; per-query gating (q dot mean-pooled block key) selects top-3 blocks; layers can toggle full/sparse. Deployed for Kimi long-context.
- Measured: README/paper report large attention speedups at 1M+ (tech report shows ~6.5x at 1M scale prefill vs full flash attention; README benchmarks vs naive implementation only). Quality parity in their trained models.
- Training: REQUIRED. README states explicitly: "MoBA requires continue training of existing models... not a drop-in sparse attention solution."
- Code: github.com/MoonshotAI/MoBA
- Paper: https://arxiv.org/abs/2502.13189
- Port: Not applicable inference-only; its training-free cousin is essentially Quest/InfLLM block selection, which you can have without retraining.

**32. NSA and DSA (context for 2025-2026 model releases)**
- NSA (Native Sparse Attention, DeepSeek, https://arxiv.org/abs/2502.11089, ACL 2025 best paper): trained-from-scratch hierarchical sparse attention (compressed + selected + sliding paths).
- DSA (DeepSeek Sparse Attention, in DeepSeek-V3.2, tech report https://arxiv.org/pdf/2512.02556): trained "lightning indexer" (ultra-light FP8 scorer) + top-k token selection, O(Lk) attention, ~50% lower long-context serving cost at quality parity with V3.1; SGLang day-0 support; llama.cpp mainline has an active implementation discussion (ggml-org/llama.cpp discussion #21183).
- Relevance: for models that SHIP with an indexer (DeepSeek V3.2+/V4 era), decode-time sparsity comes for free if the engine implements it; for your existing dense-attention fleet these are not retrofit options.

---

## E. Critique / honest-evaluation papers (the "second query" problem)

**33. SCBench** - the multi-turn critique you asked for
- Content: Microsoft benchmark (12 tasks, two shared-context modes: multi-turn and multi-request) evaluating long-context methods KV-cache-centrically across generation/compression/retrieval/loading.
- Key findings (quoted): "sub-O(n) memory methods suffer in multi-turn scenarios"; they "perform well only on the first query" because important KVs vary across queries; "sparse encoding with O(n) memory and sub-O(n^2) pre-filling computation perform robustly"; "dynamic sparsity yields more expressive KV caches than static patterns."
- Translation for your mission: SnapKV/PyramidKV-class eviction looks great on single-shot LongBench and quietly breaks agentic/multi-turn use; Quest/MagicPIG/LServe-class (full KV, sparse compute) stays robust.
- Paper: https://arxiv.org/abs/2412.10319 (ICLR 2025); code/data: github.com/microsoft/MInference (SCBench)

**34. The Pitfalls of KV Cache Compression (ACL 2026)**
- Content: Chen, Geh, Grover, Van den Broeck, Israel. Multi-instruction prompting study (IFEval-style) of StreamingLLM, SnapKV, TOVA, H2O, K-Norm on Llama-3.1-8B and Qwen2.5-14B: "certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored"; demonstrates system-prompt leakage as a concrete failure; shows instruction-position and eviction-bias effects; proposes eviction-policy tweaks (e.g., protecting system-prompt spans).
- Paper: https://arxiv.org/abs/2510.00231
- Relevance: direct evidence that "PPL fine, behavior broken" extends to instruction adherence, the axis your GODWIT work cares about.

**35. The Sparse Frontier (Cohere + Edinburgh, NeurIPS 2025)**
- Content: Largest controlled study of TRAINING-FREE sparse attention (6 methods, models to 72B, 128K sequences, sparsity to 0.95). Findings: isoFLOPS analysis shows for long sequences large-sparse beats small-dense; decoding tolerates much higher sparsity than prefill (token/page selection generalizes best); BUT no method is uniformly safe: moderate sparsity levels routinely degrade at least one task per configuration, so "sparse attention is not a free lunch" and per-task validation is mandatory.
- Paper: https://arxiv.org/abs/2504.17768; code: github.com/PiotrNawrot/sparse-frontier

**36. Additional honest-data points**
- "Can LLMs Maintain Fundamental Abilities under KV Cache Compression?" (https://arxiv.org/abs/2502.01941): cross-task study; arithmetic/math reasoning degrades far earlier than QA under compression; proposes ShotKV.
- "KV Cache Compression, But What Must We Give in Return?" (https://arxiv.org/abs/2407.01527): early comprehensive benchmark of long-context-capable approaches; same theme.
- "Taming the Fragility of KV Cache Eviction in LLM Inference" (https://arxiv.org/abs/2510.13334): 2025 follow-up specifically on eviction fragility.

---

## F. Surveys and engineering resources (2025-2026)

- "Efficient Attention Mechanisms for Large Language Models: A Survey" (https://arxiv.org/abs/2507.19595, July 2025, revised Feb 2026): the current best single survey covering linear attention AND sparse attention (fixed patterns, block routing, clustering, gated/native sparse), including NSA/MoBA-era production designs.
- "A Survey on Large Language Model Acceleration based on KV Cache Management" (https://arxiv.org/abs/2412.19442): most complete KV-cache-specific taxonomy (token-level eviction/merging/quantization, model-level, system-level); companion list github.com/TreeAI-Lab/Awesome-KV-Cache-Management.
- "KV Cache Compression for Inference Efficiency in LLMs: A Review" (https://arxiv.org/abs/2508.06297, Aug 2025).
- Curated paper list: github.com/October2001/Awesome-KV-Cache-Compression.
- NVIDIA kvpress (github.com/NVIDIA/kvpress) + its Hugging Face leaderboard: RULER-accuracy-vs-compression curves for 20+ methods on current models; the fastest way to sanity-check any eviction policy before writing CUDA. Headline reality check from that leaderboard: past roughly 2-4x compression, every press starts paying measurable RULER accuracy.

---

## G. Other notable 2025-2026 additions surfaced in this sweep (pointers)

- G-KV (https://arxiv.org/abs/2512.00504): decode-time eviction with globally calibrated attention scoring.
- LouisKV (https://arxiv.org/abs/2510.11292): KV retrieval optimized for long-input AND long-output sequences.
- vAttention "Verified Sparse Attention" (https://arxiv.org/abs/2510.05688): unifies top-k + sampling with (epsilon, delta) accuracy guarantees, bridging Quest and MagicPIG.
- NOSA (https://arxiv.org/abs/2510.13602): natively trained offloadable sparse attention (InfLLM-v2 lineage).
- ProxyAttn (https://arxiv.org/abs/2509.24745): block importance estimated from a few representative heads, cheaper block scoring.
- Meta's asymmetric-indexing sparse attention (https://arxiv.org/abs/2502.08246): Faiss-style partitioned KV search at inference time.
- BLASST (https://arxiv.org/abs/2512.12087): dynamic block sparsity via softmax thresholding, no extra metadata.
- 2026 crop (mostly unreviewed yet): HySparse hybrid oracle-selection + KV sharing (https://arxiv.org/abs/2602.03560); Prism spectral block-sparse (https://arxiv.org/abs/2602.08426); "Learning to Evict" (https://arxiv.org/abs/2602.10238); IndexCache cross-layer index reuse (https://arxiv.org/abs/2603.12201); HISA hierarchical indexing (https://arxiv.org/abs/2603.28458); CSAttention centroid scoring (https://arxiv.org/abs/2604.08584); MISA mixture-of-indexers (https://arxiv.org/abs/2605.07363); GRKV regression-based merge-not-drop (https://arxiv.org/abs/2605.31105); VaSE value-aware stochastic eviction (https://arxiv.org/abs/2606.03928); Nexus Sampling reservoir-based streaming eviction (https://arxiv.org/abs/2606.23961); Dustin draft-augmented sparse verification for speculative decoding (https://arxiv.org/abs/2606.24957, directly composes sparse attention with MTP-style drafting); FreqDepthKV (https://arxiv.org/abs/2607.06519).

---

## Honest-degradation synthesis at 10-25% KV budgets (permanent eviction)

- LongBench parity claims at ~12% (PyramidKV) and similar are real but misleading: LongBench answers rarely require exact-token recall. RULER multi-key NIAH, counting, and instruction-adherence break first. SCBench and Pitfalls show the failure axes standard tables hide: second-query collapse and silently ignored instructions.
- 2026 SOTA consensus (kvpress leaderboard, KVzap paper): "near-lossless" honestly means 2-4x compression (keep 25-50%), not 10x. KeyDiff's own headline is telling: the near-zero-loss operating point is only a ~23% trim.
- At 10% budgets, only specialized regimes survive: R-KV on chain-of-thought redundancy (reasoning decode), and query-aware DYNAMIC selection over retained KV (Quest/ShadowKV class at 1-2% compute budgets), which is precisely why compute-sparsity-with-full-KV beats eviction for your quality bar.
- Multi-turn/agentic: query-dependent eviction (SnapKV/PyramidKV/Ada-SnapKV/NACL, RocketKV stage-1) is unsafe on principle; query-agnostic scoring (KVzip/KVzap, Expected Attention, KeyDiff geometry, DuoAttention/RazorAttention head policies) is the safe eviction lane.

## Top 5 best-fit for the mission (llama.cpp CUDA fork, 2x96GB, quality-first, no retraining)

1. **Quest**: highest value-per-engineering-hour; page min/max metadata fits constraint (b) and the contiguous cache, no information loss, 7x attention / 2.2x e2e at 4K token budget, and it is the family SCBench/Sparse Frontier certify as robust; needs a block-gather FA decode kernel + GQA score aggregation.
2. **DuoAttention (per-KV-head two-tier cache)**: the only inference-time method here that RECLAIMS VRAM (1.67x on GQA, more with quant) without harming NIAH, so it directly buys more usable context on fixed hardware; static policy = zero score exposure; run their head-ID recipe once per fleet model.
3. **LServe blueprint (Quest dynamic pages + Duo streaming heads in one kernel family)**: proven multiplicative composition (2.9x prefill, 1.3-2.1x decode over vLLM) and the closest published architecture to what your fork would become; mine omniserve for kernel structure and the reusable-page-selection trick.
4. **KVzip (with NVIDIA KVzap as the cheap variant)**: when KV genuinely cannot fit even sparsified, this is the only eviction lane that survives the second query: 3-4x reduction, ~2x decode speedup, multi-query robust; one-time reconstruction scoring is the only place constraint (a) bites, and KVzap shows it can be approximated cheaply.
5. **XAttention**: training-free antidiagonal block-sparse prefill (up to 13.5x attention speedup at RULER/LongBench parity) attacks your 1M-token prefill wall orthogonally to all of the above, touching neither the KV contents nor decode quality.

Honorable mentions: KeyDiff (easiest first eviction experiment, score-free, honest ~23% trim); ShadowKV (6x GPU-KV reduction with CPU value store, if you open a host-offload lane); SeerAttention-R (best learned decode-gate for long reasoning if per-model gate training becomes acceptable); InfiniteHiP (mine its out-of-length generalization tricks for the YaRN-1M regime).