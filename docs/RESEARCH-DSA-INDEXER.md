# Research sweep: DSA lightning indexer + GLM/DeepSeek sparse attention (2026-07-28)

THE mission-critical sweep. Companions: RESEARCH-SPEC-DECODE-LONGCTX.md, RESEARCH-KV-QUANT.md,
RESEARCH-KV-EVICTION-SPARSE-ATTN.md.

**HEADLINE THAT REVISES OUR WEEK:** llama.cpp merged GLM-5.2 indexer support (#25407) on
2026-07-24 - four days ago. 'runs dense, ignores indexer' is now true ONLY for builds older
than that. The merged path is CORRECTNESS (indexer + top-k masking, currently SLOWER than
dense); the SPEED win waits on the sparse-FA kernel in OPEN PR #25917 (fairydreaming). Our
fleet-sync build (Jul 27) is post-#25407 = it HAS the correctness path.

**INVERTED RISK:** dense serving above 2048 tokens is the OFF-SPEC config - GLM/DeepSeek were
trained on top-2048 sparse attention, so a correct indexer implementation is closer to a
correctness fix than a quality risk. Our all-week dense-MLA GLM serving is itself an
approximation above 2048 ctx.

**THE PRIZE (quantitative, per decoded token @128K, GLM-5.2 78 layers):** DSA top-2048 =
7.7x KV traffic / 13x FLOP reduction; +IndexShare = 21x traffic / 31x FLOP. Single-stream
decode est: 1.45x@128K, 2.9x@512K, 4.7x@1M; prefill 13-30x kernel-level at 512K-1M.

All research complete. Compiling the final report.

# DSA (DeepSeek Sparse Attention) and GLM adoption: deep research report, July 2026

## 1. DeepSeek V3.2-Exp / V3.2: exact DSA spec

**Architecture (official).** DSA = a "lightning indexer" + top-k token selector bolted onto MLA. The indexer scores every prior token s against query token t as I_{t,s} = SUM_j w_{t,j}^I * ReLU(q_{t,j}^I . k_s^I), where the sum runs over H^I indexer heads; only the top-k=2048 latent KV entries then enter real attention ("we select 2048 key-value entries for each query token"). Selection operates in MLA's MQA/absorbed mode: "each latent vector (the key-value entry of MLA) will be shared across all query heads," so one top-2048 set serves all 128 query heads of a layer. Source: DeepSeek-V3.2 paper Sec. 2.1 ([arXiv 2512.02556](https://arxiv.org/html/2512.02556v1)); the V3.2-Exp tech report PDF lives at [github.com/deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) (DeepSeek_V3_2.pdf in repo root).

**Exact config values** ([HF config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json)): `index_n_heads: 64`, `index_head_dim: 128`, `index_topk: 2048`; 61 layers, 128 attention heads, `kv_lora_rank: 512`, `qk_rope_head_dim: 64` (so 576-dim latent/token/layer), hidden 7168, max_position 163,840.

**FP8 + cost of the indexer.** "The lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable." Complexity statement, verbatim: "DSA reduces the core attention complexity of the main model from O(L^2) to O(L*k), where k (<<L) is the number of selected tokens. Although the lightning indexer still has a complexity of O(L^2), it requires much less computation compared with MLA" ([2512.02556](https://arxiv.org/html/2512.02556v1)). Arithmetic check: per query-key pair the indexer costs 64 heads x 128 dims vs main attention's 128 heads x (576+512) dims, i.e. ~5.9% of dense attention FLOPs, halved again by FP8 on Hopper. Only one shared 128-dim FP8 indexer key is cached per token (MQA-style; vLLM: "The indexer keeps a small key cache of 128 per token (vs. 512 for MLA)" - [vLLM tweet](https://x.com/vllm_project/status/1972617272901644345)).

**Prefill is also sparse, with a dense fallback below ~2048.** Sparsity applies to both phases; for short sequences DeepSeek runs a masked-MHA mode: "for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions" (Sec. 2.3). Below L <= 2048 top-k selection is a no-op (everything is selected), which every engine exploits.

**Two-stage training and the key implication.** Dense warm-up: 1,000 steps / 2.1B tokens, main model frozen, indexer trained with KL-divergence against L1-normalized dense attention scores; then sparse stage: 15,000 steps / 943.7B tokens at lr 7.3e-6 with top-2048 selection live (Sec. 2.1.1, [2512.02556](https://arxiv.org/html/2512.02556v1)). Implication verified: a checkpoint that ships trained indexer weights needs zero additional training to run sparse; serving engines (vLLM, SGLang, TRT-LLM, llama.cpp's new path) implement it purely as inference kernels, and GLM-5 adapted DSA onto its own base with only ~20B tokens of continued training vs DeepSeek's 943.7B ([GLM-5 paper](https://arxiv.org/html/2602.15763v1)). Corollary that matters for us: the converse also holds - a model *trained* sparse that is *served* dense is off-distribution above 2048 tokens (see Sec. 6, llama.cpp).

**Official cost/parity claims.** API prices cut "50%+, effective immediately," with V3.1-Terminus kept on a temporary API until Oct 15, 2025 for A/B ([official announcement](https://api-docs.deepseek.com/news/news250929/)); input fell to under $0.03/M cache-hit ([VentureBeat](https://venturebeat.com/ai/deepseeks-new-v3-2-exp-model-cuts-api-pricing-in-half-to-less-than-3-cents)); current market pricing ~$0.28/M in, $0.42/M out ([OpenRouter](https://openrouter.ai/deepseek/deepseek-v3.2)). The paper's cost curves (Fig. 3) are "estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour," showing prefill cost per token nearly flat with position and decode growing only in k. Benchmark parity: MMLU-Pro 85.0 = 85.0, AIME 2025 89.3 vs 88.4, Codeforces 2121 vs 2046 ([GitHub README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)); long context: "we do not observe substantial performance degradation," AA-LCR +4 points in reasoning mode, and Fiction.liveBench "consistently outperforms DeepSeek-V3.1-Terminus" ([2512.02556](https://arxiv.org/html/2512.02556v1)). Raschka's framing of the design goal: "not to improve the performance over V3.1-Terminus but to reduce the performance degradation (due to the sparse attention mechanism) while benefiting from improved efficiency" ([Raschka technical tour](https://magazine.sebastianraschka.com/p/technical-deepseek)).

## 2. GLM-5 / GLM-5.2 (zai-org): DSA adoption confirmed, with configs

**GLM-5** (Feb 2026 paper): 744B total / 40B active, 256 experts, 80 layers, 576-dim latent KV cache, head dim raised to 256 with 1/3 fewer heads, context trained 32K -> 128K -> 200K (202,752). DSA adopted verbatim "to significantly reduce training and inference costs while maintaining long-context fidelity"; top-2048 retrieval; "DSA reduces the attention computation by roughly 1.5-2x for long sequences"; DSA adaptation took ~20B tokens. Long-context scores of the DSA base model: MQ-NIAH-128k 100.0, MV-NIAH-128k 97.0, SQuAD-128k 86.0, HotpotQA-128k 63.0 ([arXiv 2602.15763](https://arxiv.org/html/2602.15763v1); trained on Huawei Ascend chips per the [HF transformers doc](https://huggingface.co/docs/transformers/model_doc/glm_moe_dsa)).

**GLM-5.2** ([model card](https://huggingface.co/zai-org/GLM-5.2), [config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json), [blog](https://huggingface.co/blog/zai-org/glm-52-blog)): 753B total, native 1M context (`max_position_embeddings: 1048576`, rope_theta 8e6), 78 layers, hidden 6144, 64 attention heads, `kv_lora_rank: 512` + `qk_rope_head_dim: 64` (576-dim latent, same as DeepSeek), `q_lora_rank: 2048`, 256 routed experts / 8 active + 1 shared. Indexer: `index_n_heads: 32`, `index_head_dim: 128`, `index_topk: 2048`, plus **IndexShare**: `indexer_types` full/shared pattern from `index_topk_freq: 4`, `index_skip_topk_offset: 3` - the full indexer runs roughly every 4th layer and the following "shared" layers reuse its top-k indices. Official claim: IndexShare "reuses the same indexer across every four sparse attention layers, reducing per-token FLOPs by 2.9x at a 1M context length," and the MTP layer was improved for +20% speculative acceptance length. The IndexShare paper is IndexCache ([arXiv 2603.12201](https://arxiv.org/abs/2603.12201), THUDM/Zhipu authors, [repo](https://github.com/THUDM/IndexCache)): on a 30B DSA model, cross-layer reuse cuts indexer computation 75% and measures 1.82x prefill / 1.48x decode vs standard per-layer DSA with "negligible quality degradation," validated in "preliminary experiments on the production-scale GLM-5 model."

**Community long-context quality:** I found no NIAH/RULER regression reports attributing degradation to DSA in GLM-5.x or V3.2. The clearest field report is in [llama.cpp #24730](https://github.com/ggml-org/llama.cpp/issues/24730): a user running GLM-5.2 UD-Q4_K_XL on 2x Epyc CPU for overnight codebase bug-scans calls the quality "night-and-day" above MiniMax M2.7, with speed (pp 90->50 t/s, tg 5.5->3 t/s) as the only complaint. Note that run used llama.cpp's dense fallback. (Our own loop-rate data attributing GLM looping to weight quantization rather than KV or DSA is internal and consistent with this: nothing in the public record blames the sparse mechanism.) Serving engines recommended by Zhipu: vLLM >= 0.23.0, SGLang >= 0.5.13.post1, KTransformers, Unsloth ([model card](https://huggingface.co/zai-org/GLM-5.2)).

## 3. NSA and MoBA (contrast class)

**NSA** ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089), [full text](https://arxiv.org/html/2502.11089v1)): three parallel branches per query - token compression (block l=32, stride d=16), fine token selection (blocks l'=64, top-n=16 including 1 initial + 2 local blocks), sliding window (w=512), combined by learned gates. Measured: 9.0x forward / 6.0x backward at 64K training, up to 11.6x decode at 64K; decode memory-access table: 8K -> 4x, 16K -> 6.4x, 32K -> 9.1x, 64K -> 11.6x expected speedup. It is trained from scratch because "applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory"; the 27B NSA model beats full attention on 7/9 benchmarks. Contrast with DSA: NSA changes the attention module itself and must be pretrained in; DSA leaves MLA intact and adds a scorer that can be distilled onto an already-trained model in a short continued-training phase - which is exactly why it propagated to GLM in one generation.

**MoBA** (Moonshot, [arXiv 2502.13189](https://arxiv.org/html/2502.13189v1)): applies MoE-style routing to attention at block granularity - context is split into blocks (4096 at 1M scale), a gate picks top-K=12 blocks per query (95.31% sparsity), giving 6.5x prefill speedup at 1M and 16x at 10M tokens, with a hybrid recipe (90% of tokens MoBA, final 10% full attention) closing the trailing-token loss gap; "already been deployed to support Kimi's long-context requests." Block-level selection is coarser than DSA's token-level top-k; MoBA also needs training integration, unlike a shipped indexer.

## 4. Serving implementations and measured numbers

**vLLM** ([day-0 blog, 2025-09-29](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)): DSA implemented via DeepGEMM's indexer logit kernels (`deep_gemm.fp8_mqa_logits` and paged variants) plus FlashMLA's sparse attention kernels; requires block size 64. Indexer keys live in a separate FP8 cache ("first block_size*head_dim entries contain the value, the rest the scaling factor"); MLA KV cache per token is 656 bytes (512 FP8 NoPE + 16B of four float32 scales + 128B unquantized BF16 RoPE). Top-k produces a (q, 2048) integer index tensor consumed directly by the sparse kernel. The blog published no speedups. The living numbers are in the [performance tracking issue #31473](https://github.com/vllm-project/vllm/issues/31473): top-k kernels optimized ([#33680](https://github.com/vllm-project/vllm/pull/33680), long-context [#34265](https://github.com/vllm-project/vllm/pull/34265)), FlashInfer sparse MLA on B200 ([#33451](https://github.com/vllm-project/vllm/pull/33451)), gather/upconvert ([#35290](https://github.com/vllm-project/vllm/pull/35290)); a measured datapoint: at TP8/3.2K-token prefill on H200, FP8 KV step latency 353ms vs BF16 KV 280ms (FP8 KV hurts prefill ~0.8x but helps decode at low concurrency); the absorbed MQA prefill path costs ~3.36x more FLOPs than un-absorbed MHA, so a dense-MHA fallback for `seqlen <= topk` is planned (FlashMLA [PR #551](https://github.com/deepseek-ai/FlashMLA/pull/551)); indexer-MTP optimization tracked in [#35878](https://github.com/vllm-project/vllm/issues/35878) (MTP works with DSA - FlashMLA takes s_q>1 for spec decoding). Known gap: FlashMLA sparse-decode's scratch tensor OOMs on long mixed-batch prefill with FP8 KV on Hopper, currently in the way of GLM-5.2's 1M window there ([#49357 discussion in #31473](https://github.com/vllm-project/vllm/issues/31473)); DSA models also cannot yet select a backend on SM121 / DGX Spark ([#45317](https://github.com/vllm-project/vllm/issues/45317)).

**SGLang**: day-0 V3.2 ([LMSYS blog](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/)) with FlashMLA + an adapted FlashAttention-3 sparse kernel + a purpose-built NSA backend; indexer = "ultra-light FP8 scorer" with its own key+key_scale pool, page size 64; prefill/decode kernel choices include tilelang and aiter (AMD). Benchmarks in the [V3.2 docs](https://docs.sglang.io/basic_usage/deepseek_v32.html): GSM8K 8-shot 0.956 at 5226 tok/s output, stable at 1024 concurrency (2530 tok/s output). For GLM-5.2 the [cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2) publishes: B200 FP8 bs=1 TTFT 295ms / TPOT 1.85ms / 527 tok/s/GPU; bs=64 3770 tok/s/GPU; GB300 bs=1024 6039 tok/s/GPU; plus DSA-specific prefill context-parallel flags (`--enable-dsa-prefill-context-parallel`, `--dsa-prefill-cp-mode in-seq-split`).

**FlashMLA** ([repo](https://github.com/deepseek-ai/FlashMLA)): token-level sparse prefill kernel hits 640 TFlops on H800 / 1450 TFlops on B200; sparse decode with FP8 KV (BF16 math) 410 TFlops on H800; dense decode 3000 GB/s / 660 TFlops reference. Sparse API takes `indices (batch, seq_len_q, topk)` with page-encoded entries and -1 for invalid; MQA mode = head_dim_k 576 / head_dim_v 512; MTP supported via s_q>1. Sparse kernels landed via [PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98); indexer logits kernels in DeepGEMM [PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200); readable reference kernels in [TileLang examples/deepseek_v32](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32).

**Others.** TensorRT-LLM supports DSA natively (`DeepSeekSparseAttentionConfig(index_topk=...)`) alongside RocketKV and skip-softmax BLASST, with an optional Guess-Verify-Refine top-k on Blackwell ([docs](https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html); GVR paper [arXiv 2604.22312](https://arxiv.org/pdf/2604.22312)). KTransformers is on Zhipu's official GLM-5.2 support list for CPU/GPU hybrid ([model card](https://huggingface.co/zai-org/GLM-5.2); its general MoE-hybrid gains are 4.62-19.74x prefill / 1.25-4.09x decode, [SOSP'25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)). MLX: the stock mlx-lm glm_moe_dsa port is incomplete; community GLM-5.2 MLX quants require a patched mlx-lm with indexer fixes ([avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw)). Most interesting consumer datapoint: the full SGLang DSA kernel stack (lightning-indexer GEMM, top-k + page mapping, MLA sparse decode) was ported off Hopper WGMMA to Ada sm_89, serving GLM-5.2 FP8 on 32x RTX 4090-48GB at ~24 tok/s single-stream / ~725 tok/s aggregate, kernels validated to ~1e-6 ([renning22/glm-5.2-4090](https://github.com/renning22/glm-5.2-4090)).

**Third-party cost/latency and quality:** DeepInfra measured V3.2 across 9 API providers: best TTFT 0.76-0.82s, throughput up to 199 tok/s, blended price floor ~$0.29-0.31/M ([benchmark blog](https://deepinfra.com/blog/deepseek-v3-2-api-benchmarks)). No engine has published a clean same-model sparse-vs-dense A/B; the closest quantified anchors are DeepSeek's own H800 cost curves and >50% price cut, GLM's 1.5-2x attention-compute statement, and IndexCache's measured 1.82x/1.48x for the IndexShare increment. Quality regressions: none surfaced in community NIAH/RULER re-tests attributable to DSA; the one documented quality incident was an implementation bug (RoPE discrepancy in the indexer of DeepSeek's own demo code, later fixed - [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) / [Kili analysis](https://kili-technology.com/blog/data-story-deepseek-v3-2)). That is the risk profile in practice: DSA quality failures are porting bugs (RoPE style, Hadamard transforms, scale handling), not the mechanism.

## 5. The indexer's own footprint

- Per-token per-layer indexer cache: one shared 128-dim FP8 key = 128 bytes + ~4B scale = ~132 B (vs 656 B FP8 / 1152 B BF16 for the MLA latent). [vLLM](https://x.com/vllm_project/status/1972617272901644345), [FlashMLA](https://github.com/deepseek-ai/FlashMLA).
- GLM-5.2 (78 layers) totals: 128K ctx -> 1.35 GB; 512K -> 5.4 GB; 1M -> 10.8 GB if cached on every layer. With IndexShare only ~22 "full" layers need it: 0.38 / 1.5 / 3.0 GB. (llama.cpp currently allocates it for shared layers too, flagged as a follow-up in [#25407](https://github.com/ggml-org/llama.cpp/pull/25407).) Relative overhead: ~+11% on a BF16 latent cache, ~+20% on FP8.
- Indexer compute: O(L) per query token per full-indexer layer; the constant is 32 heads x 128 dims (GLM) or 64 x 128 (DeepSeek) per key vs 64 x 1088 or 128 x 1088 for dense MLA, i.e. ~5.9% of dense attention FLOPs, ~3% effective in FP8. In prefill this is the surviving O(L^2) term.
- Dense fallback below threshold is universal: selection is the identity for L <= 2048, DeepSeek's short-prefill masked-MHA mode, vLLM's planned `seqlen <= topk` dense path, and llama.cpp's observed exact-match below 2048 all encode it.

## 6. 2026 developments

- **llama.cpp (the one that matters for us).** The trail: [#19460](https://github.com/ggml-org/llama.cpp/pull/19460) merged 2026-04-11 added the GLM_DSA arch "NOTE: indexer is not yet supported" (pure dense MLA); [#24770](https://github.com/ggml-org/llama.cpp/pull/24770) (2026-06-20) loads the indexer tensors as optional, which is why our GGUF carries `indexer.attn_k/attn_q_b/proj` unused; [#23346](https://github.com/ggml-org/llama.cpp/pull/23346) (2026-05-29) merged DeepseekV32ForCausalLM with a generic DSA implementation (successor of fairydreaming's PoC [#21149](https://github.com/ggml-org/llama.cpp/pull/21149), tested in [discussion #21183](https://github.com/ggml-org/llama.cpp/discussions/21183): sparse implemented as KQ-mask masking of non-top-k tokens, dual KV cache class `llama_kv_cache_dsa` for latent + indexer keys, a Hadamard bug that cost 70%->95% accuracy on lineage-128, lineage-512 Q8_0 at 0.85); [#24231](https://github.com/ggml-org/llama.cpp/pull/24231) (2026-07-11) added `GGML_OP_LIGHTNING_INDEXER` for V3.2/V4, cutting compute buffers from 168,368 MiB to 5,808 MiB at ub=2048; and [#25407](https://github.com/ggml-org/llama.cpp/pull/25407) merged **2026-07-24** adds GLM 5.2 indexer support with IndexShare full/shared handling and interleaved (not neox) rope, existing unsloth GGUFs working unconverted. Critical caveats from the author: greedy output matches the old dense path only for short sequences ("indexers are essentially a no-op the first 2048 tokens"), long sequences are coherent but not KLD-verified, and decode is currently *slower* than the dense path because the sparse speed kernel ([#25917](https://github.com/ggml-org/llama.cpp/pull/25917), sparse KV indices in the MMA FA kernel, explicitly enabled for "DeepSeek V3.2 and similar models (GLM 5/5.1/5.2)" and V4 CSA layers) plus a gather op ([#21458](https://github.com/ggml-org/llama.cpp/pull/21458)) are still open. fairydreaming's quality note stands: "DSA attention works exactly like MLA up to 2048 tokens, then they may start slightly diverge" - i.e., our current dense-MLA serving of GLM-5.2 is itself an approximation above 2048 tokens of context, since the model's final 944B-token-scale training regime (and GLM's 20B adaptation) only ever saw top-2048 attention. Community formulation of the same point ([#24730](https://github.com/ggml-org/llama.cpp/issues/24730)): "a proper DSA implementation I think (positively) affects output quality, not (positively) output speed" for CPU-bound rigs. Third-party Mesh-LLM patches for DSA + IndexShare on CPU and Metal are being cleaned up for upstreaming per the same thread. No dedicated Ollama/LM Studio threads found; they inherit llama.cpp's state.
- **Post-hoc sparse retrofits onto dense models** are now a research lane: SeerAttention (Microsoft, learned intrinsic sparse gates, [repo](https://github.com/microsoft/SeerAttention)); SpotAttention (plug-in block-sparse routing for pretrained transformers, KL-distilled selector against the frozen dense teacher with dual top-p, [arXiv 2606.22874](https://arxiv.org/html/2606.22874v1)); MISA (mixture-of-indexer replacing the lightning indexer with an MoE router, [arXiv 2605.07363](https://arxiv.org/pdf/2605.07363)); HISA (hierarchical indexing, [arXiv 2603.28458](https://arxiv.org/pdf/2603.28458)); StreamIndex (memory-bounded streaming top-k, [arXiv 2605.02568](https://arxiv.org/pdf/2605.02568)); MTraining (dynamic sparse attention extending Qwen2.5-3B/Llama-3.1-8B from 32K/128K to 512K, [arXiv 2510.18830](https://arxiv.org/pdf/2510.18830)); plus the tradeoff survey The Sparse Frontier ([arXiv 2504.17768](https://arxiv.org/pdf/2504.17768)).
- **DeepSeek V4** (public April 2026): 1.6T Pro and 284B Flash, 1M context, goes beyond DSA to a per-layer hybrid: sliding-window-128 mixed with either top-512 sparse over 4:1-compressed KV ("C4," still lightning-indexer-driven) or dense over 128:1-compressed KV ("C128"), plus mHC hyper-connections and native FP4 experts. SGLang day-0 measured near-flat long-context decode: B200 Pro 199 -> 180 tok/s and H200 Flash 266 -> 240 tok/s from 4K to 900K context, under 10% degradation ([LMSYS V4 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)). llama.cpp merged V4 in [#24162](https://github.com/ggml-org/llama.cpp/pull/24162) (2026-07-16), and #25917's sparse-FA path covers V4's CSA layers.
- GLM-5.x serving guides now standardize on sparse backends ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2), [vLLM recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3.2), [transformers glm_moe_dsa](https://huggingface.co/docs/transformers/model_doc/glm_moe_dsa)).

## Quantitative mini-model: GLM-5.2-class decode at 128K (78 layers, 576-dim latent)

Per generated token, per layer, dense MLA must stream the whole latent cache; DSA streams top-2048 plus an indexer scan on full-indexer layers (~22 of 78 with IndexShare):

| Quantity, per decoded token @128K | Dense MLA (llama.cpp today) | DSA top-2048 | DSA + IndexShare |
|---|---|---|---|
| Attention KV traffic (f16 cache) | 131,072 x 1152B x 78 = **11.8 GB** | 78 x (2048x1152B + 131,072x132B) = **1.53 GB** | 184 MB + 22 x 17.3 MB = **0.56 GB** |
| Attention FLOPs | 78 x 64h x 1088 x 131,072 x 2 = **1.42 TFLOP** | 106 GFLOP | **46 GFLOP** |
| Attention-level reduction | 1x | 7.7x traffic / 13x FLOPs | **21x traffic / 31x FLOPs** |

Scaling the same math: 64K -> dense 5.9 GB vs 0.36 GB (16x); 512K -> 47.1 GB vs 1.57 GB (30x); 1M -> 94.3 GB vs 3.0 GB. The indexer scan is the new long-context floor: it is O(L) per token but only ~132B/token/layer on ~22 layers, i.e. ~6% of dense attention FLOPs per full layer (FP8), matching DeepSeek's "much less computation" claim and Zhipu's measured 2.9x per-token FLOPs cut at 1M from IndexShare alone.

**Expected end-to-end decode speedup** depends on how much traffic is attention vs expert weights. Single-stream (our case), with ~40B active params: at ~23 GB/token weight reads (Q4-class), DSA implies roughly **1.2x @64K, 1.45x @128K, 2.9x @512K, 4.7x @1M**; at Q8 (~42 GB/token) roughly 1.1x / 1.3x / 2.1x / 3.2x. Batched serving amortizes weights, so gains approach the attention-level 8-30x, which is what makes DeepSeek's >50% price cut, GLM's 1M window, SGLang's 527 tok/s/GPU on GLM-5.2, and V4's near-flat 4K->900K decode curves coherent with each other. Measured anchors bracketing these estimates: NSA 11.6x decode @64K attention-level ([2502.11089](https://arxiv.org/html/2502.11089v1)), IndexCache 1.48x decode / 1.82x prefill end-to-end for the IndexShare increment on a 30B DSA model ([2603.12201](https://arxiv.org/abs/2603.12201)), GLM's own "1.5-2x" attention-compute figure at long sequences ([2602.15763](https://arxiv.org/html/2602.15763v1)). Prefill at 128K: DSA cuts attention FLOPs to ~3-7% of dense (indexer O(L^2) at ~6% constant on 22/78 layers + main attention at 2048/L), so attention-dominated long prefill improves up to ~13-30x at the kernel level - at 512K-1M this, not decode, is the larger absolute win.

**Quality risk, inverted:** for a model trained with top-2048 sparse attention, implementing the indexer locally is closer to a correctness fix than a risk - dense serving above 2048 tokens is the off-spec configuration, official and community evals show parity-or-better for sparse vs the dense predecessor, and the real hazards on record are porting details: rope style (GLM uses interleaved, DeepSeek neox), the Hadamard transform, FP8 scale handling, and keeping indexer weights at high precision when quantizing (pcuenca ships them Q8_0 in his test GGUF).

Sources: [DeepSeek-V3.2 paper](https://arxiv.org/html/2512.02556v1), [V3.2-Exp repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), [V3.2-Exp config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json), [official announcement](https://api-docs.deepseek.com/news/news250929/), [VentureBeat](https://venturebeat.com/ai/deepseeks-new-v3-2-exp-model-cuts-api-pricing-in-half-to-less-than-3-cents), [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2), [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json), [GLM-5.2 blog](https://huggingface.co/blog/zai-org/glm-52-blog), [GLM-5 paper](https://arxiv.org/html/2602.15763v1), [transformers glm_moe_dsa](https://huggingface.co/docs/transformers/model_doc/glm_moe_dsa), [IndexCache](https://arxiv.org/abs/2603.12201), [NSA](https://arxiv.org/html/2502.11089v1), [MoBA](https://arxiv.org/html/2502.13189v1), [vLLM day-0 blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html), [vLLM #31473](https://github.com/vllm-project/vllm/issues/31473), [vLLM #45317](https://github.com/vllm-project/vllm/issues/45317), [LMSYS V3.2](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/), [LMSYS V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), [SGLang GLM-5.2 cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2), [SGLang V3.2 docs](https://docs.sglang.io/basic_usage/deepseek_v32.html), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200), [TileLang examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32), [TensorRT-LLM sparse attention](https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html), [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/), [glm-5.2-4090](https://github.com/renning22/glm-5.2-4090), [DeepInfra benchmarks](https://deepinfra.com/blog/deepseek-v3-2-api-benchmarks), [Raschka technical tour](https://magazine.sebastianraschka.com/p/technical-deepseek), [Raschka IndexShare note](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html), llama.cpp: [#19460](https://github.com/ggml-org/llama.cpp/pull/19460), [#24770](https://github.com/ggml-org/llama.cpp/pull/24770), [#23346](https://github.com/ggml-org/llama.cpp/pull/23346), [#24231](https://github.com/ggml-org/llama.cpp/pull/24231), [#25407](https://github.com/ggml-org/llama.cpp/pull/25407), [#25917](https://github.com/ggml-org/llama.cpp/pull/25917), [#21458](https://github.com/ggml-org/llama.cpp/pull/21458), [#21149](https://github.com/ggml-org/llama.cpp/pull/21149), [#24162](https://github.com/ggml-org/llama.cpp/pull/24162), [discussion #21183](https://github.com/ggml-org/llama.cpp/discussions/21183), [issue #24730](https://github.com/ggml-org/llama.cpp/issues/24730), retrofit papers: [SeerAttention](https://github.com/microsoft/SeerAttention), [SpotAttention](https://arxiv.org/html/2606.22874v1), [MISA](https://arxiv.org/pdf/2605.07363), [HISA](https://arxiv.org/pdf/2603.28458), [StreamIndex](https://arxiv.org/pdf/2605.02568), [MTraining](https://arxiv.org/pdf/2510.18830), [Sparse Frontier](https://arxiv.org/pdf/2504.17768), [GVR top-k](https://arxiv.org/pdf/2604.22312), [MLX GLM-5.2 quant](https://huggingface.co/avlp12/GLM-5.2-Alis-MLX-Dynamic-2.56bpw), [Kili V3.2 analysis](https://kili-technology.com/blog/data-story-deepseek-v3-2), [vLLM DSA thread](https://x.com/vllm_project/status/1972617272901644345).

One action-relevant headline beyond the brief: llama.cpp merged GLM-5.2 indexer support (#25407) on 2026-07-24, four days ago. The premise "llama.cpp runs dense MLA and ignores the indexer" is now true only for builds older than that. What merged is the correctness path (indexer + top-k masking, currently slower than dense); the speed win waits on the sparse flash-attention kernel in open PR #25917 by fairydreaming.
---

## 9. Implementation status: gathered top-k attention, ported and validated (2026-07-29)

This section records what was built after the research above, and corrects two things the
research phase got wrong.

**The waste, confirmed in our own source rather than inferred.** `llm_graph_context::build_attn`
for the DSA input (`src/llama-graph.cpp`, the `llm_graph_input_attn_k_dsa` overload) does exactly
this: `ggml_fill(kq_mask, -INFINITY)` over the full `[n_kv, n_batch]` mask, `ggml_set_rows` to
punch zeros at the top-k indices, `ggml_add` to combine with the causal mask, then
`build_attn_mha` runs stock flash attention over **all** n_kv keys. The shortlist is computed and
then thrown away. This is the mask-shaped vs gather-shaped distinction from Sec. 6, now verified
in the tree we serve from.

**What was ported.** `GGML_OP_FLASH_ATTN_EXT_DSA` with a CUDA implementation
(`ggml/src/ggml-cuda/fattn-dsa.cu`), transplanted from ik_llama.cpp's `dsa_attn.cu`
(MIT, Iwan Kawrakow): gather mask columns, gather K rows (and V rows unless V aliases K),
stage Q to f16, batched `cublasHgemmStridedBatched` for KQ, f16 softmax, second batched gemm for
KQV, copy out to f32, in chunks of at most 32 query rows. The mainline precedent for an
FA-family op carrying a sixth source tensor is `ggml_flash_attn_ext_banded`, which is what made
the op-registration side of this low risk.

**One genuine bug found in the source kernel.** ik's `dsa_attn.cu` never calls `cublasSetStream`,
so its gemms run on the default stream while the surrounding kernels run on the backend's stream.
Our port sets it. ik does call it in `indexer_topk.cu`, so this reads as an oversight rather than
intent.

**Numerical validation (48/48 case-runs, 9 shapes x 4 seeds).** Against stock `ggml_flash_attn_ext`
with a mask that unmasks the identical key set, and independently against an fp64 host reference:

| comparison | rel_L2 range |
|---|---|
| gather vs stock FA | 6.9e-4 to 9.9e-4 |
| gather vs fp64 reference | 5.9e-4 to 9.1e-4 |
| stock FA vs the same fp64 reference | 3.1e-4 to 4.1e-4 |

Max absolute difference 2.4e-4 on outputs of scale ~0.15. The MLA case that matters
(d576/512 with V constructed as a view into K, at 8, 40 and 128 head configurations) is covered.
Sensitivity control: a reference computed over a **different** random top-k set scores rel_L2
1.3 to 3.4, three to four orders of magnitude worse, so the test is measuring something. Mask
entries at non-selected positions were poisoned with random values in [-8, 8], indices were
shuffled rather than sorted, and `dst` was pre-filled with NaN so a partial write cannot pass.

**Where the equivalence claim is genuinely false: duplicate indices.** If the indexer emits a
repeated key in a row, the gather double-counts it and the mask formulation cannot. Measured
rel_L2 for that case is 0.335. `ggml_top_k` returns distinct positions and the graph-side guard
forces `n_kv >= 4*top_k` so the selection never clamps, but this is a live precondition on
anything that ever replaces the top-k source, not a theoretical one.

**Residual error was the fp16 accumulate floor, and fp32 accumulate removes it for free -
now the default.** Measured: under fp16 accumulate the error tracked top_k (rel_L2 9.1e-4 at
top_k 512, 1.06e-3 at 12032); under `cublasGemmStridedBatchedEx` with `CUBLAS_COMPUTE_32F` it is
flat at 3.08-3.22e-4 regardless of top_k - the top_k dependence disappears, and the gather path
becomes slightly *more* accurate than stock flash attention against the same fp64 reference
(0.78-0.99x of its error). Cost: median -0.8 percent over 42 paired measurements at GLM shapes,
i.e. free; at 128 heads the Ex API was actually 12-15 percent faster at prefill. One unproven
attribution: the Ex-API-with-fp16-compute arm was not run, so "Ex API is faster" versus "fp32
accumulate is faster" is not separated. The remaining 3.1e-4 is the fp16 KQ/KQV storage floor.
`LLAMA_DSA_F32ACC=0` restores the fp16-accumulate path.

**A latent aliasing bug, fixed.** `dsa_v_is_k_view()` decided that V aliases K purely from the
pointer landing inside K's first row, without checking row stride. The reuse path then indexes
the gathered K buffer with K's stride, so a V view based inside K's first row with a different
`nb[1]` would silently gather wrong rows with no guard tripping. It now returns false on stride
mismatch, falling back to the slower separate-V gather rather than aborting.

**Wiring.** The path is selected at graph-build time, where n_kv, top_k and tensor types are all
known, so a shape the backend would refuse never reaches it. Guards mirror the CUDA
`supports_op` conditions one for one. Selection is gated on `LLAMA_DSA_GATHER=1` so a single
binary runs both arms and an A/B carries no build-difference confound.

**Constraint worth stating plainly: the gather path requires an F16 KV cache.** `-ctk q8_0` and
below are refused by the type guard. This trades KV bytes for attention work, which cuts against
the FIT-first priority, so whether it is net positive is an open measurement rather than a claim.

### Correction to Sec. 6: the fused Lightning Indexer is enabled on our split, not disabled

A hypothesis circulated that `resolve_fused_ops` (`src/llama-context.cpp`) disables the fused
Lightning Indexer globally on a multi-GPU split, because it breaks on the first
`device_fused != dev_layer(il)` mismatch and then sets `enabled = false` for the whole model.

This is refuted by evidence already on disk. `server_greedy_clean.log` from the MTP PR work is a
genuine two-card GLM-5.2 run (`arch = glm-dsa`, `model type = 744B.A40B`, layers assigned across
both CUDA0 and CUDA1) and logs `resolve_fused_ops: Lightning Indexer enabled`. Three other server
logs agree, and no `is assigned to device` warning appears anywhere.

The mechanism explains why it cannot fire: the scheduler places a node on the device holding its
weights, and `indexer_score` for layer `il` is built from layer `il`'s own tensors, so
`device_fused == dev_layer(il)` by construction. A layer split never separates them. The guard
exists for genuinely missing backend support, not for tensor splits.

This matters because the `[n_kv, 32, n_ubatch]` F32 score tensor blamed for the compute-buffer
ceiling exists **only in the unfused branch** of `src/models/glm-dsa.cpp`. The fused branch is a
single `ggml_lightning_indexer` call. At 128K context and ubatch 512 that is roughly 8.6 GB
against 0.27 GB, a factor of 32, and we are already on the cheap side. Any compute-buffer ceiling
derived from the unfused shape needs re-deriving on the fused graph.

### Kernel-level measurement: gather vs mask-shaped at GLM shapes (2026-07-29)

Standalone benchmark (`tests/bench-fattn-dsa.cpp` on `fable-dsa-harvest-box`), single GPU,
head dim K 576 / V 512 with V as a view into K, top_k 2048.

**These figures were measured at n_head 128, but GLM-5.2 has 64** (`glm-dsa.attention.head_count
= 64`, read from the served GGUF) - an error in the benchmark brief. The corrected sweep at
n_head 64 landed and the speedups drop materially:

| shape | n_head 128 (table below) | **n_head 64 (GLM-correct)** |
|---|---|---|
| 8K decode | 2.14x | **1.77x** |
| 32K decode | 5.55x | **3.47x** |
| 128K decode | 19.68x | **11.75x** |
| 8K prefill | 1.33x | **1.02x (buys nothing)** |
| 32K prefill | 4.81x | **3.68x** |
| 128K prefill | 18.40x | **12.99x** |

**Quote 11.8x decode / 13.0x prefill at 128K, not 19.8x/18.3x.** The mechanism was not the
predicted one: the gather GEMMs did not get relatively worse - the mask path got much better.
Mask cost scales near-linearly with n_head (roughly -42 to -50 percent when halved) while the
gather path barely moves at decode (-4 to -9 percent), because gathering top_k K rows of DK
elements has no n_head term. The denominator shrank, so the ratio collapsed. The n_kv-flatness
of the gather path is unchanged, and the compute-buffer decoupling is unaffected.
Median of 20 to 2000 iterations per cell after warmup. The mask arm includes its
`ggml_fill` + `ggml_set_rows` + `ggml_add` construction cost, since that work disappears in the
gather path.

**Decode, one query token:**

| n_kv | gather (ms) | mask + full FA (ms) | speedup |
|---|---|---|---|
| 8,192 | 0.0204 | 0.0433 | 2.1x |
| 16,384 | 0.0204 | 0.0669 | 3.3x |
| 32,768 | 0.0204 | 0.1139 | 5.6x |
| 65,536 | 0.0204 | 0.2078 | 10.2x |
| 131,072 | 0.0205 | 0.4050 | 19.8x |

The gather cost is flat: 0.0204 ms at 8K and 0.0205 ms at 128K, sixteen times the context for the
same time. The mask path grows linearly. A top_k=256 control drops the gather to 0.0142 ms and is
equally flat, confirming cost tracks top_k rather than n_kv.

**Prefill, 512 query tokens:** 1.34x at 8K, 2.56x at 16K, 4.82x at 32K, 9.03x at 64K,
and 18.26x at 128K (4.79 ms against 87.52 ms).

**Compute buffer, which is the constraint that actually sets the context ceiling.** At 512-token
prefill the gather path allocates 134,217,728 bytes at every context from 8K to 128K, while the
mask path grows from 155,189,248 to 272,629,760. The gather path decouples graph scratch from
context length entirely; the mask path does not.

Net device memory is a more mixed picture and the compute-buffer figure alone overstates it.
Counting kernel pool scratch as well, the gather path saves roughly 20 MiB at every decode shape
and 104 MiB at 128K prefill, but **costs about 8 MiB more at 8K prefill**, where its gathered-K
scratch outweighs a small mask. The win is real and grows with context, but it is not a clean win
at the short end.

Correctness controls inside the benchmark: `absmean` of the two paths agrees to five or six
significant figures at every shape (for example 0.0127705023 against 0.0127689639 at 128K decode),
and `nonfinite` is zero everywhere.

**What this number is not.** It is one attention call, not a token. GLM decode is dominated by
expert weight traffic, so the end-to-end gain is bounded by the fraction of token time spent in
attention. At 128K that is roughly 79 layers x 0.405 ms = 32 ms per token going to about 1.6 ms,
but only a model-level run settles what that is worth. Not yet measured: the real model, output
quality, NIAH, and multi-GPU behaviour. Two of the 64 result rows carry corrupt memory counters
(negative values from harness counter wraparound); the `compute_buf` figures quoted above are
consistent across all 64 rows.

**Why the speedup falls short of the work ratio, and where the remaining headroom is.** At
131,072 / 2,048 the work ratio is 64x but the measured speedup is 18.26x. The gap is entirely
execution efficiency: the stock flash-attention mma kernel sustains 214 to 226 TFLOP/s (roughly
85 to 90 percent of dense fp16 peak) while the gather path's batched cuBLAS HGEMM sustains 61 to
73 TFLOP/s (24 to 29 percent). 64 divided by 3.52 is 18.2, against 18.26 measured, and the same
arithmetic reproduces the decode figure. The cause is cuBLAS batch inefficiency at skinny shapes:
the kernel chunks queries into 32 rows and issues GEMMs of m=top_k, n=n_head, k=576, which does
not fill the machine at top_k 2048. Gather throughput climbs 7.5 to 36 to 85 TFLOP/s as top_k goes
256 to 2048 to 8192, reaching flash-attention-class efficiency only near 8192. **Closing that gap
is worth more than anything else in this lane**, since it is the difference between 18x and
something approaching the 64x the work ratio allows.

Related, and useful when comparing against the mask path: the mask path structurally cannot
exploit its own sparsity. `flash_attn_mask_to_KV_max` only truncates a fully masked suffix, never
interior blocks, and it is not launched at all unless `Q->ne[1] >= 1024 || Q->ne[3] > 1`, so at
decode and 512-token prefill it does full work regardless of how sparse the mask is.

**A latent crash, found by the benchmark sweep and now fixed.** `dsa_attn_layout_ok()` did not
check the softmax shared-memory requirement, so `supports_op` accepted shapes that then hard
aborted in `dsa_soft_max_f16_cuda` on `GGML_ASSERT(shmem < smpb)`. Measured bracket on this box:
top_k 12,032 runs, 12,288 aborted (smpb 49,152). The fix gives the shared-memory requirement a
single definition used by both the launcher's assert and the guard so they cannot drift; the
previously aborting shapes (12,288 through 16,384) are now rejected cleanly at `supports_op` with
zero aborts, and 2,048/12,032 still run correctly. Two permanent regression cases added
(14/14 pass in both accumulate arms).

### Head-to-head: upstream PR #25917 (mma sparse) vs the cuBLAS gather (2026-07-29)

fairydreaming's draft PR #25917 modifies the stock MMA flash-attention kernel to gather at
tile-load time (`KV + top_k[i]*stride_KV`), activated by `ggml_flash_attn_ext_add_top_k` on the
standard FA call. Architecturally attractive: per-token top-k rows (`ncols1 = 1`), graceful CPU
fallback (the CPU backend ignores `src[5]` and computes the mask semantics), and his
`llama-graph.cpp` wiring covers the glm-dsa overload. The branch was brought into the fork as
`fable-sparse-mma-25917`, built, and his backend-ops suite passed 2888/2888 on the box's
Blackwell card, including his sparse top-k cases.

Measured head-to-head at GLM shapes (DK 576 / DV 512, V a view into K, 64 heads MQA, top_k 2048,
same protocol and PRNG as the gather benchmark, zero timing overlap with other GPU work):

| shape | mma sparse | cuBLAS gather | verdict |
|---|---|---|---|
| decode, any n_kv (both flat) | 0.0293 ms | 0.0188 ms | gather 1.56x faster |
| prefill 32K | 4.94 ms | 3.06 ms | gather 1.62x faster |
| prefill 128K | 7.48 ms | 3.63 ms | gather 2.06x faster |

**The mma-sparse path loses at every measured GLM shape.** The reason is the cost of entering
the sparse mode: `use_top_k` forces `ncols1 = 1` and disables cp_async and multi-stage
pipelining, giving up roughly 7.4x per-tile efficiency. Net efficiency lands at about 13 percent
of the theoretical n_kv/top_k work ratio, against 18 to 48 percent for the gather - the gather is
roughly twice as close to theory. His prefill gate also fires too early on this hardware: at
n_kv 8192 and top_k 2048 the sparse path is a 1.86x slowdown against the mask path it replaces
(the crossover is a ratio of about 8, i.e. 32K at top_k 2048). The kernel is numerically clean
everywhere (22/22 shapes, and a dense-fingerprint control proved the sparse path was genuinely
active, not silently falling back). The defect is gate tuning and pipeline loss, not correctness,
matching the TODO in his own dispatch code.

Conclusion: the cuBLAS gather stays the merge candidate. The "close the GEMM efficiency gap"
lane resolves as parked - the alternative that promised to close it measures worse end to end.

**A tree-level finding with strategic weight, from the benchmark's own controls:** the sparsemma
tree's mask path is itself 1.2 to 1.5x faster than the dsaport tree's mask at the same shapes
(42.98 vs 66.36 ms at 128K prefill) - upstream's flash attention improved between the two bases.
A model-level A/B run in the dsaport tree would therefore compare the gather against a stale
mask baseline and overstate the win. Consequence: the gather commits were cherry-picked onto the
current fleet base (`fable-dsa-fleet` = fleet-sync a91c47d49 + the 8 DSA commits, all clean), so
the A/B binary is the merge candidate itself, with the newest mask path as the honest baseline.


### Model-level A/B on the real GLM-5.2 744B: the gather path end to end (2026-07-29)

Harness: one binary (`fable-dsa-fleet`), two env-gated arms, identical weights
(IQ1_S + blk.78 q4_K), F16 KV both arms, greedy, six short prompts plus one deep needle leg
per context size. Path proof per arm: kernel-execution line
(`ggml_cuda_flash_attn_ext_dsa: EXEC (n_kv=8448, top_k=2048, f32acc=1)` - engages at the first
256-block past the 4x2048 guard) and graph structure (6510 vs 7134 nodes; the 624 delta is
exactly 78 DSA layers x 8 mask-construction ops).

| depth of needle leg | mask decode t/s | gather decode t/s | gain | mask prefill | gather prefill |
|---|---|---|---|---|---|
| 24K (ctx 32K) | 43.91 | 50.84 | +15.8% | 412.9 | 438.6 |
| 50K (ctx 64K) | 38.51 | 49.53 | +28.6% | 345.9 | 419.1 |
| 100K (ctx 128K) | 29.30 | 49.68 | +69.6% | 243.0 | 385.9 |

**The gather arm's decode is flat across the sweep - 50.8 / 49.5 / 49.7 t/s from 24K to 100K
deep - while the mask arm degrades linearly.** Decode stops paying a context-depth penalty at
these depths. Compute buffer at 128K: 1552/1548 MiB against 1692/1716 (the gather buffer grows
with ubatch only). Needle found in both arms at every size; short prompts (below the guard
threshold, mask path in both arms) byte-identical.

Long-leg outputs differ between arms at every size (same length class, both correct on the
needle): the arms are equivalent only to fp tolerance and 64 greedy tokens through 79 layers can
flip an argmax. Quality parity therefore requires the NIAH-sweep and loop-rate gate rather than
hash identity; that gate is a merge precondition.

One instrumentation note: a one-shot graph-build INFO line for path proof never surfaced because
llama-context mutes the logger during probe reserves and the static latched during the muted
window. The execution-time fprintf proof replaced it. The harness grep for the exec line also
misfired twice while the line was demonstrably in the file; the structural node-count proof is
the load-bearing check.
### The context ceiling, measured (2026-07-29)

Read off the 128K model-level A/B loads (`GLM-5.2-ours-IQ1_S-prot-blk78q4`, two RTX PRO 6000,
F16 KV, ubatch 512). No dedicated measurement run; these numbers are in the serve logs.

| component | measured |
|---|---|
| weights | 161.7 GiB (78,992 + 86,579 MiB) |
| KV at 128K | 13.58 GiB total, i.e. **108.6 KiB per token** |
| main KV (MLA latent, K only, V is a view into K) | 88.9 KiB/token, matches 576 x 2 x 79 exactly |
| indexer KV | 19.8 KiB/token, matches 128 x 2 x 79 exactly |
| compute buffer, gather path | 1,084 / 1,820 / 3,100 MiB at 32K / 64K / 128K |
| compute buffer, mask path | 1,484 / 2,124 / 3,408 MiB at the same sizes |
| VRAM in use at 128K | 183,894 of 194,494 MiB, leaving 10.4 GiB |

**KV dominates the compute buffer by more than four to one at 128K** (13.6 GiB against 3.1).
Any ceiling analysis that treats graph scratch as the binding constraint on this model is
looking at the wrong term.

The compute buffer scales with context as well as with ubatch: at fixed ubatch 512 it grew from
1,084 to 3,100 MiB across a 4x context increase. A linear fit on the two widest points gives
roughly **540 MiB + 20.0 MiB per 1K of context**.

**Ceiling arithmetic.** After weights, 28.2 GiB remains for KV plus compute:

| KV precision | maximum context | compute buffer there | decode behaviour |
|---|---|---|---|
| F16, the only precision the gather kernel accepts | **~221K** | 4.8 GiB | flat, about 50 tokens/s at any depth |
| q8_0, mask path only | **~366K** | 7.7 GiB | degrades linearly, 29 tokens/s at 100K deep |

So the gather path currently trades roughly 145K of maximum context for decode that does not
degrade with depth. Neither column dominates the other.

**The change that removes the trade.** The kernel refuses quantized KV only because the gather
copies raw f16 rows. It already touches every selected element exactly once while gathering, so
dequantizing in that same pass costs close to nothing. Teaching `fattn-dsa.cu` to dequantize on
gather would give both the larger context and the flat decode rather than forcing a choice, and
it moves the ceiling itself rather than the clock.

Labelled as extrapolation rather than measurement: the fitted compute buffer at 1M context is
about 20.9 GiB.
