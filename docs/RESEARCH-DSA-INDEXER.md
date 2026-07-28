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

**Residual error is the fp16 floor, not a logic error.** Both gemms use fp16 accumulate; one fp16
ulp is 4.9e-4 relative. Scaling confirms the mechanism: head dim 128 to 576 barely moved the
error, but top_k 256 to 512 moved it by about sqrt(2), consistent with accumulation growth in the
KQV reduction. Extrapolating (labelled as extrapolation) to the real top_k of 2048 gives roughly
1.5e-3 per layer, which on a 1-bit-class quant across all layers is worth measuring rather than
assuming. `cublasGemmStridedBatchedEx` with `CUBLAS_COMPUTE_32F` removes it.

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
