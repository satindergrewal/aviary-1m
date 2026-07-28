# Research sweep: squeezing K3-class (1TB+) MoE onto our box (2026-07-28)

THE ENDGAME MAP. Satinder's directive: Kimi-K3-class is the real target; GLM 5.2 is the
warmup. Companions: RESEARCH-DSA-INDEXER.md + 4 others.

**THE PHYSICS:** K3 = 2.78T total / 104B active, native MXFP4-QAT, 1.56TB on disk (= our
1.5TB download). Static fit IMPOSSIBLE (even 1.0bpw = 348GB > 320GB total fast mem). Per-expert
NVMe paging is the ENTRY TICKET, not an optimization.

**PROVEN ON OUR EXACT HARDWARE:** loFT LLC ran DeepSeek-V4-Flash (284B, native FP4+FP8, 146GB)
on 2x RTX PRO 6000 Blackwell Max-Q 96GB at PP 38.3 / TG 35.7 tok/s, fully in VRAM, GPUs at
30-40% util. V4-Pro (1.6T) is the stepping stone between GLM and K3 (~400GB @2bpw = light
paging away).

**llama.cpp K3 STATUS:** PR #26185 (pwilkin) OPEN + WORKING - loads and generates coherently
with GPU+CPU offload, logit correlation 1.0, lossless MXFP4 repack. GGUFs already exist
(GrEarl/Kimi-K3-GGUF 928GB @2.671bpw). One PR from loadable.

**THE CRITICAL PATH:** per-expert paged serving (VRAM<-RAM<-NVMe, async io_uring, LFRU
hub-protection) + speculative-decode-driven expert prefetch (MoE-SpeQ: 90.9% expert-prediction
accuracy, 96-99% hit -> DSpark/MTP is the prefetch ORACLE, not just speed) + REAP (50% expert
prune NEAR-LOSSLESS, proven on Kimi-K2 itself). Recipe R1 estimate: ~5-12 t/s decode.

**KDA insight (answer to 'is K3 more fittable at long ctx than GLM?'):** YES on context - KDA
linear layers carry FIXED ~432MB state at ANY length + 27GB MLA KV at 1M; 3 of every 4 layers
carry zero per-token state. KDA removes the CONTEXT wall, leaving only the WEIGHTS wall for
paging to break. llama.cpp already has KIMI_LINEAR + gated-delta-net + MXFP4 type.

All research lanes are closed. Composing the full structured report now.

---

# DEEP RESEARCH REPORT: Squeezing K3-class (1TB+) MoE onto 2x96GB Blackwell + 128GB RAM + NVMe

## Executive summary (the physics first)

Kimi K3 weights dropped July 27 (2 days ago): 2.78T total / 104B active, native MXFP4 QAT, 1.56TB on disk. **That is exactly the 1.5TB already on our NVMe.** The three-lab mining converges on one conclusion: **K3-class can never statically fit 320GB (192 VRAM + 128 RAM) at any sane bpw (even 1.0bpw = 348GB), so per-expert NVMe paging is not an optimization, it is the entry ticket.** The good news: every ingredient now exists with code and numbers, including a working llama.cpp K3 arch PR, a llama.cpp expert-paging PoC with measurements, a vLLM RFC with 97-100% hit rates, a lab-published speculative-prefetch method with 90.9% expert-prediction accuracy, and a K2-proven 50%-expert-pruning tool. K3's KDA hybrid also makes 1M context nearly free on memory (~27GB KV + 0.4GB state), so unlike GLM, context is NOT the enemy; weights are.

**Mission-fit ranking:**
1. Per-expert paged serving (VRAM cache <- RAM <- NVMe, async I/O, LFRU/SLRU)
2. Speculative-decode-driven expert prefetch (DSpark/MTP as the oracle)
3. REAP expert pruning (50% near-lossless, proven on Kimi-K2 itself)
4. Resident-compressed-fallback experts (FloE pattern; dovetails with our two-profile quant lane)
5. KDA/hybrid state economics (answer to question c)
6. Lab-native 4-bit QAT formats (MXFP4 repack, don't requantize blindly)
7. EPLB/MoonEP-style hot-expert replication/pinning planner
8. Mooncake-style KV/prefix NVMe tier (matters for GLM at 1M more than K3)
9. PHASE-2: FlashKDA kernel port, MoonEP zero-copy/static-shape tricks, raw DSpark speed

---

## LEAD 1: MoonEP — https://github.com/MoonshotAI/MoonEP

- **What it is:** "A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts" (MIT, 2026, authors Yutian Chen, Cong Li, Yucheng Wang, Ming Wei). EP *communication* library for training + inference of MoE across ranks. Not expert paging.
- **Mechanisms:** (1) **Dynamic redundant experts**: hot experts are replicated online so every rank always computes exactly S×K tokens per layer regardless of router skew; redundancy is "planned online and prefetched before computation" (there is a literal `prefetch_weight()` API). (2) **Zero-copy dispatch**: tokens written directly to final expert-grouped positions on remote ranks, comm-buffer views handed straight to compute. (3) **Static shapes**: "statically known shapes eliminate per-layer MoE host synchronization", no fragmentation, no OOM.
- **Numbers:** vs DeepEP v2 on H20 EP=8: comm time flat at every imbalance (maxvio) level while DeepEP degrades steadily. Hardware: NVIDIA + "Zhenwu PPU (under review)".
- **Single-box port path:** don't port the RDMA comm; port the *ideas*. (a) Hot-expert **replication across our 2 GPUs** to kill PCIe cross-traffic (no NVLink on RTX Pro 6000). (b) The static-shape/no-host-sync doctrine is the lab-validated version of exactly the disease we diagnosed in our MTP graph-reuse work (the ~18ms/tok sync-stall): Moonshot considered per-layer MoE host sync worth eliminating at datacenter scale. (c) `prefetch_weight()` is first-party evidence Moonshot prefetches expert weights ahead of compute.
- **Mission-fit: 6/10 as concept mine, 2/10 as code.** PHASE-2 for its comm tricks.
- Announcement: https://x.com/nathancgy4/status/2081762692109639859

## LEAD 2: FlashKDA — https://github.com/MoonshotAI/FlashKDA

- **What it is:** CUTLASS implementation of Kimi Delta Attention kernels (MIT, SM90+, CUDA 12.9+, PyTorch 2.4+, K=V=128 head dims required). Drop-in backend for `flash-linear-attention` >= 0.5.0, auto-dispatched from `chunk_kda`. 
- **KDA itself:** linear attention refining Gated DeltaNet with **channel-wise (per-channel) forget gates**; the core of Kimi Linear 48B-A3B (paper: https://arxiv.org/abs/2510.26692, claims ~75% KV reduction and up to ~6x decode throughput at 1M) and of K3.
- **Numbers:** 1.72x-2.22x prefill speedup over fla's Triton baseline on H20 (https://x.com/Kimi_Moonshot/status/2046607915424034839; benchmarks exist for H20 and GB200). The *decode* side is a separate fused kernel (conv + recurrent update + gate + norm) co-developed with NVIDIA, integrated in vLLM/SGLang, not in this repo.
- **What KDA implies for memory:** each KDA layer carries a **fixed dk x dv state per head instead of a KV cache**. For K3: SGLang gives ~54MB per request per TP=8 shard covering all 69 KDA layers, i.e. **~432MB total FP32 state per sequence, constant at any context length** (my math: 96 heads x 128x128 x 4B x 69 layers ≈ 434MB, checks out).
- **llama.cpp KDA support today:** better than expected. Mainline already has `LLM_ARCH_KIMI_LINEAR` graph + converter, `ggml_gated_delta_net` already supports KDA's per-channel gate mode, hybrid recurrent/attention memory infra exists (Qwen3-Next lineage), and pwilkin has "Add Kimi Linear to unified delta net" work in CI (https://github.com/ggml-org/llama.cpp/actions/runs/22125274911; feature issue https://github.com/ggml-org/llama.cpp/issues/16930). What llama.cpp does NOT have is a fused KDA decode kernel; it runs on generic ggml ops.
- **Single-box port path:** state management is a solved problem in llama.cpp; a CUDA fused-decode-kernel port of FlashKDA-style fusion is a real fork opportunity. **PHASE-2** (speed, not fit). Mission-fit: **7/10** (enabler for K3; the fit benefit comes from KDA architecture itself, see question c).

## LEAD 3: Kimi K3 — https://www.kimi.com/blog/kimi-k3

**Architecture (confirmed across blog + HF card https://huggingface.co/moonshotai/Kimi-K3 + runpod FAQ https://www.runpod.io/articles/guides/kimi-k3-technical-faq):**
- 2.78T total / **104.2B active** per token. 93 layers: **69 KDA + 24 Gated MLA, 3:1 interleave** (3 KDA then 1 MLA, final layer MLA). Hidden 7168, 96 heads, NoPE (position from KDA gating/decay, that's how 1M works without RoPE rescaling). Vocab 160K, ctx 1,048,576. Vision: MoonViT-V2 401M.
- MoE: **896 routed experts, 16 active, +2 shared**; "Stable LatentMoE": router projects tokens to a **latent dim 3584 (half of 7168) before dispatch**, expert hidden dim 3072. My math: ~33M params/expert/layer, ~99% of all weight mass is experts (~2.75T), ~17.5MB per expert at MXFP4. Per token, 16x93 = 1488 routed expert-layer hits ≈ **26.1GB of expert bytes touched per token** (zero-reuse upper bound).
- SiTU-GLU activation, AttnRes (cross-depth residual attention, ~25% training-efficiency gain at ~2% compute), Quantile Balancing (expert allocation from router-score quantiles; note: this *trains toward balance*, which weakens static hotness skew, an important caching caveat), Per-Head Muon. Claimed 2.5x scaling efficiency over K2.
- **Quant: MXFP4 weights / MXFP8 activations with QAT from SFT onward.** Disk: **~1.56TB MXFP4** vs 5.6TB BF16 (this is our 1.5TB download). MXFP4 = 4-bit FP + per-32-block E8M0 scale ≈ 4.25bpw, native on Blackwell.
- **MTP/draft:** a pre-trained MTP layer fine-tuned into an **EAGLE-3-style draft** fusing features from the 1st/4th/final AttnRes blocks; served via **DSpark** speculative decoding (see DeepSeek section; yes, Moonshot adopted DeepSeek's framework).
- **Serving economics:** $0.30/MTok cache-hit vs $3.00 cache-miss input, $15 output; >90% cache hit in coding; official recommendation "supernode configurations with 64 or more accelerators"; minimum 8x B300 (2.3TB) per runpod FAQ. Recommended engines: vLLM, SGLang, TokenSpeed (+ cookbooks). Tech report PDF: https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf. License: "Kimi K3 License" (modified-MIT-style).

**Day-0 serving intel:**
- vLLM day-0 (https://vllm.ai/blog/2026-07-27-k3, preview https://vllm.ai/blog/2026-07-22-kimi-k3-preview): FlashKDA prefill + fused NVIDIA decode kernel; MXFP4 TRTLLM-Gen + DeepGEMM paths with SiTU wired in; bs1: 111 tok/s (TP8) -> **331 tok/s with DSpark = 3.14x**; hybrid prefix caching over recurrent KDA state via copy-on-write state blocks (off by default, `--enable-prefix-caching`).
- SGLang day-0 (https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support): **the authoritative memory math**: "one small MLA KV block per token (about 27 KB, covering all 24 MLA layers)" and "one large KDA state block per request (about 54 MB under TP=8)". Unified pool: KDA states fill from one end, MLA KV from the other. bs1 ~113 -> **~423 tok/s with DSpark**; ReplaySSM cuts draft-window state memory 32x (512KB -> 16KB); 2808 tok/s per GPU disaggregated. MXFP4 served natively (no restore-to-BF16), rollout peak 225GiB/GPU.
- AMD day-0 on MI355X: https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html

**llama.cpp status (the lane that matters for us):**
- Groundwork discussion: https://github.com/ggml-org/llama.cpp/discussions/26041 (gap analysis: KIMI_LINEAR + gated-delta-net + hybrid memory + `GGML_TYPE_MXFP4` all already in mainline).
- **PR #26185 "Kimi K3 (text)" by pwilkin: OPEN and WORKING** (https://github.com/ggml-org/llama.cpp/pull/26185). Implements KDA variant (full-rank `ssm_g` gate), Gated MLA output gate, AttnRes (`attn_res_block_size`), LatentMoE (`n_expert_latent`), SiTU, **lossless MXFP4 byte repack** (0 mismatched elements / 11,010,048 tested), `LLAMA_MAX_EXPERTS` raised 512 -> 1024. Independent tester @thispwd (Jul 28): converted checkpoint **loads and generates coherently with hybrid GPU+CPU offload `-ngl 99`**. Logit correlation 1.0 vs transformers+fla reference. Limitations: `ggml_dsv4_hc_pre` op is CPU+CUDA only (note: it reuses a DeepSeek-V4 hyper-connection op, evidence of shared plumbing); awaiting CISC/ggerganov approval.
- **GGUFs already exist:** GrEarl/Kimi-K3-GGUF (https://huggingface.co/GrEarl/Kimi-K3-GGUF): Q2_K experts / Q4_K dense = **928GB at 2.671bpw**, built against PR #26185, "No released llama.cpp can load this" yet; also Kuberwastaken/Kimi-K3-GGUF and unsloth/Kimi-K3-GGUF (https://huggingface.co/unsloth/Kimi-K3-GGUF, quant files not yet listed at fetch time). Caveat to carry: Q2_K re-quantized *from MXFP4 QAT weights* is double quantization; our own loop-rate finding (weight bit-width drives looping, PPL blind to it) applies with force at <=2.7bpw; evals mandatory.
- **Mission-fit: 10/10.** This is the target model and the port surface.

---

## KEY QUESTION (a): Expert offload/paging, the `--n-cpu-moe` evolution

**What exists in llama.cpp land (all open, none merged):**
1. **RFC+PoC discussion #23324 "MoE offload to disk with on-demand paging"** (kisasexypantera94, May 19, 2026, https://github.com/ggml-org/llama.cpp/discussions/23324). Design: per-MoE-layer compact **pool tensor of N slots**; kernel publishes selected expert IDs; **CPU sidecar thread does LRU + pread + slot remap** via shared memory; `MUL_MAT_ID` unchanged. Needs `--no-mmap` + `--no-warmup`. Numbers (Qwen3-30B-A3B-Q6_K, M3 Pro): 80 slots = 29.1 tok/s saving 8.3GiB; 32 slots = 22.1 tok/s saving 16.6GiB; ran on M1 Pro 16GB at 13 tok/s. Constraint: `ub * n_expert_used <= n_slots`. Author found SLRU no better than LRU. **CUDA datapoint (koren1712, July 2026, RTX 4080): mmap paging 0.43 t/s -> async reads + pinned blocks 0.75 t/s; overlapped reads sustained 2.80 GB/s vs 377 MB/s through mmap page faults. That 7.4x I/O gap is the whole argument for building real paging instead of leaning on mmap.** No maintainer response yet.
2. **Issue #20757 "Two-tier GPU+RAM expert cache"** (e1n00r, Mar 19, 2026, closed unimplemented, https://github.com/ggml-org/llama.cpp/issues/20757). Tier1 VRAM persistent slot buffer (SLRU: ~20% probationary / 80% protected), Tier2 pinned RAM, Tier3 mmap NVMe with `POSIX_MADV_WILLNEED`/`DONTNEED`. Python PoC on GPT-OSS-120B, 8GB GPU: cold 1.9-2.5 t/s at 48-56% hit -> **steady state 12-14 tok/s at 98-100% hit** vs 0.5-1 t/s plain CPU offload. Author explicitly lacked C++ chops; free design for us.
3. **vLLM RFC #38256 "Incremental MoE Expert Offloading"** (https://github.com/vllm-project/vllm/issues/38256, PR 1 at #37190, CI passing): per-expert GPU cache + pinned-CPU backing; **LFRU** scoring `freq / (clock - last_access + 1)` to protect "hub" experts (measured: GPT-OSS-20B experts E13@L11, E26@L15 carry 50%+ of traffic); async H2D on CUDA streams + **cross-layer prefetch using layer-N router logits to prefetch layer N+1**; NVMe mmap tier in PR 3; fixed slot addresses + persistent mapping tensor for CUDA-graph/torch.compile compat (zou3519's guidance). Production numbers: GPT-OSS-20B MXFP4 on an 8GB RTX PRO 2000: 30 tok/s stable, **97-100% hit**, expert loads cut O(seq x topk) -> O(n_experts).
4. Baseline practice today: `--cpu-moe` / `--n-cpu-moe` static splits + `-ot` regex (DocShotgun guide https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide, gist https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0). Static-by-layer only; no per-expert anything. Also: plain mmap already gives OS-page-cache NVMe paging at 4KB granularity, which is the day-0 "it technically runs" mode for the 928GB GGUF (expect ~0.5-1.5 t/s per koren1712's mmap datapoint scaled to our Gen5 drive).

**What a fork implementation needs (synthesis of the three designs):**
- Per-expert slot pools per layer on each GPU (granularity 17.5MB for K3 at MXFP4); MUL_MAT_ID row remap; fixed addresses for CUDA-graph safety.
- Async I/O thread pool: io_uring/pread at high queue depth from NVMe into pinned staging, side-stream `cudaMemcpyAsync` H2D. (Our hardware: 9100 Pro ~14GB/s, plus the free Gen5 x4 slot from the MonkeyAMD storage audit -> a second drive striped ≈ 28GB/s of expert bandwidth. GPUDirect Storage/cuFile is the direct NVMe->VRAM option on Blackwell.)
- Eviction LFRU/SLRU (hub protection), per-layer partitioned.
- Prefetch: cross-layer router correlation (free, same-token) + speculative lookahead (below).
- Papers for the design space: **MoE-Infinity** (activation-aware, sequence-level; https://arxiv.org/html/2401.14361v2), **FineMoE/EuroSys'26** (fine-grained selection patterns + semantic prompt hints: **-47% latency, +39% hit rate**; https://arxiv.org/html/2502.05370v2, PDF https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf), **Two-Stage Expert Offloading** (temporal+spatial+**domain** locality, cross-layer routing correlations; https://www.researchgate.net/publication/400888358_Two-Stage_Expert_Offloading_for_Domain-Aware_MoE_Inference), **Diff-MoE SC'25** (priority-driven differential caching for batching; https://dl.acm.org/doi/10.1145/3712285.3759903), **PIPO** (pipelined consumer-device offload; https://arxiv.org/pdf/2504.03664), **FloE ICML'25** (below), **SMoE expert substitution** (https://arxiv.org/pdf/2508.18983).
- **KTransformers** (https://github.com/kvcache-ai/ktransformers, K2.5 doc doc/en/Kimi-K2.5.md): the industrial precedent: 1T-class K2.5 on 2x4090 (48GB) + **600GB RAM**, RAW-INT4, `--kt-num-gpu-experts 30`, AVX512F CPUs. Its assumption (all experts fit RAM) is inverted on our box (big VRAM, small RAM), which is precisely why the NVMe tier work is ours to do. Note their AMX fast path is Intel-only; Zen4/5 AVX-512 still works.

**Mission-fit: 10/10. This is the critical path.**

## KEY QUESTION (b): Active-parameter serving + expert locality numbers

The honest, slightly uncomfortable evidence:
- **Natural next-token temporal locality is WEAK on well-balanced MoE.** MoE-SpeQ (https://arxiv.org/html/2511.14102) measured expert-activation entropy "consistently high, close to the theoretical maximum" (Phi-3.5-MoE, Qwen1.5-MoE, DeepSeek-V2-Lite). K3's Quantile Balancing explicitly trains routing toward balance, so expect weak *static* skew too.
- **But prediction beats locality.** MoE-SpeQ's INT4 self-draft predicts the big model's experts at **90.9% accuracy**, and speculative lookahead prefetch yields **96.25-99.85% cache hit** (32GB->16GB cache) and up to **3.3x / 2.34x** vs offloading baselines on PCIe4 (32GB/s). This is the "speculative-decode-aware prefetch" the mission asked about, published with numbers.
- **Hub experts exist in some models regardless:** vLLM RFC measured 2 experts carrying 50%+ of traffic (GPT-OSS-20B), and e1n00r's GPT-OSS-120B PoC hit 98-100% steady-state. Model-dependent; must be *measured per model*, which for us is a one-day llama.cpp router-logging patch on K3 (nobody has published K3 routing stats yet, first-mover data opportunity).
- **Session/domain locality is the second lever:** FineMoE (+39% hit via prompt-semantic hints), Two-Stage (domain-aware prefetch), VisMMOE (visual-expert affinity, https://arxiv.org/pdf/2605.05899).
- **EPLB** (https://github.com/deepseek-ai/EPLB) is indirect lab evidence of skew at datacenter scale: DeepSeek replicates "heavy-loaded experts" from moving-average load stats (hierarchical + global policies). Port path: same rebalance math, but output = "which experts get pinned in VRAM vs RAM vs NVMe" on one box.
- **The strategic synthesis for our fork:** we already have DSpark (PR #25173 lane) and MTP infrastructure. The draft model's routing IS the prefetch oracle: while verifying token t, the draft's tokens t+1..t+k tell you which experts to pull. K3 even ships its own EAGLE-3-style MTP draft. Speculative decoding stops being PHASE-2 speed here; it is the fit-enabling I/O scheduler.

**Mission-fit: 9/10.**

## KEY QUESTION (c): KDA/linear state at 1M, is K3 more fittable than GLM at long context?

Numbers (SGLang day-0, authoritative):
- K3 @ 1M tokens: MLA latent KV ~27KB/token total across 24 MLA layers (~1.125KB/layer/token, consistent with DeepSeek-style kv_lora 512 + 64 rope at BF16) = **~27GB at 1M (BF16), ~13.5GB at FP8**. KDA state: **~432MB per sequence, constant**. Conv states + AttnRes prefill chunk add single-digit GB. (Disregard the "256GB at 1M" figure floating in secondary blogs, e.g. kingy.ai; the SGLang per-token block math contradicts it.)
- GLM 5.2 (744B, full/DSA attention in every layer): O(N) KV in all layers; at 1M it is tens-to-hundreds of GB depending on KV quant, and it competes with weights for the same 192GB VRAM.
- **Verdict:** per-token-of-context K3 is radically cheaper (3 of every 4 layers carry zero per-token state, plus NoPE extrapolation designed for 1M). But it does NOT flip the overall fit verdict: GLM 5.2 fits VRAM at ~1.9bpw with room for KV; K3's weights keep it NVMe-bound regardless of context. Correct framing: **KDA removes the second wall (context), leaving only one wall (weights) for the paging work to break.** Also note Qwen3.5's GDN hybrid (same 3:1 pattern) claims ~4x KV reduction and 8.6x/19x decode at 32K/256K vs Qwen3-Max (https://qwen.ai/blog?id=qwen3.5), and Kimi-Linear claims 75% KV cut / ~6x decode at 1M (https://arxiv.org/abs/2510.26692). The whole frontier is converging on hybrid-linear precisely to make 1M serveable.

**Mission-fit: 8/10 (it is why K3-class + 1M on one box is even thinkable).**

## KEY QUESTION (d): Lab-native extreme quant

- **Moonshot:** K3 = **first frontier-scale MXFP4-QAT open release** (MXFP4 W / MXFP8 A from SFT onward; 1.56TB). Precedent: Kimi K2 Thinking shipped native INT4 QAT (~594GB). Implication: **4-bit is now the labs' floor and it is FREE quality (QAT'd); everything below 4-bit remains community territory** (our IQ/TQ ladder). llama.cpp K3 PR does **lossless MXFP4 byte repack** into `block_mxfp4`, zero requant loss; same pattern as gpt-oss and the DeepSeek-V4 converter. Rule for us: preserve MXFP4 where it fits, requantize only the tiers that must shrink, and eval loop-rate (double-quant from FP4 masters is uncharted).
- **DeepSeek:** V4 ships **native "FP4 + FP8 Mixed"** (experts FP4, rest FP8), MIT (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). The loFT run below consumed the native FP4+FP8 GGUF at 4.39bpw.
- **Qwen:** official ladder for Qwen3.5-397B-A17B: FP8 block-128 (https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8), GPTQ-Int4 (https://huggingface.co/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4), NVIDIA NVFP4 (https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4). No official sub-4-bit from any of the three labs.
- **Mission-fit: 7/10** (mostly confirms our sub-4-bit R&D lane is uncontested space).

---

## LAB MINES

### DeepSeek
- **V4 family:** V4-Pro **1.6T total / 49B active**, V4-Flash **284B / 13B**; hybrid **CSA (compress 4 tokens -> 1, 8-token window) + HCA (attention over 128x-compressed tokens + sliding window)** + lightning indexer + mHC hyper-connections + Muon; at 1M context needs only **27% of FLOPs and 10% of KV vs V3.2** (https://deepseek.ai/deepseek-v4, https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). DSA lineage: V3.2 paper https://arxiv.org/pdf/2512.02556; FlashMemory lookahead-sparse long-context paper https://arxiv.org/pdf/2606.09079.
- **llama.cpp: V4 is MERGED upstream** (PR #24162 by am17an, merged June 29, 2026: CSA, HCA, indexer, iSWA; follow-up #24231 indexer ops; https://github.com/ggml-org/llama.cpp/pull/24162). Unsloth GGUFs exist (https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF).
- **Twin-box proof:** loFT LLC ran V4-Flash on **exactly our hardware** (2x RTX PRO 6000 Blackwell Max-Q 96GB): native FP4+FP8 GGUF, 4.39bpw, 146GB, fully in VRAM, **PP 38.3 / TG 35.7 tok/s** with FA disabled and GPUs loafing at 30-40% util (https://loftllc.dev/en/docs/tech/llm-research/deepseek-v4-flash-llama-cpp-blackwell-local-inference/). Community field data: heiketu serves V4-Flash Q2 at **~30 tok/s on dual Xeon + 2x3090** with a modified llama.cpp + DSpark (https://huggingface.co/moonshotai/Kimi-K3/discussions/59).
- **V4-Pro 1.6T is the natural stepping stone between GLM and K3:** at ~2bpw ≈ 400GB, only ~80-100GB over our static budget; light paging or REAP closes the gap on an architecture that is already merged upstream.
- **DSpark/DeepSpec:** confidence-scheduled semi-autoregressive speculative decoding (anchor-based block attention + Markov head), **+60-85% per-user on V4-Flash, +57-78% on V4-Pro over MTP-1**; full MIT training/eval stack released as DeepSpec (paper https://arxiv.org/pdf/2607.05147, code map https://deepwiki.com/deepseek-ai/DeepSpec/4.1-dspark-architecture, https://venturebeat.com/orchestration/deepseek-open-sources-dspark-a-new-framework-to-speed-up-llm-inference-by-up-to-85). Adopted by K3 (3.14x vLLM, ~3.7x SGLang bs1). For us: PHASE-2 as speed, PHASE-1 as the expert-prefetch oracle.
- **EPLB** https://github.com/deepseek-ai/EPLB (see question b). **3FS** (NVMe-centric FS with KVCache tier) is cluster tech; the single-box takeaway is only the io_uring/O_DIRECT high-QD discipline.

### Moonshot
- MoonEP, FlashKDA, K3: covered above.
- **Mooncake** (https://github.com/kvcache-ai/Mooncake, FAST'25 best paper): KV-centric disaggregation; **Store SSD-offload RFC #171** (https://github.com/kvcache-ai/Mooncake/issues/171): local mode = DRAM buffer + SSD spill on one node, POSIX vs SPDK paths, microsecond-class lookup penalty; SGLang HiCache uses it as L3. Port path for us: llama.cpp session/prefix KV on NVMe (save/restore + reuse at 1M) primarily for **GLM-class** models; K3's KV is small enough not to need it. Mission-fit 5/10.
- Kimi-Linear repo: https://github.com/MoonshotAI/Kimi-Linear.

### Qwen/Alibaba
- **Qwen3.5-397B-A17B** (our Ornith lineage): 60 layers, 512 experts (10 routed + 1 shared active), hidden 4096, **GDN linear-attention hybrid 3:1**, 262K ctx, 397B/17B; decode 8.6x/19x vs Qwen3-Max at 32K/256K (https://huggingface.co/Qwen/Qwen3.5-397B-A17B, https://huggingface.co/blog/mlabonne/qwen35). Same hybrid playbook as Moonshot, independently confirming the trend.
- **Qwen3.6** (27B dense + 35B-A3B so far): GDN hybrid + **MTP out of the box, 1.4-2.2x** low-latency decoding; vLLM recipes + Unsloth guides (https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://recipes.vllm.ai/Qwen/Qwen3.6-27B, https://unsloth.ai/docs/models/qwen3.6). Their serving guidance normalizes MTP-default-on, same direction we bet on.
- **Expert pruning:** **REAP (Cerebras, ICLR 2026)**: one-shot router-weighted expert pruning, **50% of experts removed near-losslessly on Qwen3-Coder-480B AND Kimi-K2** for generative/code/agentic tasks, no retraining, code at https://github.com/CerebrasResearch/reap. Live example: OpenMOSE/Qwen3.5-REAP-262B-A17B (397B -> 262B, https://huggingface.co/OpenMOSE/Qwen3.5-REAP-262B-A17B). The K2 result is direct evidence it transfers to Moonshot's fine-grained MoE lineage. Related 2026 papers: SlimQwen (https://arxiv.org/pdf/2605.08738), AIMER calibration-free (https://arxiv.org/pdf/2603.18492), ConMoE (https://arxiv.org/pdf/2605.29350), unified expert-scoring (https://arxiv.org/pdf/2606.15716).
- **FloE (ICML'25)** (https://arxiv.org/abs/2505.05950): keep ~9.3x-compressed expert copies resident + low-cost sparse prediction; 48.7x vs DeepSpeed-MII on a 3090, Mixtral in 11GB, 4.4-7.6% quality cost. The portable idea: **never stall decode; if the MXFP4 expert hasn't arrived from NVMe, compute with the resident ultra-low-bit copy.** Math for K3+REAP-448: an ~1.8bpw resident copy of all remaining experts ≈ 310GB ≈ exactly our VRAM+RAM, with MXFP4 "quality tier" streamed from NVMe. This merges cleanly with our existing one-file/two-profiles quant-frontier lane.

---

## THE SQUEEZE MATH FOR OUR BOX (K3, derived, flag as estimates)

- Static fit: impossible. 2.671bpw GGUF = 928GB; even 1.0bpw = 348GB > 320GB total fast memory.
- Naive mmap paging: ~21.7GB of expert misses/token at 17% residency -> ~0.4-1.5 t/s. Day-0 curiosity only.
- **Recipe R1 (near-term, all parts exist):** REAP 50% (896->448 experts, ~730GB at MXFP4; or ~464GB at 2.67bpw) + per-expert paged cache (~250GB across VRAM+RAM = 34-54% residency) + LFRU + async NVMe pipeline: uniform-random floor 1.6 t/s, with hub/session locality + cross-layer prefetch at 85-95% hit -> **0.9-3.9GB/token from NVMe -> ~5-12 t/s decode**, DSpark on top multiplies accepted tokens per fetched working set. Quality gate needed (REAP x sub-4-bit double-quant is unmeasured; loop-rate eval mandatory).
- **Recipe R2 (the fork thesis):** full 896 experts, MXFP4 preserved, FloE-style two-tier experts (resident ~1.5-1.8bpw fallback + streamed MXFP4) + DSpark-driven prefetch. Novel enough to be a first-of-its-kind local-serving result.
- Hardware note the data forces: every working K3-class local setup found (KTransformers 600GB, loFT 768GB, heiketu 512GB+, Laststar 1.5TB) is RAM-heavy. 128GB RAM is our binding constraint; the second Gen5 NVMe slot (stripe to ~28GB/s) is the cheapest bandwidth doubler if RAM stays fixed.

## PHASE-2 (pure speed, park until fit is solved)
FlashKDA fused-kernel port to ggml-CUDA; MoonEP zero-copy/static-shape EP for the 2-GPU pair; DSpark throughput tuning beyond prefetch duty; fused MXFP4 grouped-GEMM paths (TRTLLM-Gen/DeepGEMM equivalents); TokenSpeed (Moonshot's third recommended engine, no public repo found worth mining yet).

**Bottom line:** the field has already built and measured every component of a K3-class single-box serving stack except the integration, and llama.cpp is one open PR away from loading the model. The critical path is per-expert paged serving with speculative prefetch; REAP and two-tier expert precision are the levers that turn it from ~1 t/s physics into ~5-12 t/s reality.