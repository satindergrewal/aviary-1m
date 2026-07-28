# Research sweep: speculative decoding at long context (2026-07-28)

Mission frame: big models in expected state on local iron, context to its limits. This sweep
covers the drafting/verification side. Companion sweep (KV compression + DSA sparse attention)
lands separately. Sources include papers with and without code, serving-stack PRs, and
llama.cpp-family issues. Findings tagged PHASE-2 SPEED are banked, not actioned, per the
fit-first ruling.

## The regime math that frames everything

Speculation value at depth = accepted_len x T_target / (T_draft + T_verify), where T_target at
depth = weights-load + KV-load per token.

- **GLM-5.2 IQ1_S (~165GB weights, MLA KV ~100KB/tok)**: weights-bound even at 128K (~12.5GB
  KV). Tiny drafters (MTP/DSpark) remain the right tool; compressed-KV self-drafting adds
  little at bs1.
- **Qwen3.6-27B Q8 (~28GB weights, GQA KV ~160KB/tok f16)**: KV-bound well before the context
  limit (~80GB KV at 500K). Depth wins live in drafter-training and self-spec-with-small-KV;
  sparse verification is the eventual 1M endgame.

## Ranked candidates (mission-fit vs port cost)

### 1. Depth-robust drafter training bundle (VERY HIGH / medium cost, we own the pipeline)
Apply to our DeepSpec/DSpark training:
- **EAGLE 3.1 attention-drift fixes** (vLLM PR #42764, May 2026): FC-normalize target hidden
  states before fusion + feed post-norm states into subsequent draft steps. Evidence: up to 2x
  acceptance length at long context. Runtime delta small (our fork has EAGLE3); the real prize
  is the training-side adoption. https://vllm.ai/blog/2026-05-26-eagle-3-1
  Supporting: "Attention Drift" https://arxiv.org/pdf/2605.09992 , KVShot caution on KV-reuse
  drafting https://arxiv.org/abs/2604.26412
- **LongSpec** (sail-sg, ICLR/ACL 2026): constant-sized draft KV (sliding window + anchors),
  position-index scheme = train short, serve at 500K-1M. 3.26x evidence. Code + data released.
  https://github.com/sail-sg/LongSpec
- **BudgetDraft** (Jun 2026): acceptance-aware multi-budget training = budget-robust drafter.
  6.55x@4K / 2.10x@16K. Training-recipe only. https://arxiv.org/pdf/2606.00144
  Same philosophy as our quant-tax/GGUF-capture doctrine, applied to KV sparsity.

### 2. SuffixDecoding / SAM upgrade of the ngram type (HIGH / low-medium cost)
Suffix tree over prompt + generated outputs; ~20us/token CPU drafting, depth-invariant, zero
VRAM, exact distribution. vLLM-merged (arctic-inference, Nov 2025). **Unclaimed in
llama.cpp-land** (ik_llama #1602 is an implementation sketch nobody built). Agentic loops =
best case. v0 = greedy linear path into existing propose/verify; v1 = tree mask.
https://arxiv.org/abs/2411.04975 https://github.com/ikawrakow/ik_llama.cpp/issues/1602
SAM-Decoding's adaptive retrieval-vs-model-drafter selector is worth copying as a meta-policy:
https://github.com/hyx1999/sam-decoding

### 3. Self-spec with compressed KV (HIGH for dense/KV-bound; training-free)
- **MagicDec** (ICLR 2025): target as own drafter with StreamingLLM cache (sinks+window ~512);
  draft cost constant in depth; 2.51x at 32K-100K batch>1. Port: second context on same
  weights (MTP machinery precedent) + sink-keeping ring eviction (context-shift exists).
  Mind second-context compute buffers (our measured MTP-context ceiling effect).
  https://github.com/Infini-AI-Lab/MagicDec
- **QuantSpec** (ICML 2025, stretch): hierarchical INT4/INT8 shared KV, draft reads INT4 half,
  verify reads INT8, no duplicate cache; >90% acceptance, ~2.5x. Kernel-heavy.
  https://github.com/SqueezeAILab/QuantSpec

### 4. Depth quick wins (MEDIUM-HIGH / small cost) — partially DONE
- ~~Reproduce #23658 (MTP acceptance collapse at ctx boundaries)~~ **ALREADY CLOSED BY US
  (F3, 2026-07-26): bit-identical acceptance across the issue's -c values on BOTH draft-dspark
  AND draft-mtp paths; issue numbers not reproducible; closed as noise.** The sweep flagging it
  independently confirms our F3 was worth running.
- **PEARL-style async draft/verify overlap** (ICLR 2025): pre-verify + post-verify remove the
  mutual stall — the paper-shaped version of our measured ~18ms/tok sync stall (worst
  dual-GPU). With drafter pinned GPU1 / target GPU0 the phases genuinely overlap. [PHASE-2
  SPEED] https://github.com/smart-lty/parallelspeculativedecoding
- Adaptive draft-length-at-depth policy: cheap, composes with everything.
- ik_llama #732: draft-context smaller than target context degrades hard at depth — check our
  fork's -cd handling at depth.

### 5. Sparse verification lane (HIGHEST 1M ceiling / high cost, no reference code) [R&D]
Dustin (draft-augmented sparse verification: 9.17x e2e at 32K, near-lossless, no code)
https://arxiv.org/html/2606.24957v1 ; SpecPV (partial-KV verify + periodic full flush, 6x,
not strictly lossless) https://arxiv.org/abs/2512.02337 ; Vegas; SpecSA. Needs ggml per-head
sparse-attention kernels — same kernel family as Quest/TriForce. Watch for code releases;
pairs with the DSA-indexer question in the companion sweep.

## Watchlist (revisit on trigger)
- **MoE-offload x speculation** (SpecMoEOff 2.5x, SP-MoE expert prefetch via draft tokens):
  THE lane for 2T+ giants on 192GB+RAM when those weights land.
- **Mamba/SSM drafters** (depth-invariant state; llama.cpp runs the archs already; a
  vocab-matched small SSM drafter = near-zero-cost experiment via --model-draft).
- GliDe/KVShot KV-reuse drafting (blocked: acceptance gains don't yet convert e2e).
- TokenSwift (100K-token marathon generation), RAPID (RAG-drafter), SpecExtend, CLaSp.

## Ecosystem intel
- vLLM: suffix decoding merged; EAGLE 3.1 shipped; speculators lib growing; dLLM RFC #36155.
- SGLang: spec roadmap #23005; EAGLE acceptance bugs at scale (#16274, #18251).
- Nobody in llama.cpp-land has MagicDec/LongSpec/suffix-tree drafting. First-mover room real.
