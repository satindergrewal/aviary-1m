# Making llama.cpp serve parallel agents properly: research and work list

Opened 2026-07-28 on Satinder's brief. The driving case: running the solar prompt
under `ultracode --xhigh + workflows`, which fans out many agents in parallel. With
`-np 1` those agents serialised behind one another and the run took multiples of the
time it should have, while nvtop showed **40-58% GPU utilisation**. Idle bandwidth
plus blocked callers is a serving-concurrency limit, not a compute limit.

His hypothesis, which this document is meant to test: **the GLM 5.2 64K setup we
already have would handle that workflow comfortably if the concurrency path were
done properly.**

---

## 1. What we already have, and what it does not cover

`--kv-unified` is now the mainline default and it is better than we credited:

```c
// src/llama-context.cpp
if (cparams.kv_unified) { cparams.n_ctx_seq = cparams.n_ctx; }   // per-seq cap = FULL n_ctx
else                    { cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max; }
```

Measured on the GLM server: 4 slots, each reporting the full 65,536. So requests
share one arena instead of getting fixed equal partitions, and an idle slot's space
is reusable.

**What it is not.** A contiguous cell arena with per-cell `seq_id` tags. There is no
block table, no non-contiguous allocation, no copy-on-write, no GPU/CPU swap. It is
dynamic *sharing*, not dynamic paged *allocation*. Its ceiling is still one `n_ctx`
worth of cells in total.

The gap this leaves is exactly what the benchmark shows: unified OOMs at 26
sequences, paged reaches 247.

---

## 2. State of the paged-KV work upstream

matiaslin's draft PR **#22569**. Read the thread directly rather than trusting
summaries:

**Declared Phase-1 restrictions:** single CUDA device only, full offload only,
`n_batch == n_ubatch`, no SWA architectures (gemma3, llama4...).

**Benchmarks (A10G, Llama-3 8B f16, n_batch=n_ubatch=1024):**

| path | n_seq | agg tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| unified | 25 | 496 | 159 | 50 |
| unified | 26 | **OOM** | | |
| paged | 25 | 479 | 163 | 50 |
| paged | 247 | **1256** | 791 | 179 |
| paged | 248 | **livelock** | | |

So ~2.5x aggregate throughput peak-vs-peak, ~3% overhead at low concurrency, and a
**livelock at 248** that is an unfixed bug, not a limit.

**Maintainer position, which matters for planning.** ngxson called the PR "slop",
argued paged attention mainly helps when requests share a prompt prefix, and said it
"will likely be rejected". am17an questioned whether ">25 parallel requests" justifies
the complexity since "most users of llama.cpp are not going to run that many".
ggerganov, slaren, JohannesGaessler and compilade were pinged and never replied. The
bot also flagged it for touching two backends in one PR and for undisclosed AI use.

**Consequence: nobody upstream is going to fix any of this.** If we want it, we own it.

**Explicitly unimplemented (author's own Phase 2 list):** copy-on-write, prefix
caching, `seq_cp` / `seq_keep` / `seq_div` / `seq_add`, `state_write` / `state_read`.

---

## 3. The multi-GPU blocker, and why it is tractable

The PR states "single CUDA device only" as a Phase-1 *restriction* and gives **no
technical reason**. Reading the code, there is not much of one.

`llama_kv_cache_paged::init` already allocates **one tensor per layer**:

```cpp
for (uint32_t il = 0; il < n_layers; ++il) {
    auto * t = ggml_new_tensor_4d(ctx_gpu, type, head_dim, block_size, 2*n_heads_kv, n_gpu_blocks);
    kv_gpu_layers.push_back(t);
}
ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
```

Every layer already has its own tensor. They are simply all placed in a single
context bound to a single backend.

And llama.cpp's `--tensor-split` splits **by layer**, not by attention head. That is
strictly easier than vLLM's tensor-parallel case, where each GPU holds a head-slice
of every layer.

**Design (mirrors vLLM: one logical block table, per-device physical storage):**

1. Replace the single `ctx_gpu` / `backend_gpu` with one context per device.
2. Place layer `il`'s tensor in the context belonging to `model.dev_layer(il)`.
3. Allocate each context against its own backend.
4. Size `n_gpu_blocks` from the **most constrained** device so a single block id is
   valid on every device. Uniform block count keeps the block table device-agnostic.
5. Make the CPU-swap path target the correct device per layer.
6. Delete the rejection in `validate_paged_kv_placement`.

Block ids, block tables, the scheduler and the attention kernel are **unchanged**,
because each layer is already addressed as `kv_gpu_layers[il]` regardless of where it
lives. This is a contained change to allocation, not a redesign.

---

## 4. What the mature engines do that we do not

Ordered by expected value for agentic fan-out, which is our actual workload.

**a. Prefix caching / RadixAttention.** Every agent in a workflow shares the same
system prompt, tool definitions and often a large shared context. vLLM hashes blocks
and reuses them across requests; SGLang keeps a radix tree over the KV cache so
common prefixes are computed once. For our case this is probably the single largest
win available, and it is precisely the thing ngxson said paged attention is *for*.
Not implemented in the PR (it is Phase 2).

**b. Continuous batching with chunked prefill.** Both vLLM and llama.cpp run a
tick-driven loop, but chunked prefill splits a long prompt into pieces and interleaves
them with other slots' decode tokens, so one long prompt cannot freeze every other
request. Trade-off is real: smaller chunks smooth inter-token latency but raise TTFT
and cut total throughput (Sarathi-Serve). Worth measuring, not assuming.

**c. Copy-on-write block sharing.** Forked sequences (beam search, parallel samples,
agent retries from a common state) share blocks until one writes. Phase 2, and it
exists in the author's local branch.

**d. Centralised scheduler with admission control.** The PR has an FCFS scheduler with
token-budget admission and deadlock detection, which is more than mainline has. The
livelock at 248 suggests the admission logic has a hole.

**e. GPU/CPU block swapping.** Present in the PR's block manager. Untested by us.

---

## 5. Work list

Ordered so each item is verifiable before the next depends on it.

| # | item | why | state |
|---|---|---|---|
| P0 | Port onto current mainline | 294 commits of drift | **done**, `fork/kv-paged`, control-verified |
| P1 | CUDA build + run the test suite | separates "CPU reference is sloppy" from "paged read path is wrong" | approved, not started |
| P2 | Metal build + run | Satinder wants Apple Silicon covered too | approved, not started |
| P3 | **Multi-GPU (layer-split) support** | hard requirement; GLM needs both cards | designed above, not started |
| P4 | Diagnose the token-1 e2e divergence | pre-existing, reproduces in the author's own branch | open |
| P5 | Diagnose the livelock at 248 seqs | admission-control hole | open |
| P6 | Prefix caching across agents | biggest expected win for workflow fan-out | not started |
| P7 | Profile and optimise the CPU reference path | author labels it unoptimised | not started |
| P8 | Chunked prefill interaction | measure TTFT/ITL trade-off | not started |
| P9 | Security review | Satinder asked explicitly: attack vectors, loose ends | not started |

---

## 6. Adjacent finding: the TurboQuant fork

`AtomicBot-ai/atomic-llama-cpp-turboquant`, built on `TheTom/llama-cpp-turboquant`.
No paged-KV or multi-GPU work, but it lands on three of our existing lanes and we
already have a tree at `/mnt/nvme0/llama.cpp-tq`.

**TurboQuant is WHT (Walsh-Hadamard) rotated low-bit quantization**, which is our C0
vhad lane and our B2 rotate-then-recalibrate lane, already productized:

- KV formats `TURBO2_0` / `TURBO3_0` / `TURBO4_0` (2/3/4-bit, ~6.4x / 4.3x / 3.8x vs
  F16), with a Metal `TurboFlash` FA decode kernel
- Weight formats `TQ3_1S` / `TQ4_1S`, WHT-rotated **Lloyd-Max**, block_size 32.
  Claimed "~25-35% size reduction vs Q8_0 with single-digit-% PPL deltas". Direct
  comparison target for our KT trellis quants.

**And they have already built what `mtp-fix-plan.md` proposed.** Depth-2 async
pipeline via `llama_decode_mtp_async` / `llama_decode_mtp_wait` so draft compute
overlaps verification, plus "in-graph argmax reduces host transfers to 4 bytes per
step vs full vocab row". That is Fix A from our plan, shipped. Claimed +28-36% on
Qwen 3.6 35B-A3B MoE, +30-50% short-prompt on Gemma 4.

Unverified by us. But if it holds, it is a reference implementation for work we had
scheduled as "hours to days" of our own effort.

---

## Sources

- [llama.cpp PR #22569, paged KV cache](https://github.com/ggml-org/llama.cpp/pull/22569)
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM paper)](https://arxiv.org/pdf/2309.06180)
- [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/pdf/2412.19442)
- [From Tensor Buffer to Distributed Memory Hierarchy: KV Cache Management survey](https://arxiv.org/pdf/2607.02574)
- [vAttention: Dynamic Memory Management without PagedAttention](https://arxiv.org/pdf/2405.04437)
- [From Tokens to Layers: Stall-Free Scheduling for MoE Serving with Layered Prefill](https://arxiv.org/pdf/2510.08055)
- [FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity](https://arxiv.org/pdf/2602.16603)
- [LLM Serving Optimization: Continuous Batching, PagedAttention, Chunked Prefill](https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/)
- [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant)
- [AtomicChat/Kimi-K3-GGUF](https://huggingface.co/AtomicChat/Kimi-K3-GGUF)

---

## TurboQuant re-research (2026-07-29): read from their quantizer, not their card

Satinder pushed back twice on my downgrade. He was right to. From
`ggml-turbo-quant.c` in the tree on the box:

**The pipeline:** normalize, forward WHT rotation, then quantize against **fixed
Lloyd-Max centroid tables** (16-entry for 4-bit, 8 for 3-bit, non-uniform spacing
tight near zero: a Gaussian-optimal codebook). Post-WHT values are near-Gaussian by
the central limit theorem, so **the codebook is fitted to the post-rotation
distribution**. Third independent confirmation of why our B2 failed: we rotated and
then used codebooks built for unrotated weights.

**The trick I missed when I downgraded it** (line 594):

```c
/* No inverse WHT, dequant stays in the rotated domain.
 * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores. */
```

WHT is orthogonal, so dot products are preserved under it. They rotate Q once per
step (tiny) and compute attention **in the rotated domain**, meaning the K cache is
never unrotated at decode. My "adds an inverse-WHT node per attention layer" cost
criticism applies to the **V path only** (`turbo_cpu_fwht_inverse` exists for it);
the K path is genuinely free.

**What to adopt into our stack:**

1. **The post-rotation codebook principle.** Settled; any future rotation work fits
   the codebook after rotating, never before.
2. **Rotated-domain K attention.** Design insight worth carrying into any KV-quant
   work of ours, independent of TurboQuant's formats.
3. **A real open test:** the KV survey says never take K below q8_0, but that
   evidence is for UNROTATED K. Rotated K is more Gaussian and quantizes better, so
   "3-bit rotated K vs q8_0 unrotated K" is a live question, not settled by the
   survey. If rotated-3-bit holds quality, K-cache drops ~2.7x at 1M context, which
   serves the north star directly.

**Still true from the first pass:** upstream rejected the PR; the weight-format gain
over our types is modest. The value is the KV formats and the two ideas above, not
the weight quants.

---

## StreamIndex scoping (2026-07-29): the 1M-context indexer lane, measured claims

From arXiv:2605.02568 directly, not the sweep's summary.

**What it is.** A chunked partition-merge top-k driver for sparse-attention
indexers. Instead of materializing the full `[B, S, H_I, T]` score matrix, it
streams chunks and merges top-k results. This is EXACTLY the tensor that we
identified as GLM-DSA's compute-buffer ceiling (the indexer score tensor that FA
never covers).

**Measured claims (theirs, H200):**
- 1,048,576 context at **6.21 GB peak HBM**, where the materialize approach OOMs
  already at 65,536 (a 256 GB intermediate at V4-Flash dims)
- Set-overlap recall vs materialized ground truth: bit-exact at small S,
  **0.9980 minimum recall** across their sweeps
- End-to-end at S=262,144 with a TileLang attention kernel: 1.97 s at 18.56 GB
  peak, where materialize OOMs

**Reference implementation:** Triton, `github.com/RightNow-AI/StreamIndex`,
targets H200.

**Their own stated limitation, keep it honest:** "Our contribution targets the
indexer step; we make no claim of a faster attention kernel or of real-checkpoint
end-to-end behavior." So it is an indexer-memory fix, not an attention speedup.

**Port considerations for us:**
1. Triton is Python-side; llama.cpp needs it as a CUDA (or Metal) kernel. 349 LOC
   of Triton is a readable spec, not a drop-in.
2. The recall caveat: 0.9980 minimum is NOT bit-exact at large S. For a top-k
   indexer feeding sparse attention, a 0.2% index miss is probably benign but must
   be MEASURED on GLM output, not assumed (cell zero: compare generations, not
   just recall).
3. Convergence with Fable-DSpark's DSA lane: their fairydreaming sparsemma arm
   validates the sparse top-k FA path on Blackwell (2888/2888). StreamIndex would
   feed that same path with a bounded-memory indexer. The two compose.
4. The realistic sequencing: first re-measure our ACTUAL compute buffer on
   GLM-5.2-ours-IQ1_S (provenance of the 22.8 GiB figure is bad), then decide
   whether the indexer term is still the binding one before porting anything.
