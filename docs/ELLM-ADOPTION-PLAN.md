# eLLM for our fork — thought-experiment pass, before any code

Paper: *eLLM: Elastic Memory Management Framework for Efficient LLM Serving*, arXiv 2506.15155
v2 (2026-05-06), DAC '26. PDF held locally at `_private/papers/ellm-2506.15155.pdf` (gitignored).
**No public source code.** Built on vLLM v0.5.5, ~4000 lines of C++ and Python.

Following the owner's process: **thought experiments and planning first, kill the bugs and the
wrong assumptions before writing code**, converge, then implement, then diff code against plan,
then verify. This document is the first pass. **Nothing here is implemented.**

---

## 1. What eLLM actually claims

**The problem it names: "space-wise internal fragmentation."** Activations and KV cache live at
*different abstraction levels* — activations are static tensors from the DL framework, KV is
page-virtualized (PagedAttention). Because of that split the two spaces **cannot share memory**:
when KV is exhausted while activation space sits idle, the idle space cannot be repurposed.
Cost: **~20% throughput**.

Their quantification (A100, MTP-30B): raising max context **2K → 200K** pushes the activation
reservation from **0.3% → 30.8% of GPU memory**, while **99% of ShareGPT prompts are under 2K**
and >90% of prefill batches use <30% of the max context. In decode, activation utilisation is
**~1%**.

**Three components:**

1. **eTensor** — virtual tensor abstraction, two flavours.
   - *KV eTensor*: reserves virtual address space equal to **max context length**; physical
     blocks mapped **on demand during writes**. Justified by KV being large, regular,
     predictable, low-access-frequency, persistent.
   - *Activation eTensor*: variable-sized virtual segments, small blocks, short lifespans, high
     access frequency.
   - Both align to physical block granularity ("tensor slots"). Slots are **not unmapped at end
     of life** — marked mapped+available and reused. KV uses **Best-Fit** (smallest slot ≥ s);
     activations keep the framework's **Best-Fit with Coalescing**.
2. **Elastic memory** — *inflation*: KV allocation short ⇒ borrow from the activation pool via
   lightweight GC that unmaps physical chunks of inactive activation eTensors, transfer
   ownership, remap to KV virtual addresses. *Deflation*: the reverse, done **lazily**.
   Plus **GPU↔CPU offload/fetch**, using CPU DRAM as an elastic buffer.
3. **Lightweight scheduling** — phase-specific (prefill = activation-heavy ⇒ offload KV to CPU;
   decode = activation-light ⇒ fetch KV back), plus **SLO-aware logical buffer scaling**: a
   *logical* size inside a fixed physical buffer, so it can shrink **without evicting data**.
   TPOT violation ⇒ shrink (throttles prefill); TTFT violation ⇒ grow. A violation fires when
   the metric breaches SLO **3 times in a 5-iteration window**; tuning factor α (default 2).

**Results:** up to **2.32×** throughput; TTFT up to 295×/140× better than vLLM/vLLM-CP on
Llama3-8B-262K, one A100; 2.5×/2.26× goodput. Overheads hidden via *decoding speculative
pre-mapping* (<50 MB) and *asynchronous unmapping* (one physical chunk mapped to several virtual
addresses so the new slot is usable immediately).

---

## 2. ⛔ THOUGHT EXPERIMENT 1 — does their problem even exist in llama.cpp? **Largely NO.**

This is the assumption that decides whether the whole paper is worth implementing, so it gets
checked first rather than last.

**Verified in our tree** — `src/llama-context.cpp:698`, `llama_context::sched_reserve()`:

```cpp
const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
```

**llama.cpp reserves the compute (activation) buffer for `min(n_ctx, n_ubatch)` tokens.** With
`-ub 512` that is **512 tokens regardless of whether `-c` is 8K or 1M**. vLLM reserves activation
space sized to the *maximum context*; that difference is the entire fragmentation eLLM attacks.

**Consequence, and it is decisive:**

- Their borrowable pool is **30.8% of GPU memory** at 200K context. Ours is the compute buffer at
  ub=512 — hundreds of MB against a KV pool of tens of GB at long context, i.e. **order 1%**.
- Inflation/deflation between activation and KV would therefore recover **~1% of memory, not
  ~30%**. The headline **20% throughput / 2.32×** rests on the large number.

⇒ **eLLM's central mechanism does not transfer.** We do not have the disease it cures. Adopting
the eTensor + inflation/deflation architecture would be ~4000 lines to chase a percent.

### ✅ MEASURED — and my reasoning above was half wrong, in an interesting direction

I flagged "is the compute buffer flat in `-c`?" as the assumption to kill. It **is not flat on
the static path**, so the reasoning as written was incomplete. But measuring it produced a
better result than the argument it broke.

`-ub 512` held fixed, qwen3-4b:

| path | ctx=8,192 | ctx=131,072 | growth |
|---|---|---|---|
| STATIC | 75.01 + 18.01 = **93 MiB** | 195.01 + 138.01 = **333 MiB** | **3.6x** |
| **PAGED** | 67.01 + 10.01 = **77.02 MiB** | 67.05 + 10.05 = **77.10 MiB** | **1.001x — FLAT** |

**Mechanism:** `sched_reserve` uses `min(n_ctx, n_ubatch)` for the token *count*, but the graph
still contains tensors dimensioned by context — above all the **attention mask, `n_ubatch x
n_kv`**, which grows linearly with `-c`. Both static buffers grew by exactly +120 MiB over the
same +123K of context, which is the signature of one mask-shaped term.
**The paged path has no dense mask** — it uses block tables — so its activation footprint does
not move with context at all.

⇒ **This inverts the eLLM analysis in our favour.** eLLM's "space-wise internal fragmentation"
(activation reservation growing with max context) **does exist in llama.cpp — on the STATIC
path.** And **our paged path already solves it structurally**, by construction, for free.

⇒ Extrapolating the linear mask term to **1M context**: static would carry roughly **+2 GB** of
activation memory that paged does not. That is a real, measured, north-star-relevant advantage of
the paged path which was not previously recorded anywhere.

⇒ And it settles the adoption question: **we do not need eLLM's eTensor machinery to get eLLM's
main benefit, because we already have the benefit.** The remaining gap to their 30.8% figure is
theirs to close, not ours.

(Related prior finding: [[graph-nodes-not-compute-buffer]] — `n_tokens*40` is a graph NODE
budget, ~30 MB, not the compute buffer. Distinct from this measurement; do not merge them.)

## 3. THOUGHT EXPERIMENT 2 — does the SLO scheduler help *the owner's* workload? **Mostly no.**

The SLO policy throttles **admission of prefill requests among concurrent requests** to protect
TPOT. Its lever is *how many prefills to admit*.

At **`-np 1`** — the owner's configuration, and *required* for his 1M Ornith runs (otherwise the
context is split across slots) — there is exactly one sequence. **There is nothing to schedule.**
The policy would run, observe, and have no admission decision to make.

⇒ Valuable for **multi-tenant serving**; near-zero for a single-user long-context box, which is
the north star. Keep it, but rank it far below anything that helps a single long request.

**⚠ Nuance worth preserving:** the *logical buffer* idea — shrink usable size **without evicting
stored data** — is elegant and reusable. It is the right shape for S2 (pressure reclaim): under
memory pressure, stop *handing out* blocks before you start *taking back* blocks. That is a
hysteresis primitive we want regardless of eLLM.

## 4. THOUGHT EXPERIMENT 3 — what *does* transfer?

**(a) GPU→CPU KV offload with layer-wise overlap. ← the one real win.**
Their argument is independent of the activation question: KV migration is **O(N)** while prefill
self-attention is **O(N²)**, and the transformer's layer structure lets transfer overlap
compute. They report the overhead **completely hidden** on A100 + Llama3-8B.

We already have the *mechanism*: a CPU block pool (`n_cpu_blocks`,
`cpu_to_gpu_blocks_ratio` default 0.25) and a block manager with a watermark. **What is missing
is the policy and the overlap** — when to spill, when to fetch, and doing it asynchronously
per-layer instead of synchronously.

⚠ On **unified memory** the calculus changes and must be re-derived, not copied: there is no PCIe
hop, "GPU" and "CPU" memory are the same physical RAM. Offload may be nearly free (just a
bookkeeping move) **or entirely pointless** (no capacity is actually freed). **This is the single
most important open question for us**, and it is cheap to answer with a measurement.

**(b) Best-Fit slot reuse + do-not-unmap-immediately.** Applies directly to our block manager,
independent of VMM. Small, safe, portable.

**(c) Speculative pre-mapping for decode.** Needs VMM ⇒ blocked on Metal (sparse is texture-only,
see the research doc), open on CUDA/DGX Spark.

## 5. Open questions to resolve *before* any implementation

1. ~~Is the compute buffer flat in `-c`?~~ **ANSWERED — see §2.** Static grows 3.6x over a 16x
   context increase (the attention mask); **paged is flat**. The paged path already avoids
   eLLM's fragmentation, worth ~2 GB at 1M context versus static.
2. **On unified memory, does CPU offload free anything real?** If GPU and CPU blocks are the same
   RAM, the answer may be no — which would delete §4(a) too, leaving very little.
3. **Does the paged path even support `-np > 1` well today?** If not, the SLO scheduler is moot
   twice over.
4. What is our actual block-allocation failure mode under pressure — does the block manager
   currently *block*, *evict*, or *fail*? Determines whether a logical-buffer throttle is even
   expressible.
5. MLA and SWA sizing (already flagged in the research doc) — unrelated to eLLM but blocks any
   claim that the pool sizing is universally correct.

## 6. Provisional ranking — deliberately *not* a commitment

| rank | item | why | blocked? |
|---|---|---|---|
| 1 | Answer open questions 1 and 2 | they can delete most of this document | no |
| 2 | S2 pressure reclaim + logical-buffer throttle | helps single-user *and* multi-user; the elegant bit of eLLM | no |
| 3 | CPU offload policy + layer-wise overlap | real win **iff** Q2 says offload frees capacity | gated on Q2 |
| 4 | Best-Fit slot reuse | small, safe, portable | no |
| 5 | SLO-aware scheduling | multi-tenant only, near-zero at `-np 1` | no, but low value |
| 6 | eTensor / inflation / deflation | **the paged path already avoids the fragmentation** (measured flat) | VMM-blocked *and* now redundant |

**The honest summary: eLLM is an excellent paper solving a problem we mostly do not have.** Its
2.32× is not available to us because the fragmentation it removes is an artifact of vLLM
reserving activations for max context, which llama.cpp does not do. What survives is the CPU
offload policy (conditional on Q2) and the logical-buffer idea. **That is worth a few hundred
lines, not four thousand.**

Credit: Xu, Xiong, Zhang, Guo, Liu, Zhou, Hu, Wu, Li, Zhao, Guo, Zhu, Zhou, Leng — eLLM,
arXiv 2506.15155, DAC '26.
