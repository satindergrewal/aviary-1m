# Elastic memory for the paged KV cache: what the world already knows

Research commissioned by Satinder after he noticed a 2.3 GB model filling a 128 GB Mac. His
framing — *"this should be elastic; if the system is asking for resources, free some; if
resources are free, expand; go learn how VMs and swap systems actually do it"* — turns out to
match where the field is actively moving, and to name a problem the big engines have **not**
solved.

## 1. The state of the art in LLM serving is *static*, and everyone is unhappy about it

**vLLM** pre-allocates a fixed fraction of GPU memory at startup via `gpu_memory_utilization`.
Once reserved it is **never released** and cannot change without a process restart.
**SGLang** does the same thing with `--mem-fraction-static` (default 0.9), with the documented
remedy for OOM being "use a smaller value" — i.e. hand-tuning.

Concrete user pain, from the vLLM forum:

- 128k context ⇒ ~20 GB KV + ~14 GB weights + 2–3 GB overhead ⇒ **~37 GB reserved up front**.
- `cpu_offload_gb` "significantly degrades throughput" because layer choice is arbitrary.
- KV-cache quantization "not available in v1".
- Lowering `max_model_len` or `gpu_memory_utilization` works but **sacrifices the context
  length**, which is the whole point.
- Official answer: *"there is no v1 feature to dynamically shrink the KV pool or offload only
  the KV cache without performance loss."* **Restart is the only way to reclaim.**

**vLLM issue #18125, "Elastic KV memory management based on usage", was closed as
`not planned`.** The two pain points it names are exactly Satinder's two directions:

1. **Idle:** pre-allocated memory can't be freed for other workloads.
2. **Burst:** allocation can't grow to absorb a spike.

The stated technical difficulty is **which blocks to evict, and under what policy**.

> **So: this is not a gap in our fork's competence. It is an open problem in the leading
> engines, and one of them explicitly declined to solve it.**

## 2. The emerging answer is OS-style virtual memory — which is precisely Satinder's model

**kvcached** (ovg-project) — *"brings OS-style virtual memory abstraction to LLM systems"*:
decouple **GPU virtual addressing** from **physical allocation**. Reserve virtual address
space up front, back it with physical memory only when blocks are actually touched. That
enables on-demand growth and multi-model GPU sharing.

**eLLM** (arXiv 2506.15155) — the academic treatment, and the most directly instructive:

- **eTensor**: KV tensors reserve virtual address space for the *maximum* context, with
  physical blocks committed on write. Activation tensors use variable-sized virtual segments.
- **Inflation / deflation**: when KV space runs short it *borrows* from the activation pool via
  the GPU virtual memory manager; deflation returns it lazily to avoid churn.
- **SLO-aware scaling policy** — this is the part worth stealing: **three TPOT violations in a
  scheduling window ⇒ shrink; TTFT violations ⇒ expand.** Feedback control on observed latency,
  not on a guessed constant.
- **Overhead: <1% CPU scheduling, 1–5% for VMM operations**, via mapping reuse and async ops.
- Quantified waste in the static approach: raising max context 2K → 200K pushes the activation
  reservation from **0.3% → 30.8% of GPU memory** while **99% of real prompts are under 2K**,
  costing **~20% throughput**.

## 3. The virtualization world solved a version of this decades ago — with known traps

**virtio-balloon** (KVM/QEMU, Firecracker, Cloud Hypervisor, Proxmox): the host "inflates a
balloon" inside the guest; the guest allocates those pages and reports their addresses; the
host then decommits the backing memory. **Free page reporting** inverts it — the guest
proactively tells the host which pages it has freed.

The failure modes are documented and directly relevant:

- **Ballooning lies to the guest.** Windows guests see the ballooned memory as *in use*, report
  ~99% RAM usage, and respond by **dropping their own caches to disk** — a performance
  collapse caused by the reclaim mechanism itself.
- **"Inflating too much slows down the guest; not enough slows down the host."** There is no
  free lunch; it is a control problem with a cost on both sides.

**Lesson for us:** a naive "give memory back under pressure" loop can trigger exactly the
thrash it was meant to prevent. Any reclaim must be *hysteretic* (different thresholds for
shrink and grow) and rate-limited, or it oscillates.

## 4. On unified memory, the balloon problem is *inverted* — and macOS hands us the signal

On a discrete GPU, VRAM is a private pool; grabbing it all is defensible and is why vLLM does
it. On Apple unified memory **there is no separate pool — we are the guest *and* the host**,
and nobody else will reclaim on our behalf.

macOS provides the exact hook: **`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`**, delivering
`NORMAL` / `WARN` / `CRITICAL`. Apple's own guidance maps cleanly onto a block cache:

| level | Apple's guidance | our action |
|---|---|---|
| normal | no action | may grow toward the ceiling |
| warn | "relax your caching policy" | stop growing; evict cold blocks to CPU; return free blocks |
| critical | "expect latencies, consider dropping all caches" | release every unpinned block immediately |

**Two documented limitations we must design around:**

- If the process starts while pressure is **already elevated**, it will *never* be notified.
- **There is no way to query the pressure level "right now"** — it is edge-triggered only.

⇒ Therefore the startup budget must still be computed conservatively (we cannot ask the OS how
bad things are), and the pressure source is a *runtime correction*, not the primary control.
Linux equivalent: cgroup v2 **PSI** (`memory.pressure`) plus `memory.high` as a soft limit.

## 5. Problem list — what people actually want fixed

1. Reclaim without restart (vLLM: impossible today).
2. Grow under burst without over-reserving for the worst case.
3. Offload *only* the KV cache, without the arbitrary-layer throughput cliff.
4. Stop trading context length for memory safety.
5. Share a GPU between models/processes (kvcached's motivating case).
6. Avoid the activation-vs-KV fragmentation that eLLM measures at 20% throughput.
7. Not lie to, or thrash, the surrounding system (the balloon trap).
8. **Ours, and not on anyone else's list because unified memory is niche in serving:** do not
   take the user's desktop down. Static llama.cpp already refuses cleanly; paged did not.

## 6. What we apply, in order — each independently shippable and gateable

**S0 — bound and refuse (DONE today, fork `9f7006b9` + reserve policy).**
Pool sized from `n_ctx`, not free VRAM. Reserve is an absolute floor on unified memory
(`max(8 GiB, 15% RAM)`) instead of 5% of total, which was 6.4 GiB on a 128 GB Mac and an absurd
0.8 GiB on a 16 GB one. When the request cannot fit: refuse and **name the largest `n_ctx` that
would**, matching how the static path behaves, with `--paged-pool-clamp` to shrink instead.

**S1 — commit-on-demand. ❌ MEASURED AND REFUTED as a cheap win (fork `a391331c`).**
The pool *is* eagerly resident: RSS 58.67 GB before any request on a 20x-headroom pool, 58.70
after 3000 tokens. But `ggml_backend_buffer_clear` is **not** the cause — skipping it entirely
gives 58.67 vs 58.68 GB, i.e. no change. **The residency path wires it**: `rset addAllocation`
+ `requestResidency` in `ggml-metal-device.m` make the buffer resident whether or not anything
touches it. So there is no free lazy commit here; S1 requires **sparse or placement `MTLHeap`**,
which remains unresearched and is now the gating question for this whole direction.

**What S1 did establish, and it is worth having:** a poison probe (`DS4P_KV_POISON`, fills every
pool with NaN) proves **nothing ever reads an unwritten KV block** — output sha bit-identical to
the zeroed control, markers confirming the fill ran. The scalar path bounds its token loop
instead of reading-then-masking. **Every lazy-allocation or reclaim scheme depends on exactly
this property**, and it is now tested rather than assumed.

⚠ Also learned: **`test-paged-vs-cpu` never constructs a paged KV cache at all.** It is a kernel
gate. It cannot answer allocation questions in principle, and three separate "ALL PASSED"
results from it were meaningless before a marker exposed that.

**S2 — pressure-driven reclaim.** Subscribe to `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` (macOS)
and cgroup PSI (Linux). warn ⇒ stop growing + spill cold blocks to the CPU pool we already
have; critical ⇒ release all free blocks. **Hysteresis and rate limits are mandatory** — the
Windows-guest balloon collapse is the cautionary tale.

**S3 — grow on demand.** Start at a modest fraction of the budget and expand toward the ceiling
as occupancy demands, instead of committing the maximum up front. Pairs with S1.

**S4 — SLO feedback (eLLM's contribution).** Drive S2/S3 from *observed* TTFT/TPOT rather than
memory thresholds alone. Deferred until S1–S3 are measured; it is the most speculative piece.

## 7. Honest caveats

- kvcached's exact CUDA VMM calls are not on its README; the `/csrc` directory and its arXiv
  paper are the next read if we pursue S4-style VA/physical decoupling.
- Metal has no direct `cuMemCreate`/`cuMemMap` analogue. `MTLHeap` (placement/sparse) is the
  candidate and is **unresearched** — do not assume it works.
- eLLM's 1–5% VMM overhead is *their* number on *their* hardware; it is a plausibility argument
  for us, not a prediction.

**Sources:** vLLM issue [#18125](https://github.com/vllm-project/vllm/issues/18125) ·
[vLLM forum thread on non-reclaimable KV](https://discuss.vllm.ai/t/vllm-v1-forces-me-to-pre-allocate-a-huge-non-reclaimable-gpu-kv-cache-for-long-contexts-and-none-of-the-current-offload-or-quantization-options-solve-the-resulting-vram-bloat-without-crippling-speed/1502) ·
[vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/) ·
[kvcached](https://github.com/ovg-project/kvcached) ·
[eLLM (arXiv 2506.15155)](https://arxiv.org/html/2506.15155v2) ·
[SGLang memory management](https://deepwiki.com/sgl-project/sglang/2.3-memory-management-and-caching) ·
[Firecracker ballooning](https://github.com/firecracker-microvm/firecracker/blob/main/docs/ballooning.md) ·
[Cloud Hypervisor memory techniques](https://www.cloudhypervisor.org/blog/memory-management-techniques/) ·
[virtio-balloon guest-cache trap](https://github.com/virtio-win/kvm-guest-drivers-windows/issues/568) ·
[DISPATCH_SOURCE_TYPE_MEMORYPRESSURE](https://developer.apple.com/documentation/Dispatch/DISPATCH_SOURCE_TYPE_MEMORYPRESSURE) ·
[xnu memorystatus_notify](https://github.com/apple-oss-distributions/xnu/blob/main/doc/vm/memorystatus_notify.md)


---

# Round 2 — Satinder's questions answered, and a correction I owed him

## 0. ❗ CORRECTION: unified memory is NOT niche. It is the direction of the industry.

Round 1 called unified memory "niche in serving". **That is wrong.** DGX Spark and DGX Station
are unified CPU/GPU memory, Satinder already runs a two-Spark cluster, and Windows-on-ARM is
converging on the same architecture. Apple was simply first at volume.

⇒ This **raises** the priority of everything below rather than making it a Mac curiosity, and
it means a solution must be **portable**, not Metal-specific. Note that DGX Spark is CUDA, so
the CUDA VMM route is open *there* even though it is closed on Apple.

## 1. Should we adopt kvcached instead of building our own? — **No, and not because of NIH.**

Satinder's instinct ("why reinvent the wheel, and I don't want to fork/maintain it") is right in
principle. Three hard blockers, from its own docs:

1. **It is a Python library that monkey-patches Python engines.** `pip install kvcached`, then
   `ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1`. "No engine changes needed" means *no changes to
   vLLM/SGLang* — both Python. **llama.cpp is C++.** There is nothing to autopatch.
2. **It is CUDA-only**: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`.
   No Metal path exists, so it cannot serve the Mac at all.
3. It is a **daemon + library** for sharing one GPU between *several* engine processes. Our
   problem is one process sizing itself sanely, which is a different problem.

**So there is no seamless integration to have, and forking it would mean rewriting it in C++
for a GPU API it does not target — exactly the maintenance burden he wants to avoid.**

**What we take instead: the technique, which is published and unencumbered.** The same idea has
three independent write-ups — kvcached, **vAttention** (arXiv 2405.04437), and **vTensor**
(arXiv 2407.15309) — and there is an open vLLM request for it (issue #17612). We implement the
*idea* where our platform allows, and we cite them.

## 2. eLLM — paper only, and yes it is worth planning for

**No public source code.** arXiv 2506.15155, v2 revised 2026-05-06, 14 authors, no repository
found. So "fully applying it" means implementing from the paper.

Reported results: **2.32x higher decoding throughput and 3x larger batch at 128K inputs.**
Its three parts split cleanly by cost to us:

- **Virtual tensor abstraction** — needs the same VMM primitives kvcached needs. Blocked on
  Metal (see §3); open on CUDA/DGX Spark.
- **Elastic inflation/deflation with CPU memory as an extensible buffer** — *we already have a
  CPU block pool* (`n_cpu_blocks`, `cpu_to_gpu_blocks_ratio`). This is the closest fit.
- **SLO-aware scheduling** (3 TPOT violations ⇒ shrink, TTFT violations ⇒ expand) — **portable,
  cheap, and needs no VMM at all.** It is pure policy over counters we already have. This is
  the piece to steal first, and it is S4 in the plan.

## 3. ❌ Metal sparse is TEXTURE-ONLY — the VA-reservation route is closed on Apple

Checked the Metal headers directly rather than the docs (the docs page is a JS app):

- `MTLHeapType`: Automatic, Placement, **Sparse** (macOS 11+).
- `MTLResourceStateCommandEncoder` exposes **only** `updateTextureMapping` and
  `updateTextureMappings`; the mode enum is `MTLSparseTextureMappingMode`.
- **There is no sparse *buffer* API anywhere in the headers.**
- Placement heaps do allow `newBufferWithLength:options:offset:`, but a placement heap's memory
  is committed when the **heap** is created — that controls *where* buffers land, not *whether*
  pages are backed.

⇒ **Reserve-VA-then-commit is not available for buffers on Metal.** S1 as originally conceived
is dead on Apple, and this is a platform limitation, not an implementation gap.

**The portable path that remains: a CHUNKED pool.** Allocate the KV pool as *many* smaller
buffers instead of one monolith, create them on demand, and release them when free. That gives
real grow-and-shrink at chunk granularity, needs no sparse or VMM support, and works identically
on Metal, CUDA and CPU. Less elegant than VA reservation, and it fragments at chunk granularity,
but it is **the one design that works on every backend we care about** — including DGX Spark.
**This becomes the new S1/S3.**

## 4. "Will this break other models that handle KV differently?" — partly already guarded

Legitimate worry, and one guard already exists: the paged path **refuses hybrid architectures
loudly** (`kv_paged is not yet supported for hybrid architectures`), added by this lane after it
used to be silently ignored and OOM. So Mamba-hybrids like Nemotron are rejected, not corrupted.

What my sizing change does and does not touch:

- It changes only the **block COUNT**. `bytes_per_block` is pre-existing and already carries a
  fix for the Qwen3 head-dim assumption (`n_head*head_dim != n_embd`), so any model whose
  per-block size was right stays right.
- The `n_ctx x n_seq x headroom` estimate **over-allocates** for sliding-window models, where
  not every layer needs full context. Over-allocation is the SAFE direction — it cannot
  under-provision and corrupt — but it is waste worth fixing when SWA support lands.
- **Not yet verified: MLA (DeepSeek-style compressed KV).** Its per-token footprint differs
  structurally. ⚠ Do not assume the sizing is right there; test before claiming it.

## 5. Revised plan

| stage | status |
|---|---|
| **S0** bound + reserve + refuse | **DONE** (fork `9f7006b9`, `bf25ebd5`) |
| **S1** ~~lazy commit via VA~~ → **chunked multi-buffer pool** | redesigned; portable, unblocked |
| **S2** pressure-driven reclaim (macOS pressure source, Linux PSI) | unblocked, needs only free-block release |
| **S3** grow-on-demand | falls out of S1-chunked |
| **S4** SLO feedback from eLLM | portable, no VMM needed, cheapest of the eLLM ideas |

**Extra sources this round:** [kvcached PyPI](https://pypi.org/project/kvcached/) ·
[kvcached.org](https://kvcached.org/) ·
[vAttention (arXiv 2405.04437)](https://arxiv.org/pdf/2405.04437) ·
[vTensor (arXiv 2407.15309)](https://arxiv.org/pdf/2407.15309) ·
[vLLM issue #17612 (vAttention request)](https://github.com/vllm-project/vllm/issues/17612) ·
[eLLM abstract](https://arxiv.org/abs/2506.15155) ·
[MTLHeapType](https://developer.apple.com/documentation/metal/mtlheaptype) ·
[Metal resource heaps guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/ResourceHeaps/ResourceHeaps.html)
