# Elastic memory for the paged KV cache: what the world already knows

Research commissioned by the owner after he noticed a 2.3 GB model filling a 128 GB Mac. His
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
`not planned`.** The two pain points it names are exactly the owner's two directions:

1. **Idle:** pre-allocated memory can't be freed for other workloads.
2. **Burst:** allocation can't grow to absorb a spike.

The stated technical difficulty is **which blocks to evict, and under what policy**.

> **So: this is not a gap in our fork's competence. It is an open problem in the leading
> engines, and one of them explicitly declined to solve it.**

## 2. The emerging answer is OS-style virtual memory — which is precisely the owner's model

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

**S1 — verify commit-on-demand. ⚠ Measure before designing.**
The 88.79 GB figure was **RSS**, i.e. *resident*, which means something is touching the whole
pool at startup rather than leaving it lazily committed. If true, this is the single cheapest
win available: an over-large pool becomes harmless, because untouched blocks cost address space
rather than RAM. This is kvcached's core idea, and on unified memory we may get most of it free
from the OS's own lazy commit. **Not yet measured — do this first.**

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
