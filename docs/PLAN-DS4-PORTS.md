# PLAN — ds4 → llama.cpp PORTS (DeepSeek V4-Flash serving machinery)

**Status: DRAFT v2 for the owner's review. NO implementation until he says go (restated by him 2026-08-02).**
Author: Fable-DSpark | Date: 2026-08-02 | Work surface (exists, untouched): branch `ds4-ports` off
`dsv4-kvpaged` @ `3376ebbc8`, worktree `<BOX>/wt-ds4-ports` (box).

## ★ MISSION (his directive, 2026-08-02)
**Bring llama.cpp's serving layer to neck-to-neck parity with vLLM/SGLang's architecture — one
shared KV pool, no static lanes, dynamic admission, queue-instead-of-reject, prefix sharing,
preemption — and then keep going past parity** (ds4's extras: session forking, disk KV banks,
yield-quench). "I want to shatter that wall." Nothing here is a closed door: mechanisms land
behind runtime flags/kill switches so every model can A/B them, and deferred items stay on the
record with revisit triggers rather than being rejected.

## 0. His directive (verbatim, 2026-08-02)

"do deep research about that project, and make a list what can be ported and/or copied or
coded/modified in our fork of llama.cpp. make a copy of the current branch serving our current
deepseek v4 flash, and you can work on that copy of branch to make the changes you think helps make
better work in our setup. Also assess how much time/resources/efforts it is to make you work on them.
and don't start work on it until I have read your plan."

## 1. What ds4 is (provenance, verified by reading the source)

- `github.com/Entrpi/ds4`, branch `batched-serving` @ `82d2a6f` (v0.5.2), fork of antirez's
  DwarfStar (`e16ead1`, 495 commits ahead of upstream). Upstream has NO serving/batching — all of
  that is fork-only.
- Self-contained C+CUDA engine (~87k lines: `ds4.c` 40,944 / `ds4_cuda.cu` 29,484 /
  `ds4_server.c` 18,829 / `ds4_kvstore.c` 1,606 / `tools/ds4_weight_server.cu` 1,873).
- Serves **DeepSeek V4-Flash ONLY**, on **its own custom ~81 GiB quant** (IQ2_XXS gate/up,
  Q2_K down, Q8_0 dense, F16 LoRA/compressor) — that is how it fits a DGX Spark 128 GB.
- Claims (theirs, unverified by us): ~20 t/s decode on Spark, 6.2× throughput batch 1→128,
  766K concurrent tokens, ~7× TTFT from warm prefix records, ~49× fork fan-out at N=4,
  MMLU 79.5 / GSM8K 96.4 / HumanEval 88.4 on their stdlib harness.
- Research clone: Mac `/tmp/ds4` (read-only). No box copy exists anymore.

**Why mine it:** it is the only open-source engine that has solved, for THIS EXACT MODEL, the
multi-agent serving problems the owner hit (static slots, prefill stalls, dead clients burning
compute, no prefix reuse across sessions, RAM-only cache). Two independent research agents mapped
it: serving machinery (done) and CUDA kernels (in flight — §5 sections marked ◐ fill in when it
lands; the serving items stand on their own).

## 2. THE CANDIDATE LIST — ranked by value-per-effort for OUR box

Effort units: focused agent-days on the `ds4-ports` worktree, box GPU windows for gates.
Every item ships behind a kill switch and a measured gate; nothing touches `fleet`.

---

### P0-1. Yield-quench: per-request speculation governor — EFFORT 1–2 days — VALUE: makes DSpark/MTP self-tuning

- **What:** per-bank EWMA of speculative yield (α=0.125) + cumulative regret debt vs a calibrated
  guard of 2.16 break-even tokens/verify-step; after 4+ bad steps the request's speculation is
  quenched TERMINALLY (re-arming measured to lose). Forced-quench identity 1.000–1.004×;
  adversarial workload floor 0.72×→0.96× vs ungoverned.
- **ds4:** ds4.c yield-quench block (verify-loop instrumentation, EWMA/debt state per bank).
- **Our tree:** `src/llama-speculative.cpp` + per-slot state in the server. Our measured pain is
  exactly this: DSpark on the fast Q4_K_XL target = 0.40–0.65× (verify doesn't amortize) while the
  same head wins 1.65× on the slow Mac target — and today the only control is a global on/off.
  A per-request governor means: leave DSpark ON, requests that don't pay get quenched individually.
- **Kill gate:** (1) forced-quench byte-identity vs no-spec baseline; (2) A/B at acceptance 0.60
  (our V4 head) must auto-quench to ≤1.01× of baseline; (3) A/B at acceptance 0.85+ must NOT quench.
- **Risk:** LOW. Self-contained, no cache/graph changes.

### P0-2. Engine-side prefix re-validation — EFFORT 0.5–1 day — VALUE: closes the wrong-reuse class forever

- **What:** ds4 never TRUSTS a cache claim: warm records are looked up by byte-LCP, then the engine
  re-validates with memcmp before reuse; a mismatch degrades to cold compute, never to wrong reuse.
- **ds4:** warm-record lookup + engine memcmp path (ds4.c prefix-match code).
- **Our tree:** the prompt-cache match path (llama-memory + server slot logic). We cache TOKENS,
  so the port is even simpler than byte memcmp: re-validate the claimed token-prefix array against
  the request's actual tokens before accepting a match. This is the suspicion class from the solar
  loop night (cross-session cache churn) — exonerated but never structurally closed.
- **Kill gate:** zero measurable overhead on hits (one array compare per request); a corrupted
  cache file must degrade to recompute, not serve.
- **Risk:** VERY LOW.

### P0-3. Dead-client abort probe — EFFORT 0.5–1 day — VALUE: killed agents stop burning prefills

- **What:** zero-byte `MSG_PEEK|MSG_DONTWAIT` probe on the client socket at every abort point in
  the compute loop; a dead connection aborts that sequence's work at the next batch boundary.
- **ds4:** ds4_server.c (probe at every abort point; `alive()` callback into the engine).
- **Our tree:** tools/server — the slot/task loop already has cancellation; the gap is POLLING the
  socket during long prefills so a Ctrl-C'd Claude Code session doesn't run a 256K prefill to
  completion (his ECONNRESET night: the server kept working for a dead client).
- **Kill gate:** kill a client mid-256K-prefill → slot frees within one ubatch; live clients
  unaffected (probe cost is one syscall per batch).
- **Risk:** LOW.

### P0-4. Methodology adoption (process, not code) — EFFORT 0.5 day — VALUE: compounds on everything

- Same-boot ABBA A/B for every perf claim (kills frequency/governor drift confounds).
- Per-feature env kill switches (their `DS4_CUDA_FP8_KV=0` pattern) → our `DS4P_*` env gates so any
  ported feature can be A/B'd and killed without a rebuild.
- Byte-exactness counters as first-class gates (their "0/84M re-encode mismatches" style) — a port
  is not "working" until an exactness counter says so, not a log line (our announce-vs-verify scar).
- Standing gate scripts in-repo (`speed-bench/*.sh` equivalent) so gates re-run identically forever.

---

### P1-5. Disk-persisted, content-addressed KV banks with partial restore — EFFORT 2–4 days — VALUE: the 1M agentic unlock

- **What:** KV snapshots named `<sha1-of-rendered-prefix>.kv`; warm admit of an 80K prefix = 3.1 s
  vs 125 s cold prefill; v0.5.1 partial restore by byte-LCP (salvages ≥1/8 of a stale snapshot);
  6-hour hit-half-life eviction; payload v3 stores packed fp8/fp4 natively (2.3–3× smaller).
- **ds4:** ds4_kvstore.c (1,606 lines, self-contained) + KVC 48-byte header format.
- **Our tree:** we have slot state save/restore primitives and the `-cram` RAM cache (measured:
  8 GiB cap holds ONE 64K state; save #2 evicts #1 — the pain is real and quantified). The port:
  content-addressed disk tier behind the RAM cache, with partial-prefix salvage. For agentic V4 at
  256K–1M this turns "re-prefill the world per session" into "load the snapshot" (16-min 1M
  prefill → seconds).
- **Kill gate:** (1) restored state produces byte-identical logits vs in-RAM state;
  (2) warm admit ≥5× faster than prefill at 64K; (3) partial restore exactness counter = 0
  mismatches on the salvaged span.
- **Risk:** MEDIUM (new on-disk format + eviction policy; correctness must be proven by counters).

### P1-6. Session forking (prefix fork for agent fan-out) — EFFORT 2–4 days (on kv-paged) — VALUE: N agents from one prefix

- **What:** fork a live sequence at a partial prefix: rewind to replay base R=(n_cached−4) aligned
  to the compress ratio (128), then D2D-copy the KV (~10 MiB/1K tokens); refused above a 65,536-token
  pin threshold. Their number: ~49× cheaper than N independent prefills at fan-out N=4.
- **ds4:** ds4.c fork path (aligned rewind + D2D copy).
- **Our tree:** kv-paged gives us block-level KV management; fork = clone the block table for the
  shared span + copy-on-write or outright D2D copy. THE primitive for "orchestrator + N subagents
  sharing a 100K system+task prefix" — the owner's actual workload.
- **Kill gate:** forked session's next-token logits identical to an unforked continuation; fork
  cost < 5% of re-prefill at 64K.
- **Risk:** MEDIUM-HIGH (touches the paged cache's ownership model).

### P1-7. Chunked-prefill interleave audit ("R4 overlap-lite") — EFFORT 0.5-day audit, fix if gap — VALUE: prefills stop stalling decodes

- **What:** cold prefill chunked at 4096, live sessions interleave at 512-token quanta so one big
  prefill stalls live decodes by at most one quantum; non-final chunks skip the output head
  (no logits computed for prefill-only positions).
- **ds4:** ds4.c:26771–26785.
- **Our tree:** `-b/-ub 2048` already chunks prefill and interleaves decode tokens of other slots —
  this may already match. Audit: (a) measure decode t/s of a live slot WHILE a 256K prefill runs on
  another; (b) verify logits are only computed where requested. Fix only what the audit shows.
- **Kill gate:** live-slot decode degradation < 10% during a concurrent max-size prefill.
- **Risk:** VERY LOW (audit-first).

---

### P2-8. Continuous batching beyond static `-np` — EFFORT 5–10 days — VALUE: the "100s of agents" answer

- **What:** up to 128 banks (`DS4_MULTISEQ_MAX_SEQ`, ds4.c:94), `ds4_batch_ctx_create_fit` sizes the
  bank pool from FREE MEMORY at startup; FCFS admit pass; one batched step per loop for all live
  banks; coalescing of non-streaming requests (cap 16).
- **ds4:** ds4.c batch-context machinery.
- **Our tree:** server-context scheduler surgery — replace fixed slot array with a dynamic
  admit/evict pool on top of kv-paged's dynamic block allocation (the fitter already sizes the
  block pool; slots become just sequence state). This is the genuine vLLM-parity item and the
  biggest single effort. NOTE: kv-paged already removed the static-KV waste; what remains is
  dynamic slot COUNT and admission.
- **Kill gate:** 16-agent fan-out workload: total throughput ≥ 6× single-stream (their 6.2×
  claim at 1→128); per-agent latency ≤ 1.5× single-stream.
- **Risk:** HIGH (scheduler rewrite; where subtle bugs live). Recommend doing LAST, after P0/P1
  items have hardened the gate culture around this branch.

### P2-9. Compressed-KV native tier for V4 (bit-lossless e4m3) — EFFORT 3–6 days — VALUE: KV ÷2.76, deep decode −19–26%

- **What:** V4's compressed-KV lanes are NATIVELY e4m3-quantized values produced by the model
  itself, so ds4 stores 448 e4m3 codes + 64-lane F32 rotary + 7 F32 pow2 scales = 704+28 B/row
  vs 2048 B F32 — "bit-lossless" because storage matches what the model emits. Exactness hinges on
  `frexpf/ldexpf` pow2 scales (fast-math `exp2f(ceilf(log2f))` corrupted 10.5% of fp4 lanes) and
  pinned in-order `__fmaf_rn` chains. Verified 0/84M re-encode mismatches; deep decode −19%/−26%
  at 240K/516K ctx.
- **ds4:** ds4_cuda.cu KV encode/decode kernels.
- **Our tree:** new KV type in the paged cache + CUDA dequant-in-attention path. ◐ KERNELS MAP
  PENDING to reconcile with our measured 7.4 KiB/token V4 KV — if our F16 storage is 2× the native
  codes, this is pure win; if our MLA cache already stores compressed, the win shrinks.
- **Kill gate:** byte-exactness counter 0 mismatches; decode curve 64K–256K must not regress vs
  the kv-paged baseline curve (short 63.6 | 64K 52.3 | 256K 38.5).
- **Risk:** MEDIUM-HIGH (kernel work + numerics exactness).

---

## 3. DEFERRED — parked with revisit triggers, NOT rejected (his directive 2026-08-02:
## "don't shut the doors so early... find a way to inherit the advancements")

Principle he set: everything below lands eventually, behind runtime flags (`--ds4-*` family or
equivalent), default-off until gated, so ANY model can A/B them — a mechanism written for
DeepSeek today must be arch-generic where the physics allows.

- **D2R GEMM / SoA repack kernels** (was "not porting"): REVISED per his call. Value hypothesis
  unchanged (their edge = repack-once layout; the MTP small-batch dead zone is the prime target)
  but the door stays open: implement as an OPTIONAL kernel path (flag-gated), measure per model —
  some archs/quants may gain far more than MMQ gives. Revisit trigger: after P0/P1 lands, re-run
  the small-batch (2–8 col) GEMM curve on the then-current tree; if the dead zone persists, this
  becomes P1. Also: the repack-at-load pass itself is arch-agnostic infrastructure.
  **Technique library for the revisit (2026-08-02, see `docs/RESEARCH-HF-KERNELS-HUB.md`):**
  HF kernels-hub MoE small-batch kernels to mine BEFORE writing our own — `kernels-community/
  vllm-moe`, `drbh/yamoe`, `drbh/fused-moe`, `axolotl-ai-co/sonic-moe`, `nCompass-tech/
  triton-moe` (flashrt's grouped-moe-gemv is binary-only, skip).
- **Weight server (VMM `cuMemCreate` + `SCM_RIGHTS` fd-passing):** deferred, not dead. We run one
  server today; the moment we run TWO+ engines on the box (llama.cpp + a trainer, or dual-model
  serving), zero-copy weight sharing becomes real. Revisit trigger: any second resident engine.
- **DSML tool-call rax byte-replay:** deferred. Revisit trigger: any observed agentic cache churn
  (we now have the re-validation gate from P0-2 to DETECT it — measure first, then decide).
- **Their quant format/pipeline:** deferred as a runtime path; their published imatrix + recipe
  knowledge stays mined for our own quant work regardless.
- **Upstream glm5.2 branch (Metal SSD expert-streaming, hot-expert preload):** Mac-lane item,
  parked; revisit when the Mac needs to run models that don't fit its RAM.

## 3b. NEW ITEM — kv-paged for hybrid/conv archs (inkling-class) — EFFORT 1–3 days — BLOCKS his concurrency goal
- **What:** inkling on the paged path allocates the FULL -c KV statically (measured 2026-08-02:
  4M tokens × exactly 12 KiB = 48 GiB alloc vs fitter's 19.5 GiB pool estimate → OOM) — the arch
  doesn't actually page. Fix = make the paged cache support inkling's hybrid layout (global-attn
  KV + conv state) properly, incl. the fitter's bytes_per_block for this arch family (same bug
  class as the DSV4 head_dim fix d6a35ba78). Without this, inkling concurrency is static slots.
- **Kill gate:** inkling serve with --kv-paged creates llama_kv_cache_paged, pool sized on-demand,
  decode curve 8K→512K within 5% of non-paged.
- **Risk:** MEDIUM (cache-internals work, but bounded and we fixed the V4 one before).

## 3c. NEW DELIVERABLE — FORK-DELTA DOCUMENT (his order 2026-08-02)
`ornith-1m/docs/FORK-DELTA.md`: everything our llama.cpp fork carries that upstream doesn't —
kv-paged + fitter fixes, DSV4 arch/dflash work, inkling support, MTP ports, KT quants, gates,
each with branch/commit and status (merged-fleet vs branch-only). Living doc, updated on every
merge. Effort: hours to draft from git log + board, then maintained as part of each change.

## 4. Sequencing + total effort

| Phase | Items | Effort | Gate before next phase |
|---|---|---|---|
| 0 | 3c FORK-DELTA.md (documentation, hours) | hours | doc exists, he reads it |
| 1 | P0-4 methodology + P0-2 re-validation + P0-3 abort probe | 1.5–2.5 d | gates green, zero regression on decode curve |
| 2 | P0-1 yield-quench + **3b kv-paged hybrid-arch fix** (his ask) | 2–4 d | the 3 acceptance-regime A/Bs + inkling paged serve |
| 3 | P1-7 audit → P1-5 disk banks → P1-6 forking | 5–8.5 d | exactness counters 0; warm-admit ≥5× |
| 4 | P2-9 compressed KV + (D2R GEMM if revisit trigger fires) | 3–6 d | byte-exact + no curve regression |
| 5 | P2-8 continuous batching — THE PARITY ITEM | 5–10 d | 16-agent throughput gate |

**Total: ~2.5–4.5 weeks of focused work** (his assessment: possibly hours-to-days per item rather
than days-to-weeks — noted; the ranges above are focused-work estimates including measurement
gates, and P0/P1 items in particular tend to land at the fast end when the codebase cooperates).
All on `ds4-ports`, each phase independently shippable and kill-switchable. GPU needs: gates only
(the box already serves V4; each gate = a serve window of hours, not days). No cloud spend.
Nothing touches `fleet` without per-push approval (standing).

## 5. KERNELS SECTION — mapped inline (agent lane died twice to permission interruptions; finished by hand)

Source: `/tmp/ds4` @ 82d2a6f, targeted reads of ds4_cuda.cu / Makefile / README / speed-bench.

### 5.1 D2R GEMMs (the decode engine)
- Aligned-SoA repacked quant artifacts (IQ2_XXS kind 4 / Q8_0 kind 5 / Q2_K row-pair-SoA kind 6)
  read IN PLACE by custom GEMMs; integer tensor-core path (`mma.sync.aligned.m16n8k16...s32.s8.s8.s32`,
  ds4_cuda.cu:924+). Their own comparisons vs `mul_mat_q<8,128,0>` per-launch: [4096x2048] 3.2×,
  gate/up [2048x4096] 1.3× (comments near ds4_cuda.cu:16776).
- **Portability verdict:** llama.cpp's MMQ already does fused-dequant integer MMA on STANDARD GGUF
  layouts; their edge comes from the once-at-load SoA REPACK. Our tree has precedent for repack-at-load
  (repack.cpp, CPU side). Relevance = the MTP Stage-1 small-batch (2–8 col) dead zone. MEDIUM value,
  HIGH effort (CUDA kernel work + repack pass). Only attempt after P0/P1 items land and ONLY if the
  small-batch GEMM measurement still shows the dead zone on our current tree.

### 5.2 CUDA graphs per MoE layer
- Capture once per (layer-shape, weight-offset), single cudaGraphLaunch per MoE layer, ~8 CPU↔driver
  round-trips eliminated per replay; runs on a dedicated g_moe_stream overlapped with the rest;
  requires pre-sync bracketing (async replay hazard they hit and fixed). ds4_cuda.cu:16776–16964.
- **Portability verdict:** ggml-cuda already captures whole-decode graphs for supported models; our
  V4 decode curve doesn't show launch-bound behavior (66.8 t/s, graph capture on). LOW value for us.

### 5.3 FP4 indexer + fp8 KV row kernels (reference implementations)
- `indexer_hadamard_fp4_row_kernel` + `fp8_kv_quantize_row_kernel`, per-layer emit step mutating one
  row; exactness counters exported (`ds4_cuda_fp4_index_enabled`, `..._read_path_blocks`).
  ds4_cuda.cu:297–2851.
- **Portability verdict:** directly relevant ONLY to port item P2-9 (compressed-KV tier). Keep as the
  reference when/if that item is scheduled.

### 5.4 VMM weight arena — skip (confirmed in-process cuMemCreate+cuMemMap arena, 2 MiB pages,
  ds4_cuda.cu:418–527; exists to serve the multi-process weight server we don't need).

### 5.5 Arch targeting discipline (worth copying)
- Makefile: sm_121a needs the explicit `-gencode arch=compute_121a,code=sm_121a` PAIR — bare
  `-arch=sm_121a` silently emits `.target sm_121` and ptxas rejects the MMA; nvcc 13.0's default
  compute_75 PTX silently JITs onto sm_121 (their Makefile:31–118). DS4_CUDA_HAVE_MXF4 gates the
  MXF4 kernels. SAME FAMILY as our box's nvcc-12.9-for-compute_120a lesson. Zero-cost adoption:
  our fleet builds already pin arch; add the "silent PTX fallback" check to build docs.

### 5.6 Their measured numbers (README + speed-bench CSVs) — calibration for OUR expectations
- RTX PRO 6000 Blackwell 96 GB, q2 (81 GiB, single card): 85.2 t/s prefill(short) / **53.3 t/s decode**.
  Ours for comparison: llama.cpp UD-Q4_K_XL (155 GiB, dual card) = 66.8 t/s decode, kv-paged curve
  63.6→38.5 at 256K. Their decode at 11.7K ctx: 21.5–21.9 t/s on M3 Max class, and the Mac table
  (M3 Max 128 GB q2: 58.5/26.7 short) explains why the owner rejected the Mac lane (his call stands:
  the class is slow regardless of engine).
- DGX Spark (GB10): ~800 tok/s prefill claim, 2.6× their July start.
- Standing gates: abort_paths_gate, bank_churn_soak, bank_persist_gate, deep_ctx_gate,
  kv_crossmode_gate, launch_defaults_gate, needle_sweep, serial_rightsize_gate, teb_gates,
  vmm_trim_gate + per-hardware CSVs (gb10, m2/m3/m4_max, pro6000_blackwell). **This gate suite is
  the single most portable artifact in the repo** — the names alone are a checklist of every way a
  serving engine lies to you (churn, cross-mode KV corruption, launch-default drift, VMM trim
  regressions, abort paths). P0-4 covers adopting the pattern.

### 5.7 Quant pipeline answer (was an open question)
- RESOLVED by antirez's HF repo: their GGUFs are STANDARD types (IQ2_XXS/Q2_K/Q4_K/Q8_0/F16);
  gguf-tools/deepseek4-quantize.c produces them and the engine repacks to SoA at load. So: their
  kernels are format-locked to their REPACK, not to exotic GGUF types; and their GGUFs load in
  stock llama.cpp (verified structurally: the Mac q2 served fine on our yaniss build).

## 6. Standing rules for the work (re-stated)

- Work only in `<BOX>/wt-ds4-ports` (branch `ds4-ports`). No `fleet` pushes — per-push
  approval every time. Research stays Mac-local (`/tmp/ds4`); the box never sees research clones.
- Every feature behind an env kill switch; every claim behind a measured gate with an exactness
  counter. Same-boot ABBA for perf numbers.
- V4 production server (`dsv4-srv` :8331) is the owner's until he declares V4 testing done; port
  gates take GPU windows around his usage, never during.
