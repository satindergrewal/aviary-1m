# RESEARCH — Hugging Face Kernels Hub: what's usable for OUR stack

Author: Fable-DS4 | Date: 2026-08-02 (the owner's ask: "check and find if there's any kind of
kernel that can help speed boost our work... shortlist and queue them... if there is any idea we
can port to our code"). Provenance: hub pages fetched today; per-repo file trees verified where
marked; everything else is catalog-level.

## 0. What the hub is (and what "integrate" means for us)

`huggingface.co/kernels` — a hub section (~236 kernels) of GPU kernels loaded at runtime by the
`kernels` pip library as **PyTorch extension binaries** (per-torch-version `.abi3.so` builds,
built by their Nix kernel-builder; Triton kernels ship as source). Platforms: CUDA, ROCm, Metal,
XPU, CPU.

**Reality check for our stack:**
- **llama.cpp (the fleet) CANNOT consume these directly** — they are torch extensions, not ggml
  backends. Our mode is **mine the source, port the technique** into ggml-cuda/ggml-metal behind
  a flag with a measured gate (exactly the ds4-ports methodology).
- **Direct pip use IS possible in our PyTorch lanes**: DeepSpec head training, abliteration
  tooling, any future vLLM revival.
- Source availability varies BY AUTHOR: `kernels-community` repos ship full source (verified on
  metal-flash-sdpa: 66 KB `.metal` shader + bindings + tests + bench). Random-author repos can be
  **binary-only** (verified: flashrt/grouped-moe-gemv has NO source, no README — unusable for
  porting, benchmark-datapoint at best).
- Licenses seen: Apache-2.0 (port-friendly with attribution; verify per repo before any port).

## 1. SHORTLIST — ranked by value to our stack

### K-1. FP4 GEMM family → native FP4-tensor-core path for MXFP4 experts on SM120 — ✅ CLOSED SAME NIGHT: no gap
**RESOLVED 2026-08-02 ~23:55 (code-read, `wt-ds4-ports` @ 70ef59be1): our tree ALREADY runs
native FP4 MMA on Blackwell** — `ggml/src/ggml-cuda/mmq.cu:131` `use_native_fp4 =
blackwell_mma_available(cc) && (MXFP4 || NVFP4)`, dedicated `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4`
tile layout (`mmq.cuh:128`), and a `GGML_TYPE_NVFP4` type exists in ggml. The gate below fired
exactly as written: the assumed gap does not exist. Nothing to port. (Kept for the record —
the hub FP4 repos remain reference material only.)
- Repos: `kernels-staging/nvfp4-gemm`, `flashrt/fp4-gemm` + `fp4-fused-ops` (source presence
  UNVERIFIED), `Atlas-Inference/nvfp4-moe` / `nvfp4-dense-gemm` / `nvfp4-paged-attention`,
  `phanerozoic/fp4-train`.
- **Why us:** our RTX PRO 6000 Blackwells have FP4 tensor cores. We SERVE MXFP4 models today
  (Inkling-Small MXFP4_MOE 148G) and K3's QAT truth is MXFP4. If llama.cpp's CUDA path for
  MXFP4 experts runs int8-MMQ after dequant, a native FP4-MMA GEMM could be a real decode/prefill
  lever on exactly the models we run.
- **Gate on the idea (do FIRST, no GPU needed):** read our tree's actual MXFP4 CUDA path — if it
  already uses FP4-native MMA, this collapses to nothing. Do not assume the gap exists.
- Action: queued research note (K-1 below). Port effort if real: HIGH (CUDA kernel work), so it
  enters as a deferred item with a measured revisit trigger, same class as D2R.

### K-2. `kernels-community/metal-flash-sdpa` — ✅ DIFF DONE 2026-08-03: real graft targets
- **Full analysis: `docs/K2-METAL-FA-DIFF.md`** (line refs both sides). TLDR: our Metal FA
  prefill streams K/V **4× more often** than MLX Steel (Q=8 vs BQ=32 tiles), round-trips O/P
  through threadgroup memory every block, and pays ~1 GiB of causal-mask reads per layer-op
  at 1M ctx that Steel avoids analytically. Plausibly tens-of-percent on Mac long-context
  prefill (bandwidth arithmetic, UNMEASURED). Ours is BETTER at quantized-KV, split-KV
  decode, arbitrary masks, MLA head shapes — grafts must not regress those.
- Disposition: headline work item for the future dedicated kernel-only sessions (the BQ
  re-tile is a redesign, not a patch). Beneficiaries: Ornith-35B Mac daily driver, V4-on-Mac
  revival. Companion references: `mlx-quantization-metal-kernels`, `gpt-oss-metal-kernels`.

### K-3. MoE small-batch family — feeds the EXISTING D2R revisit trigger
- Repos: `kernels-community/vllm-moe` (211 likes, vLLM lineage = source), `drbh/yamoe` +
  `fused-moe` (HF staff), `axolotl-ai-co/sonic-moe`, `nCompass-tech/triton-moe`,
  `bassrehab/moe-dispatch`, `phanerozoic/grouped-gemm-moe`; flashrt's `grouped-moe-gemv`/`
  grouped-moe-gemm` are BINARY-ONLY (skip for mining).
- **Why us:** the deferred D2R item's prime target is the MTP/small-batch (2–8 col) GEMM dead
  zone. When the post-P0/P1 revisit measurement re-runs the small-batch curve, these repos are
  the technique library to mine BEFORE writing our own.
- Action: no new work now — attach this list to the existing D2R revisit trigger in
  PLAN-DS4-PORTS.md §3.

### K-4. `kernels-community/mamba-ssm` (+ `phanerozoic/mamba3`) — hybrid/SSM reference
- **Why us:** inkling's short-conv hybrid path (3b work) and the parked Nemotron-3 hybrid-Mamba
  plan. Reference for llama.cpp's SSM/conv ops if hybrid-arch perf work opens after 3b.
- Action: parked reference, revisit trigger = hybrid-arch optimization after 3b lands.

### K-5. `kernels-community/liger-kernels` — training-side fused ops (direct pip use)
- **Why us:** DeepSpec head training runs PyTorch on the box; Liger's fused CE/RMSNorm/rope cut
  training step time and VRAM. Small but nearly free — it's a pip install + config flag in the
  training venv.
- Action: suggestion handed to the DSpark lane (their venv, their call), post-regen.

### K-6. `kernels-community/paged-attention` — reference only
- vLLM's classic paged-attention kernels. Our kv-paged is our own design; their block-table→
  attention fusion details are useful reading for 3b/P2-8, nothing to integrate directly.

## 2. NON-FINDINGS (searched, absent — saves future sessions the look)

- **No hadamard/rotation/QuaRot kernels** on the hub → TurboQuant/WHT lane unaffected.
- **No SM120 paged flash-attention** → the vLLM-on-SM120 inkling wall (A8) stays closed from
  this direction too. flash-attn3/vllm-flash-attn3 are Hopper-class (SM90); hub cards list
  H100/A100/MI300/M3/M4 targets, Blackwell workstation (SM120) appears in no card we read.
- flashrt's interesting-sounding small-batch repos: binary-only, no docs — nothing to learn.

## 3. QUEUE (proposed; none of it touches Phase A order or the box during the regen)

| id | what | cost | when |
|---|---|---|---|
| K-1 | verify our MXFP4 CUDA path (MMQ-int8 vs FP4-native), write feasibility note | hours, code-read only | any time |
| K-2 | metal-flash-sdpa vs ggml-metal FA technique diff | hours, Mac-local | any time |
| K-3 | attach MoE-kernel list to D2R revisit trigger | minutes (doc edit) | done with this doc |
| K-4 | mamba-ssm reference | parked | after 3b |
| K-5 | Liger suggestion to DSpark lane | one room message | post-regen |

Integration doctrine for anything that graduates: flag-gated port into OUR tree, same-boot ABBA
A/B, exactness counter, kill switch — no hub binaries linked into the fleet, ever (sovereignty).
