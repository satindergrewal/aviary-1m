# INKLING-SMALL @ 1M CONTEXT — THE WALL MAP (2026-08-02, Fable-DSpark)

**His order:** "map out all the walls you hit trying to make it load and run with 1M context, and
then we work on to fix them." This is that map: every wall, exact numbers, root cause, and the
fix path for each. Status: no code changes yet — fix work is gated on his go.

**Model:** Inkling-Small 276B/12B active, 42 layers, hybrid iSWA (sliding-window) attention +
per-layer packed shortconv state, dense rel-position attention (CORRECTED 2026-08-03 by
Fable-DS4's code read, #1667: NO Lightning Indexer anywhere in inkling.cpp — the "Lightning
Indexer enabled" log line prints from the fused-ops resolve table in llama-context.cpp, not
from this model. My earlier "DSA-family sparse attention" claim was the log line misleading me),
native 1,048,576 ctx. Served artifact:
unsloth MXFP4_MOE GGUF, 148 GiB, `<BOX>/bigmodels/inkling-small-gguf/`.
**Box:** 2× RTX PRO 6000 Blackwell (SM120), 95.3 GiB usable each, 191 GiB total.

---

## ROUTE A — vLLM 0.26.0: DEAD END (engine-level, unfixable by us)

| # | Wall | Exact evidence | Root cause | Fix path |
|---|---|---|---|---|
| A1 | No inkling support in 0.25.1 | grep of vllm25-venv = 0 hits | support landed in 0.26.0 | new venv `<BOX>/vllm26-venv` (built) |
| A2 | Post-NCCL 100%-CPU spin, 35–66 min, zero I/O | `SymmMemCommunicator: Device capability 12.0 not supported` | symm-mem + custom allreduce + P2P on mixed-SKU pair (6000 + Max-Q) | `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 --disable-custom-all-reduce` (WORKED) |
| A3 | "SM 12.x requires CUDA >= 12.9" (flashinfer JIT) | flashinfer/compilation_context.py | system CUDA 12.0; toolkit not on PATH | `CUDA_HOME=/usr/local/cuda-12.9` + PATH (WORKED) |
| A4 | `Errno 2: 'ninja'` | worker traceback | FlashInfer CUTLASS JIT needs ninja | `pip install ninja` into venv + venv/bin on PATH (WORKED; one-time 20-min compile, 378-file cache) |
| A5 | V2 runner ignores `cpu_offload_gb` | resolved args show 8.0, available KV pinned at 7.36 GiB | V2 memory accounting doesn't apply offload | `VLLM_USE_V2_MODEL_RUNNER=0` (V1 honors it: 7.36→13.5 GiB) |
| A6 | KV floor: ~11 GiB FIXED per sequence | 12.51 GiB needed @65K, 13.39 @131K; fp8 KV saved only 1 GiB | short-conv per-sequence state, context-independent | cpu-offload (V1 only) |
| A7 | cute-dsl warmup assert | `SM120 forward only supports num_splits=1` | FA4 warmup sweep tries configs SM120 forbids | `--kernel-config.enable_cutedsl_warmup=False` (WORKED to boot) |
| A8 | **HARD WALL: first request dies** | `AssertionError: Paged KV not supported on SM 12.0 in this PR` — vllm_flash_attn/cute/interface.py:1091 | TML's FA4 rel-attention kernel: paged-KV path implemented for SM100 (B200) only | **NONE in vllm 0.26.0.** Watch for a later vllm; SGLang untested (same TML kernel family suspected). Route abandoned. |

**Verdict:** vLLM cannot serve Inkling-Small on SM120 today. Not config — kernel support.

---

## ROUTE B — llama.cpp (fleet binary): serving, but 1M is blocked by ONE wall

| # | Wall | Exact evidence | Root cause | Fix path |
|---|---|---|---|---|
| B1 | **mmproj crash** (vision blocked) | `mtmd/models/inkling.cpp:82: GGML_ASSERT(dmel_embd_w->ne[0]==n_embd)` | fleet's inkling projector code was written for the 975B; Small's projector layout differs | mtmd patch for Small's projector (S, hours) — vision only; text unaffected |
| B2 | **KV budget** | KV = exactly 12 KiB/token bf16, allocated on ONE device; 4×1M = 48 GiB won't fit beside 148G weights; 2×1M = 24 GiB OOM'd CUDA0 by ~1 GiB | full-precision KV + single-device KV placement | **CONFIG-SOLVED with a COUPLING (revised 2026-08-03, DS4 #1691):** `-ctk q8_0 -ctv q8_0` halves KV **BUT disqualifies banded flash (KV type gate) → silently causes B4's compute explosion at big context.** Safe at ≤262K total ctx (fallback plane is tiny there), a TRAP at 1M-scale. Real solve = B4's gate-widening (q8-capable banded kernel); until then, prefer f16 KV + fewer slots over q8 at big ctx. Deeper fix = paged on-demand pool (B3) or fp8/nvfp4 KV |
| B3 | **kv-paged doesn't page for inkling** | paged path allocated FULL -c KV statically: 48 GiB @4×1M (= 4,194,304 × 12 KiB exactly) vs fitter pool estimate 19.5 GiB → OOM; `-b/-ub` changed nothing | **ROOT CAUSE FOUND (Fable-DS4, DESIGN-DS4-P0): `create_memory`'s kv_paged branch sits in the NON-HYBRID leg — inkling never reaches the paged constructor; `--kv-paged` is SILENTLY IGNORED for hybrids = the measured 48 GiB static wall.** Plus the fitter budgets all-layers uniformly | paged attention sub-cache inside `llama_memory_hybrid` + cache-owned bytes-per-block (plan item 3b, 1–3 d, design doc written) |
| B4 | **★ THE 1M WALL: compute buffer ∝ ubatch × total context** | measured line: 115,108.71 MiB @ (ub 2048 × 2,097,152 tok); 28,996.71 @ (ub 512 × 524,288); ~3 GiB @ (512 × 262,144). **Formula: compute_GiB ≈ 55.3 × ub × ctx_millions / 10³** — fits all three points. FA on/auto identical → NOT the FA score path | **DIAGNOSIS REWRITE (Fable-DS4 #1691, source + synthetic measurement): the chunked mechanism ALREADY EXISTS — `use_banded_flash` (inkling.cpp:235-246) avoids the [n_kv, n_tokens] planes entirely, but is GATED OFF by exactly our serve configs: (1) requires KV ∈ {F32,F16,BF16} — q8_0 DISQUALIFIES, so B2's q8-KV "solution" silently CAUSES B4 (halves KV, explodes compute); (2) requires `n_kv_pos_contiguous > 0`, which is 0 for ANY multi-sequence ubatch (-np>1 falls back regardless of FA — explains FA on/auto identical). Measured ladder @128K synthetic: fa-off 2057 MiB (~32 B/pair), fa-on fallback 261 MiB (~4 B/pair = the I32 rel_idx plane), banded = the zero-plane regime | **Fix reframed: WIDEN THE GATE + IMPLICIT MASK (DS4 #1694 regime ladder, synthetic-measured; scale corrected per Grok #1695):** unfused fallback ~32 B/pair, FA-fused fallback 4 B, banded 4 B — **even banded has a floor: the KQ mask (F16 [n_kv, n_tokens] + cont-copy) = 4 B/pair ⇒ ~15.3 GiB at ub2048×2M (~7.6 GiB mask-only at 2 B/pair).** Two parts: (a) widen the banded gate (multi-seq + q8-capable kernel — kills the 32→4 B/pair fallback penalty on his real configs); (b) the mask must go implicit/analytic (exactly K-2's MLX-Steel finding #4: Steel does causal analytically, zero mask traffic) — the two investigations converge on one mechanism. Real-model 56 B/pair = the q8+multi-slot fallback regime, confirmed. Free config datapoint: f16-KV single-slot serve should already dodge B4 (verify in a window). Same class as GLM's 22.8 GiB O1 |

### The VRAM arithmetic every config must close (per device, 95.3 GiB usable)
`model_share + compute(55.3·ub·ctx_M/10³) + KV(12 or 6 KiB/tok × ctx) + ~3 misc ≤ ~92`
and the model share is bounded by the OTHER card: `model_G1 ≤ ~92`.

### What serves TODAY (all measured):
- **4 × 64K** (-c 262144, default ub): 71–73 t/s decode, 234 prefill — the first working config,
  and the one left RUNNING for him (port 8340).
- 4 × 128K: DOES NOT CLOSE — the compute reserve is per-device, proportional to the graph's
  device split (measured: 22,654.74 MiB demanded on CUDA1 at ub 384, OOM with the 56% model
  share). Both cards need compute headroom, so 148 weights + ~44 total compute + KV ≈ 201 GiB
  at ub 384; ub 128 closes it (~172) but prefill crawls. Not worth it vs the B4 fix.
- 1M anything: blocked by B4. 2×1M bf16-KV: blocked by B2 (marginal) THEN B4 anyway.

### Estimated post-B4 landscape (if the compute op gets chunked):
KV q8 @1M = 6 GiB; compute back to ~3–5 GiB flat → **2×1M at 71+ t/s fits trivially,
4×1M fits with q8 KV**, and kv-paged (B3 fix) makes slot count dynamic. B4 is the key that
turns the whole landscape.

---

## Fix priority (proposed — his go gates implementation)
1. **B4 gate-widening** (the 1M gate; also advances GLM O1): multi-seq banding + q8-capable
   banded kernel — NOT "build chunking", the mechanism exists (DS4 #1691). ~2–5 days, core.
2. **B3 kv-paged hybrid-arch support** (dynamic slots for inkling) — 1–3 days, plan item 3b
   (root cause found: paged branch in create_memory's non-hybrid leg; synthetic fixture exists).
3. ~~B1 mtmd Small projector~~ — ✅ FIXED 2026-08-02 (upstream 4-line fix, fleet 8a3b616cb).
4. B2 stays config-solved WITH THE B4 COUPLING NOTED (q8 only at ≤262K total ctx); P2-9
   compressed-KV tier is the long-term answer.
5. Route A (vLLM): park; re-check when vLLM ships SM120 paged-FA4.

Related: plan `docs/PLAN-DS4-PORTS.md` (items 3b, P2-9, P2-8); memory [[inkling-small-serve]]
(the full attempt log + vLLM chain); board item O1.
