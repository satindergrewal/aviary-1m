# DESIGN — 3b: kv-paged for hybrid/SWA archs (inkling-class)

**Status: GROUND ZERO READ (2026-08-03 ~18:00). Design forming; no code yet.**
Author: Fable-DS4 | Lane: ds4-ports | Prior: B4 wall CLOSED (arcs 1-3, see DESIGN-B4-ARC3-*).

## The problem (measured 2026-08-02, plan §3b)
Inkling with --kv-paged allocates the FULL -c KV statically: 4M tokens × 12 KiB = 48 GiB
actual vs the fitter's 19.5 GiB pool estimate → OOM. The arch doesn't page at all.

## Ground zero (WITNESSED 2026-08-03 ~18:40 — corrects the first inferred version)
The memory dispatch (llama-model.cpp create_memory) for inkling is NOT the SWA branch:
- **inkling is `llm_arch_is_hybrid`** → routes at ~2166 to **`llama_memory_hybrid_iswa`**
  (SWA case, ~2193) or `llama_memory_hybrid` — the iswa attention cache in the boot log is
  created INSIDE this wrapper. The recurrent (shortconv) side is `recurrent_rs_size =
  n_seq_max` = **n_seq-bounded, tiny** (step-6 shortconv question ANSWERED).
- Pure-SWA non-hybrid archs (gemma-class) take the `swa_type != NONE` branch at ~2279 →
  `llama_kv_cache_iswa` directly.
- Only the plain else branch (no SWA, non-hybrid) consults `cparams.kv_paged`.
⇒ the flag silently no-oped for BOTH hybrid and pure-SWA archs (fa-on-disables-the-guard
class). First witnessed lesson: my initial ground-zero read INFERRED the SWA branch from
swa_type and was wrong — the witness run (guard in SWA branch never fired, server booted
normally) caught it. Guards are now in BOTH branches.
**3b-0 SHIPPED**: loud-refuse guards (LLAMA_LOG_ERROR + GGML_ASSERT) in the hybrid branch
(~2167) and the SWA branch (~2283); witnessed firing on the synthetic fixture
(scratchpad kvp-witness2.txt: E-line + assert at llama-model.cpp:2175).
**⇒ 3b composition point = llama_memory_hybrid_iswa's ATTENTION sub-cache** (+ the pure-SWA
iswa for gemma-class later); the paged pool replaces the attn side only, recurrent state
stays as-is.

## Composition facts (llama-kv-cache-iswa.h)
- `llama_kv_cache_iswa : llama_memory_i` COMPOSES two sub-caches:
  `std::unique_ptr<llama_kv_cache> kv_base` (:96) + `kv_swa` (:97) — concrete static type.
- `llama_kv_cache_paged` is a SIBLING class (not a `llama_kv_cache`), so it cannot slot in
  today. → the plan's "interface extraction" = make kv_base/kv_swa hold the interface both
  static and paged caches implement, then iswa-of-paged composes for free.

## Design shape (A, preferred)
1. Extract/verify the common interface the iswa wrapper actually calls on its sub-caches
   (init_batch/apply/get_k/get_v/cpy_k/cpy_v/seq_* / the arc-1 rel-API:
   get_n_kv_pos_contiguous_stream). List call sites in llama-kv-cache-iswa.cpp first.
2. Make llama_kv_cache_paged implement the missing members (or wrap it in an adapter).
3. Dispatch: in the SWA branch, when cparams.kv_paged → iswa with paged sub-caches.
   SWA sub-cache page budget: SWA layers only need ~n_swa live cells per seq — the paged
   pool sizing must reflect that or the whole win evaporates.
4. Rel-API over the paged block table: banded needs the CONTIGUOUS MONOTONIC TAIL property
   per stream. Paged blocks are physically scattered → either (a) the gather/copy the paged
   path already does for attention makes the tail logically contiguous at graph level
   (check how flat-arch paged builds its K/V views), or (b) banded needs a block-table-aware
   variant. READ NEXT: llama-kv-cache-paged.{h,cpp} attention view construction.
5. Fitter: bytes_per_block for the hybrid family (same bug class as DSV4 head_dim 1.60x fix
   d6a35ba78; kv-paged-facts memory has the details).
6. Shortconv state: inkling also carries conv state (SHORTCONV_KERNEL=4) — find where it
   lives (hybrid memory? separate buffer?) and confirm it is n_seq-bounded, not n_ctx-bounded
   (it should be tiny either way). Unresolved.

## Kill gate (from plan §3b, unchanged)
Inkling serve with --kv-paged creates the paged hybrid, pool sized on-demand, decode curve
8K→512K within 5% of non-paged. Toy-scale first on the synthetic fixture (the arc-1/3 gate
harness reuses directly: banded_equiv + buffer A/B + boot-topology log).

## Read #1 DONE — the true interface surface (2026-08-03 ~18:05)
Wrapper-side calls on kv_base/kv_swa (iswa.cpp): clear, seq_rm/cp/keep/add/div,
seq_pos_min/max, prepare, state_write/read, memory_breakdown, get_size, get_can_shift.
**The leak**: `get_base()/get_swa()` return CONCRETE `llama_kv_cache *` (iswa.h:90-91; context
variant :140-141) and the GRAPH CODE calls through them — inkling.cpp:443-453 uses
get_k/get_v/cpy_k/cpy_v(k_idxs) + get_n_kv_pos_contiguous_stream (arc-1 rel-API). The
extraction surface = memory_i ops PLUS this graph-facing API, not just what iswa.cpp calls.

## Read #2a DONE — paged's actual surface (llama-kv-cache-paged.h)
- `llama_kv_cache_paged : llama_memory_i` (NOT a llama_kv_cache). Implements init_batch/
  init_full/init_update/clear/seq_rm/seq_pos_min/max; **STUBS**: seq_cp (no-op comment),
  seq_keep/add/div empty, state_write/read stubbed, get_can_shift=false.
- Graph-facing accessor is **`get_kv_tensor(layer_idx)`** + block machinery
  (allocate/free_blocks/swap_in/swap_out/set_paged_batch_info/concat_block_ids/do_block_copy)
  — a COMPLETELY different shape from llama_kv_cache's get_k/get_v/cpy_k/cpy_v.
⇒ shape-A composition is NOT a drop-in: needs either (i) an ADAPTER implementing
llama_kv_cache's graph API over paged storage, or (ii) the inkling graph builder learning
the paged path natively (as flat archs presumably do). Which is cheaper depends on read #2b.
⚠ stubbed seq_cp also collides head-on with P1-6 prefix sharing (seq_cp IS the fork
primitive) — whatever 3b builds must leave a real seq_cp path open or P1-6 pays for it.

## Read #2b PARTIAL — paged graph consumption is a SEPARATE native path
llama-graph.cpp:923+ — `llm_graph_input_attn_kv_paged` is a dedicated input class feeding
write_slots / block_table / context_lens / batch_offsets / batch_lens tensors per ubatch.
Flat archs consume paged NATIVELY through this machinery (paged-attention style), NOT via
the llama_kv_cache get_k/get_v API. Consumers: llama-graph.cpp, llama-context.cpp,
llama-model.cpp, llama-paged-scheduler{,-impl}.cpp.

## Read #2b DONE + **DECISION RESOLVED: NATIVE (ii)** (2026-08-03 ~18:15)
The flat paged path is `build_attn_mha_paged` (llama-graph.cpp:2431) → **`ggml_paged_attn`**
— ONE fused op taking (q, k_cur, v_cur, k_cache, v_cache, block_table, write_slots,
context_lens, batch_offsets, batch_lens, kq_scale, block_size, max_blocks). It fuses the KV
WRITE (write_slots) with block-table attention; causality is implicit via context_lens —
**there is NO mask tensor in the paged op at all**. Implemented on BOTH backends
(ggml-cpu/ops.cpp + ggml-cuda/pagedattn.cu) ⇒ toy-scale gates stay on Mac.

**Why native wins on arithmetic:** the adapter path would gather scattered blocks into
contiguous K/V views per layer per ubatch = O(n_kv) copy bandwidth every step — exactly the
cost paging exists to avoid. Dead.

**Why arc-3 is the enabler:** ggml_paged_attn has no mask input, and after arc-3 the banded
path doesn't need one — the analytic band (rel_dist + visibility_window) is computed from
positions, which survive block scattering untouched (block_table maps logical→physical;
rel_dist is LOGICAL). So 3b's kernel work = **`ggml_paged_attn_banded`**: ggml_paged_attn's
signature + rel_logits tensor + rel_extent + visibility_window op_params (op_params[4]/[6]
layout from arc-3 step 1 carries over verbatim). The rel lookup and analytic band drop into
the paged kernel's inner loop exactly as they sit in fattn-banded today.

## Resulting 3b work plan (supersedes "shape A" §above where they conflict)
1. **Cache side — REFINED after the hybrid_iswa read (2026-08-03 ~19:05)**: the concrete
   chain is `llama_memory_hybrid_iswa` → `unique_ptr<llama_kv_cache_iswa> mem_attn`
   (hybrid-iswa.h:89, ctor .cpp:34) + `unique_ptr<llama_memory_recurrent> mem_recr` (:53);
   prepare calls `mem_attn->get_base()->prepare()` / `get_swa()->prepare()` (.cpp:104-117)
   and `get_n_stream()` (:81); graph reaches `get_mem_attn()` (.h:83). THREE concrete layers
   ⇒ do NOT make the chain polymorphic. Instead: hybrid_iswa gains an optional **parallel
   paged attention member** (one `llama_kv_cache_paged` pool, or base+swa pair) active under
   the DS4P flag; init_batch/prepare/seq ops route to it when active; `mem_attn` stays null
   in paged mode. The graph side branches on paged natively (build_attn_inp_kv_paged style),
   so NO API unification is needed — lifecycle routing only. SWA pool budget ~n_swa
   cells/seq.
2. **Graph side**: inkling banded branch, behind DS4P flag: build_attn_inp_kv_paged-style
   inputs per sub-cache + ggml_paged_attn_banded call (rel view per stream as in arc-1).
3. **Kernel side**: ggml_paged_attn_banded CPU + CUDA — paged_attn inner loop + the arc-2/3
   rel/window additions (element loads already type-aware from arc-2 patterns).
4. **Fitter**: bytes_per_block for the hybrid family (head_dim 1.60x bug class,
   kv-paged-facts memory).
5. **Gates**: synthetic fixture boot topology (paged hybrid created, pool on-demand),
   banded_equiv paged-vs-static bit-compare (CPU), decode-curve 8K→512K within 5% (box),
   test-backend-ops paged-banded cases.
⚠ seq_cp is stubbed in paged — P1-6 prefix sharing needs it real (block-table COW is the
natural paged implementation and is BETTER than static's copy — note for P1-6, don't lose).

## Part-4 map (graph builder; grounded 2026-08-03 ~20:15)
- `build_attn_inp_kv_paged` (llama-graph.cpp:3286) makes 5 I32 input descriptors sized from
  a **`llama_kv_cache_paged_context`** (get_n_tokens/get_batch_size/get_max_blocks);
  `set_input` (:923) fills them from the same context (get_write_slots/get_block_table/...).
  ⇒ the hybrid_iswa CONTEXT must carry an optional child paged context when paged mode is
  active (mirror of mem_attn_paged nesting in the wrapper); its init_batch creates it.
- In-tree TODO right below (llama-graph.cpp:~3311) literally anticipates this work:
  "separate the inner implementation ... once sliding-window hybrid caches are a thing."
- Inkling builder branch (paged-active): ONE set of paged inputs (single pool spans all
  layers) + per-stream rel views (arc-1 pattern) + `ggml_paged_attn_banded` with
  **window = is_swa ? n_swa : 0** — paged is implicitly causal (token ≤ q_pos by loop
  construction), so base/global layers need NO analytic window; only SWA layers pass n_swa.
  Simpler than the static banded path.
- The paged write path replaces cpy_k/cpy_v (write_slots do the store) — the inkling
  builder's cache-store calls are skipped in paged mode.
- Kernel-layer state feeding this: ggml_paged_attn_banded COMPLETE both backends
  (5e594336/a2da43a3/7bdeb5c2/2e0e51bc), CPU witnessed 2.4e-07 vs the banded-FA reference,
  CUDA witness = box run pending micro-slot.

## Part-4a OPEN QUESTION (bank before writing code — 2026-08-03 ~20:20)
Context plumbing target: `llama_memory_hybrid_iswa_context` (hybrid-iswa.h:107) holds
`ctx_attn` (iswa context) + `ctx_recr`; paged mode adds an optional
`llama_kv_cache_paged_context` child + accessor, and the graph builder branches on it.
**BUT: ubatch-split ownership conflict.** The hybrid's init_batch splits balloc itself
(split_seq/split_equal per recurrent constraints) then preps recr + both static attn
sub-caches on those ubatches; `llama_kv_cache_paged::init_batch(balloc, ...)` ALSO consumes
balloc directly (its own iteration + block allocation via llama-paged-scheduler). Both
cannot consume balloc. Resolve by READING llama-kv-cache-paged.cpp init_batch +
llama-paged-scheduler-impl.cpp FIRST. Options:
(i) paged gains an init variant accepting pre-split ubatches (hybrid keeps split ownership
    — recurrent constraints must win anyway, so this is the likely shape);
(ii) hybrid delegates the whole split to paged, then feeds the SAME ubatches to recr
    prepare (only valid if paged's split satisfies the recurrent equal-split constraint).
Do NOT start 4a until this is resolved; the recurrent side's split constraints are the
binding ones (that's why hybrid exists).

**RESOLVED (same read): option (i) confirmed viable.** llama_kv_cache_paged::init_batch
(paged.cpp:345-374) is a THIN wrapper: split_simple over balloc → context wrapping the
ubatches → `set_batch_data(*last_paged_info)`. Block allocation is NOT in init_batch — it
happens externally via the paged SCHEDULER calling set_paged_batch_info before decode
(GGML_ASSERT(last_paged_info) proves the ordering contract). So a pre-split-ubatch init
variant is ~10 lines and the hybrid keeps split ownership (recurrent constraints win).
REMAINING TRACE before 4a code: who drives the scheduler (set_paged_batch_info /
allocate/swap call sites in llama-context.cpp + llama-paged-scheduler-impl.cpp) and whether
that path activates on cparams.kv_paged alone or is flat-arch-gated — the hybrid needs the
same scheduling to run for its pool.

**4a trace result (2026-08-03 ~20:30): the scheduler is HOST-DRIVEN, and only
`examples/paged/paged.cpp` drives it.** The scheduler is public API
(llama.h:1658+ llama_paged_scheduler_init/add_request/prepare_batch/update);
prepare_batch → set_paged_batch_info (impl.cpp:78) → paged init_batch's
GGML_ASSERT(last_paged_info) ordering contract. llama-server does NOT call
llama_paged_scheduler_* anywhere. ⚠ OPEN before 4a: how does server-mode --kv-paged
(un-gated to fleet as 5751cfdde, the 19/17 curve) actually function without the scheduler —
different path, or does server+paged rely on something else setting batch info? VERIFY
FRESH with a 2-min witness: flat-arch synthetic fixture + Mac CPU server + --kv-paged
(paged CPU reference exists, so it should run or die informatively). The answer decides
whether the hybrid needs server-side scheduler driving (bigger) or the same path the
flat server uses (smaller). Do not infer — witness.

**★★ WITNESSED (2026-08-03 ~20:35, results/server-kvpaged-crash-witness-20260803.txt):
llama-server + --kv-paged ABORTS AT FIRST DECODE FOR EVERY ARCH.** Flat-arch synthetic
fixture, CPU server, --kv-paged: dies at llama-kv-cache-paged.cpp:368
GGML_ASSERT(last_paged_info) during the boot probe decode. Server sources contain ZERO
paged handling (exhaustive grep); only examples/paged drives the scheduler. Consequences:
1. The fleet-un-gated server flag (5751cfdde, my lane) exposes a crash path — the
   fa-on-disables-the-guard class AGAIN, and this one is MINE to own: the un-gate shipped
   without a server+paged boot witness.
2. All prior kv-paged curves (19/17 etc.) were necessarily examples/paged runs.
3. Part 4's true scope: server-side scheduler driving (request→group mapping,
   prepare_batch before decode, update after) is REQUIRED for paged serving of ANY arch —
   flat AND hybrid. This is the same machinery as P2-8 continuous batching (the scheduler
   IS the vLLM-style engine). 3b part 4 and P2-8 partially merge.

## Part-4d map (scheduler driving; grounded from examples/paged/paged.cpp — 2026-08-03 ~23:20)
The ENTIRE drive loop is 4 public-API calls (paged.cpp:115-215):
1. `llama_paged_scheduler_init(ctx)` once; `add_request(scheduler, tokens, n, seq_id)` per request.
2. Loop: `prepare_batch(scheduler, &batch)` — fills llama_batch AND sets the cache's batch
   info (the init_batch ordering precondition — THIS is what the server never does).
3. `llama_decode(ctx, batch)` — plain decode; per-seq sampling via
   `get_batch_info()->batch_offsets[i] + batch_lens[i] - 1` (last token of each seq segment).
4. `update(scheduler, &batch, sampled_tokens, stop_flags)` — advance/free; also get_seq_state
   per request (n_prompt/n_decoded/ttft timings — the server metrics map directly).
⇒ 4d = mapping the server's slot loop onto this engine when kv_paged: requests→add_request,
update_slots' batch assembly→prepare_batch, per-slot sampling via batch info, finalize→update.
This IS P2-8's core. Sequencing note: 4c-1 (inkling builder branch) lands AFTER 4d so it is
immediately witnessable (paged context only goes live when a scheduler drives it).
**4d's one external dependency: the owner's #1831 re-gate decision** shapes the flag surface
(does --kv-paged stay exposed while 4d is built; does hybrid-paged serving land behind
DS4P_PAGED_HYBRID or a new opt-in). Building the server gate topology before his word risks
building it twice.
