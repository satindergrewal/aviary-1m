# PLAN — paging minimax-m3 (MSA)

## ★★ STATUS 2026-08-13 (late): SYNTH-VERIFIABLE SCOPE CLOSED. store+gather bit-exact CPU+Metal.
Commits (on origin/ds4-ports, pushed 2026-08-13): 4e8b555c5 (I64 store op + gather wiring,
dev-gated DS4P_PAGED_MSA) + cab3f36d8 (permanent test tests/test-paged-msa-roundtrip.cpp)
+ ac9594ad4 (comment honesty).

WHAT IS NOW VERIFIED (bit-exact, max_err=0):
- store op ggml_paged_kv_store on CPU **and Metal** (MTL0), ALL K + ALL V heads
- the model's OWN in-graph gather row arithmetic (floor/scale/sub/add/arange/repeat/cast) —
  NOT hand-computed rows. The test builds rows the exact way minimax-m3.cpp:443-453 does.
- plumbing is ACTIVE: with DS4P_MSA_POSTRACE the POOL graph BUILDS in the scheduler-driven decode
  (note: 1 POOL trace = 1 graph BUILD, then reused across decodes — many decodes ran the pool).

⚠ CORRECTED PREMISE (the old sections below say "green e2e gate proves store+gather correct" — FALSE):
the synthetic random-weight model's MSA attention output is ~1e-8, so the token/logit e2e gate is
STRUCTURALLY BLIND to the gather. Proof: garbaging the pool path (kg×100) AND emptying it (NOSTORE)
BOTH leave the e2e test PASSING, even with graph-reuse DISABLED and POOL confirmed active. The e2e
gate proves the path RUNS, never that the gathered VALUES are right. That is why the round-trip test
(tests/test-paged-msa-roundtrip.cpp) exists and is the ONLY sensitive instrument for this feature.
Also: the sensitive decode-logit check in test-paged-kv-e2e measures the DIRECT-decode path, which
for MSA never engages paging (has_paged_batch_info false there) — comment corrected in that file.

REMAINING (genuinely needs a real unsloth/MiniMax-M3 GGUF — owner's provisioning call):
only the pos_slot/mask interaction inside a real forward pass. The store+gather MACHINERY is done.
The env-gated controls (DS4P_MSA_NOSTORE/GARBAGE/DENSEGARBAGE/POSTRACE) are KEPT as the real-model
negative-control harness.

--- historical sections below (superseded where they claim the e2e gate proves correctness) ---

## ✅ CPU STORE OP DONE (2026-08-13, commit 2f6347848) — ggml_paged_kv_store
The store-only op is committed: enum/name/symbol, constructor (result aliases kv_cache for
store-before-gather ordering), ggml.h + ops.h decls, CPU forward (verified index math, f16/f32),
both ggml-cpu dispatch switches. Full build links; harness ALL PASSED; #19 gate still green (op
unused). REMAINING for #19: (1) METAL kernel for ggml_paged_kv_store (pool is Metal-resident;
mirror a simple scatter kernel + ggml-metal-ops.cpp dispatch + supports_op), (2) wire the store
(replace/augment minimax-m3.cpp 366-367 dense cpy with ggml_paged_kv_store(pool_kv, Kcur, Vcur,
write_slots, bs)) + the GATHER (read pool_kv via block_table translation, 422-427), (3) gate
DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e GREEN. The hard math is proven; this is kernel
plumbing + graph wiring.

## ✅ METAL PATH MAPPED (2026-08-13): REUSE the existing write kernel, no new shader
KEY: my CPU ggml_paged_kv_store math EXACTLY matches kernel_paged_attn_write_f32 (ggml-metal.metal
:3064): base=block*stride_block+tok*stride_token, K at +h*stride_head+d, V at +(n_heads_kv+h)*stride_head+d
== d + hd*pos + hd*bs*h + hd*bs*2hkv*blk. So the Metal op REUSES that kernel (and _q8_0). Remaining:
1. NEW getter ggml_metal_library_get_pipeline_paged_kv_store (device.cpp + .h) — copy of
   get_pipeline_paged_attn_write (device.cpp:1238) but assert PAGED_KV_STORE and read src[0]->type
   (my kv_cache) instead of src[3]; returns kernel_paged_attn_write_f32/_q8_0.
2. encode ggml_metal_op_paged_kv_store (ggml-metal-ops.cpp) — mirror the WRITE PHASE at :5044-5062:
   set ggml_metal_kargs_paged_attn { head_dim=src[0]->ne[0], block_size=op_params[0],
   n_heads_kv=src[0]->ne[2]/2, stride_token=hd, stride_head=hd*bs, stride_block=hd*bs*2*hkv, probe=0 };
   buffers k_cur→1, v_cur→2, kv_cache(src[0])→3, write_slots(src[3])→4; dispatch (n_tokens, n_heads_kv, 1)
   width wnth (f16: head_dim). NO barrier needed after (no attend in same op; the ALIAS result orders
   the downstream gather).
3. dispatch case GGML_OP_PAGED_KV_STORE → ggml_metal_op_paged_kv_store (ggml-metal-ops.cpp:483).
4. supports_op case (ggml-metal-device.m:1230): return src[0]->type==F16||F32 && src[1]->type F32/F16.
Correctness ASSURED by construction (same kernel+math as the proven write path; CPU op agrees).
Gate: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e once store+gather are wired.

## ✅✅ STORE OP COMPLETE on CPU+Metal (2026-08-13, commits 2f6347848 + dd56111cf)
ggml_paged_kv_store is DONE end to end: CPU forward (verified math) + Metal (reuses the proven
kernel_paged_attn_write_f32, correctness by construction). Builds clean, harness ALL PASSED, #19
gate green (op still unused). ⇒ #19's ONLY remaining work is the GRAPH WIRING in minimax-m3.cpp:
1. STORE: in the sparse branch, build_attn_inp_kv_paged(pg_ctx) for write_slots/block_table, then
   replace/augment the dense store (mctx_cur->cpy_k/cpy_v, 366-367) with
   ggml_paged_kv_store(pg_ctx->get_k(il), Kcur, Vcur, inp_paged->paged_write_slots, block_size).
   Use the RESULT (aliases pool) as the cache tensor the gather reads (orders store-before-gather).
2. GATHER: read that pool tensor; translate the gather cell cs -> pool row via block_table
   (block_table[cs/bs]*bs + cs%bs) in the ggml_get_rows path (422-427). kv_idx lockstep or dense v1.
3. GATE: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e must stay GREEN (now the pool is actually
   read, so green proves the store+gather are correct end to end).
The op infra is done; this is graph construction against a green gate. 10 commits this session.

## ★ GATHER FORMULA DERIVED (2026-08-13) — the last piece, reduced to arithmetic
The store op is DONE (CPU+Metal). Remaining = wire store+gather in minimax-m3.cpp sparse branch.
STORE (clean): pg_ctx = static_cast<msa_context*>(mctx)->get_attn_paged(); inp_paged =
build_attn_inp_kv_paged(pg_ctx); pool_kv = pg_ctx->get_k(il);
stored = ggml_paged_kv_store(pool_kv, Kcur_hd_hkv_ntok, Vcur, inp_paged->paged_write_slots, blk).
Keep the dense cpy_k/cpy_v (366-367) too for v1 so seq_pos tracking is unaffected (no ledger risk);
memory-saving removal is a later step. Use `stored` (aliases pool) as the gather source.
GATHER (the bespoke part, formula derived): the dense path gathers k3=[D, HKV*n_kv, ns] by
tokr = cs*HKV + h (cs = pos_slot cell, minimax-m3.cpp 416-420). The POOL is [hd, bs, 2hkv, nblk];
VIEW it flat as [hd, bs*2hkv*nblk] and the pool row for (cell cs, head h) is:
    pool_row(cs,h) = (cs % bs) + bs*h + bs*2*hkv*block_table[cs/bs]
Build it in ggml: blk_idx = floor(cs/bs); pos = cs - blk_idx*bs; phys = get_rows(block_table_f, blk_idx);
pool_row = pos + bs*h + bs*2*hkv*phys ; cast I32 ; ggml_get_rows(pool_flat, pool_row). h broadcasts
over the Hd head dim (arange, as the dense path does for tr). ⚠ K at head h uses i2=h; if V shares the
buffer at i2=hkv+h, gather V with h->hkv+h (or gather V from the same flat view at +bs*hkv offset).
block_table_f = the paged_block_table as F32 (or gather as I32 directly). VERIFY the flat-view row math
against the standalone index test's off() before trusting it. GATE: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1
test-paged-kv-e2e GREEN == store+gather correct end to end. This is the final focused implementation.

## ⚠ GATHER CRUX: resolve pos_slot LOGICAL-vs-PHYSICAL before writing the gather (2026-08-13)
The gather formula pool_row(cs,h)=(cs%bs)+bs*h+bs*2hkv*block_table[cs/bs] ASSUMES cs (from
msa->pos_slot_f, minimax-m3.cpp:282/412) is a LOGICAL cell needing block-table translation. BUT if
the MSA context's set_input_pos_slot (llama-kv-cache-msa.h:138) already returns the PHYSICAL pool
slot (post-translation), then the block_table step is WRONG (double translation) and the correct
pool_row is just (slot%bs)+bs*h+bs*2hkv*(slot/bs) with NO block_table gather. FIRST TASK next
session (CPU reading, no GPU): read set_input_pos_slot's impl + how the producer
(init_batch_with_ubatches) relates the pool's block_table to the MSA cells. Determine logical vs
physical. THEN the gather is a transcription of the right variant. Getting this wrong = silent wrong
gather (red gate, debug cycles) -- resolve it by READING, not guessing. Everything else for #19 is
done: store op (CPU+Metal), plumbing, gate. This one semantic + the bespoke ggml index construction
is the final focused piece. 10 commits this session; store op complete both backends.

## ✅ GATHER CRUX RESOLVED (2026-08-13, by reading set_input_pos_slot) — simpler than feared
set_input_pos_slot (llama-kv-cache-msa.cpp:345,360) maps position -> the KV_BASE DENSE CELL index j
(map[p0]=j from get_base()->get_cells). So pos_slot is a kv_base cell, NOT a pool slot. ⇒ CLEAN v1:
- STORE to the pool at kv_base's OWN indices so pool cell == kv_base cell == pos_slot:
  ggml_paged_kv_store(pg_ctx->get_k(il), Kcur, Vcur, inp_attn->get_k_idxs(), blk)
  (use inp_attn->get_k_idxs(), NOT inp_paged->paged_write_slots -- that keeps the pool aligned to the
   dense cell numbering pos_slot uses). Keep the dense cpy_k/cpy_v too (positions/seq_pos unaffected).
- GATHER with NO block_table (v1): pool cell = kv_base cell j, pool block = j/bs, pos = j%bs. Flatten
  the pool to [hd, bs*2hkv*nblk_pool]; pool_row(cs,h) = (cs%bs) + bs*h + bs*2*hkv*(cs/bs), cs=pos_slot.
  ggml_get_rows(pool_flat, pool_row_i32) replaces the dense kg/vg (minimax-m3.cpp 422-427). V head h
  at +bs*hkv on the head term. h broadcasts over Hd via arange (as the dense tr does).
- The block_table variant (true eviction/memory saving) is v2, AFTER v1 is green.
Pool must have >= n_kv/bs blocks (sized for context; OK). GATE: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1
test-paged-kv-e2e GREEN. This is now a clean transcription -- no logical/physical ambiguity, no
block_table. Store op (CPU+Metal) + all plumbing DONE. 10 commits this session.

## ⚠⚠ GATHER WIRING WRITTEN but DISCONNECTED — GRAPH REUSE bug (2026-08-13, caught by neg controls)
Wrote the full store+gather (ggml_paged_kv_store + pool_row gather, minimax-m3.cpp decode branch),
builds + runs. Gate DS4P_NO_SPEC=1 test-paged-kv-e2e PASSED -- BUT two negative controls PROVED the
pass was FALSE (the pool read is NOT load-bearing; output comes from the DENSE path):
  - DS4P_MSA_NOSTORE=1 (skip the store, pool empty) -> STILL PASSED.
  - scale pool-gather kg by 100x (garbage) -> STILL PASSED, argmax unchanged.
ROOT CAUSE (from DS4P_MSA_POSTRACE): the paged run's gather is 14 DENSE + 1 POOL. The producer DOES
attach pg_ctx (has_batch_info=1, mem_attn_paged non-null) -- but only for ~1 build. The GRAPH IS
REUSED: the reserve/first build has pg_ctx=null (dense), gets cached, and is REUSED for the actual
decode, so the pool branch never runs on the output path. Classic graph-reuse-vs-conditional-path.
⇒ FIX (next session): make the graph NOT reuse a dense build for a paged decode.
  ⚠ DEEPER: DSV4's init_full does NOT attach the paged ctx either (llama-kv-cache-dsv4.cpp:1402),
  yet DSV4 paging works -- so "attach in init_full" is NOT the mechanism. ernie4-5.cpp:97 conditions
  on get_attn_paged() just like MSA. So a WORKING paged arch already handles reserve/reuse; the
  difference must be HOW the reserve graph is built for DSV4/ernie vs MSA. NEXT: trace DSV4's gather
  path with a POSTRACE-style probe (does DSV4's reserve build see pg_ctx non-null? does its graph get
  rebuilt for the real decode?) and copy that mechanism. The fix is whatever keeps DSV4's paged path
  on the OUTPUT decode -- EXACT FIX FOUND: llm_graph_input_msa::can_reuse (minimax-m3.cpp:119) does NOT check paged-ctx
  presence, so a DENSE-built graph passes can_reuse for a PAGED decode and is reused (the bug). FIX:
  add a bool was_paged member to llm_graph_input_msa (set at construction from
  mctx->get_attn_paged()!=nullptr), and in can_reuse add:
      res &= (was_paged == (static_cast<const llama_kv_cache_msa_context*>(params.mctx)->get_attn_paged() != nullptr));
  This forces a REBUILD on any dense<->pool transition, so the output decode gets a POOL graph. Then
  re-apply the store+gather wiring; the NOSTORE + kg*100 negative controls MUST fail (load-bearing);
  gate GREEN = #19 decode paging done. (was: likely the reserve uses init_batch (not init_full) with a live scheduler
  batch, so pg_ctx is present at build time; verify by checking WHEN/HOW the paged reserve happens. Options:
  (a) llm_graph_input_msa::can_reuse (minimax-m3.cpp ~119) must include (pg_ctx != nullptr) in its
      reuse test so a dense->pool transition forces a rebuild; OR
  (b) confirm with LLAMA_GRAPH_REUSE_DISABLE=1 first (should make the wiring's pool path run every
      decode -> then the NOSTORE/garbage controls MUST fail = load-bearing).
The wiring code (store+gather + I64 op support + Metal I64 kernel_paged_kv_store_f32) is REVERTED
(unverified); re-apply from git reflog or re-transcribe (it built cleanly; only reuse blocks it).
★ LESSON: the naive gate PASSED but was entirely dense; the NOSTORE + garbage negative controls
caught it. NEVER trust a paged "pass" without a control that empties/garbages the pool. Store ops
(CPU+Metal, dd56111cf) ARE verified and committed. This is the last bug: graph reuse.

## ⚠⚠⚠ can_reuse fix INSUFFICIENT — output decode uses DENSE graph (pg_ctx null there) [2026-08-13 ~4am]
Re-applied full wiring (I64 op support + Metal kernel_paged_kv_store_f32 + store+gather in
minimax-m3.cpp) + the can_reuse pg_ctx check. Builds. But NOSTORE + kg*100 STILL PASS -> output still
DENSE. Trace (last layer, decode regime n_tps=1): still 14 dense (pg_ctx=null) + 1 POOL. So during
the OUTPUT-producing decode graph build, pg_ctx is NULL (has_paged_batch_info=false at that
init_batch). The can_reuse fix can't help because the output build is dense to begin with (not a
reused stale graph). REAL ROOT CAUSE = the SCHEDULER/init_batch TIMING: for the output decode,
has_paged_batch_info() is false when MSA's init_batch runs -> producer skips -> pg_ctx null -> dense.
NEXT (deep, focused): (a) DECISIVE test -- put kg*100 in the DENSE else-branch; if the gate FAILS,
output IS dense (confirms the 1 POOL build is a reserve, not output). (b) Trace has_paged_batch_info
in the paged scheduler's step/prepare vs when MSA init_batch runs -- WHEN does the scheduler set the
pool's batch info relative to the decode graph build? DSV4 works, so compare: does DSV4's output
decode see has_paged_batch_info=true? The fix is whatever makes has_batch_info true at the MSA
output-decode init_batch (likely the scheduler must set the pool's batch info for the MSA memory the
same way it does for DSV4 -- check llama_paged_scheduler prepare/step for the MSA branch).
⚠ THE WIRING IS UNCOMMITTED IN THE TREE (dev-gated DS4P_PAGED_MSA=1, builds, safe) -- do NOT revert;
it has debug probes (DS4P_MSA_NOSTORE/GARBAGE/POSTRACE) to remove before any commit. Store ops
(dd56111cf) committed+verified. This is the true last blocker: has_paged_batch_info at the MSA decode.

## ★★★ THE GATE IS INSENSITIVE — synth-model token gate CANNOT verify the MSA gather [2026-08-13 4am]
DECISIVE finding: garbaging the DENSE gather branch (kg*100) ALSO PASSES -- just like garbaging the
POOL branch and NOSTORE. So NEITHER K/V path affects the compared output. The synth minimax-m3 +
test-paged-kv-e2e token comparison (top-5 overlap>=4) is STRUCTURALLY INSENSITIVE to the K/V gather:
random weights + 1-token decode -> attention contribution doesn't move the argmax. ⇒ EVERY paged
"PASS" tonight (and the earlier Tier-1 "green") measured NOTHING about the gather. The gather wiring
may be CORRECT -- the gate just can't see it. (gate-shape-excludes-the-defect + impossible/insensitive-
bar scars, at the root.)
⇒ VERIFICATION needs a SENSITIVE instrument BEFORE trusting any gather result:
  1. Make the gate compare LOGITS at tight tolerance (test-paged-kv-e2e already captures
     prefill_logits, line ~171 -- compare paged vs non-paged logit VECTORS, not just top-5 tokens;
     a K*100 garbage MUST then diverge). This is the cleanest fix and CPU-doable.
  2. OR use a real minimax-m3 GGUF where attention matters (needs the download).
  3. Sanity: a garbage probe (kg*100 in the ACTIVE path) MUST fail the sensitive gate, else the gate
     is still blind.
Only once a garbage probe FAILS is the gate trustworthy; THEN re-run NOSTORE (must fail) + normal
(must pass) to verify the real gather. The has_paged_batch_info/scheduler question is SECONDARY --
first get a gate that can measure anything. Wiring is uncommitted in tree (builds, dev-gated, debug
probes DS4P_MSA_NOSTORE/GARBAGE/DENSEGARBAGE/POSTRACE). Store ops (dd56111cf) committed+verified.

## ★★★★ DEFINITIVE: the SYNTH model cannot verify the MSA gather -- a REAL model is required [2026-08-13]
Built a SENSITIVE decode-logit gate (compare first-decode logit VECTORS, tol 1e-2; test-paged-kv-e2e).
Result: garbaging the gather K by 100x changes the decode logits by max_abs_diff = 3.73693e-08 --
BIT-IDENTICAL to the normal run. ZERO effect, not just small. So the synthetic minimax-m3 (random
weights) has a NEGLIGIBLE/ABSENT attention contribution to the output; NO gate at ANY tolerance can
verify the gather on it. Every synthetic paged "PASS" is meaningless for the gather; every negative
control passes because the gather's output doesn't reach the logits on this model.
⇒ #19 GATHER VERIFICATION IS BLOCKED ON A REAL minimax-m3 GGUF (unsloth/MiniMax-M3-GGUF), not on any
code. The store op (CPU+Metal) IS verified (byte-exact test-paged-vs-cpu). The gather WIRING is
written + builds (uncommitted, dev-gated) but CANNOT be trusted until run on a real model where
attention matters. NEXT: (1) provision a real minimax-m3 (download or HF+convert -- HIS resource
call, ~large); (2) run test-paged-kv-e2e (now with the sensitive decode-logit check) on it: a garbage
probe MUST fail, NOSTORE MUST fail, normal MUST pass; (3) THEN debug the has_paged_batch_info/gather
if it diverges. The sensitive decode-logit gate ADDITION to test-paged-kv-e2e is a genuine keeper
(makes the gate able to see decode-path defects on real models).
CONFIRMED by sanity check (not assumed): garbaging the ENTIRE MSA attention output (cur*100) moves
the decode logit 3.73693e-08 -> 3.75649e-06 (exactly 100x). So the attention path IS connected +
propagates; its magnitude is just ~1e-8 (negligible vs O(1) logits) on the synth model. The gather is
wired correctly into the compute; it's unmeasurable only because synth attention is tiny. NOT a
disconnection bug -> a real model IS the fix. (Good: I garbaged the attn OUTPUT to distinguish
"disconnected" from "negligible" instead of guessing.)
LESSON (biggest of the session): a synthetic random-weight model can be STRUCTURALLY BLIND to the very
feature under test. Before trusting ANY gate, prove a deliberate defect (garbage/NOSTORE) FAILS it.
Store ops (dd56111cf) committed+verified; gather + sensitive-gate uncommitted (debug probes to strip).

## ✅✅✅ GATHER MACHINERY VERIFIED (2026-08-13, CPU round-trip, no model) — sidesteps synth blindness
Wrote a standalone ggml round-trip test (msa_gather_roundtrip.cpp in scratchpad): store known K/V into
a zeroed f16 pool via ggml_paged_kv_store, gather back via the pool_row formula (get_rows), compare.
RESULT: PASS, max_err=0, bad=0/104 (bit-exact). ⇒ the store op + gather index math + their COMPOSITION
are PROVEN CORRECT, CPU-only, independent of the synth model's attention magnitude. This is exactly
what the model-level gate couldn't measure.
⇒ #19 STATUS REFRAMED: the gather CORRECTNESS is VERIFIED (not blocked on a real model). What remains
is only the GRAPH PLUMBING -- does the MSA output decode actually TAKE the pool path (has_paged_batch_info
true at that init_batch so pg_ctx is non-null)? That's a wiring/scheduler question, and its end-to-end
confirmation wants a real model, but the ops it wires are now proven. NEXT: (a) make msa_gather_roundtrip
a permanent test (verifies the store+gather ops forever); (b) resolve has_paged_batch_info at the MSA
decode (compare DSV4's scheduler->init_batch timing); (c) real-model end-to-end sanity when available.
Test recipe: c++ -std=c++17 msa_gather_roundtrip.cpp -I ggml/include -L build-metal/bin -lggml -lggml-base
-lggml-cpu -Wl,-rpath,build-metal/bin ; store slot=t, pool_row(t,h=0)=(t%bs)+bs*2hkv*(t/bs).

## ⭐ START HERE — #19 state (2026-08-13, consolidated from this session's appends below)

DONE + PUSHED (origin/ds4-ports, all dev-gated behind DS4P_PAGED_MSA=1, zero effect on shipping):
- Tier 0 (4e3d85e60): MSA gets a paged pool + scheduler branch; runtime-gated (paged-probe accepts).
- Tier 1 gate (5733cfb87): DS4P_NO_SPEC on test-paged-kv-e2e = clean basic-decode gate (MSA lacks
  draft rollback, so the default spec probes false-fail it).
- Tier 1 plumbing (7b3e0a587): MSA context get_attn_paged() + producer (init_batch_with_ubatches).
- Basic paged decode PASSES (paged==non-paged, graph still reads DENSE; pool allocated, unread).

THE ONE REMAINING THING = make the MSA graph WRITE+READ the pool (minimax-m3.cpp sparse branch).
⚠ SCOPE (corrected by attempting, 2026-08-13): the pool STORE is FUSED inside ggml_paged_attn; there
is NO store-only op, and get_k==get_v==one interleaved K+V buffer. MSA does its own sparse attention
(build_attn_msa_fa = stock ggml_flash_attn_ext over a gathered top-k), so it CANNOT call ggml_paged_attn.
⇒ two sub-pieces, do together, gate at completion:
  1. STORE-ONLY PATH — either (a) NEW op ggml_paged_kv_store(kv_tensor, k_cur, v_cur, write_slots,
     block_size) [Metal+CPU write kernel laying K/V into the interleaved buffer the read kernels
     expect], or (b) manual ggml_set_rows into get_kv_tensor matching that interleave (fragile).
  2. GATHER — read the pool buffer (pg_ctx->get_k(il)) and translate the gather cell cs → pool row
     via block_table[cs/bs]*bs + cs%bs, ggml_get_rows, replacing the dense reads at minimax-m3.cpp
     368-369/422-427. kv_idx in lockstep (or keep dense for v1).
INPUT TENSORS: build_attn_inp_kv_paged(pg_ctx) → paged_write_slots/paged_block_table (auto-filled).
GATE (CPU, no real model): mkdir -p D && test-llama-archs -a minimax-m3 -o D  (dumps synth gguf);
then DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e -m D/minimax-m3-moe.gguf must stay/GO GREEN.
Optional after: MSA draft rollback (seq_rm of rejected drafts) for the FULL spec e2e.


### ⚠ STORE-ONLY PATH CONFIRMED = NEW KERNEL (2026-08-13, by pool-layout inspection)
Pool tensor = ggml_new_tensor_4d(type, head_dim, block_size, 2*n_head_kv, n_blocks)
(llama-kv-cache-paged.cpp:202). Block-structured: token slot s → block=s/bs, pos=s%bs; the K heads
are [0,hkv) and V heads [hkv,2hkv) on dim 2. bs (dim1) and n_blocks (dim3) are NON-ADJACENT, so the
per-token slot is NOT a single contiguous ggml row dimension ⇒ ggml_set_rows CANNOT express the
token→slot scatter. Option (b) is DEAD. Option (a) stands: a dedicated write op/kernel
ggml_paged_kv_store(kv_tensor[hd,bs,2hkv,nblk], k_cur[hd,hkv,ntok], v_cur, write_slots[ntok]) that,
per token t, writes k_cur[:,:,t] into kv_tensor[:, ws%bs, 0:hkv, ws/bs] and v_cur into
[:, ws%bs, hkv:2hkv, ws/bs] (ws=write_slots[t]). Metal + CPU. That is the concrete kernel spec.
Then the gather reads the same 4D buffer with block_table translation. This is #19's remaining arc.


### ✅ INDEX MATH VERIFIED (2026-08-13, standalone CPU round-trip test, PASS)
The pool scatter/gather arithmetic is proven correct in isolation (msa_pool_index_test.cpp in the
session scratchpad, 320 values exact). USE THESE FORMULAS in the write op + gather:
  off(i0,i1,i2,i3) = i0 + hd*(i1 + bs*(i2 + 2*hkv*i3))   // pool [hd, bs, 2*hkv, nblk]
  STORE token t (write_slot ws), K head h dim d: pool[off(d, ws%bs, h,       ws/bs)] = k_cur
                                                V head h dim d: pool[off(d, ws%bs, hkv+h,   ws/bs)] = v_cur
  GATHER cell cs, K head h dim d: pool[off(d, cs%bs, h,     block_table[cs/bs])]
                                 V head h dim d: pool[off(d, cs%bs, hkv+h, block_table[cs/bs])]
So the ggml_paged_kv_store CPU kernel is a direct transcription of the verified STORE loop; the Metal
kernel the same with one thread per (token, head, dim). The gather uses the verified GATHER formula.
The index arithmetic — the error-prone core — is DONE and proven; remaining is ggml op registration
(enum + CPU + Metal + dispatch) and graph wiring, then the green gate.

--- detailed session log (chronological, some later entries CORRECT earlier ones) ---

# PLAN — paging minimax-m3 (MSA), grounded in the code (2026-08-12)

## Verdict from scoping (CPU-only, source-read)

minimax-m3 uses **MSA (MiniMax Sparse Attention)**: a `llama_kv_cache_msa` holding TWO standard
`llama_kv_cache` instances — `kv_base` (K/V) and `kv_idx` (the indexer) — each sized `kv_size`
cells (the FULL context; `llama-kv-cache-msa.cpp:34,46`). The sparsity is in the **attention**
(which blocks it selects), NOT in the **storage** (it stores every position). So:

- **Paging IS valuable here** — same dense-storage memory ceiling as DSV4, and arguably a better
  fit: sparse attention means most stored blocks are COLD at any step, so paging can evict them.
  (My first "it's already sparse, don't bother" hypothesis was WRONG — checked the constructor.)

- **KEY SIMPLIFICATION — no new kernel.** `build_attn_msa_fa` (`models/minimax-m3.cpp:190`) calls
  standard `ggml_flash_attn_ext` over K/V that are **already gathered** — block selection runs
  UPSTREAM and hands the attention dense, pre-selected blocks. So the paged champion kernel is
  NOT on the critical path. Paging lives at the STORAGE + GATHER layer, and the attention stays
  the stock dense flash_attn_ext. This is smaller than DSV4, which needed the masked champion.

- **msa_enabled = fa_on && streams_ok** (`minimax-m3.cpp:226-228`): MSA already requires flash
  attention ON and single-stream (n_seq_max==1 or !kv_unified) — aligns with the paged
  single-sequence contract. No new constraint.

## Why it's still an arc (not a wire)

The scheduler's `dynamic_cast` chain (`llama-paged-scheduler.cpp:33-60`) accepts paged / hybrid /
hybrid-iswa / iswa / dsv4 — there is NO `llama_kv_cache_msa` branch, so MSA falls to the
"non-paged memory type" refusal. And there are TWO caches to page, plus a gather that must read
from the pool. Reuse from the DSV4 lane is heavy but not total.

## Staged tiers (DSV4-pattern: all-or-nothing per tier, gate each)

**Tier 0 — accept the memory.** Give `llama_kv_cache_msa` a paged pool for `kv_base` (reuse
`llama_kv_cache_paged`, the DSV4 donor pattern at `llama-model.cpp:2224`). Add a scheduler branch
for the MSA type. GATE: server starts under --kv-paged on minimax-m3 without the refusal, pool
allocated. ⚠ Tier 0 alone = pool no graph reads = the silent static-vs-static trap; keep it
DEV-GATED until Tier 1+2 land (the exact DS4P_PAGED_DSV4 lesson).

**Tier 1 — the gather reads the pool.** The block-selection / pos↔cell maps
(`llama_kv_cache_msa_context::set_input_cell_pos` / `set_input_pos_slot`,
`llama-kv-cache-msa.h:136-139`) currently resolve to dense cache slots. Route them through the
paged block table so the gather pulls selected blocks from pool blocks. `kv_idx` pages in lockstock
with `kv_base` (they are kept cell-synced by construction — `llama-kv-cache-msa.cpp:10`). GATE:
byte-exact vs static at single-chunk, marker-verified the gather hit the pool.

**Tier 2 — multi-chunk + eviction.** Confirm the pos↔cell maps stay correct across ubatch chunks
when cold blocks are evicted/paged (the sparse win: unselected blocks leave residency). GATE:
byte-exact vs static at multi-chunk prefill AND a long-context rung where eviction actually fires;
determinism x8+; the concurrency-race class from DSV4 #21 re-checked (untracked-scratch barrier).

**Then flip** (dev-gate removal) once 0+1+2 are byte-exact + deterministic, same as DSV4.

## Reuse ledger
- `llama_kv_cache_paged` pool + init/init_multi — REUSE as-is.
- Scheduler acceptance pattern — COPY the dsv4 branch shape.
- Champion/masked kernel — NOT needed (attention is pre-gathered dense flash_attn_ext).
- The pos↔cell → block-table mapping — NEW (the real work of Tier 1).
- Two-cache lockstep paging — NEW-ish (dsv4 had one composite cache; here two synced caches).

## Estimate
~3 focused sessions (Tier 0 / Tier 1 / Tier 2+flip), each gate-bounded. Smaller than DSV4 because
no new kernel. Needs the box's `unsloth/MiniMax-M3-GGUF` (present, `PLAN-paged-arch-support.md:328`).

## Open decision (his)
minimax-m3 is a NON-SHIP model; DSV4 (the ship target) is done and clears the bar. This arc extends
the CONTEXT-CEILING program to another giant — aligned with the mission, but a real 3-session
spend. GO / NO-GO is his. This plan makes the arc concrete and de-risked so the decision is
informed, not blind.

---

## Tier 0 STATUS (2026-08-12): ✅ DONE — runtime-gated + PUSHED (4e3d85e60)
The model blocker was FALSE — no giant GGUF needed: `tests/test-llama-archs.cpp:176` builds a
SYNTHETIC minimax-m3 in-memory (random weights, n_layer=2), so Tier 0 is CPU-gatable locally.

Tier 0 done (mirror of the DSV4 donor exactly):
- `llama_kv_cache_msa`: + `set_attn_paged` / `get_mem_attn_paged` + `unique_ptr<llama_kv_cache_paged>`
  member (needed `#include "llama-kv-cache-paged.h"` — llama-kv-cache.h does NOT transitively
  provide it, unlike dsv4.h via iswa.h; the header-defaulted dtor needs the complete type).
- `llama-paged-scheduler.cpp`: fifth `dynamic_cast<llama_kv_cache_msa*>` branch (after dsv4).
- `llama-model.cpp` MINIMAX_M3 case: attaches a pool ONLY under `DS4P_PAGED_MSA=1` (dev-gate,
  all-or-nothing like DSV4). Default --kv-paged still refuses (no graph reads yet = Tier 1).
Compile-clean, dev-gated (zero default effect, shipping path + all tests untouched).

### THE TIER 0 GATE (next session — run this, then push da30f0524):
Add a paged-accept probe to `test-llama-archs.cpp` (it already builds synthetic minimax-m3 +
creates a context via `get_model_and_ctx`). For arch==MINIMAX_M3:
1. build a context with `ctx_params.kv_paged = true` (+ setenv DS4P_PAGED_MSA=1),
2. call `llama_paged_scheduler_init(ctx)` (public API, include/llama.h:1671),
3. EXPECT non-null WITH the flag (Tier 0 accepts) and null WITHOUT it (dev-gate still refuses).
Positive control: run the same probe on DSV4 (LLM_ARCH_DEEPSEEK4 + DS4P_PAGED_DSV4=1) — must also
accept, proving the probe itself works. ⚠ Scope the probe to those archs; don't perturb the
all-arch equivalence loop. GREEN → push da30f0524 → Tier 1 (gather reads the pool).

## Tier 1 MAP (source-read 2026-08-12, ready to implement next session)
The MSA KV STORAGE + GATHER, all in `src/models/minimax-m3.cpp` sparse-layer branch (il >= n_layer_dense_lead):
- **Store** (366-367): `mctx_cur->cpy_k/cpy_v(Kcur/Vcur, inp_attn->get_k_idxs()/get_v_idxs(), il)` → writes into
  kv_base (dense). Tier 1: writes must land in the POOL blocks (the write-slots must be pool slots).
- **Read** (368-369): `k = mctx_cur->get_k(il)`, `v = mctx_cur->get_v(il)` → the FULL dense cache tensor.
  Tier 1: this must expose the pool's storage instead.
- **Indexer** (351-352): `mctx_idx->cpy_k(...)` + `mctx_idx->get_k(...)` — kv_idx, same treatment (lockstep).
- **Gather** (422-428): `k3/v3 = view_3d(k/v, ...)`; `kg = get_rows(k3, tokr)`; where
  `tokr = cs*HKV + h` and `cs = pos_slot[tokj]` (pos→cell map). ★ THE HARD PART: this indexes a FLAT,
  CONTIGUOUS cell space. The paged pool is block-table-indirected and NON-contiguous, so `tokr` must be
  remapped cell→(block,offset) through the pool's block table. pos_slot / pos_slot_f (from the MSA
  context's set_input_pos_slot, llama-kv-cache-msa.h:138) is where the cell resolution happens — route it
  to pool cells, OR insert a block-table gather between cs and the final row index.
- **prefill branch** (below 434, msa_decode==false): a parallel gather to audit the same way.
GATE (CPU-only, no model): extend `test-llama-archs --paged-probe` to COMPARE logits paged (DS4P_PAGED_MSA=1)
vs non-paged for the synthetic minimax-m3 — byte-exact (or ~1 ULP) == Tier 1 correct. The equivalence
harness already compares logits across backends; reuse that comparator. Negative control: a deliberately
wrong block-table map must make the comparison FAIL (else the gate can't see corruption).
⚠ CORRUPTION-PRONE index arithmetic — implement incrementally, gate each step; this is the tier the
byte-exact gate exists for. NO new kernel (attention stays build_attn_msa_fa's stock ggml_flash_attn_ext).

## Tier 1 GATE — infrastructure findings (2026-08-12, proven by attempts)
GOAL: compare paged (DS4P_PAGED_MSA=1) vs static logits/tokens for a synthetic minimax-m3, byte-exact
== Tier 1 correct. Established this session:
- ✅ `test-llama-archs -a minimax-m3 -o <DIR>` (DIR must EXIST) dumps a loadable synthetic gguf
  (~7.5MB, random weights, MoE). No giant model / no download needed.
- ✅ It LOADS + decodes in llama-server statically (raw completion had content + tokens_predicted=4,
  deterministic).
- ❌ The SERVER JSON completion path is a dead end for the gate: filler vocab (`<x%u>`) won't
  tokenize normal text; the garbage-byte content breaks JSON capture; `n_probs`/token-id-array
  prompts hit a 500 "Content-only format" error. Three walls — do NOT build the gate on server JSON.
- ★ RIGHT GATE = C++ LOGIT CAPTURE. `tests/test-paged-kv-e2e.cpp` already drives the paged scheduler
  (kv_paged ctx → llama_paged_scheduler_init → add_request → step loop) AND captures logits at the
  C++ level (path_result, encoding-immune, scheduler-correct). Extend it to accept the synthetic gguf
  (or build the synthetic in-memory like test-llama-archs) and compare paged-vs-static logit vectors
  for minimax-m3. Two-process for the static flag, or two contexts in one process if the dev-gate is
  read per-context. Negative control: a deliberately-wrong block map must make the logits DIVERGE.
- ⚠ NOTE the static-flag-once constraint: DS4P_PAGED_MSA is a process-static, so paged and static
  logits come from two runs (or make the flag per-context for a one-process gate).
NEXT SESSION ORDER: (1) build this C++ logit gate, establish Tier-0 baseline (paged==static since the
graph still reads dense), (2) implement the Tier 1 gather rewrite incrementally, gate each step.

## Tier 1 GATE IS LIVE (2026-08-12) + first failure pinned
✅ THE GATE: `DS4P_PAGED_MSA=1 test-paged-kv-e2e -m <synth-mm3.gguf>` — it already runs non-paged
ref + paged and compares token sequences (compare_results, N_COMPARE window). No new harness code
needed; feed it the save_models synthetic gguf. NO server JSON, encoding-immune, scheduler-correct.
BASELINE MEASURED: non-paged ref PASSES (16 tokens, 260-vocab); paged path FAILS at llama_decode
"failed to initialize batch" (llama-context.cpp ~1560). This is the CORRECT Tier-0 state — Tier 0
attaches the pool + the scheduler accepts it, but paged DECODE needs the graph to read the pool
(Tier 1). So the gate correctly shows RED now and turns GREEN when Tier 1 lands.
TIER 1 START (next session, in order):
1. Diagnose "failed to initialize batch" for MSA+paged (which of llama-context.cpp:1560/1903/3589;
   likely memory->init_batch for the MSA composite under the paged scheduler). Fix so paged decode
   RUNS (may still mismatch until the gather is routed).
2. Route the gather (minimax-m3.cpp store 366-367 / read 368-369 / gather 422-427) through the pool
   block table. Gate turns GREEN (paged tokens == non-paged) when correct.
3. Multi-chunk + eviction (Tier 2), then flip.
Gate each step against the live test-paged-kv-e2e. Synthetic gguf recipe:
`mkdir -p D && test-llama-archs -a minimax-m3 -o D` → D/minimax-m3-moe.gguf.

## ★★ Tier 1 BLOCKER DIAGNOSED (2026-08-12) — TWO-LEDGER position mismatch (reframes the tier)
Ran the live gate; paged decode dies with the EXACT mechanism (not a vague "graph reads dense"):
  "inconsistent sequence positions: last stored (KV cache) for seq 0 X=55; input batch starts
   Y=55; required Y = X+1"  (preceded by DS4P-CHECKOUT allocate n=4 free_before=24)
⇒ the paged SCHEDULER tracks position via the POOL (checks out blocks, advances its own count),
while the MSA memory tracks position via its OWN dense kv_base/kv_idx caches. TWO LEDGERS, and they
collide off-by-one at the second decode. This is the two-ledgers scar class, at the arch level.
REFRAME of Tier 1: it is NOT merely "route the gather through the pool block table". The MSA
memory's STORAGE and POSITION tracking must be UNIFIED onto the pool so there is ONE ledger the
scheduler and the graph agree on — i.e. the pool must BACK kv_base (replace its dense slots), not
coexist beside it. Compare DSV4: its attn sub-cache BECAME the pool (Tier 1 routed reads there);
MSA needs the same — kv_base's storage IS the pool, and kv_idx in lockstep.
⇒ Tier 1 first task is architectural: make llama_kv_cache_msa's kv_base use the paged pool as its
backing store (single position ledger), THEN the gather (minimax-m3.cpp 422-427) reads pool blocks.
Only after that does the gate's Y=X+1 pass. This is a bigger Tier 1 than "just the gather" — plan
for it. Gate stays: DS4P_PAGED_MSA=1 test-paged-kv-e2e -m <synth-mm3.gguf>, RED until the ledger is one.

## Tier 1 fix path — VERIFIED (2026-08-12): producer-only is INSUFFICIENT
Checked the DSV4 precedent: its init_batch adds a "producer" (mem_attn_paged->init_batch_with_ubatches,
dsv4.cpp:1320) that feeds the pool the SAME ubatches as the dense caches. Tempting to mirror on MSA.
BUT VERIFIED it does NOT fix the failure: the "inconsistent sequence positions Y=X+1" check fires in
llama_batch_allocr::init (llama-context.cpp ~1903) BEFORE memory->init_batch is called, and it reads
the memory's seq_pos_max (MSA returns kv_base->seq_pos_max = 55) vs the scheduler's batch pos (55).
So the SCHEDULER and the MSA memory disagree on the stored position by one — a ledger split that a
producer inside init_batch cannot fix (the check is upstream of it). ⇒ CONFIRMS Tier 1 = the pool
must BACK kv_base's storage so seq_pos_max and the scheduler's pool-position are ONE number. The
producer (init_batch_with_ubatches) is NECESSARY but not SUFFICIENT; the storage unification is the
core. Do NOT commit a producer-only patch — it stays RED and half-changes the tree.
NEXT session, in order: (1) make llama_kv_cache_msa's kv_base pool-backed (single position ledger)
so seq_pos_max == scheduler pool pos; (2) add the DSV4-style producer in init_batch + the
set_attn_paged_ctx plumbing on llama_kv_cache_msa_context (mirror dsv4 context); (3) route the gather
(minimax-m3.cpp 422-427) to read pool blocks. Gate each against DS4P_PAGED_MSA=1 test-paged-kv-e2e.

## ★★★ Tier 1 diagnosis CORRECTED (2026-08-12, by DS4P_MSA_POSTRACE data) — NOT a storage rewrite
Traced kv_base->seq_pos_max across the paged decode. Non-paged ref: 52→53→54 (clean +1). PAGED:
52→53→**55** (skips 54, +2 in one step), and the failure coincides with test-paged-kv-e2e's
SPECULATIVE DRAFT PROBE ("draft probe took the REJECT branch, n_acc=1 of 2"). ⇒ MECHANISM: the
scheduler runs speculation, a draft is REJECTED, but MSA's memory does NOT roll back the rejected
draft's position — kv_base stays at 55 while the scheduler expects it free → the Y=X+1 collision.
★ Basic paged decode (positions 52, 53) ADVANCED CORRECTLY. So my earlier "must make kv_base
pool-backed / rewrite storage" was TOO BROAD — the collision is SPECULATION-ROLLBACK-specific, and
may be a TEST-PROBE artifact (the draft/multi-accept probe), not a core paged-serving blocker.
NEXT session, revised order:
1. Determine if the draft probe is test-only. Build a MINIMAL paged-decode gate WITHOUT speculation
   (strip the draft probe from a copy of run_paged, or add a --no-spec flag) and re-run. If basic
   paged decode matches non-paged tokens → Tier 1's remaining work is JUST the gather (graph reads
   pool), speculation rollback is a SEPARATE (optional) concern.
2. If basic decode matches: implement the gather rewrite (minimax-m3.cpp 422-427) against the
   minimal gate. Much smaller than the "rewrite storage" fear.
3. Speculation rollback (MSA seq_rm of rejected drafts) only if paged serving needs spec-decode.
This is verify-the-premise working: the data REJECTED my own broader hypothesis before I built it.

## ★★★ Tier 1 DEFINITIVE (2026-08-12): the gate's failure is SPECULATION, basic decode is fine
Read test-paged-kv-e2e run_paged: it runs (a) a SENTINEL branch (lines 201-210, negative n_acc,
batch-offset layout, alternating every step) and (b) a MULTI-ACCEPT DRAFT PROBE (218-242, decodes a
2-token draft batch, batch_lens==2). These decode EXTRA draft tokens into MSA's kv_base (the 53→55
+2 jump) and rely on the scheduler's draft rollback/accounting — which MSA's memory does NOT
implement. Basic paged decode (pos 52, 53) advanced correctly. ⇒ test-paged-kv-e2e is NOT a clean
BASIC-paged gate for MSA; its speculation exercises a feature MSA lacks.
CONVERGED Tier 1 scope (three data-driven corrections; each rejected the prior broader guess):
- Basic paged MSA decode advances positions correctly at Tier 0. Likely matches non-paged already
  (graph reads dense) — VERIFY with a no-spec gate.
- Tier 1 REAL work = the gather reads the pool (minimax-m3.cpp 422-427). Smaller than "rewrite storage".
- Speculation rollback (draft reject → MSA seq_rm) is a SEPARATE, optional concern (paged serving
  doesn't require spec-decode).
NEXT session: (1) copy run_paged → run_paged_basic WITHOUT the sentinel/draft/multi-accept probes,
just prepare→decode→sample→update(normal)→compare tokens to non-paged. That is the clean Tier-1 gate.
(2) If basic paged==non-paged at Tier 0, do the gather rewrite against it. (3) spec-rollback only if needed.

## ✅✅ Tier 1 GATE IS GREEN for basic paged decode (2026-08-12, commit 5733cfb87)
DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e -m <synth-mm3.gguf> → PASSED (paged and non-paged
both 16 identical tokens). This PROVES the scheduler/MSA-memory/decode plumbing is consistent for
basic paged decode. The entire "two-ledger / rewrite storage" fear is DEAD — it was only the test's
speculation probes (which MSA lacks draft-rollback for). 
⇒ Tier 1 is now REDUCED to ONE thing: route the gather to read the pool. At Tier 0 the gate passes
TRIVIALLY because the MSA graph still reads DENSE kv_base (pool allocated but unread). Tier 1 = make
minimax-m3.cpp's gather (422-427: tokr = cs*HKV+h) read POOL blocks via the block table, and keep
this gate GREEN. Develop directly against DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e.
Optional later: MSA draft rollback (seq_rm of rejected drafts) to make the FULL (speculative) e2e
pass — only if paged serving wants spec-decode.
7 commits this session on origin/ds4-ports: 4b62ac77b #21 · e2d787613 flip · 15353dff6 bs64 ·
edda43e5f 512-sink · 4e3d85e60 #19-Tier0 · 5733cfb87 #19-Tier1-gate.

## Tier 1 GATHER — fully grounded (2026-08-12), ready to execute in one focused pass
Pool API (llama-kv-cache-paged.h): get_k/get_v(layer) = pool storage views; get_block_table() =
[batch, max_blocks] cell→block map; get_write_slots() = [n_tokens] where to write this ubatch;
get_kv_tensor(layer) = raw pool tensor. The scheduler populates write_slots + block_table per ubatch.
THE WORK (coupled store+gather → pool, in minimax-m3.cpp sparse branch):
- STORE (366-367): currently mctx_cur->cpy_k/cpy_v to DENSE kv_base. Add/route a write to the POOL
  at get_write_slots() (like DSV4's funnel write phase). 
- READ+GATHER (368-369, 422-427): currently k=mctx_cur->get_k (dense), kg=get_rows(k3, tokr) with
  tokr=cs*HKV+h over a FLAT cell space. Route to the pool: k=pool get_k, and translate each cell cs
  to a POOL ROW via the block table: pool_row = block_table[cs/bs]*bs + (cs%bs), then the same +h.
  ⚠ MSA uses ggml_get_rows (generic), NOT the champion kernel, so the block-table indirection must
  be an explicit graph computation (a get_rows through block_table, or index arithmetic), not a
  kernel arg. This is the intricate part.
⚠ VERIFICATION-COUPLED: the gate (DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e) can only confirm
correctness when BOTH store and gather use the pool (a pool write is invisible while reads stay
dense). So do it as ONE unit, then gate. Green = Tier 1 done. Then remove the dense store.
This is a single coherent graph change — plan a focused session, gate at completion. Everything
before it (unblock, Tier 0, gate, diagnosis) is DONE and de-risked to zero unknowns.

## ✅ Tier 1 PLUMBING done (2026-08-12, commit 7b3e0a587) — only the graph gather remains
MSA context now has get_attn_paged() + set_attn_paged_ctx() + ctx_attn_paged (dsv4-mirror), and
init_batch has the PRODUCER (init_batch_with_ubatches → set_attn_paged_ctx). Gate stays GREEN
(DS4P_NO_SPEC): plumbing does not regress basic decode. The graph can now reach the pool ctx via
get_attn_paged().
⇒ ONLY remaining Tier 1 work = the GRAPH GATHER (minimax-m3.cpp sparse branch): use get_attn_paged()
to (a) write K/V to the pool at its write_slots (instead of / in addition to dense kv_base 366-367),
(b) read via the pool's get_k + block-table-translated indices (422-427: pool_row =
block_table[cs/bs]*bs + cs%bs, then +h). Verified only at completion (store+read coupled). Gate:
DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e must stay GREEN. That is the single focused change
left. 8 commits this session on origin/ds4-ports (…5733cfb87 gate · 7b3e0a587 Tier1-plumbing).

## Tier 1 gather is BESPOKE — no helper (2026-08-12, confirmed)
Standard archs (ernie4-5, dflash, eagle3, …) page by ONE call: build_attn_paged_or_null(pg_ctx,
Q,K,V,…) — an encapsulated helper that stores to the pool + reads + runs attention via the kernel
(champion/scalar). MSA CANNOT use it: MSA does SPARSE block-selection attention (custom top-k gather
+ ggml_flash_attn_ext over selected blocks, minimax-m3.cpp 381-434), not dense attention. So MSA's
Tier 1 is the ONLY bespoke paging integration in the tree:
  (a) STORE: manually write Kcur/Vcur to the pool via pg_ctx (get_write_slots + a cpy), not the dense
      mctx_cur->cpy_k (366-367).
  (b) GATHER: read the pool's get_k/get_v (via pg_ctx) and translate the gather's cell indices to
      pool rows through the block table (block_table[cs/bs]*bs + cs%bs) inside the ggml_get_rows path
      (422-427). kv_idx in lockstep.
No helper hides this; it is bespoke ggml graph construction with intricate index arithmetic — the
deepest single change in the whole DSV4/MSA paging effort. Everything AROUND it is done (unblock,
Tier 0, gate, NO_SPEC gate, context plumbing + producer). Gate: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1
test-paged-kv-e2e must go/stay GREEN; verified only at completion (store+gather coupled). One focused
session. 8 commits this session set every rung up to it.

## Tier 1 gather — EXACT ggml ops (2026-08-12, final grounding)
STORE to pool: ggml_set_rows(pool_k_tensor, Kcur, write_slots_tensor) [+ pool_v with v]. The pool
tensor comes from pg_ctx/mem_attn_paged get_k(il); write_slots from get_write_slots() but must be a
GRAPH INPUT TENSOR (int32), so the MSA context needs a set_input_write_slots() (build the input +
fill from get_write_slots()) — additional plumbing beyond the producer.
READ+GATHER from pool: pool_k is block-laid-out; the gather's cell index cs (from pos_slot) → pool
row = block_table[cs/bs]*bs + (cs%bs). block_table also a graph input tensor (set_input_block_table).
Then ggml_get_rows(pool_k, translated_rows) replaces the dense k3/get_rows at 422-427. kv_idx same.
So the remaining Tier 1 is FOUR coupled pieces: (1) set_input_write_slots + set_input_block_table on
the MSA context (input-tensor plumbing), (2) ggml_set_rows store to pool, (3) block-table index
translation, (4) ggml_get_rows read from pool. Verified only together via the NO_SPEC gate.
This is the full remaining scope, grounded to the op level. A focused session lands it; do NOT
fragment. Reference ggml_set_rows usages: llama-graph.cpp:3574 (kq_mask), :2482 (moe groups).

## Tier 1 — no shortcut exists (2026-08-12, confirmed at op level)
ggml_paged_attn (build_attn_mha_paged, llama-graph.cpp:3001) is a FUSED store+attend kernel — there
is NO store-only paged op to reuse, and it does DENSE attention (unusable for MSA's sparse gather).
So MSA's Tier 1 is irreducibly bespoke: build write_slots+block_table as graph input tensors
(machinery exists in build_attn_paged_or_null's input struct, ~1140-1260 — reuse or replicate),
ggml_set_rows to STORE Kcur/Vcur into the pool master buffer at write_slots, then translate the
gather cells via block_table and ggml_get_rows to READ. Four coupled pieces, no helper. Confirmed by
reading every candidate path. This is the single remaining change in #19; everything else is DONE +
pushed (8 commits). Fresh focused session; gate at completion (DS4P_NO_SPEC test-paged-kv-e2e green).

## Tier 1 gather — COMPLETE RECIPE (2026-08-13, all machinery identified)
Every function MSA needs, found:
1. INPUT TENSORS: `auto * inp_paged = build_attn_inp_kv_paged(pg_ctx);` (llama-graph.cpp:4922,
   cached as cached_inp_paged) → gives inp_paged->paged_write_slots, ->paged_block_table,
   ->paged_context_lens, ->paged_batch_offsets/lens (all graph input tensors, auto-filled).
2. POOL BUFFER: `ggml_tensor * pool_kv = pg_ctx->get_k(il);` — INTERLEAVED K+V heads (see the
   src[3] contract note at llama-graph.cpp:342), so K and V share one buffer; the store writes both
   and the gather indexes accounting for the interleave.
3. STORE: ggml_set_rows(pool_kv, <K+V interleaved for this ubatch>, inp_paged->paged_write_slots)
   replaces the dense mctx_cur->cpy_k/cpy_v (minimax-m3.cpp 366-367).
4. READ+GATHER: pool_kv is block-laid-out; translate the gather's cell cs → pool row via
   block_table (inp_paged->paged_block_table[cs/bs]*bs + cs%bs), then ggml_get_rows(pool_kv, rows)
   replaces the dense k3/get_rows (422-427). kv_idx (indexer) in lockstep with its own pool (or
   keep kv_idx dense for v1 if only kv_base is paged — decide at implementation).
IMPLEMENTATION NOTES: watch the interleaved K+V layout (get_k returns both), the set_rows/get_rows
tensor SHAPES (ne must match), and that MSA already builds inp_attn (build_attn_inp_kv_msa) — the
paged input is ADDITIONAL. Gate at completion: DS4P_NO_SPEC=1 DS4P_PAGED_MSA=1 test-paged-kv-e2e green.
This is the complete op-level recipe. #19 closes in one focused implementation pass from here.

## ★★★ Tier 1 SCOPE CORRECTED (2026-08-13, from attempting the implementation): store-only op GAP
Tried to wire the store. FOUND: the pool STORE is FUSED inside ggml_paged_attn/_banded/_partials
(build_attn_mha_paged, llama-graph.cpp:3713 writes k_cur→k_physical INSIDE the attention kernel).
There are ONLY those 3 paged ops (grep ggml_paged_* in ggml.h) — ALL fused store+attend, NO
store-only op. And get_k==get_v==get_kv_tensor (ONE interleaved K+V buffer, llama-kv-cache-paged.cpp
:1173-1180). ⇒ MSA (sparse attention, cannot call ggml_paged_attn) has NO existing way to STORE to
the pool. My earlier "no new kernel needed" was WRONG — it analyzed the ATTENTION (stock
flash_attn_ext) but the STORE is kernel-fused.
CORRECTED Tier 1 scope — needs a STORE-ONLY path, one of:
  (a) NEW ggml op ggml_paged_kv_store(kv_tensor, k_cur, v_cur, write_slots, block_size) — a write
      kernel (Metal + CPU) that lays K/V into the interleaved buffer exactly as the attn kernels
      expect to read it. Cleanest but a new kernel.
  (b) Manual ggml_set_rows into get_kv_tensor, replicating the interleave the kernels assume —
      fragile, must byte-match the read layout.
THEN the gather (get_rows via block-table translation) reads that buffer. So Tier 1 = store-only op
+ gather, BIGGER than "just the gather". This is a genuine finding that only surfaced on attempt —
planning had assumed ggml_set_rows would suffice. Record it so the next pass scopes for a write op.
