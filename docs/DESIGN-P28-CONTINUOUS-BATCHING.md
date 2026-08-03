# DESIGN — P2-8 continuous batching (THE parity item): server ↔ paged-scheduler map

**Status: DESIGN/MAP (2026-08-04 ~03:00). This is the 4d server-integration blueprint —
free to design now; only the FLAG TOPOLOGY choice waits on Satinder's #1831 word.**
Author: Fable-DS4 | Plan: PLAN-DS4-PORTS.md §P2-8 (16-agent throughput gate). Prior art in
this dossier set: DESIGN-3B §Part-4d map (the engine = 4 public API calls, only
examples/paged drives it; server has ZERO paged handling — witnessed crash).

## The two loops, side by side
**Server today (static -np)**: update_slots (server-context.cpp:2864) assembles one
llama_batch across slots (common_batch_add per slot token, :165-class), one llama_decode
(:3722), per-slot sampling from batch positions, slot states advance. Slots are FIXED
lanes: n_parallel contexts, each with its own seq_id and KV share.
**Paged engine (examples/paged loop)**: add_request → prepare_batch (fills the batch AND
sets cache batch info) → llama_decode → sample per seq at batch_offsets[i]+batch_lens[i]-1
→ update(sampled, stop_flags). Sequences are DYNAMIC: admitted when blocks exist, evicted/
swapped by the scheduler, no fixed lanes.

## The mapping (server-side 4d/P2-8 core)
1. **Request intake**: server task→slot assignment becomes task→`llama_paged_scheduler_add_request`
   (tokens from the task's server_tokens; request id = task id). The slot struct survives as
   the per-request bookkeeping shell (sampler, streaming state, metrics) but no longer owns
   a KV lane — rename-in-place, minimal churn.
2. **Batch assembly**: update_slots' common_batch_add loop is REPLACED (paged mode) by ONE
   `prepare_batch` call. The scheduler decides prefill-chunk vs decode mix — this subsumes
   P1-7 half-a (interleave) by construction.
3. **Decode + sampling**: llama_decode unchanged; sampling indexes via
   get_batch_info()->batch_offsets/lens (the example's exact pattern); streaming per
   request id as today.
4. **Completion/update**: stop flags from EOG/limit per request → `update()`; freed blocks
   return to the pool → next queued request admits (queue-instead-of-reject FALLS OUT).
5. **Metrics**: get_seq_state carries n_prompt/n_decoded/ttft — maps onto the existing
   /metrics fields (deposits at the same sites).
6. **Prompt cache / P1-5**: RAM tier + disk bank hook at admit time (prefix LCP against the
   scheduler queue's tokens before allocate) — design note only; not increment 1.
7. **Hybrid archs**: the SAME server path serves inkling once 4c-1/4d land (the hybrid
   context carries the paged child; ggml_paged_attn_banded consumes it).

## Increment order (post-#1831-word)
4d-1: scheduler owned by server_context when paged-mode flag set; intake→add_request;
      static path untouched (flag off = today's code byte-identical).
4d-2: update_slots paged branch = prepare_batch/decode/sample/update (the example loop
      transplanted); single-slot parity gate: same request → same output as static.
4d-3: multi-request; 16-agent throughput gate (plan: ≥6× vs -np static at fan-out).
4d-4: 4c-1 inkling builder branch + hybrid serve gate (paged-vs-static bit-compare on the
      synthetic — the harness exists).

## Kill gates
- 4d-2: single-request parity float-exact vs static path (harness = tonight's gate scripts).
- 4d-3: 16 concurrent agents, shared 8K prefix: throughput ≥6× static -np 4 (plan number),
  zero rejects (queue absorbs), exactness spot-checks float-exact.
- Crash-fix regression: the #1831 server×kv-paged abort becomes a SERVED request (the
  witness file flips from crash to response).

## Flag topology (the #1831-gated choice — NOT designed here)
Whether --kv-paged alone activates server paged-mode, or a separate --serve-paged opt-in
with --kv-paged re-gated meanwhile, is Satinder's call. Both fit the increments above
unchanged; only 4d-1's flag test differs.
