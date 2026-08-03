# DESIGN — P1-5: disk-persisted content-addressed KV banks (the 1M agentic unlock)

**Status: DESIGN (2026-08-04 ~00:35), grounded in our tree. No code yet.**
Author: Fable-DS4 | Lane: ds4-ports | Plan source: PLAN-DS4-PORTS.md §P1-5 (ds4's
ds4_kvstore.c concept, 1,606-line self-contained reference; we port the IDEA, credit on port).

## The pain (measured, prompt-cache-cram-regimes memory)
The RAM tier (-cram) holds ONE 64K state under the 8 GiB cap — save #2 evicts #1. Every
eviction throws away minutes of prefill (1M prefill ≈ 16 min; 80K ≈ 2 min). Agentic serving
re-prefills the world per session.

## Our-tree hook points (grounded tonight)
- RAM tier = `server_prompt_cache` (tools/server/server-task.{h,cpp}); save path
  server-context.cpp:286 `prompt_save` (alloc:313 → seal:323; the 168 ms real-save vs
  0.1 ms refusal scar lives here — witness saves by COST).
- **The two eviction sites = the spill points**: server-task.cpp:1764 ("making room…
  removing oldest") and :1914 ("cache size limit reached… removing oldest"). Today both
  DROP the state; P1-5 makes them SPILL to disk instead.
- P0-2 DS4P_REVALIDATE (4bea1cdc) already binds state↔claim on restore — reuse for the
  disk tier's exactness counters for free.

## Design
1. **Tier**: disk bank BEHIND the RAM cache. Directory `<--kv-bank DIR>/`, one file per
   state: `<hash>.kv` where hash = FNV/SHA1 over the TOKEN prefix (token-space, matching
   the RAM cache's longest-common-PREFIX lookup semantics — NOT ds4's byte-LCP; our lookup
   unit is tokens).
2. **Spill**: at the two eviction sites, instead of dropping: write the state file
   (existing llama_state_seq save primitives; async write via a copy so the slot isn't
   held), record `{hash, n_tokens, size, atime}` in an index header. Spill is best-effort:
   disk full/slow ⇒ log + drop (never block serving).
3. **Admit**: on prompt-cache MISS in RAM, before prefilling: scan the bank index for the
   longest token-prefix match ≥ a floor (e.g. ≥1/8 of the prompt, ds4's salvage ratio);
   load, then REPLAY the tail through normal prefill. Partial restore = load + tail-prefill,
   so correctness never depends on a full match.
4. **Eviction (disk)**: size-capped LRU with hit-half-life decay (ds4: 6 h) — an index
   sweep on save, not a daemon.
5. **Format**: v0 = the existing llama_state_seq payload + a small header
   {magic, version, model identity (build_identity, already used by seal:323), n_tokens,
   token hash}. Identity mismatch on load ⇒ ignore file (never feed a stale-model state).
   Compression (fp8 packing, ds4's payload v3) = LATER; correctness first.
6. **Flags**: `--kv-bank DIR` (off by default), `--kv-bank-cap GiB`. DS4P env for the
   bring-up arms.

## Kill gates (plan's, made concrete)
1. Restored state ⇒ BYTE-IDENTICAL logits vs the in-RAM state (same prompt, same seed;
   the banded_equiv harness pattern reuses directly on the synthetic).
2. Warm admit ≥5× faster than cold prefill at 64K (measure both arms, box slot).
3. Partial-restore exactness: replay-tail arm vs full-prefill arm bit-compare; counter = 0.
4. Eviction-site witness: spill actually fires at :1764/:1914 (log + file exists), and the
   spill path adds <5% to the eviction latency (it's off the hot decode path, but witness).

## Sequencing
Independent of 4d/scheduler (works on the static path TODAY — it extends the existing
prompt cache); also composes with paged later (a paged state file is just a different
payload). Code surface: server-task.{h,cpp} + a new server-kv-bank.{h,cpp}. Effort per
plan: 2-4 d. Gates 2-4 need one box window; gate 1 runs on the Mac synthetic.
