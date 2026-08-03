# DESIGN — P1-6 session forking (N agents from one prefix)

**Status: DESIGN/MAP (2026-08-04 ~02:35). Plan source: PLAN-DS4-PORTS.md §P1-6.**
Author: Fable-DS4. Kill gates (plan): forked next-token logits == unforked continuation;
fork cost < 5% of re-prefill at 64K.

## Finding: our tree already has a FUNCTIONAL fork path — the prompt cache IS a fork engine
With np>1, every slot that misses on a prompt LCP-loads the shared prefix from the RAM
prompt cache (state copy into its own seq) and tail-replays. Orchestrator + N subagents
sharing a 100K system prefix = each subagent's first request forks off the cached prefix
TODAY. P1-5 extends exactly this across evictions and server restarts (disk tier), and
tonight's gates already prove the fork-exactness property on this path: admit-from-bank
into a fresh process produced FLOAT-EXACT logits (gates 1+3, results/kvbank-gates-1-3).
What the plan calls "fork" is therefore ALREADY FUNCTIONAL at N-copy cost.

## The gap vs vLLM-class forking = MEMORY, not function
- cache-admit fork: N copies of the prefix KV (N × 100K × KV/token) + one state-copy per
  admit. Works today; RAM/VRAM cost linear in N.
- paged COW fork (dossier DESIGN-3B §P1-6 primitive, banked): 1 × prefix KV + block-table
  copies + refcounts; write-path COW. THE memory-right answer — sequenced AFTER 4d.

## Work plan
1. **Witness now (Mac-free)**: dual-slot fan-out gate — np=2, same long prefix, slot B's
   first request admits from the cache; both slots' next-token logits must equal a
   single-slot continuation float-exactly. (Extends tonight's gate harness; proves
   "N agents from one prefix" on the static path.)
2. **fork cost accounting**: state-copy time vs re-prefill at 64K (box, joins the bundle);
   the plan's <5% gate applies to the copy, which is a memcpy-class op — expected to pass
   trivially vs 125 s prefill.
3. **paged COW implementation**: after 4d, per the banked block-table design (refcount +
   boundary do_block_copy + write-path shared-check); its gate = fork → diverge → equals
   two independents.
4. **API surface (deferred until a real consumer asks)**: an explicit /fork endpoint is
   NOT needed for the owner's workload today — the implicit cache-admit fork covers agent
   fan-out; explicit fork-at-arbitrary-point (ds4's aligned-rewind) only matters for
   mid-generation forks, parked until a use case names itself.

## Honest scope note
The plan's EFFORT 2-4 d assumed building fork machinery; the finding above collapses most
of it — the remaining REAL work is the paged COW (already designed, 4d-gated) and the
witnesses. P1-6's Mac-free surface may close with witness (1) alone.
