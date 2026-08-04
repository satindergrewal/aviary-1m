# Paged KV for ANY model — the coverage plan

**Satinder's bar, verbatim:** *"I load ANY model, it works as I expect it to work."*
Not a supported-list. Not "some architectures". **Any model he loads.**

And this is not a new ask — **it is already my own plan.** From the 5-phase map I gave him on
2026-08-03: **"3b hybrid paging — Inkling/K3/Ornith's KV becomes paged blocks instead of static
per-slot slabs"** is **phase 2 of 5**, and the finale is **"P2-8 continuous batching — kill static
`-np` entirely: dynamic admission, queue-not-reject, evict/preempt … the actual vLLM-parity
moment: 'as many agents as fit, admitted dynamically' — no lanes, no rigid slots."**
I filed as a footnote what my own plan called a phase. That is corrected here.

## Where we actually stand — measured, not assumed

`llama-arch.h` carries **141 architectures**. The paged path splits them three ways:

| class | paged today | archs |
|---|---|---|
| **Flat attention** (plain KV per layer) | ✅ **works** | the large majority of the 141 — Qwen3, Llama, Mistral, Gemma-non-SWA, … |
| **Hybrid** (attention + recurrent/linear layers) | ❌ **refused** | `QWEN35`, `QWEN35MOE`, `QWEN3NEXT`, `JAMBA`, `NEMOTRON_H`, `NEMOTRON_H_MOE`, `FALCON_H1`, `GRANITE_HYBRID`, `INKLING`, `KIMI_LINEAR`, `LFM2`, `LFM2MOE`, `PLAMO2`, `DREAM`, `LLADA`, `LLADA_MOE` — **16** |
| **SWA** (sliding-window attention) | ❌ **refused** | gemma3, llama4, … (second guard, `llama-model.cpp:2338`) |
| **Pure recurrent** (no attention KV at all) | n/a — nothing to page | `MAMBA`, `MAMBA2`, `RWKV6/7`, `ARWKV7`, `RWKV6QWEN2` |

**Satinder's own models sit squarely in the refused column:** Ornith (`qwen35`/`qwen35moe`),
Inkling, Kimi-K3 (`KIMI_LINEAR`). So the models he actually runs are exactly the ones without
parity. That is the gap, stated honestly.

## Why hybrids are refused, and what it really takes

The paged pool assumes **every layer owns a uniform KV cache** sliceable into fixed blocks. A
hybrid has attention layers (real KV) interleaved with recurrent/linear layers (a state, not a
per-token KV). Two things follow:

1. The pool must be built over **the attention layers only** — the layer filters for this
   **already exist** (`FALCON_H1`/`INKLING`, `NEMOTRON_H(+MOE)`, `QWEN35(+MOE)`).
2. The recurrent state stays where it is — it is small (measured: Ornith-9B RS buffer = 50.25 MiB
   over 32 layers) and is **not** the thing that blows up at long context.

**Measured state of the existing scaffolding** (`DS4P_PAGED_HYBRID=1`): Ornith-9B gets **past**
the arch assert and dies at `llama_paged_scheduler_init` — *"context does not have a paged KV
cache… SWA architectures are not yet supported"* — while the log shows **`n_swa = 0`**. The
message is wrong. The real cause: **the pool bring-up lives only inside the hybrid-ISWA branch**,
so a hybrid *without* SWA takes the plain `llama_memory_hybrid` branch where no pool is built.

## The work, ordered so that each step makes the next cheaper

1. **Mirror the pool bring-up into the non-SWA hybrid branch.** Wiring, not design — the ISWA
   branch is the template. **Unblocks Ornith, Jamba, Nemotron-H, Kimi-Linear, LFM2, PLaMo2 in one
   move**, since they share the plain-hybrid path.
2. **Fix the misleading scheduler error.** It sent me down an SWA path on a model with `n_swa=0`
   and will do it to the next person. Report *which* precondition actually failed.
3. **Attention-only pool filter.** Today the pool "spans all layers" (in-code note), which
   over-allocates on a hybrid where a minority hold KV. Safe direction, wasteful — and it
   composes with the new `n_ctx`-bounded sizing.
4. **Close the hybrid decode gate** — already flagged in-code as the pending piece.
5. **Then SWA.** Separate guard, separate shape: SWA layers need only a window of KV, which is a
   *smaller* pool, not an impossible one. It is a sizing rule per layer, not a new mechanism.
6. **Generic fallback so the bar is actually met.** For any architecture not explicitly handled,
   decide by *capability* rather than by an arch allow-list: if a layer reports a
   per-token KV of known shape, it can be paged; if not, it keeps its state as today. **An
   arch-name list can never satisfy "ANY model" — a capability test can.**

## The rule this coverage plan enforces

**No allow-lists keyed on architecture names.** Every one of the 141 archs, and every arch that
lands next month, must be classified by **what its layers report**, not by whether someone
remembered to add it. Where a model genuinely cannot be paged, it must **say why in terms of the
model** ("layer 7 has no per-token KV"), never a generic "unsupported architecture".

That is the difference between a supported-list and Satinder's bar.

## Verification

`tools/ds4-gates/paged_multimodel_gate.sh` already runs static-vs-paged per model and compares
**output shas** plus RSS and system-free. Extend it to every model on the box as they arrive, and
require: **serves, output identical to static, memory sane.** A model that cannot page must fail
the gate *loudly with a model-specific reason*, not silently drop to static.
