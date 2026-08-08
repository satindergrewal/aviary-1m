# SCOPE (read-only, nothing changed): 21 architectures excluded from paging by construction

**2026-08-09 · investigation only · no code touched · binary `592ac88c2`.**

## The gap, in one line of source

`src/models/gemma3.cpp:103`:

```cpp
if constexpr (iswa) {
    inp_attn = build_attn_inp_kv_iswa();    // <- interleaved-SWA models take THIS
} else {
    inp_attn = build_attn_inp_kv_auto();    // <- the ONLY paged-aware funnel
}
```

`build_attn_inp_kv_auto()` routes on `cparams.kv_paged` and returns a paged input. **An
interleaved-SWA model never reaches it**, and `build_attn_inp_kv_iswa()` contains **zero** references
to `paged` in its entire body.

⇒ **Not "the auto funnel declines an SWA context" — it is never called.** Declining would be a
condition to relax; never-called is a branch to wire. (That mechanism was stated wrongly first and
corrected; the verdict was right either way.)

## Measured consequence

`gemma3` + `DS4P_PAGED_SWA=1`: server starts, pool constructed (`DS4P_PAGED_SWA: constructing the paged
attention pool for a pure-SWA architecture`), **5 pool checkouts, `DS4P-CONSUME` = 0.**

⚠ The zero is a **measurement**, not an absence: the same marker scored **2580** on `starcoder2` on the
same funnel in the same batch, so it demonstrably fires.

⇒ The bring-up flag **advertises more than exists**: it builds the pool and the graph then asks for the
ISWA context by construction. Third instance of *correct producer, no consumer* in this fork, after
audit finding 5 and `qwen35moe`.

## Blast radius — every model file taking the ISWA branch

```
gemma2 · gemma3 · gemma3n · gemma4 · gemma4-assistant · cohere2 · cohere2moe · phi3 · olmo2
llama4 · exaone4 · exaone-moe · openai-moe · plamo3 · laguna · mellum · mimo2 · step35
smallthinker · afmoe · dflash
```

**21 architectures.** The largest unwired group in the fork.

## ★ THE GOOD NEWS, AND IT IS THE REASON TO SCOPE BEFORE ESTIMATING

**The memory side is already built.** This is not a new subsystem:

| piece | where | state |
|---|---|---|
| pool owned by the ISWA cache | `llama-kv-cache-iswa.h:109` `std::unique_ptr<llama_kv_cache_paged> mem_attn_paged` | **present** |
| attach / read the pool | `:99` `set_attn_paged()` · `:100` `get_mem_attn_paged()` | **present** |
| the pool gets constructed | `llama-model.cpp:2573` under `DS4P_PAGED_SWA` | **present, verified running** |
| **paged context on the ISWA context** | `:155` `const llama_kv_cache_paged_context * get_attn_paged() const` | **PRESENT** |
| context setter | `:156` `set_attn_paged_ctx(llama_memory_context_ptr)` | **present** |
| **the graph reading any of it** | `build_attn_inp_kv_iswa()` | ⛔ **ABSENT — this is the whole gap** |

⇒ **Everything below the graph exists and is already exercised.** `get_attn_paged()` on the ISWA
context is the exact accessor `qwen35.cpp` uses on the hybrid context (`inp->mctx->get_attn_paged()`).

## What the change probably is — and why "probably" stays in the sentence

The shape mirrors `qwen35.cpp` / the `qwen35moe` fix committed today: thread `paged_ctx` from the input
into the per-layer call, then `build_attn_paged_or_null(paged_ctx, ...)` with a static fallback.

⚠ **What makes this NOT a copy of that fix, and why it is not a 06:00 job:**

1. **Per-layer geometry.** Interleaved SWA means layers do **not** share head geometry. The pool is
   already sized to the **majority** group (`llama-model.cpp:2597`, "paged pool majority geometry"),
   and **layers outside it must fall back, per layer.** `qwen35` had one geometry; these do not.
2. **The SWA layers themselves.** A sliding window is a different attention shape. Whether the paged
   kernel is correct for a windowed mask **has not been established anywhere in this fork.**
3. **21 files.** Even a two-line change per file is 21 chances to put it in the wrong branch, and the
   only gate that would catch a wrong one is `arch_serve_gate` — one arch at a time.

## Recommended order for whoever picks this up

1. **ONE arch first — `gemma3`** — and prove `DS4P-CONSUME > 0` with output identical to static.
2. **Then check the fallback count.** On a banded funnel, `static-path warns` in the paged arm must be
   **> 0** here, unlike every arch verified so far: the minority-geometry layers **should** fall back.
   **A zero would mean the SWA layers are being paged against a pool that is not their shape**, which
   is the `gemma4` `headdim` accident in a new costume.
3. Only then the other 20, one gate run each.

⚠ **DO NOT** wire all 21 and gate one. That is the shape that produced today's `qwen35moe` finding: a
family assumed identical to its sibling, silently disagreeing for as long as both existed.
