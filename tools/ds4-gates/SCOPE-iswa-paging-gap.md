# SCOPE: interleaved-SWA architectures and paging

**2026-08-09 · investigation · binary `592ac88c2`.**

---

## ⚠⚠ CORRECTION, SAME DAY — TWO CLAIMS BELOW ARE WRONG. READ THIS FIRST.

### 1. "21 architectures excluded **by construction**" is FALSE. At least six were already wired.

`grep -rln build_attn_paged_or_null src/models/` lists **gemma4 · gemma4-assistant · laguna · step35 ·
dflash** among others — every one of them on the ISWA branch this document calls excluded.

`src/models/gemma4.cpp:266`:

```cpp
const auto * pg_ctx = mctx ? mctx->get_attn_paged() : nullptr;
ggml_tensor * cur_pg = build_attn_paged_or_null(pg_ctx, Qcur, Kcur, Vcur,
        hparams.f_attention_scale, il,
        hparams.is_swa(il) ? (int64_t) hparams.n_swa : 0);   // <- a REAL WINDOW, per layer
```

⇒ The evidence below — *"`build_attn_inp_kv_iswa()` contains **zero** references to `paged`"* — is
true and **does not support the conclusion drawn from it.** An arch does not need the funnel to know
about paging: the input object already carries `mctx`, and `mctx->get_attn_paged()` reaches the pool
directly. I read one function body, found nothing, and wrote "by construction".
**Absence-is-not-evidence, in the file where that class is already recorded.**

### 2. "Whether the paged kernel is correct for a windowed mask has not been established" — the kernel HAS a window.

```
ggml/include/ggml.h:2459     visibility_window > 0 selects the analytic band: a cell is visible
                             iff 0 <= rel_dist < visibility_window
ggml-metal.metal:3317        const int lo = args.visibility_window > 0 ? ...
```

It is a shipped, implemented parameter, and gemma4 has been passing it per layer.

### 3. What that mistake then cost: a guard that disabled a working path.

Acting on claim 2, commit `cbb4c8d93` added a **blanket `hparams.is_swa(il)` rejection** to
`paged_layer_supported()`, justified in its own comment by "the paged kernel has no window
parameter". That rejection sits inside `build_attn_paged_or_null`, **upstream of the point where
gemma4's window argument reaches the op** — so gemma4's SWA layers stopped paging the moment it
landed. Output stayed correct, because it falls back to static.

⚠ **That is why nothing caught it. A correctness gate cannot see a working feature being silently
switched off** — the same indistinguishability as audit finding 5, and the *guard-for-A-disables-B*
class.

**Fixed**: the blanket reject is gone; the narrow, real hazard — a windowed layer paged with
`visibility_window == 0` — is checked at the call site where that argument is in scope, and warns
loudly instead of corrupting.

---

## What follows is the original text, kept for the record. Sections 3 and 4 stand; the headline and the "what the change probably is" section are corrected above.

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

**21 architectures.** ⚠ **NOT all unwired — see the correction at the top.** gemma4,
gemma4-assistant, laguna, step35 and dflash in this list already reach the pool through
`mctx->get_attn_paged()`. The genuinely unwired remainder is the list minus those, and
**gemma3 is now wired too** (see below).

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

---

## STATE after the correction (2026-08-09, later)

**`gemma3` is wired**, copying gemma4's proven shape rather than inventing one:

```cpp
if constexpr (iswa) {
    auto * inp_iswa = build_attn_inp_kv_iswa();
    pg_ctx   = inp_iswa->mctx ? inp_iswa->mctx->get_attn_paged() : nullptr;
    inp_attn = inp_iswa;
}
...
cur_pg = build_attn_paged_or_null(pg_ctx, Qcur, Kcur, Vcur, 1.0f, il,
        hparams.is_swa(il) ? (int64_t) hparams.n_swa : 0);
if (cur_pg) { cur = build_lora_mm(model.layers[il].wo, cur_pg, model.layers[il].wo_s); }
else        { cur = build_attn(inp_attn, ...); }
```

Three things worth carrying to the next arch:

1. **The banded funnel returns the attention core WITHOUT the output projection.** `build_attn()`
   applies `wo`/`wo_s` itself; the paged branch must apply it explicitly or the residual receives an
   unprojected tensor. This is not visible in the type system and is the easiest way to wire an arch
   wrongly while it still compiles and still produces text.
2. **The non-ISWA branch must NOT also take the banded path.** `build_attn_inp_kv_auto()` already
   returns a paged input; adding the banded call there gives one layer two paged consumers.
3. **`kq_scale` differs per arch.** gemma3 pre-scales Q by `f_attention_scale` and passes `1.0f`;
   gemma4 passes `f_attention_scale`. Copying the argument instead of reading it is a silent
   numerical error.
4. **★ CHECK `swa_type` BEFORE COPYING THIS PATTERN. `visibility_window` is a ROLLING window and
   nothing else.** There are three SWA types and two of them are a different shape:

   | type | static mask (`llama-hparams.h`) | band can express it? |
   |---|---|---|
   | `STANDARD` | masked iff `p1 - p0 >= n_swa` | **yes** — `lo = q_pos - window + 1` is the same half-open interval, verified line by line, no off-by-one |
   | `CHUNKED` | masked iff `p0 < (p1/n_swa)*n_swa` | **no** — block-aligned, jumps at chunk edges |
   | `SYMMETRIC` | masked iff `\|p1-p0\| > n_swa/2` | **no** — includes future positions, not even causal |

   **`llama4` in the list above is CHUNKED.** Wired gemma4-style it would have passed a
   presence-only check (`is_swa && window > 0`) and then attended over a rolling band it never had.
   `build_attn_paged_or_null` now rejects on the TYPE as well, so a copied pattern falls back loudly
   instead of corrupting -- but the rejection means those archs are **not wired by this pattern at
   all**, and making them paged is a kernel job, not a wiring job.

**Status: compiles clean (`-fsyntax-only` on both changed files). NOT gate-verified.** No
`DS4P-CONSUME > 0` measurement has been taken for gemma3, and per the rule above that means it is
wired, not proven. Point 2 of the recommended order — fallback count must be **> 0** on this arch —
is still the check that has to run before anyone calls it done.
