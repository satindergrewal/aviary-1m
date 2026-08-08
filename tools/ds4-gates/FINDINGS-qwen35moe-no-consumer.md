# DEFECT (FIXED): `qwen35moe` accepted the paged context and never read it

**Found 2026-08-09 · fixed the same hour · `llama.cpp-ds4ports` `fe137e502`, pushed.**

## The defect in three greps

`paged_ctx` occurs exactly **three** times in `src/models/qwen35moe.cpp`:

```
:4    #include "llama-kv-cache-paged.h"                        the header, for a type it only DECLARES
:201  ... inp->mctx ? inp->mctx->get_attn_paged() : nullptr     the producer, correct
:291  const llama_kv_cache_paged_context * paged_ctx) {         the parameter, accepted
```

and then the function goes straight to `build_attn` at `:350`. **The parameter is declared and dropped.**

Its sibling `src/models/qwen35.cpp` has called `build_attn_paged_or_null(paged_ctx, ...)` at `:334` since
the port. Two files, same family, same graph shape, disagreeing silently for as long as both existed.

## What it cost

Under `--kv-paged`, this architecture allocated a paged pool, handed out blocks, logged the checkouts,
and then computed **every attention layer on the static path, in every batch.**

| | pool checkouts | `DS4P-CONSUME` | verdict |
|---|---|---|---|
| before | 6 | **0** | VOID |
| after | 6 | **430** | PASS, static-path fallbacks **230 → 0** |

Measured with `arch_serve_gate.sh qwen35moe <Ornith-35B Q4_K_M>`, `DS4P_PAGED_HYBRID=1`.

⚠ **`qwen35moe` is Ornith-35B, Ornith-397B and Qwen3.6-35B-HauhauCS** — every large hybrid model on this
box. The 397B is the model the context-ceiling program exists for.

## Why nothing caught it

**Output was identical to static because it *was* static, twice.** Every check that compares text passed,
correctly and uselessly. The pool allocated; `DS4P-CHECKOUT` fired; the producer was right; the parameter
was threaded the whole way down. Nothing upstream was wrong.

⇒ **The only tell that works is a CONSUMER count.** An allocation, a checkout, or a producer proves the
context was *built*, never that a graph *read* it. That is audit finding 5 restated, and it is why
`DS4P-CONSUME` exists.

⇒ **It was found only because the arch matrix gained a `qwen35moe` row at all.** Prior hybrid coverage was
`qwen35` (Ornith-9B) alone, and `qwen35` has always had the line. One missing row hid it completely.

## What this does and does not invalidate

| claim | status |
|---|---|
| 225k parity (wall 0.9831, decode 1.558×) | **STANDS** — `long_context_gate.sh:37` defaults to the **9B** (`qwen35`), which consumes. Had it defaulted to the 35B, a static-vs-static ratio would have been reported as paged speedup. |
| `set_input` recurrent-fix coverage on Ornith-35B (`DS4P-RS` 813 vs 816) | **STANDS** — that counter is the **recurrent** write, which a hybrid model performs in both arms regardless of how attention is computed. |
| Any reading of that same table as *paged attention* coverage for the 35B | **DEAD** — the attention half was static in both arms. |
| Long-context paged performance of any `qwen35moe` model | **UNMEASURED.** It was previously unmeasured *while looking measured*, which was worse. |

## The fix

`qwen35.cpp`'s call verbatim — identical full-causal geometry, no mask arguments, no relative bias:

```cpp
cur = build_attn_paged_or_null(paged_ctx, Qcur, Kcur, Vcur, kq_scale, il);
if (cur != nullptr) { cb(cur, "attn_pregate_paged", il); }
else { cur = build_attn(inp, nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
       cb(cur, "attn_pregate", il); }
```

Verified with the multi-request leg and its poison control, not on text alone: 3 requests
(long, short, long) on one server all match, 43 recurrent writes after the first request, and
`DS4P_RSPOISON` applied to the **paged arm alone** changed the output — so the match is a measurement
rather than a shrug.

⚠ **Scope.** `-c 4096`, short prompts. This says the arch is **WIRED and CORRECT at small context**. It
says nothing about 256k, nothing about speed, and the first long-context paged run of any `qwen35moe`
model has still not happened.

Class: `correct producer, no consumer` — fourth code instance in this lane.
