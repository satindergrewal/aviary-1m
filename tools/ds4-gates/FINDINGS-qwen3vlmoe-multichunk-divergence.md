# OPEN DEFECT: `qwen3vlmoe` paged output differs from static on a multi-chunk prefill

**Found 2026-08-09 · reproducible · NOT fixed · binary `ae9e496e8`.**

## The measurement

Same prompt (~1571 tokens, 4 chunks at `-ub 512`), `temperature 0`, `seed 1`, `cache_prompt: false`,
`-c 4096`, `Qwen3-VL-30B-A3B-Instruct-Q4_K_M`:

| arm | reps | processes | output |
|---|---|---|---|
| static | 3 | 1 | ` on a river. What is the number` |
| static | 1 | a **second** server | ` on a river. What is the number` |
| **paged** | **3** | 1 | ` located on a river. What is the` |

⇒ **Both arms are individually deterministic, across reps AND across processes, and they disagree.**

⚠ This is the branch that was **pre-registered before the run**, in `selfconsist.sh`:
*"A stable, B == A, C differs → REAL paged divergence on qwen3vlmoe. File it."*
The competing hypothesis — the documented near-tie flakiness of cross-arm byte-equality — was the
**leading** one and is refuted: a near-tie would have made at least one arm vary. Neither did.

## What the shape rules out

| candidate | ruled out by |
|---|---|
| carried state across requests | arm C is a **fresh server, ONE request**, and still diverges |
| the 2026-08-08 recurrent defect | `qwen3vlmoe` is **non-hybrid**; recurrent writes after req 1 = **0** |
| harness near-tie / arbiter flakiness | both arms deterministic across processes |
| a wrong vehicle | loader reports `arch = qwen3vlmoe`, matching the expectation |

## Why 100+ single-request gate runs never saw it

`arch_serve_gate`'s prompt was *"The capital of France is"* — **one chunk**. This needs a **multi-chunk
prefill**. The multi-request leg added ~1.6k and ~2.0k token prompts on 2026-08-09, and this is the
first thing that has disagreed in the gate's history across 10 architectures.

## ⚠ It is NOT monotonic in length, and that is the most useful clue

Within the same gate run, on the same server:

| prompt | tokens | chunks | result |
|---|---|---|---|
| "The capital of France is" | ~6 | 1 | match |
| LONG1, geography filler | ~1571 | 4 | **DIVERGED** |
| short, "The capital of Japan is" | ~6 | 1 | match |
| LONG2, rivers/towns filler | ~2038 | 4 | match |

A *longer* multi-chunk prompt matched while a shorter one diverged. ⇒ The reading that survives is
**not** "long prompts are wrong". It is that **paged and static compute numerically different results
on this arch, and the difference only becomes visible when it crosses a token boundary.** That is a
worse statement than "one prompt is broken", and it means matching output on other prompts is weak
evidence of correctness here.

## Not universal

Nine other architectures ran the identical prompts through the same leg on the same binary and all
matched: `ernie4_5` `ernie4_5-moe` `starcoder` `starcoder2` `nemotron` `qwen3vl` `qwen3moe` `qwen35`
`qwen35moe`. Note especially:
· **`qwen3vl` (dense VL, same mrope family) PASSES.**
· **`qwen3moe` (MoE, no VL) PASSES.**
⇒ Suspicion falls on what is unique to the combination, not on VL or MoE alone. **Untested speculation,
recorded as such** — no bisect has been run.

## Next diagnostic step (not yet run)

1. Bisect the prompt length to find the chunk count at which it first diverges, at fixed content.
2. `DS4P_KVSUM_LAYERS` on both arms at the last prefill chunk — a KV content checksum read back through
   the block table says whether the divergence is in the cache or downstream of it.
3. If the KV agrees, the difference is in the attention kernel's consumption, not in what was stored.

## Scope of the claim

Reproducible, deterministic, one arch, one quant, `-c 4096`, Metal. **No claim about how wrong the
answer is** — both continuations are plausible text. **No claim that other prompts on this arch are
correct**; they are only *unmeasured*, and the non-monotonic result above is a direct argument against
reading a match as safety.
