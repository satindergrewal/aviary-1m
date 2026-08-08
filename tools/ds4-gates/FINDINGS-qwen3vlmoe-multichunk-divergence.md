# `qwen3vlmoe` paged and static resolve a COIN-FLIP TIE differently on a multi-chunk prefill

**Found 2026-08-09 · reproducible · binary `ae9e496e8`.**
**⚠ DOWNGRADED THE SAME HOUR IT WAS FILED. Originally titled "OPEN DEFECT". It is not one.**

## ★ THE MEASUREMENT THAT DECIDED IT — top-2 logprobs at the divergent position

`n_probs: 5` on `/completion`, same prompt, both arms:

| arm | top-1 | top-2 | separation |
|---|---|---|---|
| static | ` on` p=**0.16328** | ` located` p=**0.15927** | **0.0040 · 1.03× · 0.025 logprob** |
| paged | ` located` p=**0.17962** | ` on` p=**0.16509** | 0.0145 · 1.09× · 0.084 logprob |

⇒ **The same two tokens, swapped, at a 1.03× tie.** The model is functionally indifferent between them.
The measured arm-to-arm difference in logprob is ~0.01–0.12 — **small**, and it flipped an argmax that
static itself decided by **0.025**.

⇒ **VERDICT: a small numerical difference between the paged and static attention paths, made VISIBLE
because the argmax happened to be a tie. NOT evidence of a wrong answer.** Neither arm is ground truth;
nothing here says which continuation is "right", and both are plausible text.

⚠ **What my earlier determinism test did and did NOT establish.** It proved the flip is **systematic,
not random** — each arm is deterministic across processes. It could **not** distinguish *"systematic
large error"* from *"systematic tiny difference at a tie"*, because both look identical in byte
comparison. **I reported "the fifth real defect" on the strength of a test structurally unable to
support that word.** The top-2 gap is the instrument that separates them, and it cost five minutes and
zero code.

⚠ **This also explains the non-monotonic length result below without any new mechanism:** LONG2 matched
because its argmax was not a tie, not because it was safer.

## What still stands, and is worth keeping

· **Cross-arm byte-equality is a weak instrument on this arch.** A match means the tie did not fall the
  other way, not that the paths agree numerically. This is now measured rather than suspected.
· **There is a real numerical difference between the two attention paths.** Bounded here at ~0.1
  logprob at one position on one prompt. Whether it grows with context length is **unmeasured**, and at
  256k it is the thing that would matter.
· **The multi-chunk prompt was necessary to see any of this at all.**

---

# Original filing, kept for the record

## The measurement

Same prompt (~1571 tokens, 4 chunks at `-ub 512`), `temperature 0`, `seed 1`, `cache_prompt: false`,
`-c 4096`, `Qwen3-VL-30B-A3B-Instruct-Q4_K_M`:

| arm | reps | process | request position | output |
|---|---|---|---|---|
| static | 3 | server #1 | 1st | ` on a river. What is the number` |
| static | 1 | server **#2** | 1st | ` on a river. What is the number` |
| **paged** | **3** | server #3 | 1st | ` located on a river. What is the` |
| **paged** | 1 | server **#4** (the 04:17 gate run) | **2nd** | ` located on a river. What is the` |

⇒ **Both arms are individually deterministic, across reps AND across processes, and they disagree.**

⚠ **The fourth row is load-bearing and was nearly omitted.** The first draft recorded paged
reproducibility as *3 reps, one process* — **the exact evidence shape I had just rejected for static**,
since this lane's documented flip pattern is per-**invocation** and single-process stability cannot rule
out a per-process flip. The closing evidence already existed in the same night's data: the 04:17 gate
run is a **separate server process** in which LONG1 was the **second** request, and it produced the
identical divergent string. That row does double duty — it is the B-arm for paged, and it kills
carried-state independently, because the same output appears at request position 1 and 2.

⚠ This is the branch that was **pre-registered before the run**, in `selfconsist.sh`:
*"A stable, B == A, C differs → REAL paged divergence on qwen3vlmoe. File it."*
The competing hypothesis — the documented near-tie flakiness of cross-arm byte-equality — was the
**leading** one and is refuted: a near-tie would have made at least one arm vary. Neither did.
⚠⚠ **THAT SENTENCE IS WRONG AND IS KEPT ONLY TO SHOW THE ERROR.** "A near-tie makes an arm vary" is
false: a tie can be resolved **deterministically and differently** by two implementations, which is
exactly what happened (1.03×, see the header). The probe refuted *randomness*, not *a tie*, and I read
its result as refuting both.

## What the shape rules out

| candidate | ruled out by |
|---|---|
| carried state across requests | arm C is a **fresh server, ONE request**, and still diverges |
| the 2026-08-08 recurrent defect | `qwen3vlmoe` is **non-hybrid**; recurrent writes after req 1 = **0** |
| harness near-tie / arbiter *flakiness* | both arms deterministic across processes |
| ⚠ a near-tie *as such* | **NOT ruled out — see the header. This row conflated RANDOM flakiness with a TIE. The tie is real (1.03×) and each arm resolves it deterministically.** |
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

## Next diagnostic step

⚠ **The step originally written here was NOT EXECUTABLE and is corrected rather than deleted.** It said
*"`DS4P_KVSUM_LAYERS` on both arms at the last prefill chunk"*. `grep -rln DS4P_KVSUM src/` returns
exactly one file — `llama-paged-scheduler-impl.cpp` — which **only runs under `--kv-paged`. The static
arm cannot emit that checksum at all**, so the comparison has no control arm and the plan was dead on
the page. The next session would have burned time discovering that.

**What actually answered the question, with zero code:** `n_probs` on `/completion` and the top-2 gap,
recorded at the top of this file.

**Still open and worth running, in this order:**
1. **Does the numerical difference GROW with context length?** That is the only part that matters for
   the 256k–1M bar. Same two-arm top-2 comparison at 8k, 32k, 128k on this arch.
2. Length bisect at fixed content, to find where the paths first differ measurably at all.
3. Only if (1) shows growth: locate it in the kernel.

## Scope of the claim

Reproducible, deterministic, one arch, one quant, `-c 4096`, Metal. **No claim about how wrong the
answer is** — both continuations are plausible text. **No claim that other prompts on this arch are
correct**; they are only *unmeasured*, and the non-monotonic result above is a direct argument against
reading a match as safety.
