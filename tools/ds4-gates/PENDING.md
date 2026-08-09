# PENDING — what is left, and which of it needs the GPU

**2026-08-09 · the answer to "what is pending, and can it be done on CPU?"**

The split below is the useful one: **GPU-bound** means it needs the Metal device and a loaded model,
so it queues behind whatever measurement is running. **CPU-only** means it can proceed on the Mac
right now, in parallel, with one caveat recorded at the bottom that cost a measurement today.

---

## The bar

Owner's bar, verbatim: *"equal or better than the static, and the ranges are 256k to 1M context
ranges, not 2k, 32k, 64k etc."*

| rung | verdict | evidence |
|---|---|---|
| 256k | **as fast as: MET** (dirty) | ABBA, effect 0.0% vs drift 2.2% -> UNRESOLVED = tie. ⚠ The run was contaminated (see "candidate" below). |
| 512k | **as fast as: MET** | ABBA, effect 1.0% vs drift 2.6% -> UNRESOLVED = tie. Warm-vs-warm 1.0036. |
| 1M | **not measured** | Fits in memory (f16 116.1 GiB / q8_0 76.1 GiB of 128). Cost is ~20-24 h and ~5.5 tok/s decode. Owner's call. |

⚠⚠ **"AS CORRECT AS" IS NOT MEASURED AT ANY OF THESE RUNGS, AND IT IS THE BIGGER RISK.**
`FINDINGS-paged-cross-request.md` is OPEN and records **2 of 6 runs FAILING at 224,992 tokens**. The
needle used to validate every parity run *cannot see it* — the recorded failures include "fluent but
different from static", and a fluent-but-wrong answer still contains the passcode. Eight needle
passes today are eight runs of an instrument blind to this failure mode. `output_sanity.py` exists
now to close that; wiring it into the parity gate is the top CPU-only item below.

---

## GPU-BOUND (queues behind the running measurement)

| item | cost | why it needs the GPU |
|---|---|---|
| clean 256k ABBA (warm-up ON, npred=512) | ~1 h | retires the "candidate" stamp on the 256k tie |
| clean 512k ABBA (warm-up ON, npred=512) | ~3.5 h | tightens the bound + first properly-sampled decode number |
| `gemma3` paged verification | ~10 min | wired but NOT proven: needs `DS4P-CONSUME > 0` and output matching static |
| `gemma4` SWA-restored check | ~10 min | proves the guard fix re-enabled what it broke; needs CONSUME on a layer where `is_swa(il)` is true |
| paged-corruption rate at 225k | ~6 runs | the open OPEN finding; 2/6 is one data point, not a rate |
| 1M rung | ~20-24 h | owner's authorization |
| NIAH sweeps | long | **marked Pending by the owner**, deliberately deferred until paging matches static |
| inkling paged path | 975B | marker added, never gate-verified |
| CUDA/Metal parity on the box | — | needs the box's GPUs free |

⚠ **The gemma3 and gemma4 checks have a gate-shape requirement, registered before the run:** the
prompt must be **longer than `n_swa`**. Below the window a windowed layer and a full-causal layer see
identical context and produce identical logits, so a short-prompt gate is structurally incapable of
detecting a broken band. It would return a clean green that means nothing.

---

## CPU-ONLY (can proceed now)

| item | state |
|---|---|
| wire `output_sanity.py` into `paged_parity_gate.sh` | **next up.** Save the arm's text into its `.res`, grade paged against static in the ABBA summariser. Zero extra GPU cost, turns every future parity run into a corruption sampler. |
| warm-up response is never checked | **known hole.** `warmup()` sends its curl to `/dev/null`, so a 500 or an empty completion still prints "discarded (5s)". Log-line-is-not-work-done, in code I wrote today. |
| the remaining ISWA archs | `gemma2` `gemma3n` `cohere2` `cohere2moe` `phi3` `olmo2` `exaone4` `exaone-moe` `openai-moe` `plamo3` `mellum` `mimo2` `smallthinker` `afmoe` — minus anything CHUNKED or SYMMETRIC, which the band cannot express at all |
| `mimo2` sinks / `dflash` non-causal | recorded closed in `PLAN-paged-arch-support.md`; **re-verify, that file's top table was stale enough to cause a real bug** |
| lint sweep | `lint_paged_consumers` · `lint_scrub_coverage` · `lint_common_laws` — the last reports 6 of 44 gates reach the shared laws |

---

## ⚠ THE CAVEAT THAT COST A MEASUREMENT TODAY

**"CPU-only" does not mean "free while a measurement runs."** During arm 1 of the clean 256k run I
executed three `-fsyntax-only` compiles, several python fixture runs and a handful of git operations.
Arm 1 came back at **232.1 tok/s prefill against the earlier cold run's 244.9** — 5.2% slower, on the
run whose entire purpose was to be cleaner than that one. Thermals are an equally plausible cause and
I cannot separate them from here, which is exactly the problem: **the confound is unattributable
after the fact.**

⇒ The rule, and it is stricter than it looks: **while a parity arm is in flight, reads only.** No
compiles, no test fixtures, no fan-out. Doc edits and file reads are fine. Load on *some* arms is
worse than load on all of them, because that is a positional confound and positional confounds are
the one thing ABBA cannot balance.
