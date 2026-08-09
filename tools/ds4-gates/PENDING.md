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
| 256k | ⛔ **VOID** | the "paged" arm never paged — see below |
| 512k | ⛔ **VOID** | same |
| 8k, **champion** | **wall 1.0095 · prefill 0.999x · decode 0.878x** | the kernel paging exists for. One ordering, below the bar range, not a verdict. |
| 8k, scalar | wall 1.3035 · prefill 0.765x · decode 0.767x | the SLOW kernel, measured by mistake — see the champion note below |
| 1M | **not measured** | Fits in memory (f16 116.1 GiB / q8_0 76.1 GiB of 128). Cost is ~20-24 h and ~5.5 tok/s decode. Owner's call. |

> ### ⛔⛔ EVERY 35B PARITY NUMBER FROM 2026-08-09 IS VOID. THE PAGED ARM WAS NOT PAGING.
>
> From the paged arm's own log, all at WARN level:
>
> ```
> DS4P-CHECKOUT           1     pool allocated
> DS4P-SET              110     context attached to the graph
> capability contract  3610     every attention layer REFUSED
>
> "paged layer refused: layer 3: block_size x head_dim exceeds the staged-tile
>  budget (need block_size*head_dim <= 8192)"
> ```
>
> Ornith-35B is `n_embd_head_v = 256`, `n_layer = 40`, and the gate ran `--kv-block-size 64`.
> **64 × 256 = 16,384.** Refused layers 3, 7, 11, …, 39 — every 4th of 40, exactly the full-attention
> set on this hybrid; the other 30 are recurrent. **100% of attention fell back to static.**
>
> ⇒ The 256k tie (1.0003) and the 512k tie (0.9905) were **static vs static-with-an-idle-pool**, which
> explains them perfectly and retroactively: 0.3% paged drift, decode a dead heat, effect always
> inside noise. They were the same code path.
>
> ⚠ The gate's validity check asserted `n_gpu_blocks > 0`. **That proves the pool was BUILT and
> nothing else.** CHECKOUT proves allocation; CONSUME proves consumption — a distinction already
> written down in this project — and the instrument was built on the wrong one.
>
> ⚠ **The 9B numbers are unaffected**: that harness ran `--kv-block-size 16`, and 16 × 256 = 4,096,
> inside the budget by accident of history. **The one parameter never re-derived when the model
> changed is the one that broke**, and because a refused layer falls back to static, the output stayed
> correct the whole time.
>
> **Fixed**: block size is now derived from `n_embd_head_v` by a 10 s geometry probe, and the paged
> arm VOIDs on the engine's own WARN-level no-consumer alarm.
>
> ### ⚠⚠ AND THE FIRST FIX TRADED A SILENT NO-OP FOR A SILENT DOWNGRADE.
>
> The 8192 bound is the **scalar** kernel's. `paged_layer_supported` already relaxes it for the
> champion (`llama-graph.cpp:4548`):
>
> ```cpp
> champ_geometry = champ_on && block_size == 64 &&
>                  (head_dim == 64|96|128|192|256);
> if (!champ_geometry && block_size*head_dim > 8192) reject(...)
> ```
>
> The champion does not stage K/V tiles — flat in nsg — and **contractually requires
> `block_size == 64`**. So at head_dim 256 there are **three** states, not two:
>
> | config | result |
> |---|---|
> | `champ=1, bs=64` | **champion serves it — the configuration paging exists for** |
> | `champ=0, bs=32` | scalar serves it, slowly (wall 1.3035) |
> | `champ=0, bs=64` | **every layer refused, silently static** — 4.5 h of "parity ties" |
>
> My geometry probe clamped 64 → 32 to satisfy a bound that does not apply, **silently selecting the
> slow kernel**. Now champion-aware, and `DS4P_METAL_CHAMP` defaults ON with `champ=` stamped in the
> header.
>
> ⚠⚠ **The code comment predicting the refusal case is dated 2026-08-06:** *"at bs=64/D=256 every
> layer refused and silently took the static path, making a paged run indistinguishable from
> static."* It was found, root-caused and written down three days earlier, **in the very function
> this gate calls**, and the gate walked into it anyway. **The knowledge existed; the harness did not
> carry it.** A finding that lives only in a comment protects the next reader of that function and
> nobody else — which is the argument for encoding it in the instrument, as the geometry probe and
> the no-consumer assertion now do.

**"as correct as": MEASURED and PASSING.** 8/8 at 8k, 5/5 pre-existing grids, and 430 chunks at 225k
with needle PASS — `FINDINGS-paged-cross-request.md`, final section.

> ### ⚠⚠ RETRACTION. The first version of this file said the opposite, in bold, and it was wrong.
>
> It read: *"AS CORRECT AS IS NOT MEASURED AT ANY OF THESE RUNGS, AND IT IS THE BIGGER RISK ...
> FINDINGS-paged-cross-request.md is OPEN and records 2 of 6 runs FAILING at 224,992 tokens."*
>
> **The finding is CLOSED.** Root-caused and fixed in `6391c5e63` (2026-08-08 17:11), which is an
> ancestor of the tree these measurements ran on: oversized final prefill chunk → compute-buffer
> layout → leftover activation floats under `s_copy` → read as a row index into a 1-row state tensor
> → garbage recurrent state from request 2 onward. Induced on demand across five pre-registered
> arms, control included.
>
> The 2-of-6 rate is retracted **inside that same document** — *"an artefact of the harnesses"*, both
> failures coming from harnesses whose warmup made the measured request #2.
>
> ⇒ **I quoted its status header instead of reading it**, and the header had been stale since the fix
> landed. Second time in one hour: `PLAN-paged-arch-support.md` has the same shape, I found it, wrote
> a banner on it, and then repeated the mistake in the next file I opened. **When a document's status
> header and its final section disagree, the final section is the state.**
>
> ⇒ `output_sanity.py` still earns its place — a check that can fail on wrong output is worth having
> whether or not today's instance is closed — but it is **insurance, not a live alarm**, and the
> first version of this file sold it as a live alarm.

---

## GPU-BOUND (queues behind the running measurement)

| item | cost | why it needs the GPU |
|---|---|---|
| clean 256k ABBA (warm-up ON, npred=512) | ~1 h | retires the "candidate" stamp on the 256k tie |
| clean 512k ABBA (warm-up ON, npred=512) | ~3.5 h | tightens the bound + first properly-sampled decode number |
| `gemma3` paged verification | ~10 min | wired but NOT proven: needs `DS4P-CONSUME > 0` and output matching static |
| `gemma4` SWA-restored check | ~10 min | proves the guard fix re-enabled what it broke; needs CONSUME on a layer where `is_swa(il)` is true |
| ~~paged-corruption rate at 225k~~ | — | **CLOSED**, see the retraction above; no runs needed |
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
