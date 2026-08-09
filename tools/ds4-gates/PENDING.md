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
| 256k, champion, **pred_n=14** | prefill UNREADABLE · **decode 1.3692x** · wall UNRESOLVED | 4-arm ABBA. Decode cleared its drift 7.5×, but the window was **0.67 s** — see the inversion below. **Re-running.** |
| 256k / 512k, **bs=64 champ off** | ⛔ **VOID** | the "paged" arm never paged: 3,610 refusals, `64 × 256 > 8192` |
| 8k, 9B, **pred_n=128** | **prefill 1.0075 (clears)** · **decode 0.9017 (clears)** | clean box, cold-arm check +0.1%. The first decode number in this lane with a real window — **and it says paged is 9.8% SLOWER.** |
| 8k, 35B, pred_n=14 | wall 1.0095 · prefill 0.999x · decode 0.878x | decode leg inherits the short-window status; prefill leg survives |
| 1M | **not measured** | Fits in memory (f16 116.1 GiB / q8_0 76.1 GiB of 128). ~20-24 h, ~5.5 tok/s decode. Owner's call. |

> ### ⚠⚠ THE DECODE SIGN IS NOT STABLE UNDER SAMPLE SIZE. Everything decode-related is provisional.
>
> `n_predict` is a **ceiling**, not a floor. The needle prompt is answered in **14 tokens** and the
> model emits EOS (`pred_n=14, pred_ms=670, stop_type=eos`, against a requested 512). **Every decode
> number this lane produced before 19:25 was averaged over 0.67 seconds.**
>
> With `ignore_eos` and a real 128-token window, the 9B smoke **inverted**: decode went from ~0.98×
> to **0.9017×**, clearing its own drift. A short window can hold a **stable wrong sign** — the 35B's
> 1.3692× reproduced across two arms to 0.6% and is still not safe.
>
> **Two readings, registered before the re-run:**
> **(a) context-dependent** — block-table indirection is a fixed per-step cost, while static
> attention cost grows with context; at 8k the overhead dominates, at 256k the benefit does. *Both
> numbers true.* **(b)** the 1.37× is an artefact and collapses. **(a) predicts the re-run reproduces
> it.**

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
| ★ **DECODE CONTEXT SWEEP — the decisive experiment** | **<1 h total** | The 9B at 8k says paged decode is **0.9017×** (slower); the 35B at 256k says **1.3692×** (faster). **Same kernel, same block size, verified** — `head_dim=256 · CHAMPION · bs=64` on both — so it is not a kernel confound. If the fixed-per-step-indirection-vs-growing-attention-cost story is right there is a **crossover length**, and it is findable cheaply: **one model, ABBA, `ignore_eos`, at 8k · 32k · 64k · 128k.** Prefill at those lengths is minutes. A monotone curve crossing zero supports the story; a flat 0.90 everywhere with a jump at 256k refutes it and points somewhere else. **This answers "why" and gets "is it real" for free at the low end; a second 256k point estimate answers neither.** |
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
| ⛔⛔ **STALE `.res` CAN BE READ AS THIS RUN'S ARM** (found by Grok, 2026-08-09) | If an arm dies, `arm()` returns early, `static.res` is never written, the ABBA rename never happens — and **the previous run's file survives and is loaded as this arm.** Right now `static2.res` on disk is an **8k fixture from my own smoke test**: `n=3665, wall 6.5s, ok=true, needle PASS`. It parses, it passes every check the summariser makes, and it would produce **"paged is 104% SLOWER"** with full confidence. Two one-line fixes: **(a)** assert all four arms share `prompt_n`, else VOID — the field is already recorded and compared against nothing; **(b)** clear `$D/*.res` at gate start. Same class as the stale headers and the stale arch table: **an old file indistinguishable from a new one**, and this instance I created myself by running smoke tests into the measurement directory. |
| ⚠ **the `.res` omits the field a verdict depended on** | The artifact carries wall/pp/tg/n and **no `predicted_n`**, which is why artifact-first verification could not catch the `n_predict` defect — the load-bearing field simply was not there. Add `predicted_n`, `predicted_ms` and the run's own `OUT` path. **An artifact must carry every field its verdict depends on, or it launders assumptions.** |
| ⚠ **`ignore_eos` and `output_sanity.py` COLLIDE — sequence them** | With `ignore_eos: true` the model runs past its answer for the full 512 tokens, and what follows a completed answer is very often repetition. **`output_sanity.py` would grade that DEGENERATE on a perfectly healthy run** — its repetition detector fires at 0.35 and post-answer filler blows through that. Fix them together, not separately: grade only the text **up to the first EOS position**, or send a second short correctness request with `ignore_eos` off. Two fixes that are each right and jointly wrong is how the last three defects in this file were built. |
| wire `output_sanity.py` into `paged_parity_gate.sh` | **next up.** Save the arm's text into its `.res`, grade paged against static in the ABBA summariser. Zero extra GPU cost, turns every future parity run into a corruption sampler. |
| ✅ ~~`ignore_eos`~~ · ~~achieved `pred_n` in the artifact~~ · ~~stale `.res` guard~~ | **all shipped and smoke-verified 2026-08-09** (`3f02818`): `pred_n` 14 → 128 on every arm, the achieved length printed beside the requested one, `decode window actually generated: [...]` in the summary, `*.res` cleared at start **and** a VOID unless all four arms share `prompt_n`. |
| ✅ ~~LAW 6 unproven~~ | **VALIDATED by direct control** (`alarm_control.sh`): arm B proved the marker prints (240 banded, **0 auto**), arm A fired the alarm on 20 refusals. ⚠ Arm B's `auto=0` also **refuted my own published explanation** for the alarm's historic silence — eliminating three alternatives was not evidence for the fourth. That silence is now an **open loose end**, deliberately left unexplained. |
| ⛔ ~~**`ignore_eos` — `PP_NPRED` has never taken effect**~~ (fixed, kept for the record) | **highest-value queued fix.** The request sets `n_predict: 512`, which is a **ceiling, not a floor**, and the model answers the needle in **14 tokens** then emits EOS: `pred_n=14, pred_ms=670, stop_type=eos`. **Every decode number in this lane is sampled over ~0.6 seconds**, which is *shorter* than the 48-token case the change replaced. Fix is one field, `"ignore_eos": true`. Also print the ACHIEVED `predicted_n` on the arm line — the header's `npred=512` records what was *requested*, and **a parameter that silently does not take effect is worse than one that is absent, because it looks like the question was asked.** |
| decide `-lv 4` vs `-lv 5` in `paged_parity_gate.sh` | **queued, cannot edit a running script.** The gate runs `-lv 4` and proves consumption via LAW 6 (absence of the engine's WARN alarm). The sibling `arch_serve_gate.sh` runs `-lv 5` **specifically so it can read `DS4P-CONSUME` directly** — its own comment says *"At -lv 4 the marker count read ZERO on an arch that was demonstrably paging."* A direct positive count beats an inferred one; the cost is DEBUG-volume I/O during a timed run, which is itself a confound. Record whichever is chosen **as a choice**, so nobody "fixes" it back. |
| warm-up response is never checked | **known hole.** `warmup()` sends its curl to `/dev/null`, so a 500 or an empty completion still prints "discarded (5s)". Log-line-is-not-work-done, in code I wrote today. |
| the remaining ISWA archs | `gemma2` `gemma3n` `cohere2` `cohere2moe` `phi3` `olmo2` `exaone4` `exaone-moe` `openai-moe` `plamo3` `mellum` `mimo2` `smallthinker` `afmoe` — minus anything CHUNKED or SYMMETRIC, which the band cannot express at all |
| ~~`mimo2` sinks / `dflash` non-causal~~ | ✅ **re-verified in source, genuinely closed.** `mimo2.cpp:187` passes `sinks` into the paged call; `dflash.cpp:467` passes `/*causal=*/false`. Both were checked because `PLAN-paged-arch-support.md`'s top table was stale enough to cause a real bug, and "recorded closed" is not the same as closed. |
| ~~does the new `swa_type` guard break them?~~ | ✅ **no.** Both declare `LLAMA_SWA_TYPE_STANDARD`, which the analytic band implements exactly, so the tightened guard is a no-op for them. Checked rather than assumed — the last guard I added silently disabled gemma4. Window and causal are also orthogonal in the kernel (`lo` from the window, upper bound from `causal`), so `dflash` passing a window **and** `causal=false` is not a contradiction. |
| lint sweep | `lint_paged_consumers` · `lint_scrub_coverage` · `lint_common_laws` — the last reports 6 of 44 gates reach the shared laws |
| **champion-kernel arch coverage** | `arch_serve_gate.sh` passes no `--kv-block-size`, so it runs the engine default of **16** (`common/common.h:565`) on the scalar path. 16 × head_dim ≤ 8192 holds up to head_dim 512, so every arch pages and the matrix greens are valid. **But the CHAMPION is never exercised there** — 21 architectures verified on one kernel and none on the other, while the champion is the kernel the parity numbers come from. Not urgent, and not nothing. |

### Kernel selection, as a table, because getting it wrong is silent

| gate | block_size | champion | how it proves paging |
|---|---|---|---|
| `arch_serve_gate` | 16 (engine default) | off | counts `DS4P-CONSUME` directly — runs `-lv 5` **on purpose** |
| `long_context_gate` | 32 | off | LAW 6 (engine WARN alarm) |
| `multislot_gate` | 16 | off | LAW 6 |
| `warm_multislot_gate` | 16 | off | marker at `MS_LV=5`, else LAW 6 |
| `paged_parity_gate` | **64** | **on** | LAW 6 |

⚠ `paged_parity_gate` is the only one on the champion, and it is the only one whose output is a
**speed** claim. That is deliberate — the bar asks whether paging is as fast as static, and the
answer must come from the kernel paging actually ships with — but it does mean the champion's
correctness rests on far less arch coverage than the scalar path's.

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
