# PENDING — what is left, and which of it needs the GPU

**2026-08-09 · the answer to "what is pending, and can it be done on CPU?"**

The split below is the useful one: **GPU-bound** means it needs the Metal device and a loaded model,
so it queues behind whatever measurement is running. **CPU-only** means it can proceed on the Mac
right now, in parallel, with one caveat recorded at the bottom that cost a measurement today.

---

## The bar

Owner's bar, verbatim: *"equal or better than the static, and the ranges are 256k to 1M context
ranges, not 2k, 32k, 64k etc."*

## ★★★ 256k VERDICT (2026-08-10 03:20) — the bar is MET, and this is the first defensible number

The first ABBA with every instrument correct at once: champion kernel, real 512-token decode window,
warm-up prelude, per-metric drift bounds, cold-arm check, full provenance in the header.

```
pos1 static  wall 856.1  pp 235.8  tg 19.16      pred_n=512 (26.7s)
pos2 paged   wall 915.0  pp 218.2  tg 27.00      pred_n=512 (19.0s)
pos3 paged   wall 783.3  pp 256.0  tg 26.24      pred_n=512 (19.5s)
pos4 static  wall 817.8  pp 246.7  tg 20.36      pred_n=512

WALL     1.0146   effect  1.5% vs drift 14.4%  ->  UNRESOLVED
PREFILL  0.9830   effect  1.7% vs drift 14.8%  ->  UNREADABLE
DECODE   1.3471   effect 34.7% vs drift  5.9%  ->  paged FASTER, CLEARS
```

All four arms proved consumption; **zero** capability-contract refusals; needle PASS on every arm.
**The read was pre-registered before the data and matched branch for branch.**

### ⚠ THE CAVEAT, NOT BURIED: the cold-arm check fired WITH `warm=1`

pos1 ran **4.4% slower on prefill and 5.9% slower on decode** than pos4. The prelude killed the
cold-first-arm effect at 8k — **which is where I validated it** — and does not fully kill it at 256k.
**I validated a fix in the regime where it works and shipped it for the regime where it does not:
the control-set failure again, on the fix for the previous control-set failure.**

⇒ EFFECT is therefore biased toward paged, and the honest **warm-vs-warm** numbers are:

```
wall     817.8 vs 849.2  = 1.0384   paged 3.8% slower
decode    20.36 vs 26.62 = 1.3072   paged 30.7% faster   <- the number to quote
```

**Decode survives the correction.** Four independent 256k decode measurements across two sample
sizes: **1.2767 · 1.3692 · 1.4090 · 1.3471.**

### THE BAR — "equal or better", 256k, champion kernel

| metric | verdict |
|---|---|
| **decode** | **BETTER by 31-35%** — resolved, clears drift ~6×, survives the cold-arm correction |
| **prefill** | **INDISTINGUISHABLE** — a 1.7% effect inside a 14.8% band. **Not "worse". Unreadable.** |
| **wall** | tie within noise, by construction (97% prefill, and prefill is the 14.8% metric) |

⇒ **Nothing is measurably worse and decode is decisively better. The bar is MET at 256k.**
Caveats attached rather than hidden: **n=2 per arm** (a drift bound, not a variance estimate), the
**cold-arm signature is still present**, and **prefill is unmeasurable at this sample size**.

---

| rung | verdict | evidence |
|---|---|---|
| 256k, champion, **pred_n=14** | prefill UNREADABLE · **decode 1.3692x** · wall UNRESOLVED | 4-arm ABBA. Decode cleared its drift 7.5×, but the window was **0.67 s** — see the inversion below. **Superseded by the verdict above.** |
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

## ★ TASK: MAKE DEEPSEEK-V4-FLASH WORK (added 2026-08-09 on the owner's order)

Model on disk: `DeepSeek-V4-Flash-0731/UD-Q2_K_XL`, **3 shards, ~96.8 GB**
(5.3 MB + 49.4 GB + 47.4 GB).

### ✅ STEP 0 RESULT (2026-08-10): STATIC WORKS. PAGED ABORTS, ONE LAYER EARLIER THAN PREDICTED.

**STATIC — the owner's goal is MET, with no code written:**

```
loads in 42 s (96.8 GB, 3 shards) -> "model loaded", listening
/v1/chat/completions:
  content   'Paris'
  reasoning '1. The user asks for the capital of France in one word. 2. The capital of
             France is Paris. 3. "Paris" is one word.'
  finish    stop (clean EOS, not a limit)   ·   38 completion tokens
raw /completion: 31.6 tok/s prefill, 22.65 tok/s decode
```

The two upstream Metal fixes in the 42-commit merge were all it needed. **Running it before writing
anything was the right call and it cost ten minutes.**

**PAGED — `--kv-paged` hard-aborts at startup:**

```
E llama_paged_scheduler_init: context does not have a paged KV cache: found a
  non-paged memory type.
server-context.cpp:1575: GGML_ASSERT(paged_sched && "failed to init the paged scheduler") failed
```

⚠⚠ **It never reaches a graph.** The tier analysis below priced the blocker at the *kernel* level —
the missing mask for top-k. **That is still true and it is not the FIRST blocker.**
`llama_kv_cache_dsv4` is a composite memory with four sub-caches and **is not a paged memory type at
all**, so the scheduler refuses before a single layer is built. The markers reflect that:
`capability contract refusals = 0`, `took the STATIC path = 0` — **not because everything paged, but
because nothing was ever asked.**

⇒ **NEW TIER 0, ahead of everything below:** `llama_kv_cache_dsv4` must own a paged pool and
`llama_paged_scheduler_init` must accept it.

**TIER 0 IS FULLY SCOPED, and it is the FOURTH instance of an established pattern — not greenfield.**
`llama_paged_scheduler_init` accepts memory by a chain of `dynamic_cast`, and its own comment on the
most recent addition reads *"Third wrapper, same resolution"*:

```cpp
llama_kv_cache_paged        -> direct
llama_memory_hybrid_iswa    -> get_mem_attn_paged()
llama_memory_hybrid         -> get_mem_attn_paged()
llama_kv_cache_iswa         -> get_mem_attn_paged()     <- "third wrapper, same resolution"
llama_kv_cache_dsv4         -> ** the fourth branch, to be written **
```

Four mechanical pieces, each with a line to copy from:

| # | change | copy from |
|---|---|---|
| 1 | `unique_ptr<llama_kv_cache_paged> mem_attn_paged` + `set_attn_paged()` / `get_mem_attn_paged()` on `llama_kv_cache_dsv4` | `llama-kv-cache-iswa.h:99-109` |
| 2 | construct the pool for DSV4 and attach it | `llama-model.cpp:2322` + `:2352` (`hybrid_iswa->set_attn_paged(paged_attn)`) |
| 3 | `get_attn_paged()` / `set_attn_paged_ctx()` on `llama_kv_cache_dsv4_context` | `llama-kv-cache-iswa.h:155-156`, and `llama-kv-cache-iswa.cpp:238` for where the ctx is set |
| 4 | fourth `dynamic_cast` branch in the scheduler | `llama-paged-scheduler.cpp:50-59` |

### ⚠⚠ AND TIER 0 MUST NOT SHIP ALONE EITHER — IT WOULD BUILD THE SILENT-FALLBACK STATE ON PURPOSE

Tier 0 makes the **scheduler** accept DSV4. It does **nothing** about the graph. So on its own it
converts today's **loud abort** into a server that **starts cleanly, allocates a pool, and pages
zero layers** — the exact state that produced 4.5 hours of static-vs-static "parity" on 2026-08-09,
and the exact state LAW 6 exists to catch.

⇒ **Today's failure mode is BETTER than what Tier 0 alone would produce.** An abort is honest. A
green server with an unread pool is not.

⇒ **DSV4 paging is all-or-nothing: Tier 0 + 1 + 2 land together, or nothing lands.** The
`gate_assert_paged_consumed` law is the check that would catch a partial ship, and the acceptance
bar stays what it was: **capability-contract refusals == 0 on every attention layer**, paged ≡ static.

### ⚠ SEPARATE CHEAP DEFECT: a user-passed flag CRASHES the server

`--kv-paged` on any arch whose memory is not paged-capable hits `GGML_ASSERT` and aborts. **A
designed refusal — "this model's memory type does not support paging; remove `--kv-paged`" — is
strictly better than an assert**, and this lane already refuses by design in three other places.
Small, and squarely inside "make the model work in expected ways".

### Step 0 — RUN IT BEFORE WRITING ANY CODE

The two upstream Metal fixes DSV4 needs are **already in the 42-commit merge**:

| commit | what it fixes |
|---|---|
| `e40bf8864` | `threadgroup half4x4[]` is a **COMPILE ERROR** in MSL (matrix types have no zero-arg constructor) — inside `kernel_lightning_indexer`, the DSV4 indexer kernel |
| `a194a75b7` | NORM/RMS_NORM drop partial-simdgroup sums → **wrong mean and variance for the whole row** |

⇒ **So "static DSV4 works" may already be true.** The first action is a serve + one-shot
completion, not a code change. **Writing code before running the binary is how a day gets spent on a
problem that was fixed upstream two days ago.**

### Step 1 — STATIC path, if step 0 fails

Diagnose against the actual error. Known-adjacent risk, rescued from a deleted clone
(`historical/antirez-dsv4-metal/`): **the graph NODE budget.** `GGML_ASSERT(obj_new)` fired during
context reserve at `-ub 1024` on the other DSV4 implementation because its floor only covered ≤512.
Ours is `max(n_tokens*40, 32u*n_tensors())` — **different shape, no floor** — and
`LLM_ARCH_DEEPSEEK4` is in that arch list. **Unmeasured on our code, not excluded.**

### Step 2 — PAGED path, and it splits three ways

`deepseek4.cpp` never calls `build_attn`; it calls **`build_attn_mha` directly, three times**:

| site | shape | pageable? |
|---|---|---|
| `:877` PLAIN | `k = mctx->get_k(ctx0, il)`, implicit causal mask | **YES — this is the shape our funnel already serves** |
| `:786` CSA | `k_all = concat(raw_k, csa_k)`, `kq_mask = concat(raw_mask, **top_k_mask**)` | no — see the blocker |
| `:841` HCA | `k_all = concat(raw_k, hca_k)`, `kq_mask = concat(raw_mask, hca_mask)` | no — same |

⇒ **THE BLOCKER, verified at the signature and not asserted:** `ggml_paged_attn_banded` takes
**no mask tensor**. Visibility is analytic — `causal` + `visibility_window` + `context_lens` — and
`rel_logits` is a distance-indexed bias bounded by `rel_extent`. **A top-k selection is arbitrary
per (q,k) pair; an analytic band cannot express it.** Missing kernel capability, not a wiring gap.

⚠ The earlier framing "dual cache, different pool geometry" was **directionally right and
mechanically wrong**: the two-cache concat is a graph-level `ggml_concat` that paging could feed
from two pools. **The MASK is the homeless piece.**

| tier | scope | cost |
|---|---|---|
| **1** | plain layers only | **small** — the gemma3 pattern. ⚠ **PRECONDITION: read DSV4's actual `swa_type`.** `get_raw()` returns `llama_kv_cache_iswa*`, and today's guard only covers `LLAMA_SWA_TYPE_STANDARD`. **Inferring the type from the cache class is exactly the move that silently disabled gemma4's paging this morning.** |
| **2** | CSA + HCA | **the real work** — optional explicit mask input on `ggml_paged_attn_banded`, the Metal kernel, a CPU reference, `test-paged-vs-cpu` coverage |
| **3** | LID indexer | **skip** — `get_lid()->get_k()` feeds `ggml_mul_mat` + relu + `top_k`. A **scorer**, not attention. |

### ⚠⚠ THE GOAL IS SUPPORT PARITY, NOT THROUGHPUT — AND MY FIRST COST-BENEFIT ANSWERED THE WRONG QUESTION

The owner, verbatim: *"the point is to have the model runnable on the machine in expected states,
and have kv paged supported too as you are doing this work already. **context is not the issue.**
This is just to make the model work in expected ways on Mac."*

The paragraph that used to sit here argued **against** paging on memory math — 96.8 GB of 128 leaves
~31 GB, so the KV pool is small either way. **That reasoning is correct and it answers a question
nobody asked.** It priced paging as a long-context *performance* lever because that is the metric I
had spent the day measuring. **The frame was mine, not the task's** — the same mistake as filing the
GGUF-conversion fix "irrelevant" because one download happened to be prebuilt.

⇒ **Support parity is a COVERAGE claim**, and the acceptance test changes with it: not a wall-clock
ratio, but **`fails the paged capability contract` == 0 on every attention layer**, with paged
output matching static.

### ★★ AND THE LAYER SPLIT SETTLES THE TIER ARGUMENT — MEASURED FROM THE GGUF, NOT ESTIMATED

Read straight out of `deepseek4.attention.compress_ratios` in the metadata shard (5.3 MB, no need to
touch the 96 GB of weights):

| ratio | layers | path | pageable today |
|---|---|---|---|
| 0 | **5** | PLAIN, single cache, implicit causal mask | **yes** |
| 4 | **21** | CSA — concat of two caches + **top-k mask** | no |
| 128 | **20** | HCA — concat of two caches + mask | no |

**Tier 1 covers 5 of 46 layers. Eleven per cent.**

⇒ **That is exactly the state this lane spent 2026-08-09 proving is indistinguishable from static** —
a pool allocated, a handful of layers reading it, the rest silently falling back, and a green that
means nothing. **Shipping Tier 1 alone would be a misleading green, not an increment.**

⇒ **So for the owner's actual goal, Tier 2 is not optional. It is 89% of the model.**

Other facts from the same read, both load-bearing:

- `attention.sliding_window = 128`, and `deepseek4.cpp:67` sets **`swa_type = LLAMA_SWA_TYPE_STANDARD`**.
  ⇒ **The Tier 1 precondition is RESOLVED, by reading the arch rather than inferring from the cache
  class**: STANDARD is exactly the rolling window the analytic band implements, so the
  `visibility_window` path covers those layers.
- `head_count = 64`, `head_count_kv = 1` — MLA-style. GQA ratio 64, integer, so the paged capability
  contract's GQA check passes.
- `indexer.top_k = 512` — the sparse selection that has nowhere to go in the banded kernel.

⚠ Also live upstream and unfixed: **issue #26694, "DeepSeek-V4-Flash degenerates into repetition and
leaks special tokens in long agentic chats (Metal)"** — Mac Studio, `-ngl 99 -fa on`, 262k ctx.
Follow-ups narrow the trigger to **prompt content, not the client**. Budget for it before promising
agentic use.

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
| ⛔ **`"out"` in the `.res` records NOTHING** (found by Grok) | I added it in the same commit that fixed `npred=512` recording a request that never took effect — **and made the identical mistake one line away.** The writer reads `os.environ.get("OUT","")` inside a python heredoc, but the shell never exports `OUT`, so it is always `""`. **A field added specifically to make artifacts self-identifying, which identifies nothing.** Fix: pass it as `argv`, the way every other value in that heredoc is passed. Queued — script executing. |
| ⚠ **stamp the model's STORAGE LOCATION in the result header** | **queued, script running.** The header records only the GGUF's basename. On 2026-08-09 the model tree under `~/Documents/GitHub/ornith-models` was reorganised **mid-run** (mtime 19:54:40) and the re-run had to be pointed at `/Volumes/KING4TB/...` instead — **a USB volume.** Two result files with identical headers can therefore describe runs whose weights came off different storage. The mmap page-fault path differs, and although the warm-up prelude pulls 21 GB into page cache on a 128 GB box (so all four arms are equal), **"the arms are equal" and "this run is comparable to yesterday's" are different claims.** Record the directory — `/Volumes/...` carries no username, so the scrubber leaves it intact. ⚠ **Downgraded after measuring**: the warm-up arms took **5 s and 6 s** off that USB volume, and a cold 21 GB USB read cannot finish in five seconds — so the pages were already in page cache from the *previous* run reading the *old* path, which is only possible if **both paths are the same physical file**. Inference from timing, not a filesystem fact (the old path is gone, so no inode to compare). The stamp is still worth having: next time this should be readable rather than reconstructed from how fast a warm-up ran. |
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
