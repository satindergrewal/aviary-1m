# 256k parity: the first long-context measurement, and why run 1 does not answer the question

**2026-08-08.** Ornith-1.0-9B-1M IQ2_M, `-c 262144`, 225k-token needle prompt, `-b 512 -ub 512`,
M3 Max 128 GB. First measurement this lane has made above 40960.

## Static baseline — VALID

Single arm, flash attention, unaffected by the confound below.

```
found=TRUE   prompt_n=224992   prefill=1133.2s (198.5 tok/s)   decode=15.8 tok/s   wall=1134s
answer: ' MAGENTA-7742\n\nNote 4914 contains the'
```

Prefill curve, fixed 20k bins (binning rule pre-registered *before* the data landed):

| bin | tok/s | | bin | tok/s |
|---|---|---|---|---|
| 0 | 609.3 | | 120k | 171.3 |
| 20k | 473.0 | | 140k | 141.7 |
| 40k | 376.3 | | 160k | 141.2 |
| 60k | 285.4 | | 180k | 139.8 |
| 80k | 236.8 | | 200k | 123.4 |
| 100k | 205.4 | | 220k | 114.7 |

**5.3x throughput loss** across the run. There is a genuine flat shelf at 140k-180k (141.7 / 141.2 /
139.8, under 2% spread over 40k tokens) but it does not hold — 200k falls to 123.4.

⚠ Three mid-flight shape claims made from adjacent-sample differences ("steepening", "knee at 122k",
"plateau") were each refuted by the next sample. **Adjacent samples are ~512 tokens over ~2s and their
noise is not structure.** The binning found the shape; the narration invented it.

## ⚠⚠ Run 1's paged arm is CONFOUNDED — it does not measure paging

Stack sample of the live server (`sample <pid> 5`), non-invasive:

```
4123 of 4127 samples:
  update_slots_paged -> llama_context::synchronize -> ggml_metal_synchronize
```

CPU contributes nothing; it is GPU-bound. Which sent me to check the kernels:

| arm | attention kernel |
|---|---|
| static | `flash_attn` |
| paged | **`DS4P-MMA OFF (scalar path)` D=256 bs=16, champion matches: 0** |

**Static ran flash attention. Paged ran a scalar fallback.** The arms differed in *two* things — KV
cache *and* attention kernel — so the wall-time gap cannot be attributed to paging.
`arms-must-differ-in-ONE-thing`, violated at the most basic level.

⚠ `DS4P-MMA OFF (scalar path)` was in the paged log from the first minute. I read past it three times,
filing `D=256 bs=16` as geometry trivia. **It was the answer.**

### What survives run 1
- **The default `--kv-paged` configuration is ≥1.89x slower than static at 225k** (2145s elapsed and
  still no answer vs static's 1134s). Real and user-facing: it is what `--kv-paged` gives you today.
- The static curve above.

### What does NOT survive
- Any claim that *paging* costs the time. Cause is at least partly the kernel and is unapportioned.
- "Parity at 256k FAILS" as a verdict on the paging **architecture**. It is a verdict on the current
  **default configuration** only.

## Corrected run (`$CLAUDE_JOB_DIR/tmp/parity_fair.sh`)
1. `--kv-block-size 64` + `DS4P_METAL_CHAMP=1` so paged uses the **champion** kernel (its abort message
   documents the preconditions: block_size 64, n_seq 1, head_dim instantiated; D=256 qualifies).
2. **curl-wall for both arms.** Run 1 timed paged from process start (including model load) and static
   from curl.
3. `--metrics` on both arms.
4. **The runner asserts and prints the kernel before each request.** A check that must be remembered
   is not a check.

## Instrumentation gap — blocks the pre-registered gate
The paged branch of `update_slots` emits **no per-batch progress**, only a final `print_timings()`. So
the 20k-bin curve exists for static and is **structurally uncomputable for paged**. A single average
per arm cannot show a **crossover**, and a crossover is the whole reason the 256k-1M band matters:
"paged is a dead end at long context" and "paged costs fixed overhead that pays off above N" imply
opposite next moves. Emitting the same per-batch timing from the paged loop is a small change and
unlocks every future long-context comparison.

## Harness defects found by this run (all in the measurement path, none in the code under test)
1. **`curl --max-time 900`** in `long_context_gate.sh`. Prefill at 230k is ~40 min. **That gate could
   never have produced a result at the sizes it exists to test.** Fixed: scales with `FILL`.
2. **tokens-per-line 12, measured 23.4** — every long-context run built ~2x the prompt it claimed.
3. **15 MB/s idle log storm** at INFO, once per scheduler step, 898 MB/min — and it landed on the
   **paged arm only**, so it would have shown paged losing for a non-paging reason.
4. **`>` not `>>`** on server logs: every restart truncated the log, hiding restarts.
5. My own monitoring regex required one space after `t =`; the log pads for alignment, so it reported a
   healthy run as "0 samples".

⚠ And I **killed two healthy ~40-minute runs** after mistaking *issuing* a background sleep for time
having *elapsed*, then reading young server ages as restarts. The "gate is thrashing" diagnosis was
retracted: the second PID was the shell wrapper (`pgrep -f` matched the script path; `pgrep -x` shows
one server). The append-mode log I had just added to make restarts visible showed one model load — and
I argued against my own instrument.

## Rival engines
vLLM and SGLang are CUDA/ROCm only — **cannot run on this Mac**, so no comparison is possible here.
On the box (RTX PRO 6000, SM120) it is feasible for models vLLM supports; the Inkling-Small dead end
was specific to that model's FA4 paged-KV path, not a blanket SM120 failure. Not apples-to-apples
(different silicon, different quant formats), but it would legitimately answer "does a real
PagedAttention engine hold up at 256k". **Premature until the kernel confound is fixed** — otherwise we
would be benchmarking our scalar fallback against their production path.

---

# REVISION 2026-08-08 — the verdict above is SUPERSEDED. Read this section.

Everything above was true when written. It is no longer the conclusion. Kept intact rather than
rewritten, because the sequence of wrong turns is the useful part.

## The corrected pair PASSES the owner's bar

Same binary, curl-wall both arms, needle required, kernel identified before the request.

| arm | config | kernel | wall | prefill | decode | needle |
|---|---|---|---|---|---|---|
| static | — | flash | 1102 s | 204.4 tok/s | 15.0 tok/s | found |
| paged | bs=64 + CHAMP + LAZY | **CHAMP-PAGED ACTIVE** | 1105 s | 203.7 tok/s | **25.9 tok/s** | found |

**Prefill parity (0.997x). Decode 1.73x FASTER. Needle retrieved on both arms.**

## BLOCK SIZE IS THE CATASTROPHE — and 16 is the DEFAULT

Cleanly attributed. `run 1` and `arm A` differ in **exactly one thing**: `--kv-block-size`.
Same binary, scalar kernel in both, no lazy sync in either, same prompt.

| run | block size | kernel | result |
|---|---|---|---|
| run 1 | **16** (default) | scalar | **never completed** — >2200 s, >1.94x static, still climbing |
| arm A | **64** | scalar | 1139 s, 197.7 tok/s prefill, needle found |

⇒ `--kv-block-size 64` turns an unusable paged path into a roughly-parity one **with nothing else
changed**. This is the single highest-value configuration fact in this document.

⚠ Mechanism NOT measured. Plausibly 4x more blocks per attention call and 4x the block-table
indirection; the champion's `C == bs` contiguity requirement hints the scalar path has a similar
locality cliff. **Hypothesis with evidence, not a conclusion.**

## Arm A also shows block size does NOT explain the decode win
Arm A decodes at 14.2 tok/s — slightly *worse* than static's 15.0, nowhere near 25.9. So the decode
result comes from the champion and/or the lazy-sync fix. Arms B (`bs64+CHAMP`, no lazy) and C
(`bs64+LAZY`, no champ) isolate it; each differs from the passing config in one thing.

## ⚠ "The champion did not engage" was WRONG — a case-sensitive grep

`grep -c 'champ'` → 0. `grep -c 'CHAMP'` → 3. The log said
`CHAMP-PAGED ACTIVE D=256 bs=64 nsg=4 Q=8 C=64`. The champion **did** engage, and `C=64` is exactly
the `C == paged bs` condition recorded in `memory/champion-paged-port.md` three days earlier.

**A false retraction costs as much as a false claim.** It went unchallenged because it pointed at
self-criticism, which felt like rigour.

## The kernel assert took THREE iterations, each defeated by a different assumption

| version | defeat | consequence |
|---|---|---|
| v1 | wrong **case** | retracted a correct finding |
| v2 | wrong **time** — checked at health-ready, before any attention dispatch | UNKNOWN on a healthy run |
| v3 | wrong **variant** — knew `CHAMP-PAGED ACTIVE`, not `CHAMP-VEC` | UNKNOWN on an active champion |

Root defect in all three: **exact-string matching against output I do not control and never
enumerated.** Fixed by matching the family and *printing which variant was seen*, so a number is
traceable to log evidence rather than to intent.

**Zero false greens across all three.** Every failure was a false red that cost a re-run. A check
biased toward refusing is recoverable; one biased toward passing is how run 1 carried a confounded
verdict for eleven hours.

## Two A/B design errors, both mine
1. I believed the passing config changed **two** things. It changed **three** (`bs=64`,
   `DS4P_METAL_CHAMP`, `DS4P_LAZY_SYNC`). Neither original arm isolated the champion. *I did not know
   how many variables I had changed*, while building the run whose purpose was to separate them.
2. Arm C was left at `bs=16` — the setting arm A had just proven catastrophic — making it a
   three-variable arm that would likely never return. Caught only because arm A's result forced a
   re-read of the design.

**A precondition and a flag are not the same knob.** `bs=64` is *required by* the champion, but it is
also an independent switch that massively helps the scalar path. I collapsed the two.

---

# ATTRIBUTION SETTLED — the champion kernel is the cause

All arms: same binary, same prompt, curl-wall, needle required, kernel identified *before* the
request was sent.

| arm | config | kernel | wall | prefill | decode | needle |
|---|---|---|---|---|---|---|
| static | — | flash | 1102 s | 204.4 tok/s | 15.0 tok/s | found |
| run 1 | bs=16, no champ | scalar | **>2200 s** | — | — | **never finished** |
| A | bs=64, no champ, no lazy | scalar | 1139 s | 197.7 tok/s | 14.2 tok/s | found |
| B | bs=64, **CHAMP**, no lazy | CHAMPION | **1091 s** | **206.4 tok/s** | **25.5 tok/s** | found |
| PASS | bs=64, CHAMP, LAZY | CHAMPION | 1105 s | 203.7 tok/s | 25.9 tok/s | found |

**Two clean one-variable comparisons:**

- **A vs B** — differ *only* in the champion → decode **14.2 → 25.5 tok/s (1.80x)**.
  ⇒ The champion kernel produces the decode win. Settled.
- **B vs PASS** — differ *only* in lazy sync → 25.5 vs 25.9 decode, 1091 s vs 1105 s wall.
  ⇒ **Lazy sync has no measurable effect.** The arm *without* it was nominally faster.

## ★ Paged + champion BEATS static on every axis

| | static | paged+champion | |
|---|---|---|---|
| wall | 1102 s | **1091 s** | 1.01x faster |
| prefill | 204.4 tok/s | **206.4 tok/s** | marginally faster |
| decode | 15.0 tok/s | **25.5 tok/s** | **1.70x faster** |
| needle | found | found | — |

Against the bar — *"equal or better than static, 256k to 1M"* — this is **better**, not equal.

## ⚠ A committed fix of mine measured ZERO

`3da26e179` (lazy sync) was reverted by `72ef1fd16`. The observation was correct — ~440 redundant
synchronizes versus static's ~1, found by stack sampling — and the conclusion that they were the
bottleneck was wrong. The CPU sat in `ggml_metal_synchronize` because the GPU was genuinely busy.
**A correct observation about redundant work is not evidence that the work is the cost.**

Reverted rather than left as an inert env-gated branch: it measured zero, and a flag that does
nothing later reads as an untried idea.

## What is still NOT established
- **n=1 throughout.** One model, one context, one box, one prompt, one depth, no repeats.
- **Mechanism of the bs=16 collapse is unmeasured.** Hypothesis only (block-table indirection /
  locality cliff).
- Arm C (lazy alone) aborted on the kernel assert and was not re-run — B vs PASS already answers it.
- Graph reuse (`can_reuse` hard-returns false for paged) still **specified and unbuilt**.
- Per-batch timing for the paged loop still missing, so the 20k-bin curve remains uncomputable there.
- Metal/CUDA parity, 512k–1M ladder, multi-model: untouched.

---

# DRIFT MEASURED — cross-arm comparisons made hours apart are partly measuring the clock

Replication in **reverse arm order** (paged first, static second) to test reproducibility and arm
order together.

| arm | position | wall | prefill | decode | needle |
|---|---|---|---|---|---|
| static | 1st (forward pair) | 1102 s | 204.4 tok/s | 15.0 tok/s | found |
| static | **2nd (reverse pair)** | **1187 s** | 189.7 tok/s | 14.5 tok/s | found |
| paged+champ | 2nd (forward pair) | 1091 s | 206.4 tok/s | 25.5 tok/s | found |
| paged+champ | **1st (reverse pair)** | **1064 s** | 211.6 tok/s | 24.0 tok/s | found |

**Position 2 costs 2.5% (paged) to 7.7% (static).** The box has drifted over ~15 hours at sustained
99% GPU. This was pre-registered as the outcome that would invalidate banked results, *before* the
number existed.

## What survives

**Paged wins in BOTH orders:**

| pair | comparison | paged advantage |
|---|---|---|
| forward | static(pos1) 1102 vs paged(pos2) 1091 | 1.0% — **won from the disadvantaged slot** |
| reverse | paged(pos1) 1064 vs static(pos2) 1187 | 10.4% |

True wall advantage is roughly **5-6%**, not the 1% or 10% either single pair suggests.

**Decode is robust to drift and is the largest effect:**
static 15.0 / 14.5 tok/s vs paged 25.5 / 24.0 tok/s → **~1.68x, both orders, n=2 each.**

## What this costs the champion attribution

Arms A and B were one-variable by **configuration** but **hours apart** in wall clock, with unrelated
runs between them. With drift now measured at 7.7%, that gap is a second variable and the attribution
inherits it. Re-running A and B once more reproduces the same flaw with fresh numbers.

**Fix is interleaving, not repetition:** `A B A B` back to back, reporting *paired differences*. Drift
slow relative to a pair cancels in the difference; drift fast enough to matter shows up as
disagreement between pair 1 and pair 2 — which is the detector. (This is what the lane's own scar file
already prescribed: "interleave + rotate + fresh + repeat control". Reverse-order alone is half of it.)

## The kernel assert reached iteration 6, and iteration 5 was a different axis

| # | defeat | axis |
|---|---|---|
| 1 | wrong case (`champ` vs `CHAMP`) | how it reads |
| 2 | wrong time (checked before any dispatch) | how it reads |
| 3 | wrong variant (`CHAMP-PAGED ACTIVE` vs `CHAMP-VEC`) | how it reads |
| 4 | no case for the STATIC arm (legitimately has no paged kernel) | how it reads |
| 5 | **identify vs verify** — added an *expected* kernel per arm | **what it reads for** |
| 6 | 1-token warmup takes the vec path; scalar emits no marker there | how it reads |

Iterations 1-4 and 6 were all the same class: assumptions about a marker's exact form. **Iteration 5
is the one that matters** — run 1's confounded verdict was not "we didn't know the kernel", it was
"the arm ran a different kernel than I believed". Identification cannot catch that; only a declared
expectation can. It would have aborted run 1 at minute one.

**Zero false greens across all six.** Every failure was a refusal.

---

# ⚠⚠ RETRACTION — "block size is the catastrophe" is WRONG. Paged only works with the champion.

Direct short-context test, `-lv 5`, `--kv-block-size 64`, one variable (`DS4P_METAL_CHAMP`):

```
nochamp   CHAMP=0   DS4P-MMA=0   DS4P-CONSUME=0     <- ZERO layers consume the paged cache
champ     CHAMP=3   DS4P-MMA=0   DS4P-CONSUME=32    <- 32 layers consume it
```

**Without the champion kernel, `--kv-paged` allocates a pool that nothing uses.** Every layer falls
back to static attention. `initializing paged KV cache` prints, `paged_pool=1` reports success, and
not one layer reads the pool.

## What this kills

`arm A` (bs=64, no champ — 1139 s / 197.7 / 14.2 tok/s) is a **mislabelled static arm**: static
attention carrying a wasted allocation. So the earlier claim —

> *"bs=16 → bs=64 is cleanly attributed, ONE variable: never-finishes → parity"*

— was comparing **real paged** (bs=16, which logged `DS4P-MMA OFF (scalar path)`) against
**not-paged-at-all**. That is not a block-size result. It is the same kernel confound as run 1, one
level down, and it had already been committed here as cleanly attributed.

## The real finding, and it is a user-facing trap

The scalar paged kernel **does not implement attention sinks** — its own abort message says so and
names the fix ("use a geometry the champion serves"). Rather than corrupt, layers degrade to the
static path. The champion implements sinks. Therefore on a sinks model:

| config | what actually happens |
|---|---|
| `--kv-paged` alone | pool allocated, **silently unused**, static performance, memory wasted |
| `--kv-paged` + `DS4P_METAL_CHAMP=1` + `--kv-block-size 64` | genuinely paged, and it **beats static** |

⇒ A user enabling `--kv-paged` on such a model today gets no paging, no error, and pays the memory.
Both success indicators (`initializing paged KV cache`, `paged_pool=1`) report fine. This is the
**reports-success-for-a-no-op** shape. Every "paged" number produced in this lane before the champion
arms was measuring static attention.

## What survives — and it was always a champion-vs-static comparison

**Decode ~1.68x**: paged 25.5 / 24.0 tok/s vs static 15.0 / 14.5 tok/s. Both arm orders, n=2 each,
`CHAMP` + `DS4P-CONSUME` confirmed on the paged arms and `STATIC` on the controls, needle found on
every arm. This is the one number that has survived every check.

Wall advantage ~5-6% after drift correction; prefill is inside the noise.

## How it was found
The **expected-kernel assert** refused to produce a number for an arm whose kernel it could not name.
Without that refusal the interleave would have returned four clean-looking numbers built on an arm
that was not paged at all. Six iterations of that check, zero false greens — every failure a refusal.

⚠ `DS4P-CONSUME` is `LLAMA_LOG_DEBUG` and requires `-lv 5`. At `-lv 4` it reads 0 on arms that are
demonstrably paging. Reading that 0 as evidence is how this went unnoticed for hours.

---

# ⚠⚠⚠ CORRECTION 3 — my own config advice was HARMFUL. `--kv-block-size 64` alone disables paging.

Clean one-variable test, Ornith-1.0-9B-1M, same binary, `-lv 4`, `--no-kv-unified`, only the block
size differing. The new no-consumer guard is the instrument:

| config | pool | guard | verdict |
|---|---|---|---|
| default bs (16), no champion | 1 | 0 | **genuinely paging** |
| `--kv-block-size 64`, no champion | 1 | **1** | **POOL BUILT BUT DEAD** |
| `--kv-block-size 64` + `DS4P_METAL_CHAMP=1` | 1 | 0 | genuinely paging (and fastest) |

⇒ **`--kv-block-size 64` WITHOUT `DS4P_METAL_CHAMP` SILENTLY TURNS PAGING OFF.** The scalar kernel
serves bs=16 but not bs=64; the champion serves bs=64. With neither, every layer falls back to static
and the pool is dead weight.

⚠ **The champion is opt-in only** (`getenv("DS4P_METAL_CHAMP")` — no default-on, no auto-detect). So
the failing combination is reachable by following this document's own earlier advice.

## What this retracts
The earlier headline — *"`--kv-block-size 64` is the highest-value config fix; the default of 16 is
the catastrophe"* — is **wrong and backwards**. A user who set bs=64 on my recommendation, without
knowing an undocumented env var exists, got **less** paging than doing nothing.

Correct guidance:

| what you want | flags |
|---|---|
| paging that works, no env vars | `--kv-paged` (default block size) |
| paging that works and is fastest | `--kv-paged --kv-block-size 64` **plus** `DS4P_METAL_CHAMP=1` |
| paging silently disabled | `--kv-paged --kv-block-size 64` alone ← **avoid** |

## Arch table intact
Re-audited with `-lv 4` so the pool marker can appear, and `--no-kv-unified` so paging is not
disabled by the unified-KV conflict:

`qwen35` · `starcoder` · `ernie45` · `nemotron` — all `pool=1 guard=0`, **genuinely paging**.
4 of 7 positively confirmed. The verified-arch table is not invalidated.

## Three states, and why two of them look identical
`guard=0` alone is **ambiguous**: it means *either* "paging fine" *or* "paging never enabled".
Distinguishing requires **both** signals:

| pool | guard | state |
|---|---|---|
| 0 | 0 | paging never enabled (e.g. `kv_unified` conflict) |
| 1 | 1 | pool built but dead |
| 1 | 0 | genuinely paging |

⚠ `initializing paged KV cache` is `LLAMA_LOG_INFO` and needs `-lv 4`+. Reading its absence at default
log level produced a `pool=0` column that was pure artifact — on **every** row, including runs that
demonstrably paged. Third instance tonight of reading a 0 from a filtered log line
(`DS4P-CONSUME` at `-lv 4`, `DS4P-MMA` at bs=64, this).

⚠ Three runs tonight were silently killed by **zsh not word-splitting unquoted parameter expansions**
(`DP_FLAGS`, the ngram `--spec-type`, and this `--kv-block-size 64`). The server rejected the joined
argument and the run produced a plausible-looking zero. Pass server flags inline, never via `$var`.

---

# Champion verified on five architectures — the default-on case

`--kv-block-size 64`, `DS4P_METAL_CHAMP=1`, `-lv 4`, `--no-kv-unified`, 4K context, greedy:

| arch | pool | guard | `CHAMP-PAGED ACTIVE` | output |
|---|---|---|---|---|
| ornith9b | 1 | 0 | yes | coherent; **needle found at 225k** |
| qwen35 | 1 | 0 | yes | **byte-identical to the static fallback** |
| starcoder | 1 | 0 | yes | coherent code continuation |
| ernie45 | 1 | 0 | yes | `[Paris, London, Berlin]` — correct |
| nemotron | 1 | 0 | yes | coherent |

Three of these (qwen35, starcoder, nemotron) the champion had never served before.

## The trap generalises beyond one model
`qwen35` at `--kv-block-size 64` **without** the champion is also `pool=1 guard=1` — pool built but
dead. Second model, same failure. This kills the attention-sinks explanation offered earlier: it is a
property of the **block size / kernel pairing**, not of one model's sinks.

The failing arm logs `CHAMP-PAGED REFUSED`. So the champion is being *asked* and correctly declining —
and then **nothing else serves bs=64**. The precise bug is not "the flag disables paging"; it is
**no fallback exists for the geometry the champion declines.**

## Recommendation (not applied — default-behaviour change, owner's call)
Make `DS4P_METAL_CHAMP` default **true**, keep `=0` as explicit opt-out:

```cpp
ggml_metal_paged_champ_enabled() -> return e ? atoi(e) != 0 : true;
```

**For:** five archs page correctly under it · measured faster (decode 1.68x, n=2, both arm orders,
prefill parity) · it is the only kernel that serves bs=64 · it already **fails safe**, declining to
the scalar path when preconditions do not hold (that refusal machinery is already written and
exercised) · the current default is the one that silently breaks.

**Against:** these are 20-24 token greedy completions at 4K — a smoke test, not a correctness suite.
One long-context correctness datapoint under champion (ornith9b, 225k) and none for the other four.
A default flip changes the kernel for every paged request.

With the default flipped, both geometries work: bs=16 → champion refuses → scalar serves;
bs=64 → champion serves. The only broken combination today is the one the env var currently prevents
fixing.

---

# MECHANISM — it is a shared-memory budget boundary, and head_dim decides which side you land on

The scalar paged kernel stages a tile in threadgroup memory (`ggml-metal-ops.cpp:5603`):

```
smem_stage = (stage_v ? 2 : 1) * block_size * head_dim * sizeof(uint16_t)     budget 32,768 B
```

| head_dim | block size | tile | vs budget |
|---|---|---|---|
| 256 | 16 | 16,384 B | fits |
| 256 | **64** | **65,536 B** | **2x OVER** |
| 128 | 64 | 32,768 B | exactly at budget — fits |

Exceed the budget and the layer is **refused upstream**, so it takes the static path and the op never
dispatches — which is why the kernel's own `"paged scalar tile exceeds threadgroup memory -- would
corrupt silently"` assert never fires. The champion's layout is different (`smem=16384/32768` in its
own log line), so it serves bs=64 where the scalar cannot.

## Prediction, stated before testing, then measured

Predicted from the arithmetic: D=256 refused, D=128 marginal, D=64 fits.

| model | head_dim | bs=64, no champion | |
|---|---|---|---|
| ornith9b | 256 | `guard=1` | **TRAP** — pool built, dead |
| qwen35 | 256 | `guard=1` | **TRAP** — pool built, dead |
| starcoder | 128 | `guard=0` | no trap, pages on scalar |
| ernie45 | 128 | `guard=0` | no trap, pages on scalar |

Clean split on head_dim, exactly at the computed boundary.

## The rule

> `--kv-block-size 64` is **safe on head_dim ≤ 128** and **silently disables paging on head_dim 256**
> unless `DS4P_METAL_CHAMP=1` is set.

## What this does to the options
- **"Make the scalar path serve bs=64"** — dead. The tile is 2x the hardware budget at D=256; it would
  need a fundamentally different staging layout, which is what the champion already is.
- **Default-on champion** — strengthened. On head_dim 256 it is the *only* way bs=64 works at all; on
  head_dim 128 it costs nothing, since the champion serves bs=64 there too. **There is no
  configuration where the flip makes things worse.**
- **Refuse loudly at startup** — still the minimal alternative.

## Method note
Seven retractions this session came from generalising off observations. This finding went the other
way: arithmetic from the source first, a falsifiable boundary stated with numbers, then the
measurement. It is the only mode that produced durable results tonight — the same shape as the
dose-response that settled the SSM mechanism.

## Regression-risk case measured (not argued)

The flip only *changes* behaviour where the champion would newly serve a request. On `head_dim <= 128`
**both** kernels already work, so the flip swaps a working path — that is where regression risk lives,
not on `D=256` where the alternative is no paging at all.

Tested directly. `starcoder`, `head_dim=128`, `bs=64`, same binary, same prompt, 40-token greedy:

| arm | `CHAMP-PAGED ACTIVE` | guard | bytes |
|---|---|---|---|
| scalar | 0 | 0 | 111 |
| champion | 1 | 0 | 111 |

**Byte-diff of the two completions: IDENTICAL.** Both genuinely paging.

Two byte-identical comparisons now, on different archs and different failure modes:

| model | head_dim | champion compared against | result |
|---|---|---|---|
| qwen35 | 256 | the static fallback | identical |
| starcoder | 128 | the scalar **paged** path | identical |

⇒ Knowing *which* arm to test came from the arithmetic, not from guessing. Without the shared-memory
derivation the natural instinct is to keep testing D=256 models — where the comparison is meaningless,
because there is no working scalar path to regress from.

## head_dim <= 64: untested, no vehicle
Swept every GGUF on the box (`attention.key_length`, falling back to
`embedding_length / head_count`). Nothing at or below 64, so the third branch of the prediction has no
test. Recorded as **untested**, not as "obviously fine by the arithmetic".

The `D=128` case landing *exactly* on the 32,768 budget is the boundary where an off-by-one or a
differently-sized staging term would surface. It did not — mild evidence the formula is right, but
"predicted two, got two" is n=2 on the formula itself.

## Usable rule, no re-measuring needed
One GGUF field classifies any model:

| head_dim | `--kv-block-size 64` |
|---|---|
| **>= 256** | needs `DS4P_METAL_CHAMP=1`, or paging silently dies |
| **<= 128** | safe either way |

That is the difference between a finding and a rule, and it is cheap only because the mechanism is
arithmetic rather than empirical. Stopping at "bs=64 breaks Ornith" would have required a fresh run
per model.

---

# ⚠⚠⚠ The per-batch instrument corrupted the output it was measuring

`9c9631d02` added `DS4P_PP_TRACE` so a binned tok/s curve would be computable on the paged arm.
Reverted in `4a7fe8e13`.

One variable — same model, same 224,992-token prompt, same binary, same flags, champion confirmed
active on both, minutes apart:

| run | wall | needle | content |
|---|---|---|---|
| `DS4P_PP_TRACE=1` | 1053 s | **FALSE** | `' 123456789012345'` |
| trace absent | 1045 s | TRUE | `' MAGENTA-7742\n\nNote 4914 contains the'` |

The traced run did not *miss* the needle — it emitted a counting sequence unrelated to the prompt. A
retrieval miss looks like a plausible wrong token; degenerate counting is **corrupted context**.

⇒ **The "paged correctness defect at 225k" escalated earlier is RETRACTED.** There is no paged
corruption. There was my marker.

## The instrument was "self-validating" and that is exactly the problem

`9c9631d02` cross-checked its binned integral against the server's independently-computed
`prompt_per_second` and matched to **0.994**. That check passed *while the model emitted nonsense*,
because it validates **throughput** and is blind to **correctness**.

> An instrument can verify its own numbers and still poison what it measures — and the passing
> cross-check is precisely what makes it trustworthy-looking.

The validation covered the failure mode I imagined (wrong counters) and had no opinion on the one that
occurred (wrong text). **A check tests the premise you thought of.**

## Mechanism unknown — the two candidates differ in severity
- **(a) direct state mutation** — `llama_paged_scheduler_get_seq_state()` per batch inside the decode
  loop is the only new call; a "getter" with side effects would explain it.
- **(b) timing perturbation** — `SRV_WRN` hundreds of times per request exposing a **pre-existing
  race** in the paged path.

**(b) would mean a real defect underneath that the commit merely made visible.** Not established.
Reverted first because the risk is asymmetric and the feature is optional.

⚠ **n=1 on the failing side.** Four clean runs without the trace (three prior champion arms plus the
control), one garbage run with it. Suggestive, not conclusive. Reverted without confirming
reproduction because leaving a correctness hazard in the tree for another 18 minutes was not a trade
worth making.

## Consequence
The 20k-bin parity curve is **uncomputable on the paged arm again**. That gate returns to unevaluable
until per-batch progress can be emitted without side effects.

## The one clean pair this produced
| arm | wall | needle | prefill | decode |
|---|---|---|---|---|
| static | 1057 s | TRUE | 213.3 tok/s | 15.8 tok/s |
| paged + champion (no trace) | **1045 s** | TRUE | **215.4 tok/s** | **25.9 tok/s** |

Different runs, ~40 min apart, so drift applies — but both arms clean, both needles found, kernels
asserted. Consistent with the earlier n=2: paged wins wall and prefill marginally, decode ~1.64x.

---

# ⚠⚠⚠ The 225k corruption is INTERMITTENT — three innocent parties blamed and retracted

`paged + champion` at 225k produces byte-identical garbage `' 123456789012345'` on roughly **1 run in
3**. Every response carries `prompt_n=224992`, so the full prompt is always ingested.

## Full tally, from responses already on disk

| run | trace | headroom | result |
|---|---|---|---|
| `parity_fair/paged_opt` | off | 1.5 | pass |
| `parity_rep/rep_paged_first` | off | 1.5 | pass |
| `parity_ab/B_champ_nolazy` | off | 1.5 | pass |
| `notrace/resp` | off | 1.5 | pass |
| `parity_bins/paged_first` | **ON** | 1.5 | **FAIL** |
| `ladder/p225` | off | **1.05** | **FAIL** |

**The two failures differ in both variables that were blamed.** One had the trace marker and default
headroom; the other had no trace and low headroom. Neither hypothesis explains both.

⇒ **4 pass / 2 fail, same config family. ~33% failure rate. Intermittent.**

## Three retractions, all the same mistake

| blamed | on | retracted because |
|---|---|---|
| paging at 225k (a "correctness defect") | 1 failing run | static was clean, then the same config passed 4x |
| `DS4P_PP_TRACE` (my own instrument) | 1 failing run, 4 untraced passes | the very next untraced run reproduced it |
| `LLAMA_PAGED_POOL_HEADROOM=1.05` | 1 failing run at 1.05 | the other failure ran at the default 1.5 |

The pattern was never "wrong suspect" — it was **concluding from a single run when the comparison set
was already on disk.** The survey that settled it cost one command and no GPU.

## What the failure shape says
The occurrence is intermittent; the **failure mode is deterministic**. Both failures emit
byte-identical output, so whatever goes wrong takes the same path every time it fires. That points at
a **state-dependent trigger**, not a configuration one — a different bug shape from anything proposed
above.

⚠ `prompt_n` counts tokens **submitted**, not tokens **attended**. It cannot distinguish corruption
from silent truncation, so it rules out input-side truncation only. Same two-ledger distinction —
written vs read — that has recurred all session.

## Consequences
- **The headroom A/B is largely moot.** 70 min to test a variable the disk already exonerates.
- **The depth sweep needs redesigning.** At a ~33% failure rate, a single clean depth-90 run proves
  nothing; the design must account for the base rate before a miss can be interpreted.
- **The ~50k intermittent defect has been open on this board all session.** Same word, same lane.
  Whether it is the same bug at another scale is worth *checking* rather than assuming.
- ⚠ **Every parity number tonight came from runs that could have been silently affected.** The clean
  pairs were clean, but a 1-in-3 intermittent fault means any single unverified paged run is suspect.

---

# ⚠⚠ EVERY NUMBER IN THIS FILE IS DEFECT-ERA AND VOID (2026-08-08)

All measurements above were taken on a paged path that was **skipping a per-batch write it was supposed
to perform**. `llm_graph_input_mem_hybrid::set_input` returned early under paging (an attention-only
guard), so the recurrent input's `s_copy` was written **once at graph construction and never refreshed** —
for 24 of 32 layers, on every batch. See `FINDINGS-paged-cross-request.md` and fork `6391c5e63`.

This is not "a bug was present somewhere in the tree". **The specific work under test was partly not
happening**, and not doing it is cheaper than doing it — per batch, per layer.

⇒ The prefill parity, the decode ratio, and the comparison table are **void, not superseded**. They are
left in place, labelled, so that anyone who finds an old chart and wonders why it disagrees with a newer
one gets the reason rather than a contradiction.

⇒ Honest prior for the re-run: **paged should get slower against these figures**, and if it does that is
not a regression — it is the first measurement of the correct thing. Magnitude deliberately not estimated
in advance.

⇒ Re-run in progress: 225k needle prompt, `-c 262144`, one server at a time, PID-pinned, **needle-gated**
(a speed number from a run that got the answer wrong is not a speed number), with presence assertions that
**abort** the arm rather than warn — static refuses to run if it sees a paged pool, paged refuses if it
sees none.
