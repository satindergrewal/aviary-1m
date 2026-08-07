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
