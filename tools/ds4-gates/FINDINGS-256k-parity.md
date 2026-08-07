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
