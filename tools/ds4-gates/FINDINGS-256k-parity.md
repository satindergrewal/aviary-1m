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
