# THE BAR — paged must MATCH OR BEAT static, on BOTH Metal and CUDA

**Satinder, verbatim:** *"if your work still is not matched the parity or more than of static
is still not finished. it applies to both Metal and Cuda."*

This file exists because capability landings kept reading like progress against the bar. They
are not. **Hybrid support, memory policy, stall fixes — none of them is parity.** Parity is one
number per backend per phase, and it must be ≤ 1.0×.

## Current state — nothing here meets the bar

| backend | phase | static | paged | gap | met? |
|---|---|---|---|---|---|
| **Metal** | prefill | 1,131 ms | 1,570 ms | **1.39×** | ❌ |
| **Metal** | decode | 1,214 ms | 1,408 ms | **1.16×** | ❌ |
| **CUDA** | prefill | — | — | **1.21×** | ❌ **and STALE** |

**Metal ladder:** 6,950 → 3,334 → 3,300 → 2,940 → 1,570. Started the day at 2.60×, now 1.39×.
Real progress, **still not the bar.**

## ⚠ CUDA is stale and that is a gap in the reporting, not just the code

The 1.21× figure predates today's work entirely — last CUDA gate ran **08:02**, before the
head_dim function-constant specialisation (which gave Metal −46.5% on the scalar path), before
the memory policy, before hybrid bring-up. I have quoted 1.21× all day as if current. It is not.

**What today's work plausibly does to CUDA, none of it measured:**
- **Memory policy** (`common/common.cpp`) is **backend-agnostic** — the pool cap, the reserve and
  the refusal apply to CUDA too. On a discrete GPU the old fill-all behaviour was *defensible*, so
  the effect there is different and must be re-measured, not assumed.
- **Hybrid paged bring-up** is backend-agnostic. Should work on CUDA; **untested**.
- **head_dim specialisation is Metal-only** (Metal function constants). CUDA compiles templates,
  so the *equivalent* question — is the CUDA paged kernel specialised on head_dim, or generic? —
  is **open and worth exactly the same check that paid 46.5% on Metal.**

## Rules this file enforces

1. **Every status report states BOTH backends.** A Metal-only number is an incomplete report.
2. **A stale number is labelled stale.** 1.21× carries its timestamp until re-measured.
3. **"NOT SHIPPED" until every row above is ≤ 1.0×**, then the goal is to surpass, not to stop.
4. Capability work (hybrid, any-model, elasticity) is tracked separately and **never** reported
   as movement on this table.

## Next measurements owed

- Re-run the CUDA gate at the current tip; the 08:02 number is pre-everything.
- Check whether the CUDA paged kernel is head_dim-specialised. Metal's biggest win of the day was
  exactly this, and it cost ~30 lines.
- Metal prefill 1,570 → ≤1,131 is the remaining 1.39×; the scalar path has never been swept
  post-specialisation beyond nsg.

---

## CUDA lead, derived from the Metal result — PREDICTION, not a measurement

⚠ **I cannot test CUDA from this Mac** (no NVIDIA GPU; the 5090 box is where it runs). What
follows is a **structural read of the source with a stated mechanism**, and it must be measured
on the box before it is believed. Today already produced three retractions from claims that
outran their evidence.

**Read of `ggml/src/ggml-cuda/pagedattn.cu`:**

| kernel | head_dim | launch |
|---|---|---|
| `paged_attention_prefill_mma_kernel<HD>` | **compile-time template** ✅ | :1300 |
| `paged_attention_prefill_tiled_kernel` | runtime `dim3(head_dim)` | :1350 |
| **`paged_attention_decode_kernel`** | **runtime `dim3(head_dim)`** | :1383, :1395 |
| `paged_attention_combine_kernel` | runtime | :1310, :1390 |
| `paged_attention_write_kernel` | runtime (`head_dim = blockDim.x`) | :1175 |

**The prediction, and why it is not a guess:** on Metal, making `head_dim` compile-time was worth
**−46.5% on the SCALAR path and only −3.1% on the MMA path** — because the scalar kernel's hot
loop walks the head dimension per thread and unrolls completely once `D` is known, while the MMA
path's inner work was already in fragments.

CUDA is in **the same shape, half-done**: its MMA prefill kernel is already templated on `HD`,
and its **decode** kernel — the one whose threads walk the head dimension, `dim3(head_dim)` —
is **not**. That is the structural analogue of the Metal kernel that gained 46.5%.

**So the testable claim:** templating `paged_attention_decode_kernel` on head_dim, instantiating
for the common sizes (64/96/128/192/256) and dispatching, is the same ~30-line class of change
that paid on Metal, and it targets **CUDA decode**, which is on the parity table.

**Falsifiers to check first, on the box:**
1. Is CUDA decode actually gap-limited, or already at parity? The 1.21× figure is stale and does
   not separate prefill from decode.
2. `nvcc` may already be unrolling via `#pragma unroll` with a runtime bound — check the SASS or
   just measure; do not assume the Metal mechanism transfers.
3. Instantiation cost: Metal's function constants build one pipeline per D lazily; CUDA templates
   are compiled ahead of time and multiply build time and binary size. Bound the instantiation
   list to head dims that actually ship.

---

## Metal prefill levers — `--kv-block-size` swept, REFUTED (2026-08-04)

I predicted bs=16 was the constraint: the champion runs **C=64 keys per pass** while paged has
been on 16 all day, so widening the paged block should close ground. Swept on the specialised
scalar path, one binary, fresh server per point, best of 5:

| config | prompt_ms | vs static |
|---|---|---|
| STATIC | **1,128.1** | — |
| paged bs=16 (default) | 1,568.5 | 1.39x |
| paged bs=32 | 1,559.7 | 1.38x (−8.8 ms, at the ~5 ms noise floor) |
| paged bs=64 | 1,682.4 | worse |
| paged bs=128 | 2,166.8 | much worse |

Output sha identical at every block size (`3f50cc30`), so correctness is unaffected — this is
purely a performance knob and it does not pay. **bs=32 is not a win**; it sits at the noise floor
and would be a change with no measured benefit.

**⇒ THIRD independent refutation of the same idea.** Multi-block staging (`sb`), the MMA tile
width, and now the paged block size have each said the same thing: **our paged kernel does not
benefit from wider key tiles.** That is no longer a hypothesis worth re-testing from a new angle;
it is a property of this kernel.

### So where does the remaining 1.39x actually live?

Not in tile width. The honest structural read of what static does that paged does not:

- Static `kernel_flash_attn_ext` computes Q@K^T as a **matrix op** — 8 query rows x 8 keys per
  instruction, across 213 hand-tuned specialisations.
- Paged scalar computes **one query row at a time**, per-thread dot products reduced with
  `simd_sum`.

That is the difference a wider tile cannot fix. Closing it means the **MMA path has to become
competitive**, and today it is not: 1,960 vs the scalar's 1,570. For MMA to reach static's 1,131
it needs roughly **−42%** from where it stands.

**⚠ Do not read this as "the gap is unclosable."** It is a statement of where it is NOT: it is
not in staging width, not in block size, not in nsg (swept, default already optimal), and not in
barrier placement (swept, free). Those are closed. What remains unexamined on the scalar path is
whether it is doing **more work per key than static** — block-table indirection per key, masking
applied per token rather than per tile, and the per-row `simd_sum` reduction chain. That is the
next place to look, and it has not been measured.

### NEXT ARM (designed, NOT implemented): lane-per-KEY instead of lane-per-head-dim

The arithmetic of the current scalar inner loop, per key per query row at D=128, 32 lanes:

```
part += qv[i] * tk[t*D + d2];   // NPT = D/32 = 4 MACs per lane
sc = simd_sum(part) * scale;    // a full 32-lane reduction -- ~5 shuffle steps
```

So per 32 keys, each lane performs **~128 MACs and ~160 shuffle-equivalents**. The reduction is
executed **once per (query row, key) pair** and it dominates. This is the cost the champion
avoids entirely by expressing Q@K^T as a matrix op — and it is why widening the *tile* never
helped: the tile was never the bottleneck, the **reduction count** is.

### ⚠ CHECKED FIRST, as flagged — and the pure inversion IS dead. A two-phase form survives.

I said the V accumulation was the most likely way this fails and to check it before building.
Doing so: **it does reintroduce a reduction, and my "simd_sum leaves the key loop entirely" claim
was wrong.**

With lane-per-key, lane `l` holds the probability for key `t0+l`, but the output accumulator
`accv[i]` is indexed by **head-dim element** (`lane + i*32`, matching the final `dst` write).
Computing `acc[d] += sum_k p_k * V[k][d]` from that layout needs a cross-lane scatter — **a
reduction per `d`**, which is strictly worse than today.

**What survives is a TWO-PHASE loop, not an inversion:**

- **Phase A — lane-per-KEY.** Lane `l` computes the complete dot for key `t0+l`: D serial MACs.
  Then **two** reductions for the whole group of 32: one `simd_max` for the running max, one
  `simd_sum` for the running denominator.
- **Phase B — lane-per-head-dim (unchanged layout).** Scores are broadcast via threadgroup memory,
  and the V accumulation proceeds exactly as it does today: `accv[i] += p_k * V[k][d]`, no
  reduction, output layout untouched.

**Arithmetic per 32 keys per lane:**

| | MACs (scores) | MACs (V) | reductions |
|---|---|---|---|
| today | 128 | 128 | **32** |
| two-phase | 128 | 128 | **2** |

**Identical multiply-accumulate count; reductions cut 16x.** The win is real but it is a 16x cut
in cross-lane traffic, **not** its elimination — the corrected claim.

**The proposed structure: two phases, lane-per-KEY for scoring then lane-per-head-dim for V.**

- Lane `l` computes the **complete** dot product for key `t0 + l`: D serial MACs, reading the
  staged tile row for that key.
- Q is identical across lanes for a given query row ⇒ held in registers, broadcast-free.
- **`simd_sum` disappears from the key loop entirely.** 32 keys are scored per iteration with
  **zero cross-lane traffic**.

Per 32 keys the arithmetic becomes **~128 MACs and 0 shuffles**, against ~128 MACs and ~160
shuffle-equivalents today — same multiply-accumulate count, the reduction chain removed.

**Why this is worth an arm rather than a guess:** it is the one structural difference from static
that the three refuted tile-width experiments could never have addressed, and it explains their
failure rather than merely surviving it.

**Risks to measure, not assume:**
1. **Memory pattern.** Lanes now read *different rows* of the staged K tile instead of adjacent
   elements of one row. Threadgroup-memory bank conflicts are the obvious hazard; the tile is
   already staged so this is a pattern change, not extra traffic.
2. **Register pressure.** Q must be held per-lane across the whole key loop (D/1 values, not
   D/32) — that is the real cost, and at D=256 it may not fit.
3. **Tail handling.** Key counts not a multiple of 32 need masking that must not reintroduce a
   reduction.
4. ~~The V accumulation redistribution~~ — **CHECKED, see above.** It does reintroduce a
   reduction, which kills the pure inversion; the two-phase form avoids it by returning to the
   existing lane-per-head-dim layout for V, at the cost of broadcasting 32 scores through
   threadgroup memory (one extra barrier per 32 keys — cheap, and barriers measured free here).

Nothing here is implemented. It is written down so the next session starts from the arithmetic
rather than re-deriving it.
