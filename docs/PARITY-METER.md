# THE BAR — paged must MATCH OR BEAT static, on BOTH Metal and CUDA

**the owner, verbatim:** *"if your work still is not matched the parity or more than of static
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

### Two more obstacles found while specifying phase A — both have standard fixes

**(a) Q cannot live in registers.** Phase A needs lane `l` to read the **whole** Q vector for the
row (D values), not its `NPT = D/32` slice. At D=128 that is 128 floats per lane — a register
blowup, and fatal at D=256.
**Fix:** stage the row's Q in threadgroup memory, one D-vector per simd group — `nsg * D` floats,
**4 KB at nsg=8, D=128**. All lanes then read the same `sq[d]` each step, which is a *broadcast*
in threadgroup memory, not 32 separate reads. Cheap, and the MMA path already stages Q this way.

**(b) ⚠ The K read in phase A is a worst-case bank conflict, and the arithmetic says so.**
Lane `l` reads key row `t0+l`, i.e. address `tk[(t0+l)*D + d]`. Consecutive lanes are therefore
`D` halves apart. At D=128 that is **256 bytes = 64 four-byte words**, and `64 mod 32 == 0`, so
**all 32 lanes land in the same bank — a 32-way conflict on every access.** That would erase the
entire reduction saving and then some.
**Fix:** pad the staged tile stride, `tk[t*(D + pad) + d]`, choosing `pad` so the row stride in
4-byte words is **odd** (in halves, `D + 2` gives 65 words at D=128). Standard shared-memory
padding; costs `KT * pad` halves of smem, negligible at KT=16.
**This must be measured, not assumed** — it is precisely the kind of "the fix is obvious" step
that has been wrong three times today.

**Revised cost model per 32 keys per lane** (D=128), with both fixes applied:

| | MACs (scores) | MACs (V) | reductions | barriers |
|---|---|---|---|---|
| today | 128 | 128 | **32** | 0 in the key loop |
| two-phase | 128 | 128 | **2** | +1 per 32 keys |

Barriers measured free on this kernel (repeat-controlled), so the +1 is affordable.

**Order of work when this is built:** stage Q per simdgroup → phase A with a **padded** tile
stride → verify 12/12 → wall against 1,570 → only then tune. If the padded stride does not remove
the conflict, the arithmetic above is wrong somewhere and the arm stops there.

### ⚠ PRECONDITION discovered: the design needs `bs >= 32` — and the "refuted" sweep already paid for it

Reading the actual loop (`ggml-metal.metal:3328`): the key loop is bounded by **`bs`**, not by the
simdgroup width.

```cpp
const int tend = min(bs, n_tok - t0);
for (int t = tbeg; t < tend; ++t) { ... simd_sum(part) ... }
```

At the default **`bs = 16`** a staged tile holds **16 keys**, so a lane-per-key phase A would idle
**half of a 32-lane simdgroup**. **The two-phase design therefore requires `bs >= 32`**, where one
tile maps exactly onto one simdgroup: 32 keys, 32 lanes, one phase-A pass per block.

**This retroactively gives the block-size sweep a purpose.** It found no win — bs=32 at 1,559.7
against bs=16 at 1,568.5 sits on the noise floor — but it establishes something the design needs:
**bs=32 is FREE.** A negative result that licenses a precondition is not a wasted arm.

Revised reduction count **per block** (not per 32 keys), at bs=32:

| | reductions per block |
|---|---|
| today (bs=16) | 16 — one per key |
| today (bs=32) | 32 — one per key |
| two-phase (bs=32) | **2** |

**Updated build order:** switch the paged default to `bs=32` (free, measured) → stage Q per simd
group → phase A lane-per-key with a **padded** tile stride → 12/12 → wall against **1,570**.

⚠ Changing the default block size is **user-visible** (`--kv-block-size`) and changes the pool
block accounting, though not total pool bytes — `bytes_per_block` scales with `bs` while
`n_gpu_blocks` shrinks proportionally under the n_ctx-bounded sizing landed today. Verify that
the memory policy still reports the same footprint at bs=32 before treating it as free in
practice as well as in latency.

### bs=32 precondition — CONFIRMED on the memory axis too (not just latency)

The lane-per-key design needs `bs >= 32` or half the lanes idle. Latency already showed bs=32
free (1,559.7 vs 1,568.5, noise floor). That was only **one** axis — a design that bought speed
by doubling the pool would be no design at all, and this lane has already had one pool eat the
machine. Measured at tip `da4c628a`, one request served per arm so the pool is touched and not
merely allocated:

| bs | blocks | bytes/block | pool | RSS | sysfree |
|---|---|---|---|---|---|
| 16 | 1536 | 2,359,296 | **3.46 GiB** | 6.65 GB | 64.7% |
| 32 |  768 | 4,718,592 | **3.46 GiB** | 6.66 GB | 64.7% |

Identical, and the sizing line says why: `blocks_per_seq = ceil(n_ctx/bs)`, so
`ceil(n_ctx/bs) * bs * bytes_per_token * headroom` has **bs cancelling** whenever bs divides
n_ctx. This was derivable from the arithmetic — it was measured anyway, because "the fix is
obvious" has been wrong three separate times in this lane today.

**Precondition paid on both axes. bs=32 is free.**

⚠ Both arms hit `Abort trap: 6` at shutdown — the known **residency-set teardown assert**
(Grok reproduced 2/2). It fires on `pkill`, *after* RSS is sampled, identically in both arms,
so it does not touch this result. Stays on the open list; noted here so a later reader does not
mistake this run for the reproducer.

## Lane-per-key two-phase: BUILT, gated, and it refutes its own premise — fork `02a3b6d9`

`DS4P_METAL_LPK=1` (needs `bs>=32`). Phase A scores 32 keys with **two** reductions instead of a
full 32-lane `simd_sum` per (row,key); phase B keeps the original lane-per-head-dim V loop.
12/12 vs CPU at every head_dim with `DS4P-LPK ACTIVE` printed; all server shas identical.

**The marker earned its keep again:** the correctness test was hard-coded `BS=16`, where the
path can *never* run. `DS4P_METAL_LPK=1` would have printed **ALL PASSED for a kernel that never
executed.** Test `BS`/`NB` are now env-driven.

| arm | forward (pos) | reverse (pos) |
|---|---|---|
| STATIC | 1,483 (1st) | 1,916 (5th) |
| PAGED-SCALAR bs=32 | 2,126 (3rd) | 2,853 (3rd) |
| **PAGED-LPK** | **2,106 (4th)** | **2,704 (2nd)** |
| LPK-UNPADDED | 2,787 (5th) | 3,165 (1st) |

**1. Padding is load-bearing — robust.** UNPAD is the worst arm *even from the most advantaged
slot*. The 32-way bank conflict was predicted from arithmetic and is confirmed at 14-24%.

**2. LPK wins, small.** Better in BOTH orderings — including forward, where it ran in the
*disadvantaged* slot and still won. ~1-5%. No tighter number is honest.

**3. ★ THE PREMISE IS REFUTED BY ITS OWN FIX.** I argued: width refuted three ways, therefore by
elimination the bottleneck is the **reduction count**. I then removed ~94% of the reductions and
got **~3%**. The fix worked; the theory behind it did not. `simd_sum` is a cheap hardware
shuffle-reduce here. **Reductions are priced at ~3% — that is the condition this negative needs**
([[refuted-needs-its-condition]]). Standing suspect for the real cost: device->threadgroup
staging traffic, which no arm has isolated yet.

### ⚠⚠ HARNESS DEFECT — sequential arms drift 29% by position

STATIC measures **1,483 ms first and 1,916 ms last** in the same script. Arm order is a
confound, and it is larger than most effects this lane has chased. **Every few-percent delta
measured with sequential arms is suspect**, including my own "bs=32 is free, 0.6% noise floor".
That claim survives *directionally* — bs=32 ran later, was penalised, and still measured equal —
but the quoted noise floor was fiction.

**Rule going forward: any wall claiming <10% must run its arms in both orders**, and report the
direction where the winning arm had the *disadvantaged* slot. A single forward sweep is only
valid for effects larger than the positional swing.

**METER (this session, forward run): Metal prefill STATIC 1,483 vs best paged 2,106 = 1.42×.**
Not parity. LPK narrows it; the gap is not substantially the reduction count.

## ★★ DRIFT REMOVED — the real numbers, and my "corroborated 1.41x" was also wrong

Fix was not averaging over the drift, it was **removing** it: one warm + one measured request per
**fresh server**, arms **interleaved** (every arm sampled every round) and the round order
**rotated** so no arm owns the best slot. Tip `02a3b6d9`, 4 rounds.

| arm | mean | spread |
|---|---|---|
| STATIC | **1,478** | 1,472-1,484 |
| STATIC2 (repeat control) | **1,477** | 1,472-1,485 |
| PAGED-SCALAR bs=32 | 2,013 | 2,009-2,018 |
| **PAGED-LPK bs=32** | **1,947** | 1,942-1,950 |

**Repeat control: 1,478 vs 1,477 — a 0.07% noise floor.** That is the first real noise floor this
lane has had. Every arm's spread is non-overlapping.

**Real results:** LPK beats scalar by **-3.3%** (2,013 → 1,947), solid. Parity **1.362x → 1.317x**.

**★ AND THE CORRECTION THAT MATTERS MOST:** I posted that ~1.41x was "corroborated in both
orders, same ratio to 0.6%". The drift-free value is **1.317x**. Both sequential runs were wrong
in the same direction, and their agreement was not evidence.

> **Two confounded measurements agreeing does not de-confound them.**

Forward had STATIC in the best slot and LPK in a bad one (inflates the ratio); I then treated the
reverse run's near-identical ratio as corroboration instead of asking why a reversal did not move
it. Agreement between two runs sharing a confound is exactly what a confound produces.

**Why the drift existed at all:** the old harness ran **5 back-to-back reps inside one server**,
so heat accumulated within the burst and the later arm ate it. One measured request per fresh
server, with the restart as a cooldown, removes it — the arms no longer share a thermal history.

**METER (drift-free, tip 02a3b6d9): Metal prefill 1.317x · decode 1.16x · CUDA 1.21x STALE ·
12/12 · noise floor 0.07% · NOT SHIPPED.**

## Decode re-gated on the interleaved harness — 1.16x CONFIRMED, not superseded

Same discipline as the prefill run (fresh server per sample, interleaved, rotated, repeat
control). LPK is **not** an arm: it is gated `n_tokens > 1`, so decode takes the combine path.
Metric is server-reported `predicted_ms`, not wall — wall would fold in prefill and HTTP.

| arm | mean predicted_ms | tok/s |
|---|---|---|
| STATIC | **2,440.1** | ~105 |
| STATIC2 (repeat control) | **2,448.9** | ~104 |
| PAGED bs=32 | 2,858.4 | ~89 |

**Noise floor 0.36%** (2,440.1 vs 2,448.9) — looser than prefill's 0.07%, still far below the
effect. **Decode parity = 1.171x**, against 1.16x from the broken harness.

**This is a CONFIRMATION, and it was PREDICTED BEFORE THE RUN** (chat #2455): the drift mechanism
is thermal accumulation across back-to-back reps, and decode reps generate less heat per rep than
a 1,500-token prefill, so the confound had little to bite on. Calling it ahead of time is what
makes it a confirmation rather than a retrofit — the old number was right for the wrong harness.

⇒ **The broken harness did not corrupt everything it touched.** It corrupted the *prefill*
numbers, where reps are expensive. That is the CONDITION this negative needs
([[refuted-needs-its-condition]]): sequential-arm drift scales with per-rep cost, so cheap-rep
gates are far less exposed than expensive-rep ones.

**METER (both re-gated, drift-free): Metal prefill 1.317x · Metal decode 1.171x · CUDA 1.21x
STALE · 12/12 · floors 0.07%/0.36% · fork `02a3b6d9` dirty=0 · NOT SHIPPED.**

## ★ STAGING-TRAFFIC SUSPECT — REFUTED (pre-registered criterion, fork dirty=3 → committed)

The standing suspect for the prefill gap was device→threadgroup staging traffic. Arm: stop
staging V, read it straight from device (champion shape). **One variable** — K staging, tile
geometry, bs, nsg, LPK all held. Gated 12/12 both arms with `VSTAGE=OFF` printed; smem drops
21,632 → 13,440 at D=128, so the traffic really was removed.

| arm | mean | |
|---|---|---|
| STATIC | 1,501 | |
| STATIC2 (repeat control) | 1,500 | **floor 0.07%** |
| LPK, V staged | **2,033** | |
| LPK, V from device | **2,057** | **+1.2% — WRONG DIRECTION** |

**Pre-registered in chat #2455/#2464 before the run: CONFIRMS at ≥5%, REFUTES at <1%.**
Result is **+1.2% the wrong way ⇒ REFUTED.** Device→threadgroup staging traffic is not the
bottleneck, and this arm is filed as a negative, not spun.

**Why it is slightly WORSE, which is the useful part:** all 8 simd groups share one K/V tile but
own different query rows. A staged V is read 8× from threadgroup memory; an unstaged V is read 8×
from **device**. Staging *amortises* across simd groups — it was never pure overhead. **The
condition this negative needs:** V staging pays whenever `nsg > 1`; at `nsg == 1` there is nothing
to amortise and the answer could invert.

### Suspects remaining for the 1.35× prefill gap, ranked

1. **The honest one: our kernel vs their kernel.** STATIC runs the champion
   `kernel_flash_attn_ext` — heavily tuned, 213 template specialisations, full MMA. Paged runs
   our hand-written scalar walk. Our own MMA attempt came in *slower* than our scalar. The gap
   may substantially be that, not any single line.
2. `block_table` indirection per block (never isolated).
3. Per-block `threadgroup_barrier` structure (barriers measured free earlier, but not at bs=32
   with the LPK layout).

**Three eliminations now stand, each with its condition:** tile width (3 ways), reduction count
(~3%), staging traffic (+1.2% wrong way). That is the search space genuinely narrowing.

**METER: Metal prefill 1.354× (1,501 → 2,033) · decode 1.171× · CUDA 1.21× STALE · 12/12 ·
floor 0.07% · NOT SHIPPED.**

## ★★ BLOCK-TABLE PROBE: CONTIGUOUS — gather/indirection eliminated BY CONSTRUCTION

`DS4P_DUMP_BT=1`, the benchmark request (1,500-token prefill, `-np 1`, bs=32):
```
DS4P-BT CONTIGUOUS  max_blocks=47  first16: 721 722 723 724 725 726 ... 736
```
Offset, but **perfectly sequential**. The pool hands out blocks in order for a single sequence.

**Consequence, and it is large:** paged has been doing **near-identical memory access to static**
in every wall this lane has ever run. So:
- **`block_table` gather/indirection was never being tested** and cannot be the 535 ms. Eliminated
  *by construction* — no timing arm needed, which is the cheapest elimination available.
- The "scattered paged access is slow" intuition is **not applicable to any number on this board.**
  It would become live only under fragmentation (multi-seq, eviction, long-running reuse), and
  **that is the condition** ([[refuted-needs-its-condition]]) — the negative holds for `-np 1` on
  a fresh pool, not in general.

**Suspect list is now ONE.** Tile width (3 ways), reduction count (~3%), staging traffic (+1.2%
wrong way), gather/indirection (contiguous by construction) — all eliminated with conditions.

### What remains: our kernel vs THEIR kernel

STATIC runs `kernel_flash_attn_ext` — **213 template specialisations, full MMA, years of upstream
tuning.** Paged runs our hand-written scalar walk. Our own MMA attempt came in **slower** than our
scalar. With paging itself now shown to be free in this configuration, **the 1.317x IS the
implementation gap** and there is no paging-specific defect left to find.

**⇒ The honest path to ≤1.0x is a paged variant OF THE CHAMPION KERNEL**, not further point fixes
to ours. That is a large piece of work — the champion's specialisation machinery is the thing that
made it fast, and matching it means adopting that machinery, not out-tuning it by hand.

**This is a decision for the owner, not a task to start on my own authority:** big-kernel port vs
the capability queue (attention-only pool filter, hybrid decode gate, other hybrids, SWA pool,
capability-based fallback). Recorded here; not begun.

**METER: Metal prefill 1.317x · decode 1.171x · CUDA 1.21x STALE · 12/12 · floor 0.07% ·
NOT SHIPPED.**

## Milestone 1 of the champion path: MMA kill RE-CONFIRMED on the honest harness — and it names the defect

The MMA-is-slower verdict (scalar 1,570 vs MMA 1,960) was measured on the **sequential** harness —
the one since shown to drift 29%. A path killed on unreliable evidence had to be re-asked before
rebuilding it from scratch. 12/12 first with `DS4P-MMA ACTIVE` printed per config.

| arm | mean |
|---|---|
| STATIC | 1,483 |
| STATIC2 (repeat control) | 1,479 — **floor 0.27%** |
| PAGED-LPK | **2,019** |
| PAGED-MMA | **2,727** |

**Pre-filed: >5% worse ⇒ kill CONFIRMED. Measured +35%.** The MMA path is genuinely slower; the
old verdict was right even though the harness under it was not. Re-asking cost 12 minutes and
converted a suspect result into a solid one.

### ★ And the marker names the defect, which is the actual deliverable here

```
D=128  MMA: nsg=2  QR=16  smem=22,656/32,768
D=128  LPK: nsg=8         smem=21,632/32,768
```
**Our MMA is smem-limited to nsg=2 at D=128 — one quarter of LPK's 8 simd groups.** It is not
losing on math, it is losing on **occupancy**: the staged tile is too fat to keep simd groups
resident. At D=192 it collapses further, to nsg=1/QR=8.

**That is exactly what the champion's specialisation machinery buys** — compile-time `D` ⇒ tighter
smem ⇒ more simd groups resident. It is the same lever that paid **-46.5%** when applied to our
scalar path (head_dim as a function constant). **The champion port's first target is therefore
named and measured, not guessed: cut MMA smem until nsg reaches 8 at D=128.**

⇒ Port proceeds **from our MMA path**, not from zero — with occupancy as milestone 2, not a
line-by-line transcription of a kernel with 213 specialisations.

**METER: prefill 1.317x · decode 1.171x · CUDA 1.21x STALE · 12/12 · NOT SHIPPED.**

## Milestone 2: occupancy is REAL and worth -14.8%, but NOT sufficient — and it reopens O-in-fragments

K destaged from the MMA path (`DS4P_METAL_MMA_NOSTAGE_K=1`, requires `bs % 8 == 0` so an 8-row
fragment never straddles two physical blocks). **nsg 2 -> 4 at D=128, smem 28,928 — the exact
byte count predicted from arithmetic before the code was written.**

| arm | mean |
|---|---|
| STATIC | 1,511 |
| STATIC2 (repeat control) | 1,508 — **floor 0.20%** |
| PAGED-LPK | 2,054 |
| PAGED-MMA nsg=4 | **2,323** (was 2,727 at nsg=2) |

**Occupancy priced: doubling nsg bought -14.8% on the MMA path.** Real, large, and the mechanism
is confirmed. **But MMA still loses to LPK by 13%**, so by the pre-filed criterion this is a
**PARTIAL**: occupancy is binding but not the whole cost. Priced and filed, not spun.

**⚠ First attempt FAILED 12/12 at nmse 1e+13** — destaging K shifts every threadgroup pointer
after `tk`, and I broke the "allocation and layout move together" invariant that I had *written
in the comment directly above*. The gate caught it before any wall ran. That is the entire reason
correctness runs first, and it is the second time today a marker/gate caught a silent-wrong path.

### ★ The result reopens a design I killed EARLIER TODAY — because its condition expired

I killed **O-in-fragments** this session with: *"champion has 213 template specialisations;
**runtime D** ⇒ dynamic register-array indexing ⇒ spills"*. **That condition no longer holds.**
`D` became a **compile-time function constant** (`FC_paged_attn_D`) in the specialisation work
that paid -46.5% on the scalar path. **The blocker is gone.**

And the arithmetic now points straight at it: at nsg=4, `so` (the O accumulator) is
**16,384 of the 28,928 bytes** — the dominant term, and the only one still scaling with QR.
Getting O out of threadgroup memory is both the champion's actual shape and the largest
remaining smem term.

**⇒ Milestone 3: O in fragments, unblocked by a condition that expired.** This is exactly why a
negative must be filed WITH its condition ([[refuted-needs-its-condition]]) — filed bare, this
design would have stayed dead and the port would have gone looking elsewhere.

**METER: prefill 1.317x · decode 1.171x · CUDA 1.21x STALE · 12/12 · NOT SHIPPED.**

## ★★ M3 PREMISE KILLED BEFORE CODING — the champion keeps O in threadgroup memory

I asserted twice that O-in-fragments "is the champion's actual shape". **It is not**, and my OWN
comment at `ggml-metal.metal:3036-3039` says so:

> *"the O accumulator lives in threadgroup memory and is rescaled by scalar threads, **because
> there is no per-row scale on a simdgroup_matrix**."*

Champion source agrees: `:7495` `o8x8_t lo[NO]` is **loaded from threadgroup `sot`** at `:7501` —
fragments are transient working copies inside a block iteration, not the durable home.

**⇒ The expired condition reopened LESS than claimed.** Function-constant `D` removes the
*register-indexing* blocker. It does **not** remove the *rescale* blocker — the online softmax
needs a per-row multiply and a simdgroup_matrix has none. **Two independent blockers, and I
conflated them.** One grep away, twice.

**⇒ `so` cannot be eliminated. nsg=8 on the MMA path is permanently unreachable**, not merely
unreachable at this layout.

### The conclusion two milestones actually bought

```
MMA  nsg=4  smem 28,928  wall 2,323   <-- capped at 4 by so = QR*512, STRUCTURALLY
LPK  nsg=8  smem 21,632  wall 2,054   <-- already at 8, already 13% faster
```
**LPK is structurally better on the very axis M2 proved matters, and already wins. MMA is the
wrong substrate for the port.** Established with numbers across M1 (kill re-confirmed +35%) and
M2 (occupancy priced -14.8%, ceiling proven), not assumed.

**⇒ REVISED M3: specialise LPK, not MMA.** Function constants for the LPK loop bounds so the
D-loop and NPT unroll fully — the same lever that paid **-46.5%** on the scalar path, applied to
the path that already leads. No re-derivation needed to resume.

**BAR CORRECTION (Grok, accepted):** clean-tip bar is **STATIC 1,483**, not the 1,511 from the
dirty-tree wall. Board prefill stays **1.317x**.

**METER: prefill 1.317x · decode 1.171x · CUDA 1.21x STALE · 12/12 · NOT SHIPPED.**
