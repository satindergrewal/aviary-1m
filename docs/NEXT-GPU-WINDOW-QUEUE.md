# Queue for the next free GPU window

## ⚠ CURRENT STATE (2026-07-30 ~22:15 NZST) — CARDS ARE NOT FREE, AND NOT MINE TO GIVE

The GLM regen chain is **PAUSED at 5,220 rows** (last row verified complete JSON). Both cards are
held by the **Kimi-K3 imatrix collection** run, at the owner's direction. `HOLD_CHAIN` is in place so
`chain_autostart` cannot fire.

- **Do not remove `<BOX>/dspark-test/HOLD_CHAIN`** — that release belongs to whoever owns the K3
  imatrix work, not to this queue.
- When it *is* removed, the chain resumes from row 5,220 by itself and the items below become
  reachable again.
- **Everything below is lower priority than the K3 imatrix lane.** A working sub-2bpw K3 outranks
  every optimisation queued here.

**Why the imatrix lane took the window:** our K3 IQ2_KT carries *three* handicaps, not one — REAP80
pruning, **no imatrix**, and requantization from already-4-bit MXFP4. Every successful sub-2bpw big
quant we have built used a BF16-direct imatrix; this one used none. Bit-width was the wrong
explanation, and our own `loop-rate-quality-axis` measurement (744B @1.75bpw is fluent) is what
refuted it.

---

## ds4-ports lane items (added 2026-08-04, tip `071b124e`; ★ EXECUTED 2026-08-04 window, tip `aa3c01dd` — Fable-DS4)

**2026-08-04 window outcome (the owner's direct "do your work" order; witnesses in
tools/ds4-gates/results/*-box-20260804*):** item 1 CUDA gate PASS · warm-admit mechanism
EXACT / 5x bar honest-FAIL at 4B-GPU · 16-agent 6.39x PASS · fork-cost ~0% PASS (4 mid-gate
ships incl. the warp-parallel decode kernel: ref was 40 s/step at 8K ctx) · ub2048 2.05x ·
M3 ~12% parked · P1-7a parked with full U-curve · P2-8 finale: arm1 queue-not-reject TRUE PASS on CUDA (dec=0, f2a515df); arm2 evict/preempt = honest FAIL-AT-BAR (swap engages then LIVELOCKS; fix named: recompute-preemption + act-on-DEADLOCK; fitter-override ship 324827cb came out of it). Witness results/p28-finale-20260804.txt.
**REMAINING for a future window:** Q5 quench-econ (needs a V4+drafter serve slot — could
not co-tenant the regen) · fork-residual discriminator · tiled prefill kernel (~20x).

Current holder is the DSpark regen (~27 h to target at the 2026-08-04 03:1x measure); these
ride whatever window the owner opens, after the holder's own resume needs.

1. **CUDA hybrid_paged_gate (~20 min, cheapest decisive).** `git -C <BOX>/wt-ds4-ports
   fetch fork && git checkout 071b124e` → rebuild `build-cuda` → run ornith
   `tools/ds4-gates/hybrid_paged_gate.sh <build-cuda> <port>`. Closes the ONLY open 3b arm:
   the CUDA 4c-1 hybrid path has never executed (Mac close was CPU). The gate is
   self-witnessing (op-level + parity + it writes its own results file).
2. **P1-5 gate 2: warm-admit ≥5× @64K** (box, real model, ~30 min).
3. **16-agent ≥6× throughput gate** (the continuous-batching headline number, ~45 min).
4. **fork-cost gate** (P1-6 COW: N forks vs N cold prefills, ~20 min).
5. **quench-econ window** (`tools/ds4-gates/quench_econ_window.sh`, ~30 min).
6. **ub2048 A/B + MMA gap + fitter boot** (parked M3/M4 measurement plans, ~40 min bundle).
7. **P1-7a** (per its design doc).

---

Written 2026-07-30 while both cards were held by the GLM head chain (~34 h remaining at the
latest measured rate; see the window forecast at the bottom). Ordered by
value-per-minute. **Cheapest decisive item first.**

Deliberately a checklist, **not** an auto-runner. `boxwatch` detects the free window and alerts
(`IDLE-UNCLAIMED`, escalating at 5 min then every 15) but is read-only by design — it never
launches. An auto-seizing watcher already caused one collision in this project: two of my own
agents took both cards and OOM'd five MTP arms. Whoever is alive picks this up; nothing grabs
silicon on its own.

**How the window opens:** the chain's `llama-server` (one process, tensor-split across *both*
cards) exits. Opus's `llama-quantize` is CPU-only and does not hold a card, so both cards free
together. Confirm with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
returning nothing.

---

## 1. Fused Lightning Indexer probe — ~90 seconds, answers a standing Priority-A question

Is the fused DSA lightning indexer enabled on our `--tensor-split 49,51` serve, or does
`resolve_fused_ops` disable it on a device mismatch?

**★ CORRECTED 2026-07-30 — this item is now CONFIRMATION, not discovery, and it probably answers
"enabled".** An earlier version of this file said the probe output is `-lv 4`-only. That was too
broad, and the distinction matters:

```
mismatch -> LLAMA_LOG_WARN -> PRINTS AT DEFAULT VERBOSITY 3
success  -> LLAMA_LOG_INFO -> needs -lv 4
```
(measured both ways by forcing a mismatch: `lv=3` gives WARN=2 / INFO=0; `lv=4` gives WARN=2 /
INFO=9.)

**Consequence:** the live GLM serve log — 106,045 lines at verbosity 3 — contains **zero** fused-op
mismatch WARNs. Since WARNs *do* print at 3, **no device mismatch occurred**, so the fused
Lightning Indexer is most likely **already enabled** on the production config. That substantially
weakens Program A's "unfused indexer" premise, for free.

**What remains genuinely unknown:** the `resolving fused Lightning Indexer support:` header is INFO,
so we cannot confirm from that log that the probe *ran*. The state is **enabled-or-never-ran**,
not disabled. The `-lv 4` load below distinguishes those two.

Throwaway load, read, kill. **Do not run a long job at `-lv 4`** — that verbosity produces very
large logs.

```sh
# from the fleet tree, using the normal serve flags; kill as soon as "model loaded" appears
./build/bin/llama-server -m <BOX>/bigmodels/glm52-ours/GLM-5.2-ours-IQ1_S-prot-blk78q4.gguf \
  -ngl 99 --tensor-split 49,51 -c 4096 -np 1 -b 1024 -ub 256 \
  -ctk q4_0 -ctv q4_0 -fa on --port 8099 -lv 4 > /tmp/lid_probe.log 2>&1 &
# then:
grep -E "resolve_fused_ops" /tmp/lid_probe.log
```

**Enabled (expected):** `resolving fused Lightning Indexer support:` then `Lightning Indexer enabled`
**Never ran:** no Lightning Indexer lines at all even at `-lv 4` — that would be the surprise, and
it would mean the call is not reached for GLM_DSA.

`-fa on` is **not** a confound here: measured on Metal, `-fa on` skips only the **FA** probe; the
GDN and Lightning Indexer probes still run. See `docs/RESEARCH-KV-QUANT.md`.

---

## 2. DSA dequant-in-gather acceptance gates — the big FIT item

Branch `dsa-dequant-gather` (`42942e19b`) on the fork, off `fleet` @ `c23459b0b`. **Compiles with
symbols verified; never run on a GPU.** Full rationale in `docs/PLAN-DSA-DEQUANT-IN-GATHER.md`.
Potential: ~253K → ~572K context *with* flat decode (2.26×), extrapolated from measured inputs.

**★ ALREADY BUILT — no build time on the critical path.** `<BOX>/wt-dsa-dequant/build/bin/`,
built 2026-07-30 while the cards were busy. Verified by outcome: `libggml-cuda.so.0.17.0` 245.4 MB,
`--version` reports `42942e19b`, `--list-devices` enumerates both Blackwells, and both quantized
kernel instantiations plus the dispatcher are present in the linked `.so`.

(`llama-server` there is 17,920 bytes — a **thin wrapper**, the same size as fleet's known-good one.
Do not mistake it for a failed link; the payload is in the `.so`.)

If it ever needs rebuilding: Blackwell requires `/usr/local/cuda-12.9/bin/nvcc` and
`-DCMAKE_CUDA_ARCHITECTURES=120a`; `/usr/bin/nvcc` is 12.0 and `native` resolves wrong.

Run the gates **in this order** and stop at the first failure:

1. **Correctness.** `-ctk q8_0 -ctv q8_0` under **`-fa auto`**, never `-fa on`. Confirm the DSA
   gather path is selected, then NIAH at 32K. *Gate: matches the F16 arm.*
2. **Ceiling.** Load at 398K (q8_0) and 572K (q4_0). *Gate: fits VRAM.* A miss falsifies the
   extrapolation, not the kernel.
3. **★ FLAT DECODE — the kill gate.** Decode t/s at depths 1K / 32K / 100K / 200K. *Gate: flat
   like the F16 gather arm (~50 t/s), NOT the mask path's decay to 29 t/s @100K.* **If decode
   degrades, the dequant is not free and the change should be dropped, not tuned.**
4. **Control.** F16 arm **bit-identical** to today, proving the template did not perturb the
   existing path.

Not for `fleet` until 1, 3 and 4 pass.

---

## 3. `sa_spec +34% acceptance` — the last quarantined claim

Inherited, never verified by me. Do not let it touch `fleet` or the serve config without a local
log. Everything else in that quarantine has been resolved: the prompt-cache premise, the
`GGML_CUDA_FA_ALL_QUANTS` "25× CPU fallback" (refuted), and "no tokenization prefix reuse" (true
but a 5.6 ms non-issue).

---

## 4. GLM live-eviction event — closes the last gap in the `-cram` recommendation

Everything about the prompt cache is measured **on Qwen3-4B / Mac Metal**. The GLM byte arithmetic
now rests on a measured identity (prompt state == KV size, agreeing to 0.008%), so the one
unwitnessed thing is narrow: **does eviction actually fire on a live GLM serve at the default
8192 MiB cap?**

Three requests on the real model — long prompt A, long prompt B, then A again — with `-lv 4`.
Expect at the default cap: `making room for prompt cache entry, removing oldest entry` and a full
re-prefill on the third request. Then repeat with `-cram 12288` and expect `prompt_n` to collapse.

See `docs/PROMPT-CACHE-CRAM.md`; harnesses in `tools/prompt-cache/` port directly.

---

## Standing constraints for whoever runs these

- **`fleet` is not a draft.** Merge only on a passing gate, measured on the path that justifies it.
- Sweep KV-quant types under **`-fa auto`**; pin `-fa on` only for known-good combos — `-fa on`
  skips the support probe, so it will happily serve a configuration `auto` refuses to start.
- Do **not** rebuild into a tree whose binary a running job was launched from. Use a throwaway
  worktree with compile-only output, as `42942e19b` was built.
- KT quants (`IQ1_KT`/`IQ2_KT`) are ik trellis types **mainline cannot load**.
- Witness outcomes, not announcements: `prompt_n` over log lines, and cost over "the op was
  reached". Two of my own claims died on that distinction in one afternoon.

---

## Build-tree map — check this BEFORE planning any run (added after it nearly cost 2 h)

Opus lost time discovering that **the only K3-capable tree was CPU-only, and the only CUDA build
had no K3 arch**. Neither fact is discoverable from a model file. Verify the tree can do the job
*and* the arch before queueing work against it.

| tree | KIMI_K3 arch | CUDA build |
|---|---|---|
| `llama.cpp-k3` / `-k3kt` | **yes** | was CPU-only (Opus has since built CUDA: nvcc 12.9, `120a`, `FA_ALL_QUANTS=OFF`) |
| `llama.cpp-idxfilter` (= `fleet`) | **no** | yes |
| `llama.cpp-kvpaged` | no | — |
| `llama.cpp-kt` | no | yes |
| `llama.cpp-dspark-metal` (Mac) | no | Metal |

Blackwell needs `/usr/local/cuda-12.9/bin/nvcc` + `-DCMAKE_CUDA_ARCHITECTURES=120a`; `/usr/bin/nvcc`
is 12.0 and `native` resolves to an unsupported `compute_120a`.

**Verify a build by outcome, not by `[100%]`:** check the produced `.so` size and that the binary
enumerates the devices. `--version` proves nothing — it never loads a backend.

## Window forecast (HISTORICAL — chain is paused; recompute after resume)

Rate is **not** constant — it moves with CPU contention from concurrent quantize/build jobs:

| window | rate | note |
|---|---|---|
| ~13:00-15:00 Jul 30 | 257 rows/hr | |
| 15:09-18:57 Jul 30 | **218 rows/hr** | build ran 6 min at the end |
| **18:57-19:46 Jul 30** | **224 rows/hr** | **build gone entirely — no recovery** |

At 218/hr the chain would free both cards around **05:00 Aug 1 NZST** — but **do not plan on that
number**, because the slowdown may be transient and its cause is not established.

**I originally attributed the 15% to a concurrent quantize + CUDA build. That was withdrawn on
arithmetic**: the build was 6 min of a 229-min window at `-j 6` of 32 cores ≈ **0.49% of window
CPU-time at nice 15**, which cannot produce a 15% drop, and the quantize ran in *both* windows so it
cannot explain a *change*. Live candidates, in the order I would bet on them:

1. the quantize's **bandwidth** footprint growing as it moves from small dense tensors into ~1 GiB
   expert stacks — interference that grows while threads, load and PSI all stay flat
2. per-row cost drifting as the dataset advances
3. ~~the build~~ — **ELIMINATED by measurement.** The rate did **not** recover after the build
   ended (218 → 224 is flat, both ~13% below the 257 baseline), independently confirming the
   0.49% arithmetic. Two routes, same answer.

**Discriminator, free:** `boxwatch` already samples rows every 60 s, so any window can be computed
retroactively. Recovery *while* the quantize still runs implicates the build; recovery only *after*
it ends implicates bandwidth; no recovery after both end implicates row-cost drift.

**Current best ETA: ~32.4 h from 19:46 Jul 30 → ~04:15 Aug 1**, on the clean 224 rows/hr window.
Still assumes the rate holds; recompute rather than trust it.

**Use a window of at least ~30-45 min.** At 1.9-20 s per row a few minutes yields too few units to
resolve — quoting a rate off a too-short window is a documented failure mode here.

**Priority note:** a KLD number on our own K3 outranks everything in the queue above. If the K3
battery wants the window, it takes it — the dequant-in-gather gates are not time-sensitive.
