# KERNEL ADAPTATION MAP — our measured-win kernels × the fleet's models

Author: Fable-DS4 | Date: 2026-08-03 | Satinder's ask: "which kernels we made which gave us
excellent boost in speed... we did not push them upstream... re-code the same idea and make it
supported for other models like GLM 5.2, DeepSeek V4, Kimi K3, Qwen3.6, Inkling."
Predecessor: `_private/pr-queue.md` (2026-07-12, upstream-contribution framing — still valid
for the upstream side, which stays his per-item call). THIS doc is the fleet-adaptation
framing. Numbers below are from the recorded measurements (memory/board); arch facts verified
against the trees tonight where marked.

## 0. Attribution doctrine (his rule, 2026-08-02)
Every ported or adapted technique records its source: idea-credit in the doc + commit message
(upstream PR/repo/author), and git commits carry the usual Co-Authored-By trailer. Ports from
Apache-2.0 sources (MLX Steel, vLLM kernels, ds4) are license-clean with attribution; verify
per repo before first commit.

## 1. What we built, what it measured, where it lives

| work | measured win | lives on | upstream status |
|---|---|---|---|
| DSA gathered top-k attention (`FLASH_ATTN_EXT_DSA`, 9 commits) | serves GLM-5.2 today | fleet | fork-only |
| DSA dequant-in-gather | ~253K→~572K ctx (2.26×) projected; **gates G1 pending** | branch `dsa-dequant-gather` | fork-only |
| V4 gather-decode (stock ops) | +8-10% @16K, grows with depth (Metal) | branch `deepseek-v4-flash` 6652af2c | pr-queue #1, never PR'd |
| V4 HC fusion CUDA | +85% decode (32.7→60.5 t/s) | 7a02824e | **superseded**: upstream #25585 (am17an's own impl) |
| V4 HC fusion Metal (2 kernels + wiring) | +86% decode on M3 Max | d934fda8, eaaefa27 | **unclaimed gap**: Metal side of merged #25585 |
| V4 sinkhorn Metal kernel | +29-34% pp | cd4383dc | offered on #25421, standing |
| V4 indexer Metal + decode wiring | perf-neutral scaffold, memory-flat | branch `dsv4-indexer` | waits on #25370 |
| KT/IQK row-meta MMVQ kernel family | serves every KT quant | fleet | fork-only (type-level) |
| kv-paged multi-GPU + fitter fixes | on-demand pool vs static | fleet | fork-only |
| GLM indexer KV filter | 9.75→2.62 MiB indexer cache | fleet c23459b0b | fork-only |
| GLM MTP (NextN) | 1.37× decode | — | **UPSTREAMED** #25980 |

## 2. Applicability matrix (arch facts verified where noted)

| idea → model | GLM 5.2 | DSV4-Flash | Kimi K3 | Qwen3.6/Ornith | Inkling |
|---|---|---|---|---|---|
| DSA gather family | ✅ HAVE | ⚠ served Flash variant has NO indexer tensors (metadata only — verified 2026-08-01); applies only to a future dsv4-indexer-variant GGUF | ❌ no indexer — K3 is a KDA linear-attention hybrid (`LLM_TENSOR_SSM_G`, verified in k3kt tree tonight) | ❌ no indexer | ★ **YES — Lightning Indexer is DSA-family** (walls doc); THE new port target |
| dequant-in-gather | ★ G1 gates queued | same caveat as above | ❌ | ❌ | ★ follows the gather port |
| kv-paged (+3b hybrid fix) | ⚠ GLM_DSA has its own cache BY DESIGN | ✅ HAVE (serving today) | ★ **via 3b** (hybrid family) | ★ **via 3b** (qwen3.5 hybrid = Ornith) | ★ via 3b (the original motivator) |
| KT quants + row-meta MMVQ | ✅ | ✅ | ✅ (k3kt tree) | ✅ | ✅ |
| Spec-decode + P0-1 governor | ✅ MTP | ✅ DSpark head (regen running) | ? check GGUF for nextn tensors when relevant | ✅ (qwen3.5 MTP context exists in create_memory) | drafter TBD |
| V4 HC/sinkhorn Metal kernels | — arch-specific to V4 | Mac-only value; **Mac V4 rung REJECTED by his verdict 2026-08-02** → fleet value ≈ 0 today; worth = upstream credit | — | — | — |

**★ The 3b upgrade discovered tonight:** the hybrid-paging fix is NOT an Inkling-only item.
K3 (KIMI_K3) and Qwen3.5/Ornith are hybrids in the same `create_memory` family — one fix
covers the fleet's whole hybrid class. 3b's priority rises accordingly.

## 3. Recommended adaptation order (fleet value first; upstream = his per-item call)

1. **3b hybrid paging** — now covers Inkling + K3 + Ornith/Qwen3.5. Already in the design doc
   (`DESIGN-DS4-P0.md` §4); this map only raises its priority argument.
2. ~~DSA-gather → Inkling port~~ — **WITHDRAWN 2026-08-03 after the arch-mapping read (the
   honest first check fired).** Our tree's Inkling impl (src/models/inkling.cpp:1) is "hybrid
   iSWA attention + per-layer packed shortconv" — DENSE relative-position attention with
   sliding windows, NO indexer, no top-k selection anywhere in its attention path. The
   walls-doc description "Lightning Indexer (DSA-family)" does not match the implementation;
   the log line it cited prints from the fused-ops resolve table (llama-context.cpp:62), not
   from inkling code. There is nothing to gather. **What replaces it:** the read produced a
   concrete B4 hypothesis — the rel-attention path materializes [n_kv, n_tokens]-shaped
   structures (`rel_idx` I32 [n_kv, n_tokens], inkling.cpp:181, plus rel-bias/KQ f32
   intermediates), and B4's measured 55.3·ub·ctx_M/10³ GiB works out to **~56 bytes per
   (query-token × kv-token) pair ≈ a stack of ~14 such 4-byte planes** — consistent with
   FA on/auto being identical if the rel path bypasses FA or feeds it a dense kq_b. HYPOTHESIS
   ONLY: verify by graph dump (-lv 4 / cb_eval) in a GPU window. If it holds, B4's fix is
   chunking those planes over context — same class as FA's score chunking, and THE Inkling
   attention item (not DSA-gather).
3. **G1 dequant-gather gates on GLM** — existing queued item, GPU window; unchanged.
4. **P0-1 governor** — model-agnostic; already designed (§3 of the design doc).
5. **K3 nextn check** — one GGUF metadata read when a K3 spec-decode question arises. Cheap.
6. **V4-Metal kernel family (HC/sinkhorn/gather)** — no fleet deployment target today (Mac V4
   rejected at 7-12 t/s; box V4 runs upstream's fused HC). ⚠ HIS RULE 2026-08-03: **nothing
   goes upstream, ever — no PRs, no comments; fork-and-branches only.** The branch is KEPT as
   raw material for future **dedicated kernel-only sessions** (his idea: sessions whose sole
   target is making kernels faster, iterating/researching novel ways). Also see the Mac-V4
   revival question: he'd use V4-on-Mac at 20+ t/s — our measured Metal chain reached ~15-16
   t/s short-ctx; DSpark stacking on that base is UNMEASURED. Revival = re-download quant +
   rebase kernels onto the current #25585 three-op interface + measure. His call.

## 3b. Fleet merge status + when to queue each merge (his ask 2026-08-03)

**Already ON fleet — nothing to queue:** DSA gathered attention (incl. the generic gather
path in llama-graph.cpp), KT row-meta MMVQ family, kv-paged multi-GPU + fitter fix, GLM
indexer KV filter, MTP, Inkling arch + mm fix. (Measured in FORK-DELTA §1, not from memory.)

**NOT on fleet — queue recommendations:**

| branch work | merge trigger | notes |
|---|---|---|
| `dsa-dequant-gather` (2.26× ctx) | **G1 gates pass** (GPU window) + his approval | nearest real merge; flat-decode kill gate is the blocker by design |
| `dsv4-kvpaged` server un-gate (3376ebbc8) | next approved fleet push, or batched with first ds4-ports merge | production-proven since 2026-08-01 (his V4 serve runs it); flag-gated, low risk |
| `fleet-dsv4-dspark` (DSpark runtime) | acceptance A/B in a GPU window + approval | existing small-stuff #4 |
| V4-Metal campaign (`deepseek-v4-flash`: HC Metal +86%, sinkhorn, gather-decode) | **do NOT queue** — revisit triggers only | CUDA HC superseded upstream (#25585, already in fleet); Metal has no target (Mac V4 rejected); gather-decode targets the indexer path the served Flash variant doesn't have; branch 3+ weeks stale vs moved upstream = conflict surgery for zero current serving value. Triggers: dsv4-indexer-variant GGUF becomes a target / Mac V4 revives (his 20+ t/s bar) / a dedicated kernel-only session picks it up |
| `dsv4-indexer` branch | same triggers as above | |

**Adaptation note found during this check:** current fleet `deepseek4` still builds the
fake-sparse dense-with-mask path (`build_top_k_mask`, src/models/deepseek4.cpp:620) while OUR
generic DSA gather lives in llama-graph.cpp (`dsa_gather` engage at :2959). Wiring deepseek4's
indexer path onto the generic gather = the clean "same idea, another model" port — queued as a
design note, actionable when an indexer-variant V4 GGUF is a real serving target.

## 4. What this doc does NOT change
- Phase A gating: item 2's arch-mapping read is paper/read-only and fits Phase A; any CODE
  for it queues behind Satinder's read of the design doc and the P0 sequence.
- ⚠ `_private/pr-queue.md` is **DORMANT** (his rule 2026-08-03: NOTHING goes upstream or to
  any third-party repo — no PRs, no PR/issue comments; fork-and-branches ONLY). Kept as
  historical record.
- Nothing touches `fleet` without his explicit per-push approval (standing).
