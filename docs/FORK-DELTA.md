# FORK-DELTA — everything satindergrewal/llama.cpp carries vs upstream

**Living doc (plan item 3c). Update on every fleet merge.**
Author: Fable-DS4 | Snapshot: 2026-08-02 ~23:15 NZST | Provenance: measured from git on the box
(`<BOX>/llama.cpp-fleet`), not from memory. Branch statuses cross-checked against the task
board. Anything not measured is marked.

## 0. Position at snapshot

| ref | sha | note |
|---|---|---|
| `fork/fleet` (THE branch) | `5751cfdde` | box local `fleet` identical (0/0) |
| upstream `master` | `f5919bf45` | fetched 2026-08-02 (FETCH_HEAD only, no remote added) |

**Carried delta: fleet holds 46 commits upstream lacks** (45 as of the 2026-08-02 snapshot +
`5751cfdde`); 0 behind upstream as of the 2026-08-02 fetch. The Fable-DSpark session synced
fleet 2026-08-02 night: `9ed7e49f4` merge of upstream master + `70ef59be1` merge fixes. The
45 = the 42 feature commits enumerated below + `6fd7164d8` (CI) + the 2 sync commits.
**2026-08-03: `5751cfdde` "common: un-gate kv-paged flag family for server and cli" landed on
fleet with the owner's explicit approval** (cherry-pick of the dsv4-kvpaged `3376ebbc8` content;
CPU-build validated + flag-parse smoked on the new base before push).

## 1. ON FLEET — merged, in every fleet build

### 1.1 kv-paged: paged KV cache + multi-GPU (13 commits)
`7e0805654` core paged KV cache + attention · `627c53494` rebase onto 0e4a03622 ·
`922e30735` init_multi per-device KV alloc · `17ecf05ef` multi-GPU wiring + lift split-model
rejection · `ae3eec07b` pool sized off most-constrained GPU · `9374a18e1` build_attn reshape fix
(`n_head*head_dim != n_embd`) · `79b2c2d0e` layer→backend map logging · merge `a91c47d49` ·
`cc4059bfd` **fitter true-head_dim fix — ON FLEET** (the `d6a35ba78` cherry-pick on
`dsv4-kvpaged` is the same content off an older base).
**Status:** fleet-merged, runtime-verified (19/17 split Qwen3-4B 51.2 t/s; V4-Flash MLA paged
curve 63.6→38.5 t/s @256K). The `--kv-paged` server/CLI un-gate is ON fleet since 2026-08-03
(`5751cfdde`, his approval). Known gap: hybrid archs (inkling/K3/qwen3.5) allocate full `-c`
statically = plan item 3b.

### 1.2 DSA gathered top-k attention — `GGML_OP_FLASH_ATTN_EXT_DSA` (9 commits)
`6687df53f` op (CUDA) · `ce716a527` numerical tests · `08b24f2f6` wall-clock bench ·
`42eebcbe9` engage when shapes allow · `6c243071b` optional fp32 accum · `3ba7afa89` reject
top_k the softmax can't stage · `247adff69` bench head-count fix (GLM-5.2 = 64) ·
`88ad4bdc7` fp32-accum default + gather-path log · `976075781` one-shot execution proof.
**Status:** fleet-merged, fork-only.

### 1.3 GLM-DSA MTP (NextN) speculative decoding — UPSTREAMED
Own port `9d8ea6a1c` (merged via `4e51fc898`); **upstream merged the same work as PR #25980 =
`7be2c65dc`**, which fleet also contains via sync. Still fork-only on top of it:
`ffee9f47e` load the NextN/MTP tensors instead of skipping them (upstream's TENSOR_SKIP
discards blk.78 at load — the fix that made F2 imatrix collection possible at all).

### 1.4 MTP imatrix second pass (F2) — `46fde361a`
Collect importance data for MTP (nextn) layers via a second pass. **Status:** fork-only;
⚠ F2 is REOPENED — works on the test model, yields zero blk.78 data on the real 744B
(`docs/F2-MTP-IMATRIX.md`).

### 1.5 KT / IQK quant family from ik_llama.cpp (13 commits)
`97be5e9b4` types (IQ4_K, IQ5_KS, IQ6_K, IQ1–4_KT) · `64395ea23` ftypes/names/quantize wiring ·
`16486f7f9` gguf-py incl. per-row headers · `2633d7924` ggml-cpu dispatch (add/add1/out_prod/
get_rows) · `09dbe3e83` quantization-error test thresholds · `228c5424e` dequant kernels + cuBLAS
path · `79654323c` MMVQ vec_dot IQ4_K/IQ6_K · `769f2346e` guard row-meta types out of mul_mat_id
MMVQ · `ac538eff1` `768a003d7` `e2e98a8d7` `2dad94e14` `e704bb606` row-meta MMVQ (IQ4_KT, IQ1_KT,
IQ2/IQ3/IQ5_KT/KS) · merge `219ea306f`.
**Status:** fleet-merged, fork-only — mainline cannot load KT GGUFs (why branch `kt-quants` is
protected).

### 1.6 TML Inkling architecture
`29b02c044` arch support · `8a3b616cb` mtmd: derive decoder width per-checkpoint (fixes
Inkling-Small mmproj assert; content matches upstream PR #25731 head, which is NOT merged
upstream — UNVERIFIED whether that PR moved since 2026-08-02) · `70ef59be1` restored the inkling
preprocessor after tonight's upstream merge clobbered it.
**Status:** fleet-merged, fork-only. Vision 8/8 + audio verified on Small.

### 1.7 GLM-DSA indexer KV filter
`c23459b0b` allocate the indexer KV cache only on full-indexer layers (9.75→2.62 MiB, 78→21
layers, runtime-A/B'd). **Status:** fleet-merged, fork-only.

### 1.8 Housekeeping + sync plumbing
`6fd7164d8` CI: remove build-cann.yml (zero-job workflow producing phantom failures — the CI
issue the Fable-DSpark session worked, 2026-08-02) · sync merges `6c7cec496`, `c2152b1f6`,
`54f534a04`, `9ed7e49f4` + fix `70ef59be1` · feature merges `4e51fc898`, `219ea306f`,
`a91c47d49`.

## 2. BRANCH-ONLY — built/working but NOT on fleet

| branch | tip | what | status |
|---|---|---|---|
| `dsv4-kvpaged` | `3376ebbc8` | BOTH its commits now content-equivalent on fleet (fitter=`cc4059bfd`, un-gate=`5751cfdde`) → branch fully redundant, BUT its worktree built the RUNNING V4 serve binary | keep until the serve is rebuilt from fleet, then tag-then-delete (policy rule 3) |
| `ds4-ports` | `908d6dd9` | fleet + 30 lane commits (2026-08-03): **P0-2** DS4P_REVALIDATE state↔claim binding (`4bea1cdc`, gate 9/9) · **P0-1** DS4P_YIELD_QUENCH per-request governor (`ebb85473`, CPU gates PASS) · **inkling test fixture** — synthetic GGUF, resolves upstream TODO (`7daeca7d`, `333540a3`, `972ff5c5`) · **B4 arc-1** per-stream banded attention (`3b35188b` groundwork + `58dedb17` builder: multi-slot compute buffer 518.8→53.3 MiB = 9.7×, output bit-identical, self-witnessed gate) · **multi-stream op tests** for FLASH_ATTN_EXT_BANDED (`b7972b83`; **CUDA verdict 16/16 PASS** incl. 3 multi-stream cases, results/cuda-multistream-20260803-1516.txt — arc-1 fully closed) · **arc-2 CPU half: DS4P_BANDED_QKV admits q8_0 K into banded** (`6a5f0039`, four-arm witnessed gate results/q8-banded-*) · **arc-2 CUDA half: Q8_0-K banded kernel + split-type q8 test cases** (`8f5ffd8d`; **CUDA verdict 19/19 PASS both backends incl. all 3 q8_0-K cases**, results/cuda-q8-banded-20260803-1528.txt — ARC-2 FULLY CLOSED, q8-KV multi-agent Inkling bands on both backends) · **arc-3 analytic band: visibility_window op param + DS4P_ANALYTIC_BAND drops the kq_mask** (`0ec96d43` plumbing incl. an op_params OOB-write fix, `959f7f09` caller + no-vocab ubatch_print fix; **CPU gate: buffer 10.06→4.56 MiB with the O(n_kv×n_tokens) mask term GONE, flood PASS both arms, cross-arm BIT-IDENTICAL incl. SWA-cutoff-crossing lengths**, results/analytic-band-cpu-20260803.txt; **CUDA VERDICT 23/23 PASS both devices incl. all 4 analytic cases** (Window #3 tail 17:34, ~8 s, results/cuda-analytic-band-20260803-1734.txt) — **ARC-3 FULLY CLOSED ⇒ B4 WALL COMPLETE**: per-stream banding + q8-KV + zero mask plane, CPU and CUDA all measured) · **3b step 0: --kv-paged refuses loudly on hybrid/SWA archs** (`2c52cdd4`, closes the silent-flag hole behind the 48 GiB inkling OOM; witnessed on fixture; design doc DESIGN-3B-HYBRID-PAGED.md complete w/ resolved NATIVE decision) · **3b part 1a: paged attn pool scaffold in hybrid-iswa** (`5b1e0f20`, DS4P_PAGED_HYBRID bring-up, pool constructed+owned+inert, both arms witnessed results/paged-hybrid-bringup-20260803.txt + refuse-on-tip) · **3b part 2a: ggml_paged_attn_banded op skeleton** (`c48d275d`, GGML_MAX_SRC 10→12, rel_logits @ src[10], arc-3 op_params layout, kernels refuse-loudly until the inner-loop port; regression 0 FAIL banked) · **3b part 2b: CPU banded-paged inner loop + equivalence gate** (`5e594336` port + `a2da43a3` witness: paged-banded vs banded-FA reference ALL PASS max_abs 2.4e-07 across causal/cutoff/window-only × 2 blocks × GQA, results/paged-banded-equiv-cpu-20260803.txt; Vulkan src-table latent fix) · **3b part 2c: CUDA banded-paged kernel** (`7bdeb5c2`, uniform-skip no-divergence port; **CUDA witness DONE: ALL PASS 3/3 @ 2.4e-07 both-backend-identical** (W4 tail 22:14, ~1 s, results/paged-banded-equiv-cuda-20260803.txt) ⇒ **paged-banded kernel layer CLOSED both backends**; `2e0e51bc` cuh decl fix — the box build caught the stale signature Mac CPU builds cannot see) · **part 4a-0: init_batch_with_ubatches** (`43ea16b0`, caller-owned splits — the hybrid-keeps-split-ownership primitive; no callers yet, plain path untouched) · **part 4b: hybrid paged-context plumbing** (`3432ed40`, ctx_attn_paged carried alongside on same ubatches, conditionally dark until scheduler driving; regressions green) · **part 4c-0: context-parameterized paged input builder** (`f7a8a816`, flat path delegates exactly; 4d drive-loop mapped in dossier, gated on his #1831 word) · **P1-5 increment 1: KV-bank spill-on-eviction** (`79fbc18c`, three sites spill-then-drop, DS4P_KV_BANK bring-up; ★ WITNESSED end-to-end on a real 4B, results/kvbank-spill-witness-20260804.txt) · **P1-5 increment 2: probe/admit + tail-replay** (`28ac8663`; ★★ CROSS-RESTART witness prompt_n=1 vs cold 22 + partial-salvage lcp 22/26 + floor-miss guard, results/kvbank-admit-witness-20260804.txt — the 1M-agentic-unlock primitive WORKS end to end) · **P1-5 increment 3: LRU cap** (`b2b0fe20`, witnessed) — P1-5 card-free surface COMPLETE · **increment 4 flags** (`6c9c2554`) · ★ gates 1+3 PASS FLOAT-EXACT (full + partial restore vs cold, results/kvbank-gates-1-3) · **--no-kv-bank force-disable** (`908d6dd9`, witnessed results/kvbank-nobank-witness) · ★★ **P1-6 fan-out gate PASS FLOAT-EXACT** (N agents from one prefix via cache-admit fork, results/p16-fanout-gate; paged COW = the memory upgrade, 4d-gated) · **P2-8 design banked** (DESIGN-P28-CONTINUOUS-BATCHING.md = the 4d blueprint; queue-instead-of-reject falls out; crash witness flips to regression proof; flag topology = his #1831 blank) | active lane; merge to fleet only after full arc gates + his approval |
| `fleet-dsv4-dspark` | `fe0404a32` | yaniss dspark-dsv4 merge: DSV4 drafter (PRs #25683/#25682/#25687), RoPE pairing fix, 5 DSpark correctness fixes, `dsv4-mla-brick.cpp` build fix | builds; acceptance A/B pending GPU window |
| `dsa-dequant-gather` | `42942e19b` | DSA dequant-in-gather (~253K→~572K ctx target) | binary pre-built (`wt-dsa-dequant`); gates G1 pending GPU window |
| `fable-k3-support` | `06eec9f5c` | Kimi-K3 arch support | branch-only |
| `fable-frontend-fidelity` | `0e05b3a60` | frontend/template fidelity work | branch-only |
| `fable-mtp-batch` | `4b28e1760` | MTP batching work | branch-only |
| `fable-sparse-mma-25917` | `5890dde6d` | sparse MMA (upstream #25917 lane) | branch-only |
| `fable-dsa-harvest-box` | `915b0ebed` | DSA harvest + bench binaries | kept: live worktree |
| `fable-indexer-kv-filter` | `c23459b0b` | backs fleet-merged 1.7 | record branch |
| `inkling` | `20ba4b7fb` | inkling arch source branch | backs 1.6 |
| `inkling-mm-fix` | `9ed7e49f4`→merged | backed `8a3b616cb`; its tip became tonight's fleet sync base | record branch |
| `deepseek-v4-flash` | `7a02824e9` | V4-Flash decode work (~1.9× master @16K, 2026-07-10/11) | branch-only |
| `dsv4-indexer` | `5692575a1` | dsv4 indexer side branch | branch-only |
| `kt-quants` | `968e34444` | KT types source | PROTECTED (only loader for KT GGUFs) |
| `mtp-imatrix` | `523ebb026` | unsolved F2 work | keep until F2 closes |

## 3. UPSTREAMED — fork work that reached mainline

| PR | what | merged as |
|---|---|---|
| #25980 | GLM_DSA NextN/MTP speculative decoding | `7be2c65dc` |
| #25395 (+ follow-up #25641) | Hy3 MTP | merged (see memory `hy3-pr-25395`) |

PR-backing branches `glm-dsa-mtp` (`8cd7811d1`) and `hy3-mtp` (`a00f69af7`) are **never deleted**
(fork-branch-policy rule 2).

## 4. Not inventoried here

- The ~300 inherited upstream archive branches (`gg/*`, `sl/*`, …): frozen at fork time, kept as
  passive archive (policy rule 6). For anyone's current technique, search the `upstream` remote.
- Local-only research trees on the box (`llama.cpp-kt`, `llama.cpp-k3kt`, `llama.cpp-yaniss-dsv4`,
  `llama.cpp-mtpim`, …): scratch/experiment checkouts, not fork state.

## 5. Maintenance

- Update §0–§2 on every fleet merge AND on every `ds4-ports` push (his standing ask
  2026-08-03: keep the fork's documented differences current as the lane moves).
- fleet is 0 behind upstream at snapshot (synced tonight); keep it that way — upstream moves
  near-daily (fork-branch-policy).
- Nothing lands on `fleet` without the owner's explicit per-push approval (standing).
