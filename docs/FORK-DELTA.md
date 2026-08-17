# FORK-DELTA — everything satindergrewal/llama.cpp carries vs upstream

**Living doc (plan item 3c). Update on every fleet merge AND every `ds4-ports` push.**
Author: Fable-DS4 | **Snapshot: 2026-08-14 (ds4-ports re-measured on this Mac)** | prior snapshots 2026-08-06 (defect-closure + upstream merge), 2026-08-04 (M7 prefill arc), 2026-08-02 ~23:15 NZST.
The 2026-08-06 promise ("update on every ds4-ports push") was broken through `08f85dbfc`. This refresh is from `origin/ds4-ports` on this Mac, not from memory. Fleet / upstream SHAs below were last measured 2026-08-06 and are **not** re-measured this wake. Anything not measured is marked.

## 0. Position at snapshot

| ref | sha | note |
|---|---|---|
| `ds4-ports` (this Mac, `origin`) | `08f85dbfc` | n_batch != n_ubatch designed refuse. Tree dirty: n_gpu_blocks==0 throw, unbuilt, uncommitted. |
| live Mac DSV4-0731 serve | bin `08f85dbfc` | `127.0.0.1:8082`, UD-Q2_K_XL, `-c 8192 --kv-paged --kv-block-size 64`, champ on. Consume proven. Not a 256k-1M bar run. |
| `fork/fleet` (THE branch) | `5751cfdde` | **stale:** last measured 2026-08-06 on the box. Not re-measured this wake. |
| upstream `master` | `f5919bf45` | **stale:** fetched 2026-08-02. Not re-measured this wake. |

**Carried delta: fleet holds 46 commits upstream lacks** (45 as of the 2026-08-02 snapshot +
`5751cfdde`); 0 behind upstream as of the 2026-08-02 fetch. The Fable-DSpark session synced
fleet 2026-08-02 night: `9ed7e49f4` merge of upstream master + `70ef59be1` merge fixes. The
45 = the 42 feature commits enumerated below + `6fd7164d8` (CI) + the 2 sync commits.
**2026-08-03: `5751cfdde` "common: un-gate kv-paged flag family for server and cli" landed on
fleet with the owner's explicit approval** (cherry-pick of the dsv4-kvpaged `3376ebbc8` content;
CPU-build validated + flag-parse smoked on the new base before push).

## 0a. ds4-ports since the 2026-08-06 snapshot (measured 2026-08-14)

137 commits after 2026-08-06 on `origin/ds4-ports`. Not dumped. Load-bearing landings:

| sha | what | status |
|---|---|---|
| `4b62ac77b` | Metal barrier before champion mask-fill | bug #21 CLOSED (paged decode non-determinism was a scheduling race) |
| `e2d787613` | DSV4 pages by default under `--kv-paged` | `DS4P_PAGED_DSV4=0` is the opt-out. HYBRID/SWA/MSA stay env-gated. |
| `15353dff6` | DSV4 single-seq promotes `--kv-block-size` 16 to 64 | champion path. `-np > 1` stays scalar. |
| `480f46f25` | champion D=512 / all 43 DSV4 layers champion-servable | Metal. CUDA decode is a different kernel. |
| `4e8b555c5` + `cab3f36d8` + `ac9594ad4` | #19 MiniMax-M3 MSA store+gather | bit-exact on synth. Real-model GGUF PARKED. |
| `08f85dbfc` | refuse `n_batch != n_ubatch` instead of `GGML_ASSERT` | on origin. Same class as the still-uncommitted `n_gpu_blocks==0` throw. |

**Metal bar (Ornith-35B, later than the 2026-08-06 table):** decode paged/static 1.3471 at 262k (paged 34.7% faster), monotone cross ~128k, 512k decode 1.9662. The screenshot / PLAN line "best clean is 32k" is stale. 1M Mac ABBA not measured (owner: skip).

**CUDA bar:** NOT done. Same bar: every row ≤1.0× at 256k-1M, then past it. `paged_attention_prefill_mma_kernel<HD>` is templated 64/128. `paged_attention_decode_kernel` is runtime `head_dim = blockDim.x` with `acc_i[4]` (fits hd 64/128 only). Outer launcher guard `head_dim == 64 \|\| 128` sends everything else, including DSV4 hd 512, to `paged_attention_decode_kernel_ref`. Widening that guard without growing `acc_i` is a silent clip. Source work is Mac-doable; the ladder is box work.

**Still open, Mac-doable unless marked box:** `n_gpu_blocks==0` designed throw (source dirty, unbuilt; live serve stays on `08f85dbfc`) · CUDA decode `acc_i` sized to `head_dim/32` then the box 256k–1M ladder (a 64/128 template without nvcc is not the bar) · B4 q8 **speed / 256k–1M bar** (e2e-correctness already PASSed 2026-08-05) · P1-5 big-model value (4B wash is honest; code already ships) · champion default-on (OWNER'S CALL, one line) · P2-8 unscoped · MiniMax live GGUF parked · HYBRID/SWA/MSA family flip will not be done unasked.

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
| `ds4-ports` | `08f85dbfc` | **2026-08-14 tip (this Mac). 2026-08-06 snapshot tip `ad34b774` is gone from this clone; do not treat that SHA as live.** Post-snapshot landings are in §0a. Historical 2026-08-06 cell follows, kept as the M7 CUDA record, not as current Metal state. **2026-08-06: 256 commits ahead of upstream, 0 BEHIND — upstream/master merged (62 commits, 11 conflicts) at `ad34b774`.** Three audit defects closed this day and re-verified AFTER the merge: **#2** paged cache dropped every ggml context/buffer it allocated (RAII, `b8e89ee4`); **#3** `/slots/N?action=save` reported success for a no-op, both endpoints now refuse structurally on the paged path (`b8e89ee4`); **#1** `-np>1` corrupted AND destroyed KV, in three sub-bugs — absolute batch offsets vs local ubatch indices (`186e4c1a`), scheduler arrays aliased not copied (`a446027e`), and the residual **partial-sequence co-batching** defect (`c2f28a79`): a co-batched ubatch can carry only PART of a sequence, so `batch_lens` described 29 tokens for a 28-token dispatch and every token from index 14 on was attributed to the wrong sequence — one token of skew, and the sequence loses its own KV. Fixed by deriving offsets/lens from the UBATCH (so `sum(lens)==n_tokens` holds by construction) and building `write_slots` per token. Verified with the regime PROVEN ENTERED: 30/30 clean reps across two tools, 6,090 co-batched dispatches, 0 invariant violations, plus long-context paged 3/3 = static 3/3. ⚠ The merge's only breakage was in a file git reported CLEANLY AUTO-MERGED (`llama-kv-cache-dsa.cpp`: upstream split one `filter` param into two); the obvious repair would have silently cost 1.78 GiB at 128K / 14.2 GiB at 1M on GLM-5.2. ⚠ **CORRECTION to the M7 note below: "Metal has NO paged kernel at all" is NO LONGER TRUE** — `kernel_paged_attn_f32` + write kernels + the paged champion port now exist in `ggml-metal.metal`, which is what all of the above was measured on. Metal remains un-tuned relative to CUDA (prefill quadratic term 8.6x static's, per-KEY), not absent. Prior state: fleet + 77 lane commits, **2026-08-04 window: P2-8 CONTINUOUS BATCHING CLOSED** (CUDA re-gate 3/3: queue-not-reject 6/6, recompute-preempt 10 lossless replays, swap-preempt 4/4 swap-ins drained) and **M7 PAGED ATTENTION**: decode split-K `8af43595` 6.1x (220.7->36.2 ms/tok), then prefill. ★★★ **M7 CLOSED BOTH HALVES AND NOW SHIPPING BY DEFAULT**: prefill **3,432 ms = 12.8x** and decode **11.5 ms/tok = 3.16x** (`76234a5f` cp.async + `d8cbc0ff` device-derived decode splits), putting the paged path **1.43x / 1.57x off the TUNED static reference** (was 17.8x / 4.85x). `f63906c2` flipped the mma prefill from dark to DEFAULT after catching a banded-model regression it would otherwise have shipped (the kernel MASKED the visibility window where the old one SKIPS it = ~21x more work on SWA models, invisible to every wall in the arc because they all ran unbanded); `7e75d119` makes the unsupported-head_dim fallback log instead of being silent. Flip witnessed on the DEFAULT path: bar green with no env incl. banded cases, three-way BYTE-IDENTICAL semantic equivalence (new == old == static), P2-8 arms 3/3 now exercising the new kernel. The gate itself fell earlier at `b1b42e83`: **8,342 ms = 5.24x** (confirm run 8,320; gate >=5.00x / <=8,745; bar max_abs 4.067e-05; decode unchanged; P2-8 arms re-gated 3/3 on the same tip). Prefill took TEN dark wmma walls capped at 29-59 s before the mechanism changed: `76fc9d4a` **fragment-held mma prefill = 12,615 ms = 3.47x** over the 43,726 baseline and 2.28x over the best of those ten (28,816) -- mma.cuh typed tiles, accumulator register-resident across the whole key loop, P feeding the next mma straight out of the score registers (ported discipline: upstream `fattn-mma-f16.cuh`, Johannes Gaessler). Then `db218d9d` **split-K over the key range = 9,570** and `b1b42e83` **__launch_bounds__(128,4) = 8,342** -- both chosen by an ncu profile that measured the kernel STARVED (5.44 active warps/SM against 256 blocks x 4 warps / 188 SMs = 5.44 exactly), not slow, which refuted the cp.async prefetch that was the next planned move. Also `bc738d28` all-warps wmma 31,525 (control), `f0e02555` KV-64 FAIL 18,364 + smem overlay FLAT 12,507. Earlier ships: chunked prefill `57501615`, prefill-on-grid `db3afe58`, fork n_past cap `4a84450c`, hybrid-fork refusal `aa3c01dd`, seq_rm ordering `76da2261`, eager teardown+storm breaker `f2a515df`, explicit-wins fitter `324827cb`, KV checksum API `a84acc8c` (acquitted the fork bit-exact), candidates-gate `9669526a`, lossless recompute `23a157e5`, admission delta `99ea0134`, swap consistency `8314d655` ⚠ **BACKEND BOUNDARY — the M7 gains are CUDA-ONLY**: `GGML_OP_PAGED_ATTN` is implemented in CUDA (`pagedattn.cu`) and as a generic CPU reference (`ggml-cpu/ops.cpp:12101`); **Metal has NO paged kernel at all**, so on Apple silicon `--kv-paged` cannot run the op on the GPU -- it is not a slow fallback, it is absent (M6 parked). Post-audit (the owner's audit-log ask): head_dim 128 had NO equivalence coverage until `31a84987` (the test hard-coded D=64 while 128 is what ships); prefill splits were never swept until `6f6b3d1b` (~8%); KV depth 32 had been skipped by inference and WINS (`8a5b288c`); the re-profile's own launch_bounds-5 lever measured 27% WORSE and was reverted (`c07a2916`). head_dim 96 is BLOCKED by the REFERENCE, not the kernel -- `fattn-banded.cu:80` static_asserts D==64||128, so there is nothing to verify 96 against; the CPU paged reference has no such restriction, which is the unblock path. | | active lane; every M7 prefill kernel is DARK behind `DS4P_PAGED_QTILE` (default path bit-identical to baseline, measured 43,717 unflagged); merge to fleet only after full arc gates + his approval |
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
