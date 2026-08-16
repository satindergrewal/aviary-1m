# DS4 bar

Phrase: **ds4 bar**
When Satinder says that, read this file and answer met / not met. Do not narrate history.

## Goal (met only if ALL are true)

On a daily model (DSV4, Inkling, Ornith/Qwen3.5, GLM, or the Aug 7 short list), Metal, `-np 1 --kv-paged`:

1. One 256k–1M master stays resident in a single shared KV pool.
2. Child agents admit against the **pool**, not against a slot count.
3. Children share the master's prefix by refcount (not a copied slab).
4. If the pool is full: queue or swap. Never reject-because-slots. Never slice 1M into 4×256k.
5. A child leaving does not copy or destroy the master.

Decode may be single-width. Speed ≥ static is a later gate, not this one.

## Current verdict

**NOT MET.** 2026-08-16 09:49 GST

Landed on `ds4-ports` (not pushed to fleet):
- `4af52cc4c` Cut 1: `-np` is batch width; bookkeeping grows
- `9e0d9f165` fitter = one master + headroom, not `n_ctx × n_parallel`
- `fc940c2e7` pool-full: preempt unref tails, waiters not 500'd
- `564245b6d` finished master prefix stays parked; later children admit SHARED
- `54996eea2` session_id / POST /fork / POST /close_session
- `5cb7a0a69` unknown session_id fails loud
- `471672ff0` named session survives prefix shorter than one block
- `9f12eb160` kv_paged refuses n_gpu_blocks==0 instead of GGML_ASSERT abort
- `d11c037f3` child named /fork reports inherited tokens as HTTP cache_n (was always 0 on paged)
- `6911a61ac` paged named /fork does not bill inherited tokens as prompt_n
- `257458bdd` named/held master prefix is not an eviction victim
- `a7359a75f` named master filling GPU: children WAIT (not CPU swap, not 500) until /close_session
- `2474fd5fb` overlap crash: defer second hybrid child past n_seq_max
- `06fba44e2` --kv-paged default-on for Qwen35/QWEN35MOE only

Still missing for MET:
- DSV4 Flash 0731 8k: named /fork HTTP proven on new binary HEAD a7359a75f (child cache_n=64 prompt_n=22 tokens_evaluated=86; unknown 400; n_gpu_blocks=192 no overcommit). Parallel /fork PASS on ab2507c5e (both inherit 128/139). Decode still single-width. Not 256k.
- Qwen3.8 27B 8k: named /fork HTTP proven on HEAD a7359a75f (child cache_n=128 prompt_n=12 tokens_evaluated=140; child2 128/16/144; unknown 400; DS4P_PAGED_HYBRID=1, n_gpu_blocks=768, all attention layers paged path). Hybrid still builds attn_kv_size=n_ctx_seq first (not slab-gone). Decode gate pending. Default-on for Qwen35/QWEN35MOE as of 06fba44e2 (8k /fork env unset: cache_n=128 prompt_n=12 tokens_evaluated=140). Other hybrids still opt-in (silent-static risk). Not 256k.
- Qwen3.8 27B `-c 32768`: named /fork HTTP started on HEAD a7359a75f (child cache_n=32 prompt_n=6 tokens_evaluated=38; unknown 400; n_gpu_blocks=3072 no overcommit; RSS ~22 GiB). HONEST: 32-token prompt is not a 32k-context proof. Not 256k.
- Qwen3.8 27B two children named /fork from one 16k master: both cache_n=16384; overlap crash fixed 2474fd5fb (defer second hybrid child past n_seq_max); master still forkable after; unknown 400. JSON in llama.cpp-ds4ports/.scratch-two-child-16k/. HONEST: serial/defer at -np 1, not a concurrent usable serve. Not 256k.
- Hybrid RS widen STOPPED. Not DSV4 bookkeeping-only. Qwen RS cell is 150 MiB; 256 cells ~37 GiB. Sharing one cell breaks SSM rewind. HEAD 06fba44e2. 2474fd5fb defer remains the Qwen two-child path at -np 1. Not 256k.
- Qwen3.8 27B `-c 32768` 28k-prefix named /fork: master 28672 tokens / 306.4s; child cache_n=28672 prompt_n=9 tokens_evaluated=28681 in 1.66s; 1792 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-28k-named-fork/. HONEST: 28k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` 24k-prefix named /fork: master 24576 tokens / 241.6s; child cache_n=24576 prompt_n=9 tokens_evaluated=24585 in 1.25s; 1536 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-24k-named-fork/. HONEST: 24k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` 16k-prefix named /fork: master 16384 tokens / 135.2s; child cache_n=16384 prompt_n=9 tokens_evaluated=16393 in 1.04s; 1024 blocks by reference; unknown 400. JSON in llama.cpp-ds4ports/.scratch-16k-named-fork/. HONEST: 16k is not a 32k fill, not 256k.
- Qwen3.8 27B `-c 32768` long-prefix named /fork: master 9241 tokens / 63.8s; child cache_n=9232 prompt_n=21 tokens_evaluated=9253 in 0.92s; 577 blocks by reference; unknown 400. JSON in ornith-1m/_scratch/qwen38-fork-32k-long-18091/. HONEST: 9k is not a 32k fill, not 256k.
- concurrent agents on a daily model as a usable serve (not a one-shot e2e)

## Dated notes

### 2026-08-16 — hybrid RS widen STOPPED

- Hybrid RS widen STOPPED.
- Not DSV4 bookkeeping-only.
- Qwen RS cell is 150 MiB; 256 cells ~37 GiB.
- Sharing one cell breaks SSM rewind.
- HEAD `06fba44e2`.
- `2474fd5fb` defer remains the Qwen two-child path at `-np 1`.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 06fba44e2: --kv-paged default-on for Qwen35/QWEN35MOE only

- Landed `06fba44e2` `--kv-paged` default-on for Qwen35/QWEN35MOE only.
- 8k /fork env unset: `cache_n=128` `prompt_n=12` `tokens_evaluated=140`.
- Other hybrids still opt-in (silent-static risk).
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — two children named /fork from one 16k Qwen3.8 master

- Two children named /fork from one 16k Qwen3.8 master.
- Both `cache_n=16384`.
- Overlap crash fixed `2474fd5fb` (defer second hybrid child past `n_seq_max`).
- Master still forkable after.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-two-child-16k/`.
- HONEST: serial/defer at `-np 1`, not a concurrent usable serve. Not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 28k-prefix named /fork

- Qwen3.8 27B `-c 32768` 28k-prefix named /fork.
- Master 28672 tokens / 306.4s.
- Child `cache_n=28672` `prompt_n=9` `tokens_evaluated=28681` in 1.66s.
- 1792 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-28k-named-fork/`.
- HONEST: 28k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 24k-prefix named /fork

- Qwen3.8 27B `-c 32768` 24k-prefix named /fork.
- Master 24576 tokens / 241.6s.
- Child `cache_n=24576` `prompt_n=9` `tokens_evaluated=24585` in 1.25s.
- 1536 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-24k-named-fork/`.
- HONEST: 24k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 16k-prefix named /fork

- Qwen3.8 27B `-c 32768` 16k-prefix named /fork.
- Master 16384 tokens / 135.2s.
- Child `cache_n=16384` `prompt_n=9` `tokens_evaluated=16393` in 1.04s.
- 1024 blocks by reference.
- Unknown 400.
- JSON in `llama.cpp-ds4ports/.scratch-16k-named-fork/`.
- HONEST: 16k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 long-prefix named /fork

- Qwen3.8 27B `-c 32768` long-prefix named /fork.
- Master 9241 tokens / 63.8s.
- Child `cache_n=9232` `prompt_n=21` `tokens_evaluated=9253` in 0.92s.
- 577 blocks by reference.
- Unknown 400.
- JSON in `ornith-1m/_scratch/qwen38-fork-32k-long-18091/`.
- HONEST: 9k is not a 32k fill, not 256k.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B -c 32768 named /fork HTTP started

- HEAD `a7359a75f`.
- Qwen3.8 27B `-c 32768` named /fork HTTP started.
- `n_gpu_blocks=3072`, no overcommit.
- Child `cache_n=32` `prompt_n=6` `tokens_evaluated=38`.
- Unknown 400.
- RSS ~22 GiB.
- JSON in `/tmp/qwen38-fork-32k-18091/`.
- HONEST: 32-token prompt is not a 32k-context proof.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — Qwen3.8 27B 8k named /fork HTTP proven

- HEAD `a7359a75f`.
- Qwen3.8 27B 8k named /fork HTTP proven.
- Child `cache_n=128` `prompt_n=12` `tokens_evaluated=140`.
- Child2 `cache_n=128` `prompt_n=16` `tokens_evaluated=144`.
- Unknown 400.
- `DS4P_PAGED_HYBRID=1`, `n_gpu_blocks=768`, all attention layers paged path.
- JSON in `llama.cpp-ds4ports/.fork-proof-scratch/`.
- Does NOT stamp MET. 8k is not 256k. Bar still NOT MET.

### 2026-08-16 — DSV4 Flash 0731 8k named /fork HTTP proven

- HEAD `a7359a75f`, new binary.
- DSV4 Flash 0731 8k named /fork HTTP proven.
- Child `cache_n=64` `prompt_n=22` `tokens_evaluated=86`.
- Unknown 400.
- `n_gpu_blocks=192` no overcommit.
- Does NOT stamp MET. 8k is not 256k. Bar still NOT MET.

### 2026-08-16 — NEW pool-full HTTP proven on stories15M

- HEAD `a7359a75f`, no new code.
- Children waited; named hold not shortened (`after_fork` `cache_n=112`).
- `/close_session` then children 200.
- evicted unique-suffix from named hold: 0.
- Old 8+4 200s (master tail evicted) are dead.
- Does NOT stamp MET. 256-token ctx is not 256k. Bar still NOT MET.

### 2026-08-16 — cut a7359a75f: named master filling GPU, children WAIT not CPU swap

- Landed `a7359a75f` on llama.cpp-ds4ports `ds4-ports`.
- Named master filling GPU: children WAIT (not CPU swap, not 500) until `/close_session`.
- `allocate()` is GPU-only.
- Shared-prefix child cannot whole-table swap.
- `test_named_master_full_gpu_children_wait_not_cpu_swap` passed.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 257458bdd: named/held master prefix is not an eviction victim

- Landed `257458bdd` on llama.cpp-ds4ports `ds4-ports`.
- Named/held master prefix is not an eviction victim.
- `evict_held_prefix` skips session holds; child queues if that is the only reclaim.
- `test_named_master_not_eviction_victim`: hold stayed 4 blocks, child inherited 64 not 32.
- `test-paged-kv` ALL PASSED.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — HTTP pool-full proven on stories15M

- HEAD `6911a61ac`, no new code.
- `--kv-paged -np 1 -c 256`, 8 gpu + 4 cpu blocks.
- Session stayed; shared prefix stayed; master's unique suffix was evicted (112 → 64).
- 3 children `/fork` waited then 200, `cache_n=64`.
- after_fork inherited 64 of 112 (same session 200). Queue+preempt-tail, not full master survived.
- Logs: waiting=3, evict unique-suffix from held prefix, no 500.
- Does NOT stamp MET. 256-token ctx is not 256k. Bar still NOT MET.

### 2026-08-16 — 256 bookkeeping is seq_id / bitset width, not a leftover -np slot wall

- `DS4P_PAGED_MAX_BOOKKEEPING = LLAMA_MAX_SEQ` is seq_id / bitset width, not a leftover `-np` slot wall.
- Sequential named sessions recycle idle slots.
- 257th concurrent in-flight defers (queue), not refuse/crash/drop.
- Did not lift the cap. Did not add a loud refuse (that would create a wall).
- HEAD still `6911a61ac`. Bar still NOT MET.

### 2026-08-16 — cut 6911a61ac: paged named /fork prompt_n

- Landed `6911a61ac` on llama.cpp-ds4ports `ds4-ports`.
- Paged named /fork does not bill inherited tokens as `prompt_n`.
- stories15M child JSON: `cache_n=32`, `prompt_n=7`, `tokens_evaluated=39`. Parent cold `cache_n=0`.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut d11c037f3: named /fork HTTP cache_n

- Landed `d11c037f3` on llama.cpp-ds4ports `ds4-ports`.
- Child named /fork now reports inherited tokens as HTTP `cache_n` (was always 0 on paged).
- stories15M child `cache_n=32`.
- `test-paged-kv` ALL PASSED including `test_named_fork_n_past_is_http_cache_n`.
- Remaining: paged `prompt_n` still counts inherited tokens (`prompt_n + cache_n != n_prompt`). APC/warm `cache_n` still unset.
- Does NOT stamp MET. Bar still NOT MET.

### 2026-08-16 — cut 9f12eb160: n_gpu_blocks==0 refuse (not abort)

- Landed `9f12eb160` on llama.cpp-ds4ports `ds4-ports`.
- `kv_paged` refuses `n_gpu_blocks==0` instead of GGML_ASSERT abort. `llama_context` ctor throws; server exits 1.
- `test-paged-kv` ALL PASSED including `test_init_zero_gpu_blocks_throws`.
- Does NOT stamp MET. 32k DSV4 still FAIL. Bar still NOT MET.

### 2026-08-16 — 32k DSV4 named /fork FAIL

- HEAD `95e35281c` comments/tests only. Binary `ab2507c5e`.
- `-c 32768`: `common_fit_paged_kv_blocks` requested n_ctx=32768 needs 768 KV blocks (4.0 GiB), budget 231 (1.2 GiB). Largest n_ctx that fits ~9856. Then GGML_ASSERT `n_gpu_blocks==0` in `llama-kv-cache-paged.cpp`.
- 16k vanilla same refuse (384 vs 231).
- 16k with `--n-gpu-blocks 384`: named /fork SHARED worked (master 128 inherit, two serial children, unknown 400). RSS 92.55 GiB. Useful proof, NOT a 32k pass.
- Implication: 256k–1M DSV4 is box work. Mac ceiling after ~90G weights is ~10k KV unless overcommit.

## How we work

Allowed now (Mac in use):
- Read, plan, code, unit tests, tiny-model e2e (stories15M-class).
- Cheap "3 overlapping HTTP, `-np 1`, queue not reject" gates.

Parked until Satinder says the Mac is free:
- NIAH
- 256k / 512k / 1M ABBA ladders
- Long Metal soaks, decode-curve rungs, hours-long GPU jobs

Parked until the 2×96GB box is this seat's:
- CUDA parity, hd512 champion, hybrid CUDA witness
- 256k–1M DSV4 (Mac ceiling after ~90G weights is ~10k KV unless overcommit)

Never:
- 86-arch / 95-arch / 141-arch sweeps
- Raising `-np` to fake concurrency
- Marking P2-8 closed on a toy gate
- Calling 1k–4k "done"

## Next slice

1. ~~Land Cut 1.~~ `4af52cc4c`
2. ~~Fitter = one master, not `n_ctx × n_parallel`.~~ `9e0d9f165`
3. ~~Queue / swap / resume.~~ `fc940c2e7`
4. ~~Prefix identity that outlives the parent.~~ `564245b6d`
5. ~~Session/child API.~~ `54996eea2` (engine + routes; HTTP e2e parked)

## Daily models (these outrank the 19)

- DeepSeek V4 Flash 0731 (Mac first)
- Qwen3.8 27B (8k named /fork HTTP proven on Mac hybrid; box when free for 256k)
- GLM 5.2 (box, later)

Aug 7 19-list stays parked-to-last except where it overlaps.

## Short list (park everything else)

Aug 7: deepseek v4, dflash, eagle3, ernie4-5, ernie4-5-moe, gemma4-assistant, grok, hunyuan-moe, hunyuan-vl, laguna, mimo2, minimax-m3, nemotron, qwen3moe, qwen3next, qwen3vl, qwen3vlmoe, starcoder, step35.
