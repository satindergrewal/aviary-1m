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

**NOT MET.** 2026-08-16 01:06 GST.

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

Still missing for MET:
- DSV4 Flash 0731 8k: parallel /fork PASS on ab2507c5e (both inherit 128/139). Decode still single-width. Not 256k.
- Qwen3.8 27B: DS4P_PAGED_HYBRID=1 bring-up only. Pool 768 blocks + /fork inherit 128/140. Hybrid still builds attn_kv_size=n_ctx_seq first (not slab-gone). Decode gate pending. Not default-on.
- concurrent agents on a daily model as a usable serve (not a one-shot e2e)

## Dated notes

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
- Qwen3.8 27B (box when free; not proven on ds4-ports yet)
- GLM 5.2 (box, later)

Aug 7 19-list stays parked-to-last except where it overlaps.

## Short list (park everything else)

Aug 7: deepseek v4, dflash, eagle3, ernie4-5, ernie4-5-moe, gemma4-assistant, grok, hunyuan-moe, hunyuan-vl, laguna, mimo2, minimax-m3, nemotron, qwen3moe, qwen3next, qwen3vl, qwen3vlmoe, starcoder, step35.
