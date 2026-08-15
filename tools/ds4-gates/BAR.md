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

**NOT MET.** 2026-08-16 00:19 GST.

Landed on `ds4-ports` (not pushed to fleet):
- `4af52cc4c` Cut 1: `-np` is batch width; bookkeeping grows
- `9e0d9f165` fitter = one master + headroom, not `n_ctx × n_parallel`
- `fc940c2e7` pool-full: preempt unref tails, waiters not 500'd
- `564245b6d` finished master prefix stays parked; later children admit SHARED
- `54996eea2` session_id / POST /fork / POST /close_session
- `5cb7a0a69` unknown session_id fails loud
- `471672ff0` named session survives prefix shorter than one block

Still missing for MET:
- DSV4 Flash 0731 8k: parallel /fork PASS on ab2507c5e (both inherit 128/139). Decode still single-width. Not 256k.
- Qwen3.8 27B: DS4P_PAGED_HYBRID=1 bring-up only. Pool 768 blocks + /fork inherit 128/140. Hybrid still builds attn_kv_size=n_ctx_seq first (not slab-gone). Decode gate pending. Not default-on.
- concurrent agents on a daily model as a usable serve (not a one-shot e2e)

## Dated notes

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
