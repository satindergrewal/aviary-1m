# Paged KV: arch support plan (3b hybrid paging)

Scope set by the owner 2026-08-07: **19 named models**, everything else parked to last.

**Standing rule for this file:** anything marked DONE or PARTIAL must state, in its own row, *what is
not verifiably closed*. A row without that is not allowed to claim either.

---

## Verified against the source tree — not assumed

`paged_wired` = the file references `get_attn_paged`. `attn sites` = count of
`build_attn_inp_kv*` / `build_attn_inp_k_iswa` constructions.

| model (his name) | arch file | exists | paged wired | attn sites | wiring class |
|---|---|---|---|---|---|
| deepseek v4 | `deepseek4.cpp` | yes | no | 1 | ⚠ **special** — dual cache + top-k selection (see below) |
| dflash | `dflash.cpp` | yes | no | 3 | standard ×3 |
| eagle3 | `eagle3.cpp` | yes | no | 1 | standard |
| ernie4-5 | `ernie4-5.cpp` | yes | no | 1 | standard |
| ernie4-5-moe | `ernie4-5-moe.cpp` | yes | no | 1 | standard |
| gemma4-assistant | `gemma4-assistant.cpp` | yes | no | 1 | standard |
| grok | `grok.cpp` | yes | no | 1 | standard |
| hunyuan-moe | `hunyuan-moe.cpp` | yes | no | 1 | standard |
| hunyuan-vl | `hunyuan-vl.cpp` | yes | no | 1 | standard |
| laguna | `laguna.cpp` | yes | no | 2 | standard ×2 |
| mimo2 | `mimo2.cpp` | yes | no | 2 | standard ×2 |
| minimax-m3 | `minimax-m3.cpp` | yes | no | **0** | ⚠ **investigate** |
| nemotron | `nemotron.cpp` | yes | no | 1 | standard |
| qwen3moe | `qwen3moe.cpp` | yes | no | 1 | standard |
| qwen3next | `qwen3next.cpp` | yes | no | 1 | standard |
| qwen3vl | `qwen3vl.cpp` | yes | no | 1 | standard |
| qwen3vlmoe | `qwen3vlmoe.cpp` | yes | no | 1 | standard |
| starcoder | `starcoder.cpp` | yes | no | 1 | standard |
| step35 | `step35.cpp` | yes | no | 2 | standard ×2 |

### Discrepancies to resolve with the owner

- **`nemotron3` does not exist.** Closest files are `nemotron.cpp` and `nemotron-h.cpp`.
  `nemotron-h.cpp` has **0** attention sites (hybrid/Mamba shape). Which does he mean?
- **`minimax-m3.cpp` has 0 attention sites** — it does not construct attention the way the paged
  recipe hooks into. Not a 2-line job. `minimax-m2.cpp` has 1 and is the standard shape.
- **`starcoder2.cpp` has 0 sites**; `starcoder.cpp` has 1. He said starcoder, which is the workable one.

### Why "0 attention sites" is a stop sign, not a detail

`deepseek4.cpp` looked like a normal `build_attn_mha` call until it was read properly: it runs a
dedicated DSV4 raw cache **plus** a CSA cache, concatenates them, and builds a top-k mask per layer.
That is a different pool geometry, not a missing consumer. Counting attention sites before estimating
is the cheap version of that check — it just separated 16 straightforward wirings from 3 that need
investigation first.

---

## Mac vs box — what can be closed where

Locally present under `ornith-models/`:
`DeepSeek-V4-Flash` · `Gemma4-12B/26B-A4B/31B` · `Gemma4-HauhauCS` · `Hy3-1M` · `MLX-Qwen3.6-35B` ·
`Ornith-9B/35B/397B` · `Qwable-27B` · `Qwen3.6-35B-HauhauCS` · `QwenPaw-Flash-9B`

**Nothing in the 19-model list is currently on this Mac in a runnable GGUF except via the DSV4-Flash
directory**, which is the special case above. So every one of the 16 standard wirings needs a quant
downloaded before its serve check can run.

**TO MEASURE, not to estimate** — for each of the 16, before any wiring:
1. smallest published GGUF that exercises the arch (the wiring is arch-shaped, not size-shaped, so the
   *smallest* variant is the right test vehicle)
2. its on-disk size
3. whether it fits this Mac's working set alongside a paged pool

That list is the input to the box schedule and it does not exist yet. **It is the next measurement,
and it is cheap** — a search per model, no GPU time.

### Closure criteria per model

A model counts as wired only when **both** hold:
- it serves correct text with `--kv-paged`, and
- the paged path is confirmed **active** (marker present) — a pool that is built and never read
  produces perfect text and proves nothing. That exact trap cost an hour on Qwen3.6.

---

## Blocking dependency

All of this rides on the paged path, which **returns wrong answers near 50k tokens, intermittently and
invisibly**, on the default config with the champion off. See `FINDINGS-paged-cross-request.md`.

Consequence, stated once: models can be **wired and verified serving** while that is open. **No model
can be marked fully done**, because "serves correctly" cannot be asserted on a substrate that is
silently wrong at long context.

---

## Status

| phase | status | what is NOT verifiably closed |
|---|---|---|
| 3b hybrid paging | **NOT STARTED** for the 19-model scope | everything; 7 unrelated archs are wired from earlier work |
| B4 arcs 2-3 (q8 banded) | **IN PROGRESS** | q8 KV is refused at 3 sites on the champion path |
| P1-6 prefix sharing | gate passes | not re-verified since the ~50k defect was found |
| P1-5 disk KV banks | **NOT REOPENED** | the "measured wash" verdict was rejected and not replaced |
| P2-8 continuous batching | **NOT STARTED** | `evict()` is a 13-line body; unscoped |
| Metal parity | **NOT DONE** | bar is ≤1.0× at **256k–1M**; best clean measurement is 32k |
| CUDA parity | **NOT DONE** | stale; no NVIDIA GPU on this Mac |
