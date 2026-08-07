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

## Current state (2026-08-07, end of session)

**22 archs carry the paged consumer.** All COMPILED; none serve-verified; no local GGUFs.

| bucket | count | models |
|---|---|---|
| wired, building clean | 15 of his 19 | `qwen3moe` `qwen3vl` `qwen3vlmoe` `ernie4-5` `ernie4-5-moe` `eagle3` `grok` `hunyuan-moe` `hunyuan-vl` `nemotron` `starcoder` `laguna` `step35` `mimo2` `dflash` |
| blocked — read-only path | 1 | `gemma4-assistant` |
| need their own cache work | 3 | `deepseek4` (dual cache + top-k), `minimax-m3` (MSA cache), `nemotron-h` (SSM) |

Plus 7 wired before this session: `gemma4` `glm4-moe` `hy-v3` `kimi-linear` `inkling` `qwen35` `qwen35moe`.

### Op gaps found by reading call sites — all three closed or guarded

| gap | found on | state |
|---|---|---|
| attention **sinks** | `mimo2` | **closed on the champion**, numerically verified (finite arm + `-inf` control at `0.000e+00`). ABORT-guarded on the scalar kernel, which genuinely lacks them. |
| **non-causal** attention | `dflash` | **closed**, verified at three head dims, plus a control asserting it *differs* from causal by `2.52e-01`. `dflash` wired on it. |
| **read-only** attention | `gemma4-assistant` | **guarded, not implemented.** Skipping the write compiled and segfaulted — `k_new`/`v_new` are dereferenced for shapes elsewhere. Arm retained behind `DS4P_TEST_READONLY`. |

None of the three was on any list before the sweep. Each would have compiled and served: a dropped
sink, an imposed causal mask, a crash on the first read-only call.

### What the sinks gap actually was

Not missing math. `build_attn_kv_paged` had `ggml_tensor * /*sinks*/` — the parameter **accepted and
commented out** — while the Metal kernels carried live sink handling all along. The capability was
present at both ends and disconnected in the middle.

### Harness rules this sweep paid for

Three **silent argument shifts** in one session, all from inserting a defaulted parameter in the
middle of a signature:

| insertion | detector | cost |
|---|---|---|
| `sink_mode` mid-signature | none — types matched, compiler silent | an hour of bisecting the *library* |
| `causal` before non-defaulted params | compiler | a rebuild |
| `read_only_check` mid-signature | the no-op control | one run |

**Defaulted parameters go last.** And every feature flag gets an arm asserting it *changes* something
— that check caught two separate no-ops today.

## Measured status of the 19 — the "16 standard wirings" was never 16

Wiring 13 of them turned up **two capability gaps in the paged op itself**. Both would have compiled
and served; neither is detectable without reading the call site.

| bucket | count | models |
|---|---|---|
| wired, building clean | 13 | `qwen3moe` `qwen3vl` `qwen3vlmoe` `ernie4-5` `ernie4-5-moe` `eagle3` `grok` `hunyuan-moe` `hunyuan-vl` `nemotron` `starcoder` `laguna` `step35` |
| **blocked on the op** | 2 | `mimo2`, `dflash` |
| needs reading first | 1 | `gemma4-assistant` |
| special | 3 | `deepseek4`, `minimax-m3`, `nemotron-h` |

### The two op-level blockers

**`mimo2` — attention sinks.** Its call passes `sinks`:
```
build_attn(inp_attn, wo, NULL, wo_s, Qcur, Kcur, Vcur, nullptr, sinks, ...)
```
`build_attn_paged_or_null` has no sinks parameter. Wiring it drops the sink from every layer —
correct-looking output, quietly wrong attention, no error.

**`dflash` — non-causal attention.** `dflash.cpp:451` says so in as many words: *"cache-aware,
non-causal attention"*. The paged mask is causal-only, hardcoded `vis = (col <= q_pos)` in the Metal
source. Wiring it masks out the future half of every layer's context, silently.

**These are the real content of phase 3b**, and neither was on any list before the sweep:
1. sinks support in `build_attn_paged_or_null` and the mask kernel
2. a non-causal mode for the paged mask

### Five `wo` conventions, all silent if mishandled

| convention | archs |
|---|---|
| `wo, wo_b, wo_s` | most of the scripted batch |
| `wo, NULL, wo_s` | `ernie4-5` |
| `wo, NULL, nullptr` | `eagle3` |
| `NULL, NULL, NULL` — projection deferred past a gate | `laguna`, `step35` |
| `+ sinks` | `mimo2` (unsupported) |

The paged branch must apply `wo` in the first three and must **not** in the fourth. Getting it
backwards projects twice or not at all, and neither raises an error. This is why each file is anchored
exactly and the wiring script refuses anything it does not match rather than generalising.

## Mac vs box — what can be closed where

Locally present under `ornith-models/`:
`DeepSeek-V4-Flash` · `Gemma4-12B/26B-A4B/31B` · `Gemma4-HauhauCS` · `Hy3-1M` · `MLX-Qwen3.6-35B` ·
`Ornith-9B/35B/397B` · `Qwable-27B` · `Qwen3.6-35B-HauhauCS` · `QwenPaw-Flash-9B`

**Nothing in the 19-model list is currently on this Mac in a runnable GGUF except via the DSV4-Flash
directory**, which is the special case above. So every one of the 16 standard wirings needs a quant
downloaded before its serve check can run.

### ✅ VERIFIED archs (serve check passed)

| arch | vehicle | static | paged | paged markers | static-path fallbacks |
|---|---|---|---|---|---|
| `ernie4-5` | ERNIE-4.5-0.3B-PT Q4_K_M | ` Paris.` | ` Paris.` | **3** | **0** |
| `qwen3vl` | Qwen3-VL-4B-Instruct Q4_K_M | ` Tokyo, and the capital city of China` | identical | **3** | **0** |

⚠ **Read the arch from `print_info`, never from the repo name.** A StarCoder2-3B download loaded as
`arch = starcoder2` → `starcoder2.cpp`, which is *not* the file that was wired (`starcoder.cpp`). That
run looked like a pass — correct output — and verified nothing. The tell was `static-path-warns = 0`
in the **static** arm, meaning the wired file was never in the graph at all. A fallback counter that
can only read 0 proves nothing; `ernie4-5` read 234 and `qwen3vl` 468 before reading 0 under paging.

Both parts of the closure criterion met: correct text **and** the paged path confirmed active, with
zero layers falling back. The static arm's 234 fallback warnings are the negative control — the
counter can be non-zero, so a 0 means something.

**Not verified by this:** long context (the ~50k defect applies here as everywhere), speed, and the
other 21 archs.

Cost: **241 MB, ~4 minutes.**

### Test-vehicle sizing — the wiring is arch-shaped, not size-shaped

The paged consumer is per-architecture, so the **smallest published variant of each family** is the
correct serve-check vehicle. A 0.3B ERNIE exercises `ernie4-5.cpp` exactly as a 21B does.

| arch | smallest published GGUF vehicle | approx size | verified | side |
|---|---|---|---|---|
| `ernie4-5` | ERNIE-4.5-0.3B-PT Q4_K_M | **241 MB** | ✅ **downloaded + SERVE-VERIFIED** | Mac |
| `qwen3vl` / `qwen3vlmoe` | Qwen3-VL-4B-Instruct Q4_K_M | **~2.5 GB** (Q3_K_M ~2.08 GB) | ✅ searched | Mac |
| `hunyuan-moe` | Hunyuan-A13B-Instruct — 80B total / 13B active | **~34 GiB** at IQ3_KS | ✅ searched | **box** |
| `qwen3moe` | Qwen3-30B-A3B Q2_K | **~11.4 GB** (Q4_K_M 18.6 GB) | ✅ searched | Mac |
| `qwen3next` | Qwen3-Next-80B-A3B | sizes not published in results | ⚠ partial | **box** (80B) |
| `nemotron` | Llama-3.1-Nemotron-Nano-8B Q4_K_M | **~4.92 GB** | ✅ searched | Mac |
| `starcoder` | StarCoder2-3B Q4_K_M | **~1.85 GB** | ✅ searched | Mac |
| `grok` | Grok-1 class | not yet searched | ❌ | **box** |
| `eagle3` | draft-head arch, pairs with a target | not yet searched | ❌ | ? |
| `laguna` `mimo2` `step35` `dflash` `hunyuan-vl` `ernie4-5-moe` | — | not yet searched | ❌ | ? |

**Six verified, one partial, eight not.** The three that are verified came from one search each and took a
couple of minutes; the rest is the same work, not harder work. Recorded as ❌ rather than filled with
plausible numbers — an invented size is worse than a blank, because it silently becomes a schedule.

**Mac/box split, from what is verified so far:**

| side | vehicles | total |
|---|---|---|
| **Mac** | ERNIE-4.5-0.3B (241 MB) · StarCoder2-3B (1.85 GB) · Qwen3-VL-4B (2.5 GB) · Nemotron-Nano-8B (4.92 GB) · Qwen3-30B-A3B Q2_K (11.4 GB) | **~21 GB, five archs** |
| **box** | Hunyuan-A13B (~34 GiB) · Qwen3-Next-80B | — |

⚠ **`nemotron` needs a note.** llama.cpp issue #20570 reports Nemotron-3-Nano-30B-A3B failing at
`mamba-base.cpp:173: GGML_ASSERT` — that is the SSM family, i.e. the `nemotron-h` shape already
flagged as a special here. The Llama-3.1-Nemotron-Nano-8B vehicle above exercises `nemotron.cpp`, not
the hybrid one, which is the right target for this wiring and the wrong one for `nemotron-h`.

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
| 3b hybrid paging | **PARTIAL** — 15 of 19 wired, **2 serve-verified** (`ernie4-5`, `qwen3vl`), 22 archs total | none is serve-verified; no local GGUFs; `gemma4-assistant` blocked on read-only; 3 specials need their own cache work |
| B4 arcs 2-3 (q8 banded) | **CLOSED** — `test-paged-vs-cpu` ALL PASSED, q8_0 at the f16 error scale, champion marker asserted | q8_0 end-to-end on a real model; anything about speed |
| paged op: attention sinks | **CLOSED on the champion**, verified | scalar-kernel sinks (ABORT-guarded); no end-to-end model run |
| paged op: non-causal mode | **CLOSED**, verified with a differs-from-causal control | no end-to-end model run |
| paged op: read-only attention | **GUARDED, not implemented** | K/V shapes are still read for geometry; `gemma4-assistant` stays unwired |
| P1-6 prefix sharing | gate passes | not re-verified since the ~50k defect was found |
| P1-5 disk KV banks | **NOT REOPENED** | the "measured wash" verdict was rejected and not replaced |
| P2-8 continuous batching | **NOT STARTED** | `evict()` is a 13-line body; unscoped |
| Metal parity | **NOT DONE** | bar is ≤1.0× at **256k–1M**; best clean measurement is 32k |
| CUDA parity | **NOT DONE** | stale; no NVIDIA GPU on this Mac |
