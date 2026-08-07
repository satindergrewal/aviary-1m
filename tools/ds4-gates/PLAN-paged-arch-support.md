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

All four re-run through `arch_serve_gate.sh` **with startup reserve passes excluded** — these are the
sliced numbers, and the earlier unsliced ones (414/828/736/1104) should not be diffed against them:

| arch | vehicle | funnel | warns static→paged | `DS4P-CONSUME` | startup excluded | output |
|---|---|---|---|---|---|---|
| `ernie4_5` | ERNIE-4.5-0.3B-PT Q4_K_M | banded | 36 → **0** | **180** | 0 | identical |
| `qwen3vl` | Qwen3-VL-4B-Instruct Q4_K_M | banded | — → **0** | **576** | 0 | identical |
| `nemotron` | Nemotron-Mini-4B-Instruct Q4_K_M | banded | — → **0** | **512** | 0 | identical |
| `qwen3moe` | Qwen3-30B-A3B Q4_K_M | banded | 96 → **0** | **768** | 0 | identical |

⚠ **The `startup excluded` column is the load-bearing one.** It reads 0 for all four, which turns
"these four cannot have been contaminated by startup reserve passes" from an argument into a
measurement. `gemma4` reads **630** in that column — 21 whole graph builds before the first request —
and that is what produced a false PARTIAL until the counters were sliced.

⚠ **The arch string is `ernie4_5` with an underscore.** Earlier tables here wrote `ernie4-5`. Cosmetic
in prose, not cosmetic in a gate that does a string compare.

⚠ **These counts are NOT comparable across runs.** An earlier inline run of `ernie4_5` produced
`234 → 0` and this one produces `414 → 0`; nothing changed in the code. The absolute count is
warnings-per-layer × number of graph builds, and the graph-build count moves with verbosity, prompt
length and `n_predict`. The only meaningful form is the **transition inside one gate invocation**:
non-zero static, exactly zero paged, same binary, same model, minutes apart. A cross-run diff of these
numbers measures the harness, not the code.

⚠ **Read the arch from `print_info`, never from the repo name.** A StarCoder2-3B download loaded as
`arch = starcoder2` → `starcoder2.cpp`, which is *not* the file that was wired (`starcoder.cpp`). That
run looked like a pass — correct output — and verified nothing.

⚠⚠ **The tell I originally recorded for that retraction was WRONG, and it was wrong in a way that
would have broken eleven architectures.** I wrote that `static-path-warns = 0` proved `starcoder2.cpp`
has no paged consumer. It has one — via `build_attn_inp_kv_auto()`, which reaches paged attention
through a *different funnel* that never logs a static-path warning in either arm. Measured:
`starcoder2` produces **1140 `DS4P-CONSUME` events** and text identical to static. The retraction's
conclusion (wrong vehicle for arch `starcoder`) stands on arch identity alone; its stated *mechanism*
did not generalise, and a right verdict from a wrong mechanism survives until the mechanism is asked
to carry something else — which is exactly what happened when it was built into a gate.

**Four independent families, two `wo` conventions, dense and MoE, one recipe.** `ernie4-5` is Baidu dense
(`wo, NULL, wo_s`), `qwen3vl` is Qwen vision-language (`wo, wo_b, wo_s`), `nemotron` is NVIDIA dense
GQA (`wo, wo_b, wo_s`), `qwen3moe` is a 30B mixture-of-experts. Nothing structural is shared between
them — and the MoE case matters because its FFN half is a different graph shape, which is the obvious
place for an assumption about graph structure to surface. It did not. The recipe is arch-agnostic by
evidence rather than by assertion.

**Quant choice is part of the method.** `qwen3moe` was verified at Q4_K_M (17.35 GB), not the smaller
Q2_K (11.4 GB), deliberately: a 2-bit quant of a 30B MoE can produce degenerate output for reasons
that have nothing to do with paging, and then a paged-vs-static disagreement cannot distinguish a
wiring fault from quantisation damage. Same principle as not scoring a kernel against a baseline that
is itself looping — a broken reference cannot arbitrate.

**Not verified by any of them:** long context (the ~50k defect applies here as everywhere), speed, and
the other 19 archs.

Cost: **241 MB, ~4 minutes.**

### ⚠ A vehicle is only valid if it loads as the arch you wired

**Name match is not arch match, and it has failed twice.**

| intended arch | wrong vehicle picked by name | what it actually loads as |
|---|---|---|
| `starcoder` | StarCoder2-3B | `starcoder2` — a *different file* (wired, but on the other funnel) |
| `nemotron` | Llama-3.1-Nemotron-Nano-8B | `llama` — a Llama-3.1 derivative |

### The two paged consumers — which one an arch uses changes what can be proved about it

There is not one paged consumer in this fork, there are two, and they have different observability:

| funnel | entry | op | archs | fallback warning |
|---|---|---|---|---|
| **banded** | `build_attn_paged_or_null` | `ggml_paged_attn_banded` | 20 | yes, per layer |
| **auto** | `build_attn_inp_kv_auto` | `ggml_paged_attn` | 11 | **none, ever** |

**banded (hand-wired):** `dflash` `ernie4-5` `ernie4-5-moe` `eagle3` `glm4-moe` `gemma4` `grok`
`hy-v3` `hunyuan-moe` `hunyuan-vl` `laguna` `kimi-linear` `mimo2` `nemotron` `qwen3moe` `qwen35`
`qwen3vlmoe` `starcoder` `step35` `qwen3vl`

**auto (one-line):** `command-r` `gemma` `falcon` `gemma3` `internlm2` `llama` `mistral3` `phi3`
`qwen2` `qwen3` `starcoder2`

⚠ **The gate is strictly weaker on the auto funnel.** It has no per-layer fallback and no warning, so
for those eleven the gate proves paged *ran* and the text *matches* — it cannot prove every layer was
carried. That limitation is printed on every PASS rather than left implicit.

### `DS4P-CONSUME` — presence asserted positively, at both funnels

Before this marker existed, nothing in the fork could assert that a graph had *consumed* a paged
context. `DS4P-CHECKOUT` says the block manager handed out blocks; it is emitted by the **scheduler**
and stays true when no graph reads them. **That gap is audit finding 5 verbatim** — paged context
produced correctly, nothing consuming it, "Ornith now runs paged" retracted, and no runtime signal
that would catch a recurrence.

Both funnels now emit one line per layer per graph. The static arm is the negative control on the
marker itself:

| | `ernie4_5` (banded) | `starcoder2` (auto) |
|---|---|---|
| paged arm | 540 | 900 |
| static arm | **0** | **0** |

Needs `-lv 5`: it is `LLAMA_LOG_DEBUG` and `common/log.cpp:85` drops DEBUG below verbosity 5. At
`-lv 4` the count reads **zero on an arch that is demonstrably paging** — the gate would VOID a
working arch because its own probe was filtered out of the log it greps. Caught only by checking the
marker for *presence* on a known-good model before trusting it.

The first cost a download and a retracted "verification". The second was caught **before** downloading,
by asking which arch the repo loads as rather than trusting the brand in its name.

So every row below needs the arch it actually loads as, verified, before it counts as a plan — and
the serve check reads `print_info: arch` rather than the filename.

### Test-vehicle sizing — the wiring is arch-shaped, not size-shaped

The paged consumer is per-architecture, so the **smallest published variant of each family** is the
correct serve-check vehicle. A 0.3B ERNIE exercises `ernie4-5.cpp` exactly as a 21B does.

Every row below has its arch read out of the candidate GGUF's own header by
`gguf_arch_probe.py --expect <arch>` **before** any bytes were spent. `MATCH` means the file's
`general.architecture` string was read and compared, not that the repo name looked right.

| arch | vehicle (arch probed in-header) | size | status | side |
|---|---|---|---|---|
| `ernie4_5` | ERNIE-4.5-0.3B-PT Q4_K_M | 241 MB | ✅ **SERVE-VERIFIED** | Mac |
| `qwen3vl` | Qwen3-VL-4B-Instruct Q4_K_M | 2.50 GB | ✅ **SERVE-VERIFIED** | Mac |
| `nemotron` | Nemotron-Mini-4B-Instruct Q4_K_M | 2.51 GB | ✅ **SERVE-VERIFIED** | Mac |
| `qwen3moe` | Qwen3-30B-A3B Q4_K_M | 17.35 GB | ✅ **SERVE-VERIFIED** | Mac |
| `starcoder` | StarCoderBase-1B Q8_0 · MATCH | 1.26 GB | ⬇ downloading | Mac |
| `hunyuan_vl` | HunyuanOCR Q8_0 · MATCH | 0.58 GB | ⛔ **UNANSWERED — vehicle unusable**, see below | Mac |
| `dflash` | Qwen3.5-4B-DFlash Q4_K_M · MATCH | 0.38 GB | ⛔ **UNANSWERED — draft head**, see below | Mac |
| `eagle3` | Qwen3-8B-EAGLE3-Speculator F16 · MATCH | 2.05 GB | ⛔ **UNANSWERED — draft head** (same code path; not downloaded) | Mac |
| `gemma4-assistant` | — | — | ⛔ **UNANSWERED — not wired at all**, see below | ? |
| `gemma4` *(bonus, NOT on the 19)* | Gemma4-26B-A4B-Uncensored-1M Q4_K_M (local) | 16.80 GB | ⚠ **PASS under `DS4P_PAGED_SWA=1` only** | Mac |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-Thinking **Q4_K_M** · MATCH | **13.33 GB** | ready to fetch | Mac |
| `qwen3vlmoe` | Qwen3-VL-30B-A3B-Instruct **Q4_K_M** · MATCH | **18.56 GB** | ready to fetch | Mac |
| `laguna` | Laguna-S-2.1 Q8_0, 4 shards · MATCH | 256×4.5B | searched | **box** |
| `mimo2` | MiMo-V2.5 UD-Q4_K_S, 5 shards · MATCH | 256×8.2B | searched | **box** |
| `step35` | Step-3.5-Flash Q4_K · MATCH | 118.71 GB | searched | **box** |
| `hunyuan-moe` | Hunyuan-A13B-Instruct | ~34 GiB IQ3_KS | searched | **box** |
| `qwen3next` | Qwen3-Next-80B-A3B | 80B | searched | **box** |
| `grok` | Grok-1 class | 314B | searched | **box** |
| `minimax-m3` | MiniMax-M3 | large | searched | **box** |
| `deepseek4` | DeepSeek-V4-Flash-0731 | large | searched | **box** |

⚠ **The MoE sizes above are Q4-class and that correction matters.** They were first recorded as
~7 GB and ~9 GB — the UD-TQ1_0 and IQ1_S files, sorted to the top of a size-ordered listing and read
off. Those are **one-bit** quants, a full bit worse than the Q2_K deliberately rejected for `qwen3moe`
so that a degenerate baseline could not masquerade as a paging fault. Same rule, same day, broken by
sorting a list by size. The anchor was in this file: `qwen3vlmoe` is the same 30B-A3B base as
`qwen3moe`, measured at 18.63 GB.

⚠ **`mimo2`'s first candidate STRUCK, and the strike is the point.**
`ji-farthing/MiMo-V2.5-Pro-DFlash-draft-ik-llama-GGUF` has "MiMo" in the name and
`general.architecture = dflash-draft` in the header — an arch string that **does not exist in
`llama-arch.cpp` at all**. It would not have loaded as anything, after 5.54 GB. Same shape as the
starcoder2 mistake with different nouns.

### Three archs the gate cannot answer in its current form

These are **UNANSWERED**, which is neither verified nor broken. Recording them as failures would be
wrong; recording them as pending without the reason would lose the work.

| arch | why the gate cannot answer | what would answer it |
|---|---|---|
| `dflash`, `eagle3` | **Draft heads, not models.** `llama-context.cpp:187` throws `requires ctx_other to be set` when the GGUF carries no `tok_embd`/`output`. Confirmed empirically on `dflash` (0.38 GB, exit 3). `eagle3` is the same code path, so it was **not downloaded**. | A target+draft harness (`-md`), then the separate question of whether the *draft's* KV is paged at all |
| `hunyuan_vl` | The only vehicle found is **HunyuanOCR**, an image model. Text-only prompts return `''` or `' $ $ $ $ …'`. The static reference is degenerate *by construction*, so there is nothing to arbitrate against. | mmproj + a real image, or a different `hunyuan_vl` text vehicle if one is published |
| `gemma4-assistant` | **`gemma4-assistant` is a DIFFERENT ARCH from `gemma4`** — its own string in `llama-arch.cpp:59` and its own `src/models/gemma4-assistant.cpp`, which contains **no paged wiring at all** (`grep build_attn_paged_or_null` returns nothing). Its NextN head additionally calls `build_attn` with no new K/V, and the paged op `GGML_ABORT`s on that: read-only paged attention is not implemented. | Wire the consumer, then implement read-only paged attention (the "just skip the write" version compiled and segfaulted — `k_new` is dereferenced for shapes) |

⚠⚠ **`gemma4` is NOT the row on the owner's 19-list. `gemma4-assistant` is.** The PASS recorded above
is plain `gemma4`, a **bonus arch** in the same position as `starcoder2` — useful, and not a point on
the board. The list row stays UNANSWERED on a *different* blocker (unwired + read-only) from the one
that plain `gemma4` needed (the SWA guard).

This is the arch-identity trap for the third time, and the first time it fired at the **ledger** level
rather than the download level: right verification, right model, wrong row credited. The probe
protects vehicle selection; nothing was protecting the scoreboard.

**And even for plain `gemma4` the PASS is conditional.** It needs `AG_ENV=DS4P_PAGED_SWA=1`; under a
plain `--kv-paged` it is still refused by the SWA guard. Against the owner's actual bar — *"when I run
the latest models, they don't error out unexpectedly"* — a model that refuses `--kv-paged` without an
env var still errors out for him. So it is recorded as **verified under a development flag**, never as
verified.

⚠ **Three archs carry per-layer attention geometry**, which is the same class as the `gemma4`
`headdim` defect — an architecture with more than one head shape against a pool allocated once:

| arch | header |
|---|---|
| `laguna` | `attention.head_count` = `<array n=48>` |
| `step35` | `head_count` **and** `head_count_kv` both `<array n=45>` |
| `mimo2` | `head_count_kv` = `<array n=51>` |

Prediction on record before measurement: **PARTIAL with `headdim-fails` non-zero**, not PASS. The gate
counts `headdim` separately from `nopg` and `cap` precisely so this shows as its own number.

**Also on record:** `starcoderbase-1b` is `head_count=16, head_count_kv=1` — **multi-query**. Every
arch verified so far has `n_kv_head > 1`. If `starcoder` diverges, first suspect is the KV-head stride
in block-table addressing, not the arch wiring.

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
| 3b hybrid paging | **PARTIAL** — 15 of 19 wired, **4 serve-verified** (`ernie4-5`, `qwen3vl`, `nemotron`, `qwen3moe`), 22 archs total | none is serve-verified; no local GGUFs; `gemma4-assistant` blocked on read-only; 3 specials need their own cache work |
| B4 arcs 2-3 (q8 banded) | **CLOSED** — `test-paged-vs-cpu` ALL PASSED, q8_0 at the f16 error scale, champion marker asserted | q8_0 end-to-end on a real model; anything about speed |
| paged op: attention sinks | **CLOSED on the champion**, verified | scalar-kernel sinks (ABORT-guarded); no end-to-end model run |
| paged op: non-causal mode | **CLOSED**, verified with a differs-from-causal control | no end-to-end model run |
| paged op: read-only attention | **GUARDED, not implemented** | K/V shapes are still read for geometry; `gemma4-assistant` stays unwired |
| P1-6 prefix sharing | gate passes | not re-verified since the ~50k defect was found |
| P1-5 disk KV banks | **NOT REOPENED** | the "measured wash" verdict was rejected and not replaced |
| P2-8 continuous batching | **NOT STARTED** | `evict()` is a 13-line body; unscoped |
| Metal parity | **NOT DONE** | bar is ≤1.0× at **256k–1M**; best clean measurement is 32k |
| CUDA parity | **NOT DONE** | stale; no NVIDIA GPU on this Mac |
