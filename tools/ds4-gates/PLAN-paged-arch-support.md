# Paged KV: arch support plan (3b hybrid paging)

Scope set by the owner 2026-08-07: **19 named models**, everything else parked to last.

**Standing rule for this file:** anything marked DONE or PARTIAL must state, in its own row, *what is
not verifiably closed*. A row without that is not allowed to claim either.

---

## Verified against the source tree — not assumed

> ## ⚠⚠ THE `paged wired` COLUMN BELOW IS A 2026-08-07 SNAPSHOT AND IS NOW FALSE FOR MOST ROWS.
>
> It says **no** for `grok`, `nemotron`, `ernie4-5`, `qwen3moe`, `starcoder`, `laguna`, `step35`,
> `dflash`, `gemma4-assistant` and more — and the *"Current state"* section a few lines down says
> **22 archs carry the paged consumer**, listing those same files. **One document, two answers.**
>
> The live query, which takes a second and cannot go stale:
>
> ```
> grep -rln 'build_attn_paged_or_null\|get_attn_paged' src/models/
> ```
>
> ⚠ **THIS COST SOMETHING REAL ON 2026-08-09.** `SCOPE-iswa-paging-gap.md` was written against a
> stale reading of which archs were wired and concluded "21 architectures excluded by construction".
> Six of them were already wired. Acting on that conclusion, commit `cbb4c8d93` added a blanket
> sliding-window rejection that **silently switched gemma4's SWA layers back to the static path** —
> a working feature disabled by a guard built on an out-of-date table, invisible because the
> fallback output is correct.
>
> ⇒ Two lists of the same fact drift the instant one moves. Kept below as the historical record of
> what was true on 2026-08-07; **read it as history, never as state.**

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
| `qwen3vl` ⚠ | Qwen3-VL-4B-Instruct Q4_K_M | banded | — → **0** | **576** | 0 | identical — **TEXT ONLY** |
| `nemotron` | Nemotron-Mini-4B-Instruct Q4_K_M | banded | — → **0** | **512** | 0 | identical |
| `qwen3moe` | Qwen3-30B-A3B Q4_K_M | banded | 96 → **0** | **768** | 0 | identical |

⚠⚠ **`qwen3vl` and `qwen3vlmoe` are verified TEXT-ONLY, and that matters.** Both are
vision-language models and both were gated with a text prompt. `FINDINGS-paged-drops-images.md`
documents a defect in the **server request path**, not in any architecture: under `--kv-paged`,
`get_text_tokens()` strips every media placeholder, so an image prompt reaches the model with the
picture removed. That mechanism is arch-independent, so **their image path under paging was broken by
it too** — measured on `hunyuan_vl`, where it turned `PARIS` into `BUTTERFISH`.

Since 7b614a636 the server **refuses** multimodal prompts under `--kv-paged` rather than answering
from text alone, so the failure is now loud. But these two rows say nothing about vision, and the owner
runs VL models for vision. Recorded here so the count is not read as more than it is.

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
| `starcoder` | StarCoderBase-1B Q8_0 · MATCH | 1.26 GB | ✅ **SERVE-VERIFIED** — found + fixed a real paged bug, see below | Mac |
| `hunyuan_vl` | HunyuanOCR Q8_0 + mmproj · MATCH | 0.58 + 0.73 GB | ⛔ **FAIL → now REFUSED**: paged dropped the image (`PARIS` → `BUTTERFISH`) | Mac |
| `dflash` | Qwen3.5-4B-DFlash Q4_K_M · MATCH | 0.38 GB | ⛔ **BLOCKED — paged loop cannot speculate** (harness ran; see FINDINGS) | Mac |
| `eagle3` | Qwen3-8B-EAGLE3-Speculator F16 · MATCH | 2.05 GB | ⛔ **BLOCKED — paged loop cannot speculate.** 2.05 GB deliberately UNSPENT, see below | Mac |
| `gemma4-assistant` | gemma-4-E2B-it-assistant F16 · MATCH | 0.17 GB | ⛔ **BLOCKED — paged loop cannot speculate** (wiring done, serve-untested) | Mac |
| `gemma4` *(bonus, NOT on the 19)* | Gemma4-26B-A4B-Uncensored-1M Q4_K_M | 16.80 GB | ⚠ **PASS under `DS4P_PAGED_SWA=1`, and CHECKPOINT-SPECIFIC** — see below | Mac |
| `gemma4` *(E2B class)* | gemma-4-E2B-it Q4_K_M | 3.11 GB | ⛔ **REFUSED — KV-sharing split-brain** (`FINDINGS-paged-kv-sharing-splitbrain.md`) | Mac |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-Thinking **Q4_K_M** · MATCH | **13.33 GB** | ✅ **SERVE-VERIFIED** | Mac |
| `qwen3vlmoe` ⚠ | Qwen3-VL-30B-A3B-Instruct **Q4_K_M** · MATCH | **18.56 GB** | ✅ **SERVE-VERIFIED (TEXT ONLY)** | Mac |
| `laguna` | Laguna-S-2.1 Q8_0, 4 shards · MATCH | 256×4.5B | searched | **box** |
| `mimo2` | MiMo-V2.5 UD-Q4_K_S, 5 shards · MATCH | 256×8.2B | searched | **box** |
| `step35` | Step-3.5-Flash Q4_K · MATCH | 118.71 GB | searched | **box** |
| `hunyuan-moe` | Hunyuan-A13B-Instruct | ~34 GiB IQ3_KS | searched | **box** |
| `qwen3next` | Qwen3-Next-80B-A3B · **MATCH** | 80B | probed | **box** |
| `grok` | Arki05/Grok-1-GGUF · **MATCH** | 314B | probed | **box** |
| `minimax-m3` | unsloth/MiniMax-M3-GGUF · **MATCH** | large | probed | **box** |
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

### ⛔ DEFECT 5: KV-sharing architectures run split-brain under `--kv-paged`

Full write-up: **`FINDINGS-paged-kv-sharing-splitbrain.md`**. Guarded in the fork (`49b93a15c`) —
`--kv-paged` is now REFUSED when only some layers own KV, instead of answering wrong.

```
gemma-4-E2B-it   15 of 35 layers own KV    static " Paris."   paged " a."     REFUSED now
gemma4-26B-A4B   all 30 own KV             PASSES, 240 consume events
```

⚠ **This makes the `gemma4` bonus row checkpoint-specific.** "gemma4 passes" was true of the 26B and
false of the E2B class, on the same binary — the second time in one day the discriminating fact lived
in the **checkpoint** rather than the code (the first was fused-QKV on `starcoder`).

⚠ **And it adds a second dependency to `gemma4-assistant`.** Its only viable target is
`gemma-4-E2B-it` — `embedding_length_out = 1536` hard-couples them — so that row now needs *both*
paged speculation *and* this defect fixed.

**The real fix is scoped to machinery that already exists:** route shared layers to the paged pool as
**read-only** calls at the reuse index. Read-only paged attention landed the same day (`cce5d6959`),
and the index mapping is already implemented as the `layer_reuse_cb` at `llama-model.cpp:2431`.

### ⛔ The three draft-head rows are BLOCKED on a MISSING FEATURE, not on a harness

`draft_pair_gate.sh` exists and works — it found the blocker on its first complete run. See
**`FINDINGS-paged-no-speculation.md`**.

**Speculative decoding is structurally absent from the paged decode loop.** `update_slots_paged()`
does not call `pre_decode()`, `post_decode()` or `handle_last_sampled_token()`, which is where all
drafting lives. Measured: `draft_n` 27 → **0** between the static and paged arms, `#calls(b,g,a) =
0 0 0`, **output identical in every arm**.

| row | status | why |
|---|---|---|
| `dflash` | **BLOCKED** | harness ran end-to-end; arm (c) VOIDs at `draft_n = 0` |
| `gemma4-assistant` | **BLOCKED** | same, plus its paged wiring is still serve-untested |
| `eagle3` | **BLOCKED** | same code path |

⚠ **BLOCKED, not FAIL.** The output was never wrong. What is lost is 100% of the speculative-decode
speedup, silently — which is the *"or are slow"* half of the owner's bar, not the *"error out"* half.

⚠ **`eagle3`'s 2.05 GB is deliberately NOT downloaded, and this is the reason.** The vehicle is
identified and header-probed (`williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF`, arch `eagle3`
confirmed). Downloading it today buys nothing: the gate would reach arm (c) and VOID at `draft_n = 0`
exactly as `dflash` did, because the blocker is architectural and shared. Fetch it when the paged loop
can speculate — not before.

### Three archs the gate cannot answer in its current form

These are **UNANSWERED**, which is neither verified nor broken. Recording them as failures would be
wrong; recording them as pending without the reason would lose the work.

| arch | why the gate cannot answer | what would answer it |
|---|---|---|
| `dflash`, `eagle3` | **Draft heads, not models.** `llama-context.cpp:187` throws `requires ctx_other to be set` when the GGUF carries no `tok_embd`/`output`. Confirmed empirically on `dflash` (0.38 GB, exit 3). `eagle3` is the same code path, so it was **not downloaded**. | A target+draft harness (`-md`), then the separate question of whether the *draft's* KV is paged at all |
| `hunyuan_vl` | The only vehicle found is **HunyuanOCR**, an image model. Text-only prompts return `''` or `' $ $ $ $ …'`. The static reference is degenerate *by construction*, so there is nothing to arbitrate against. | mmproj + a real image, or a different `hunyuan_vl` text vehicle if one is published |
| ~~`gemma4-assistant`~~ | ~~unwired + read-only unimplemented~~ | **BOTH DONE** — see the read-only section below. Now: run the gate on the 0.17 GB vehicle |

### Read-only paged attention — implemented (fork `cce5d6959`), and its test arm was vacuous

`gemma4-assistant`'s NextN head calls `build_attn` with K and V **null**: it writes nothing and
attends KV the main graph already stored. The paged op's contract was fused write-then-read with no
way to express that, so the arch could not be wired at all.

**Why the previous attempt segfaulted.** "Make the write conditional" compiled and regressed nothing,
then crashed the moment a read-only call ran, because `n_heads_kv` still came from `k_new->ne[1]` —
independently, in **both** backends. Fixing one leaves the other to crash elsewhere. It was a
contract, not a branch.

Geometry now comes from the pool, which carries it exactly:

```
ne = [head_dim, block_size, 2*n_head_kv, n_blocks]      llama-kv-cache-paged.cpp:202
n_heads_kv = ne[2]/2
```

Derived **unconditionally** in both backends and cross-checked against `k_new` when present, so every
ordinary paged request exercises the new derivation. A derivation used only on the new path has its
first real test in the case nobody has run — which is how the last attempt got through.

```
DS4P_TEST_READONLY=1 DS4P_TEST_BS=64 DS4P_METAL_CHAMP=1
D= 64 read-only     max_abs=0.000e+00 PASS   max|aro|=2.499e-01
D= 64 read-only CPU max_abs=0.000e+00 PASS   max|bro|=2.499e-01
D= 96 / D=128 same.   ALL PASSED
```

⚠⚠ **The test arm was vacuous — the argument shift this file's own comment warns about, in the call
the warning exists to protect.** It read `run_paged(backend, D, with_rel, window, GGML_TYPE_F16, 0,
true)`: seven positionals, so `true` landed on `causal` and `read_only_check` stayed **false**. The
arm ran an ordinary paged call against another ordinary paged call — guaranteed `max_abs=0.000e+00
PASS`, never exercising read-only once. When `read_only_check` was moved to LAST to fix the *earlier*
shift, the declaration was corrected and the call site was not.

Two further guards came out of that: a PASS here is `max_abs == 0`, which is also what an **empty**
result prints (the loop is bounded by both sizes), so size and non-degeneracy are asserted and the
magnitude is printed beside the verdict; and a **CPU arm** was added, since read-only changed both
backends and testing only Metal leaves the other half compiled and unrun.

⚠ **Not proved: there is no mutation test.** The implementation has not been broken on purpose to
confirm the arm goes red. What is established is that the branch executes (`DS4P-READONLY` marker), in
both backends, over non-empty non-zero output, bit-identical to write-then-read at three head dims.

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

---

## 2026-08-08 — open items, each with its exact blocker

### 1. Flip `DS4P_METAL_CHAMP` to default-on — OWNER'S CALL, one line
```cpp
ggml_metal_paged_champ_enabled() -> return e ? atoi(e) != 0 : true;
```
Full case in `FINDINGS-256k-parity.md`. No configuration is made worse; the one place regression could
occur (head_dim 128, where both kernels work) tested **byte-identical**. Blocked only on approval —
it changes which kernel every paged request uses.

### 2. Graph reuse for the paged path — SPECIFIED, UNBUILT
`llm_graph_input_attn_kv_paged::can_reuse()` returns `false` unconditionally. Static reuses graphs
(measured 14 on a completed run); paged reuses zero, so every batch rebuilds **and** re-optimises the
Metal graph — the `ggml_graph_optimize` / `ggml_metal_graph_optimize_reorder` chain seen in the stack
sample.

The comment justifying it does not hold: `write_slots` is a tensor whose *contents* change, not the
topology. Shapes (`n_tokens`, `max_blocks`, `batch_size`) are constant during a uniform prefill, and
`res->set_inputs()` runs at `llama-context.cpp:1488`, **after** the reuse branch closes at `:1481` —
so a reused graph does get fresh contents.

**Exact blocker:** `set_input()` reads six fields off `mctx`. A reused graph must refresh that pointer
or it reads a stale mapping — *silently wrong attention*, not an error. The wrappers hold it privately:

| holder | accessor |
|---|---|
| `llama_memory_hybrid_context::ctx_attn_paged` | setter only, no getter |
| `llama_kv_cache_iswa` (`set_attn_paged_ctx`) | setter only, no getter |
| `llama_kv_cache_paged::get_mem_attn_paged()` | on the **cache**, not the context |

`can_reuse()` receives `params.mctx` = the hybrid/ISWA wrapper, and `dynamic_cast` to the paged type
fails on exactly the models that matter.

**Work:** add a const getter to both wrapper contexts → implement `can_reuse` to (a) refresh `mctx`
through it and (b) compare the three shape terms. **Verification required before trusting it:** a
long-context needle run under reuse, because the failure mode produces plausible text rather than an
error.

### 3. Still untouched
per-batch timing for the paged loop (the 20k-bin curve is uncomputable on that arm) · interleaved
`A B A B` attribution · ~50k intermittent defect · Metal/CUDA parity · 512k–1M ladder · multi-model ·
paged speculation checkpoint port (`spec_ckpt` + `load_tgt`, currently correct-by-refusal) ·
head_dim ≤ 64 branch of the shared-memory prediction (no vehicle on the box)

---

## Small, precise defects found 2026-08-08 (not the corruption; separate items)

**1. An unfittable pool ABORTS instead of exiting cleanly.**
`common_fit_paged_kv_blocks` emits a genuinely excellent error — the failing number, the budget, *the
largest `n_ctx` that would fit*, and four named remedies with the env var's current value:

```
requested n_ctx=524288 x 1 seq needs 12288 KV blocks (96.0 GiB), but the memory budget allows 11269 (88.0 GiB).
    Largest n_ctx that fits here: ~480768. Options: lower -c, lower -np, raise the budget with --margin,
    reduce LLAMA_PAGED_POOL_HEADROOM (now 1.50), or pass --paged-pool-clamp to shrink automatically.
```
…and the process then dies on `GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.")`
at `llama-kv-cache-paged.cpp:263`, whose text points at the wrong place. **Fix: return the error, do not
assert.** The diagnosis already exists one line above the crash.

**2. The error advertises a flag that does not exist.**
It recommends `--paged-pool-clamp`. There is **no parser entry** for it in `common/arg.cpp`;
`params.paged_pool_clamp` has no CLI binding. **Either wire the flag or stop advertising it.**

**3. The paged path emits no prefill progress.**
The static path prints `prompt processing, n_tokens = …, progress = 0.NN` per chunk. Paged prints one
allocation line, then silence — on a 17-minute 225k prefill (or ~75 min at 512k) a working run and a
stalled one are indistinguishable. Design written up (rate-limit on 5% crossings, not per chunk; do not
print during the ~21 pre-request graph builds). Site: the per-candidate batch loop in
`llama-paged-scheduler-impl.cpp`, where `DS4P_CHUNKLOG` already reads every needed value.

## ★ Headroom is a policy ceiling, not a hardware one
`LLAMA_PAGED_POOL_HEADROOM` defaults to **1.5**, applied to the bare context requirement *before* any
allocation — *"headroom for fragmentation and for the sharing/spill the paged cache exists to do"*.

| headroom | blocks for `-c 524288` | GiB | vs 88.0 GiB budget |
|---|---|---|---|
| **1.50 (default)** | 12,288 | 96.0 | **refuses** |
| 1.25 | 10,240 | 80.0 | fits |
| 1.05 | 8,601 | 67.2 | fits |

512k needs **64 GiB** of KV and the box had **88 GiB** available. ⇒ **Every "512k does not fit on this
box" in this lane should be re-read as "512k does not fit *at 1.5× headroom*."** That is a different
sentence, and it is one env var rather than an architecture change.

⚠ **Open QUESTION, not a finding:** is 1.5 measured or chosen? The purpose is documented; the value's
justification is not. If chosen as a safe-looking round number, part of this lane's context ceiling is
self-imposed. If measured, there is a low-headroom failure mode not yet met. ⚠ A 512k PASS at headroom
1.05 with `-np 1` and one request is **the least demanding configuration of that rung** — not "512k works".
Multi-slot at low headroom is untested.

## ★ The 1M rung: reachable, and it is q8-vs-q8

Back-solved from tonight's own init (`bytes_per_block = 8,388,608`):
`2 × n_heads_kv × 64 × n_layers × ggml_row_size(type, 256)` ⇒ **n_heads_kv × n_layers = 128**, i.e. 16 KV
heads over Ornith's **8 attention layers**. The pool covers those 8 only; the other 24 are recurrent and
their entire state is 50.25 MiB, which moves nothing here.

`ggml_row_size(q8_0, 256)` = 8 blocks × 34 B = **272 B** against f16's **512 B** — **53.1%**, not the
nominal half.

| context | f16 | q8_0 | q8_0 @ 1.25× headroom |
|---|---|---|---|
| 512k | 64.0 GiB | 34.0 GiB | 42.5 GiB |
| **1M** | **128.0 GiB** — over budget at *any* headroom | **68.0 GiB** | **85.0 GiB — fits in 88.0** |

⇒ **1M is reachable on this box at q8_0 banded paged KV**, with ~3 GiB to spare even at 1.25× headroom.
It is **not** reachable at f16 at any headroom, and lowering headroom cannot fix that.

⚠ **q8_0 paged is not plain `-ctk q8_0`.** The kernel rejects it unless banded:
`"paged KV cache type not supported by the kernel (f16/bf16/f32, or q8_0 with LLAMA_BANDED_QUANT_KV)"`.
The plan is **paged + q8_0 + `LLAMA_BANDED_QUANT_KV`**, and the unearned piece is exactly what the B4
closure flagged: the kernel arcs passed, **end-to-end on a real model did not run**.

⚠ **The static arm hits the same wall.** Same 8 layers, same head dim ⇒ static 1M at f16 is also 128 GiB,
without even a headroom multiplier to blame. **So the top rung is q8-vs-q8**, and the bar (paged ≤ 1.0×
static) is undefined until both arms are named q8_0.

⇒ Without that clause, a future session runs paged at q8_0, compares it against an **f16 static number
from the 256k table**, and reports a ratio that measures the KV quantisation rather than the paging —
**two arms differing in more than the thing under test**, which is this lane's most repeated error and the
exact shape of the defect-era comparisons.

Provenance worth keeping: `common.cpp:1284` records that an earlier sizing budgeted 512 B/row against an
actual 272 B/row and *"silently spent the context budget that quantising the KV cache was meant to buy
back, which is the entire point of the feature."* That fight has already been had once.

---

## ★★ DESIGN NOTE: a gate that repeats one request tests determinism, not correctness

**Thirteen gates in this lane send one prompt shape repeatedly.** That is why a three-request crash
survived weeks of testing: `long, short, long` is not an exotic sequence, it is *a conversation*, and
nothing here ever produced one.

**Every defect found on 2026-08-08 lived in the transition between requests:**
| defect | what crossed the boundary |
|---|---|
| prompt mirror | `slot.prompt.tokens` outliving the pool it described |
| context ledger | the previous request's last position, rejecting the next batch |
| recurrent input | allocator bytes under `s_copy`, read as a row index |

A suite built from repeated identical requests is blind to all three **by construction**. That is not a
gap in the gates; it is a gap in what a gate *is* here.

### The mechanical change — one line per gate
> Every gate that sends N requests should send them at **two prompt lengths, alternating.**

- **Cost:** one extra prompt file, one modulo in the request loop.
- **Catches:** the crash (`long, short, long`), the corruption (a multi-chunk request following a
  triggering one), and the whole class of *"state that survives a request boundary."*

### And the cheaper structural lesson
**Find at the cheapest context that reproduces; verify at the expensive one.** Every result that mattered
on 2026-08-08 came from **8k prompts running in ~90 seconds** — roughly twenty hypotheses tested in one
session. The 225k and 512k runs *verified generalisation*; they found nothing. A lane that tests only at
its headline context size pays 40+ minutes per hypothesis and therefore tests far fewer of them.

⚠ Written as a design note rather than applied: thirteen gate edits should not land while a measurement
holds the GPU lock, and none of tonight's work has reached the owner yet.

### ⚠ And the 1M rung costs TIME as well as memory — price both before committing

Prefill is **superlinear well past the old "knee"**. Fitted from the two static points:

| | tokens | wall |
|---|---|---|
| 225k static | 220,070 | 985.9 s (measured) |
| 512k static | 501,733 | ~4,400 s (projected floor) |

⇒ **t ~ N^1.815** — 2.28× the tokens for ~4.5× the time.

| exponent | 1M per arm |
|---|---|
| N^1.0 (linear) | 1.2 h |
| N^1.5 | 2.7 h |
| **N^1.815 (fitted)** | **4.3 h** |
| N^2.0 | 5.7 h |

⇒ **The 1M rung is an overnight spend, twice** (paged + static). Honest bracket: **2.7–5.7 h per arm,
4.3 h best fit.**

⇒ ★ **`q8_0` buys memory, not attention FLOPs** (credit: Grok). The quantisation that makes 1M *fit* in
68 GiB does nothing for what it *costs* — the decay travels with it.

⚠ **The exponent is a two-point fit and one point had not landed when it was made.** Fitting a shape to
two samples is the exact error corrected twice on 2026-08-08 (the linear extrapolation, and the defect-era
"knee then plateau"). Done deliberately here because a *bracketed* estimate beats none for a spend
decision — 1M itself is the third point that would confirm it.

⇒ **Price the 1M rung at 68 GiB *and* ~4–6 h per arm, two arms.** Both numbers travel together.

## ⚠ The defect-era prefill cost model is dead
`long-context-measurement-state` records a **knee at ~122k then a plateau ~156 tok/s**, and the planning
consequence *"256k is ~2x the cost of 128k, not 4x"*. Tonight's static 512k curve:

`~235 tok/s early → ~83 at 64% → still falling at 74%`

**There is no plateau.** The original reading was a 170k-token window taken for the shape of the whole
curve. ⇒ **Every rung above ~170k is underpriced by the current model**, which is exactly the range the
context-ceiling program exists to reach. This justifies **re-measuring the cost model**, not asserting a
new one from two points.

---

## ✅ Second model through the changed `set_input` — Ornith-35B (`qwen35moe`, 40 layers)

| arm | needle | within-arm r1==r2==r3 | `DS4P-RS` serving writes |
|---|---|---|---|
| non-paged | FOUND ×3 | **YES** | **816** |
| paged | FOUND ×3 | **YES** | **813** |

⇒ **Code-path coverage EARNED on a second model** — different layer count (40 vs 32), different FFN (MoE),
both paths, and the counter proves the rewritten function actually executed rather than merely that the
model answered.
⇒ **Correctness EARNED** — needle found by both arms on every request, each arm perfectly deterministic.

⚠⚠ **2026-08-09 — THE ARM LABELLED `paged` ABOVE WAS NOT PAGING ITS ATTENTION.** `qwen35moe.cpp` accepted
`paged_ctx` and never read it, so under `--kv-paged` every attention layer of Ornith-35B ran STATIC beside
an allocated, unread pool (`DS4P-CHECKOUT` 6, `DS4P-CONSUME` **0**). See
`FINDINGS-qwen35moe-no-consumer.md`.

**What survives and what does not, precisely:**
· **SURVIVES** — the `set_input` coverage claim. `DS4P-RS` counts the **recurrent** write, which a hybrid
  model performs in both arms regardless of how attention is computed. 813 vs 816 is real, and the
  rewritten function did execute on a second model.
· **DEAD** — any reading of that table as *paged attention* coverage. It was static-vs-static for the
  attention half.
⇒ Fixed and re-gated on the fixed binary: `DS4P-CONSUME` **0 → 430**, static-path fallbacks **230 → 0**,
sequence leg 3/3, poison control fired. **Ornith-35B pages its attention for the first time.**

⚠ **This is a second MODEL, not a second ARCH.** `qwen35moe` is the MoE sibling of `qwen35`: same family,
same recurrent design, same `key_length` 256. Header screen of every local GGUF found **15 archs, 2 hybrid,
both qwen35 family** — so hybrid-paged coverage in this fork is *one FAMILY deep by inventory*, not by bad
luck.
⇒ **Correction to the sentence that used to end here** (*"acquiring a genuine second hybrid arch is an
errand with a download"*): `qwen35moe` **is** a separate arch by the same source-file criterion that makes
`starcoder2` separate from `starcoder`, and it has been on this disk all along. What the box lacks is a
different hybrid **FAMILY** — Mamba/Jamba-shaped: `jamba`, `falcon-h1`, `granite-hybrid`, `nemotron-h`,
`plamo2`, `kimi-linear`, `lfm2`. **Do not shop for a second Qwen3.5; there are five here.**

### ⚠ Cross-arm byte-equality is the WRONG gate for two kernels
Proposed as a think-block-immune correctness test (Grok #7817) and it fails here — the arms diverge at
char 104 of every generation:

```
nonpaged: 'a long list of numbered notes (Note 0 to Note 340+).'
paged:    'a long list of "Notes" (Note 0 to Note 339+).'
```

**That is a paraphrase, not corruption.** Both reason correctly, both reach `MAGENTA-7742`, both quote the
source verbatim. Paged and static run **different attention kernels**; at temperature 0 a one-ULP
difference flips a near-tied token and the text diverges and never re-converges.

⇒ **Byte-equality is the right test for one kernel across requests, and the wrong test for two kernels
against each other.** The **within-arm** determinism check is the part that carries the claim.
⇒ Run as the verdict rather than inspected, it would have reported a **false defect**. Sixth time in this
session that reading the output beat reading the verdict line.

⚠ Two kernels producing semantically identical reasoning is strong evidence, not proof. The rigorous
version is a cross-arm **KV checksum** (`DS4P_KVSUM` exists and does exactly this) — unrun.

### ⚠ And the harness lesson: needle-in-N assumes an answer latency
The first attempt used `n_predict=16` and both arms "MISSED" — because this is a **reasoning model** that
spends those tokens opening `<think>`. Void by admission gate, and the gate was measuring my harness.
Earlier the same shape was misfiled as *"Nemotron wants a chat template"* without checking the output.
⇒ **Size `n_predict` to the model's answer latency, or the gate measures the harness.**

### ★ The paraphrase is a NEAR-TIE — measured, and cross-arm equality is *sound but flaky*
`n_probs=2` on the completion request, two **distinct** prompt lengths (the vary-length law applied):

| prompt | outcome | top-2 gap at the divergence token |
|---|---|---|
| A — 7,936 tok | diverges at char 104, token#29 | non-paged **0.0033** · paged 0.0623 |
| **B — 7,935 tok** | **generations byte-IDENTICAL, 413 chars, both arms** | — no divergence |

⇒ **Near-tie confirmed.** 0.0033 logprob between the top-2 candidates is ~0.3% of a nat — one ULP of
numerics difference flips it. The structural alternative predicted a *confident* token; it is not one.

⇒ ★ **And prompt B passes cross-arm byte-equality outright.** So the earlier verdict — *"byte-equality is
the wrong test for two kernels"* — is **too strong**. Sharper: **it is a valid test that fails
stochastically**, at whatever rate a generation encounters an early near-tie. One token of prompt length
separates a pass from a fail.

⇒ A single prompt would have given prompt A alone and the wrong conclusion; two gave the counterexample
immediately.

⚠ **But this run satisfied DISTINCTNESS, not the vary-length law** (calibration: Grok #7834). 7,936 vs
7,935 share almost the entire prefix, so the two runs hit near-ties at strongly correlated positions —
**~1.1 confirmations, not two.** Distinctness is what the gap question needed and it was sufficient here.
The law's original intent is **length coverage against boundary-class defects**, and a one-token delta
does not provide it.

⚠ Unexplained and not load-bearing: paged's gap at that token is 0.062 against static's 0.0033 — same
token index, different confidences, consistent with tiny numeric differences already accumulated by
token 29.

⚠ **Method note, self-inflicted:** the earlier second-model run sent **one prompt three times**, so
"all three requests diverge at char 104" was one measurement reported three times, and the within-arm
check was a repeat-request determinism test rather than three samples. The vary-length design note had
been written, argued and committed **four hours earlier** — and the next script written did not follow it.

### ★ THE CALIBRATED GATE RULE (joint, and the durable instrument from this exchange)

> **Cross-arm byte-equality, WITH top-2 gap logging at the first divergence.**
>
> | outcome | reading |
> |---|---|
> | **pass** | arms agree — done |
> | **fail, gap ≈ 0** | numerics near-tie, **not a defect** |
> | **fail, gap large** | **defect, and the coordinate is in hand** |

Byte-equality alone is **sound but flaky** — it fails at whatever rate a generation hits an early
near-tie, and one token of prompt length separated a pass from a fail here. **With the gap attached it is
sound *and self-diagnosing*:** the same run that reports a failure also says whether the failure means
anything.

⇒ This supersedes both earlier positions: *"byte-equality is the wrong test for two kernels"* (too strong)
and *"byte-equality is a clean correctness gate"* (too optimistic).

---

# ✅ CONCURRENCY CLOSED — warm, four-way, both models (2026-08-09)

Every cell: WARM regime (prime → N concurrent → third sequential), static control clean, `N*(N-1)`
ordered-pair contamination check, `post=N` verifying what the concurrent set leaves behind, and a
**live-paged-path assertion** so a CLEAN cannot be vacuous.

| model | `-np 2` | `-np 4` |
|---|---|---|
| Ornith-9B (`qwen35`) | CLEAN | **CLEAN** — `consume=10240` |
| Gemma4-26B (`gemma4`) | CLEAN — `consume=630`, **1c CLOSED** | **CLEAN** — `consume=18720` |

⇒ This closes the gap named as the one thing between the fixes and a recommendation: *multi-slot is
untested, and the daily driver runs `-np 4`.* It is tested now — four-way, warm, on two model families.

⚠ Scope: 3 reps per cell, `-c 8192`, one prompt set, Metal. **Not "multi-slot is proven"** — *"four-way
warm concurrency is clean on two models with verified-live paged paths."*

## ⚠ The attempt log — four harness faults, one premise change
The Gemma4 arm took **five attempts and only one was about paging.** Each of the first four printed
something that looked like a verdict:

| # | attempt | fault | what it printed |
|---|---|---|---|
| 1 | `multislot_gate.sh` | **COLD by construction** — server up → concurrent pair → down. No prior finished request, so it can never reach the regime 1c lives in | `12/12 CLEAN` |
| 2 | warm gate v1 | `n_predict=24` truncated a reasoning model mid-`<think>`; **static 0/6 too** | `1c REPRODUCES` |
| 3 | warm gate v2 | ran **Ornith**, not the model 1c was characterised on | `PASS` (scoped to the wrong model) |
| 4 | warm gate v3 | ambiguous 4th prompt → empty sequence; then raw-completion word-salad and a `la deuce` loop | `static 3 bad` |
| 5 | warm gate v4 | `MS_CHAT=1` + `--jinja` — **the premise change** | **PASS** |

⇒ **Only the VOID rule and reading the per-sequence dumps kept the first four from being reported as facts
about the paged path.** A dirty static arm means the gate is measuring itself; that rule blocked three
false verdicts in ninety minutes.

⇒ **Session ratio worth carrying: three real code defects, six harness defects, four blocked false
verdicts.** The apparatus was wrong twice as often as the code. Any green in this directory should be read
with that in mind — which is why every gate now asserts its own liveness (`consume > 0`) and voids on a
dirty control.


## 2026-08-12 SYNTHETIC BATTERY — 12/19 VERIFIED on tip (post allocator-fix + champion-partials)

Instrument: `arch_synthetic_battery.sh` — test-llama-archs fixtures (n_ctx_train=128), 100-token
4-chunk prompt (ub32), grammar-constrained greedy, token-exact 3-arm compare (static / paged-bs16 /
paged-bs64+champion) with fallback greps + champion presence counts. Vacuous-green guard: empty
token arrays report VOID, never MATCH (the first run printed 9 fake MATCHes on empty arrays —
caught by the gate-shape rule).

| his name | fixture | p16 | p64+champ | notes |
|---|---|---|---|---|
| deepseek v4 | real model | — | ✅ byte-exact | full ladder, this session |
| grok | grok-moe | ✅ | ✅ champN:4 | |
| hunyuan-moe | hunyuan-moe-moe | ✅ | ✅ champN:4 | |
| minimax-m3 | minimax-m3-moe | 🔴 REFUSED | 🔴 REFUSED | **wiring gap**: memory type not pageable even with DS4P_PAGED_HYBRID=1 — loud refusal (guard correct), fix = paged wiring for its memory class |
| nemotron | nemotron-dense | ✅ | ✅ champN:4 | |
| qwen3moe | qwen3moe-moe | ✅ | ✅ champN:4 | |
| qwen3next | qwen3next-moe | ✅ (DS4P_PAGED_HYBRID=1) | untested | hybrid; bs16 token-exact |
| qwen3vl | qwen3vl-dense | ✅ | ✅ champN:4 | text-only (image path refuses by design) |
| qwen3vlmoe | qwen3vlmoe-moe | ✅ | ✅ champN:4 | text-only |
| starcoder | starcoder-dense | ✅ | ✅ champN:4 | |
| ernie4-5 | ernie4_5-moe | ✅ | ✅ champN:4 | saver arch name is ernie4_5 (underscore) |
| ernie4-5-moe | ernie4_5-moe-moe | ✅ | ✅ champN:4 | |
| hunyuan-vl | hunyuan_vl-dense | ✅ | ✅ champN:4 | text-only |
| dflash / eagle3 / gemma4-assistant / laguna / mimo2 / step35 | — | ⏸ | ⏸ | **saver-unsupported — and box swept via ssh 2026-08-12: NO real ggufs exist for these 6 anywhere (box <BOX> + nvme + home, Mac local). They are incoming archs; the ONLY route is model-saver extensions (task #17B, inkling pattern).** |

**Score: 12 verified · 1 checked-negative (minimax-m3 wiring gap) · 6 blocked on fixtures/models.**
