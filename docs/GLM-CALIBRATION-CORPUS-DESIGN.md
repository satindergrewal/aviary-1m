# GLM 5.2 calibration corpus design

Task F4. Everything downstream depends on this: the DSpark acceptance ladder is
meaningless until the calibration question is settled, and the GLM requant target
follows from it.

---

> ## ⚠️ CORRECTION 2026-07-28: this does NOT describe the production model
>
> The analysis below is about **GLM-5.2-smol-IQ1_KT**. It was then used to justify
> replacing the imatrix for the model actually in service,
> `GLM-5.2-ours-IQ1_S-prot.gguf`. That justification does not hold, because the two
> were calibrated by different people with different data. Read straight from the
> production GGUF's own provenance keys:
>
> ```
> general.quantized_by           = Unsloth
> quantize.imatrix.file          = <BOX>/bigmodels/glm52-imatrix-unsloth.gguf
> quantize.imatrix.dataset       = unsloth_calibration_GLM-5.2.txt
> quantize.imatrix.entries_count = 1002
> quantize.imatrix.chunks_count  = 88
> ```
>
> and from the imatrix file itself, `imatrix.chunk_size = 9216`, i.e.
> **811,008 calibration tokens from a corpus Unsloth built specifically for
> GLM-5.2**. Not a wikitext split, not 1.29 MB, and no evidence of test-split
> contamination.
>
> The cost of not checking: a 25-hour replacement run was started against this
> premise and would have produced **three times less** calibration than the file
> already sitting on nvme0 (512-token chunks x 512 = 262K tokens against 811K).
> It was stopped at 7.2 hours.
>
> What survives: the corpus-design reasoning below is sound wherever a thin or
> wrong-split imatrix genuinely exists, and the measured leverage of calibration at
> low bit-width (IQ1_KT PPL 98,935 to 95.8) is real and unaffected. What does not
> survive is the claim that the served model needed rescuing.
>
> Unverified either way: whether GLM-5.2-smol-IQ1_KT's imatrix really was 1.29 MB of
> wikitext test split. That artifact is no longer on the box and the claim was never
> sourced from its provenance keys. Treat it as unconfirmed. Note the knock-on: A1's
> GLM arm used that model, so if its calibration was genuinely poor, GLM's true loop
> rate under good calibration may be *better* than the 4/8 recorded.

## The problem, precisely

GLM-5.2-smol-IQ1_KT as shipped was calibrated with an imatrix of **1.29 MB of
English-only `wikitext-2-raw` TEST split**, for a **bilingual ~744B MoE**
(⚠️ unconfirmed, see correction above).

Three separate defects, worth separating because they need different fixes:

**1. Wrong split. This is a methodology error, not just thin data.**
The calibration data came from the **test** split. If you then report perplexity on
`wikitext-2-raw` test, the quantizer has been tuned on the evaluation text. The PPL
number is optimistically biased and cannot be compared against quants calibrated on
a disjoint split. Any published PPL for that model carries this caveat.

**2. Wrong language mix.** English-only calibration for a bilingual model means the
importance matrix has no evidence about which weights matter for Chinese. Those
weights get treated as unimportant and are quantized hardest.

**3. Wrong distribution.** Encyclopedia prose, when the model is served for agentic
coding with long context and reasoning traces. The imatrix records *activation*
statistics; activations under templated multi-turn chat with `<think>` blocks and
code do not resemble activations under raw Wikipedia text.

**Scale of the thinness:** 1.29 MB is roughly 300K tokens of evidence for a 744B
parameter model. Our own measurements show what better calibration is worth at low
bit-width: IQ1_KT perplexity **98,935 → 95.8** (1,032×), and IQ3_KT **65.7 → 14.3**.
Those were the same weights at the same bit-width. Calibration is the highest-leverage
variable we have found at low bpw.

---

## Proposed corpus

Target **~50-100 MB** of text (roughly 40-80× the current corpus), assembled to mirror
served conditions rather than to maximise size.

| share | content | why |
|---|---|---|
| 30% | **Code**: multiple languages, real repositories, including diffs and tests | primary use is agentic coding |
| 25% | **Chinese technical + general prose** | the model is bilingual; currently 0% represented |
| 20% | **English technical prose**: documentation, specs, papers | the register he actually works in |
| 15% | **Multi-turn chat, chat-template formatted, with reasoning traces** | see the template note below |
| 10% | **Long documents**: 32K+ contiguous tokens | long-context activation patterns |

### Three requirements that matter more than the exact percentages

**Split-disjoint, enforced.** Calibrate on `wiki.train.raw` and any other training-side
source. Evaluate on `wiki.test.raw`. Never overlap. Record the corpus manifest with the
quant so the claim is auditable.

**Chat-template formatted where it counts.** He serves with `--jinja`. Feeding raw prose
to `llama-imatrix` produces activation statistics for a token distribution the model
never sees in production. The chat-formatted share should be passed through the same
template used at serve time, including the special tokens.

**Reasoning traces included.** GLM 5.2 reasons by default. If production output contains
`<think>` blocks and calibration data contains none, the activation statistics for
reasoning mode are unrepresented, and reasoning is exactly where long generation, and
therefore looping, happens.

---

## What this design does NOT fix

**MTP / nextn layers get zero activations in a normal forward pass**, so no amount of
corpus improvement gives them importance data. That is a separate, structural gap
(task F2, and llama.cpp PRs #23258 / #23476 / #23575). The corpus and the MTP work are
independent and both are needed.

**Looping is not primarily a calibration problem.** Measured: loop rate is driven by
weight bit-width and is *scale-independent* (27B and 744B both loop 4/8 at 1.75 bpw).
Better calibration should improve quality broadly, but nobody should expect it to move
the loop cliff on its own. Do not let this corpus work be sold as the loop fix.

---

## Validation plan

The corpus is only justified if it measurably beats the shipped one. Same model, same
bit-width, only the imatrix differs:

1. **Perplexity** on a held-out split, plus a **Chinese** eval set. The current
   corpus predicts nothing about Chinese, so that is where the largest gain should
   appear if the language-mix argument is right.
2. **Loop rate** via `tools/loop_rate.py`, expected roughly unchanged; measured so
   the claim is honest rather than assumed.
3. **Draft acceptance** (DSpark lane). This is the metric that is currently blocked,
   and the reason this design is on the critical path.
4. **Retrieval at depth**: long-document share should show up here if anywhere.

A negative result is a real result: if a 50 MB properly-mixed corpus does not beat
1.29 MB of English wikitext, that is worth knowing before anyone spends GPU hours
requantizing a 169 GB model.

---

## Decisions (made, not deferred)

**1. Chinese share: 15%, not 25%.** Every prompt in this project is English, so 25%
over-serves a use case that does not exist. It is not dropped to zero because in a MoE
the bilingual capability is not cleanly separable, since shared attention and router weights
serve both languages, and starving them of evidence damages English too. 15% buys
insurance against that without spending a quarter of the corpus on an unused capability.
The 10% freed goes to code.

**2. Code share mirrors the actual stack: 70/30.** Seventy percent Go, Rust, Swift,
Python and C++ (the languages of this project's own codebases and the llama.cpp
work); thirty percent
broad for general capability. The whole argument for this corpus is that calibration
should match served activations, and his stack *is* the served distribution. Using a
generic code mix would contradict the premise.

**3. Requant target: mixed-precision, ~1.95-2.1 bpw effective.** Uniform 2.9 bpw is
arithmetically dead (≈280 GB against ~188 GB usable). Attention and shared/router
tensors go to 6-8 bpw, experts stay at IQ1_KT/IQ2_KT. Experts are ~90-95% of an MoE's
parameters, so raising the small remainder costs little: `0.95×1.75 + 0.05×6 ≈ 1.96 bpw
≈ 180 GB`, which fits. Gated on task A2 measuring whether attention precision is what
governs loop rate.

**4. Sourcing: self-generated plus permissive only.** Three sources, no licensing risk:
- **His own repositories** for the code share, since he owns them, and they are the exact
  distribution he serves against.
- **`wikitext` TRAIN split** and open specifications/documentation for English prose
  (train split only; the shipped imatrix's use of the test split is the defect this
  design exists to fix).
- **Model-self-generated** for the chat-formatted and reasoning-trace shares: run GLM
  under production serve conditions (`--jinja`, reasoning enabled) and capture its own
  output. This is not a convenience choice, it is the approach with the strongest
  evidence behind it. The DSpark lane measured own-regenerated calibration lifting draft
  acceptance from **0.149 to 0.25-0.32**, and an imatrix-IQ1_S at 1.6 bpw holding
  **0.293**, matching Q8. Self-calibration beat generic calibration by a wide margin on
  a different metric and a different model. Chinese follows the same route where a
  permissive corpus is not readily available.

**Revised mix:** 40% code (70/30 his-stack/broad) · 15% Chinese · 20% English technical ·
15% chat-formatted with reasoning traces · 10% long documents.

---

*Design complete. Assembly begins once A2 returns the mixed-precision result, since
decision 3 sets the bit-width the corpus is calibrating for.*
