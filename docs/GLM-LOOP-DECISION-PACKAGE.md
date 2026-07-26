# GLM 5.2 looping: the answer, the fix, and the options

The end-to-end result of the loop investigation, written as a decision document.
Everything below is measured — on the production GLM-5.2-smol-IQ1_KT itself where
possible, on a Qwen3.6-27B ladder where scale made GLM impractical. No projections
without being labeled as such.

---

## The problem, restated

The production GLM (169 GB, 1.99 file-level bpw) collapses into repetition loops on
long generations. Observed originally at 264K context in real agentic use; reproduced
here on demand.

## Root cause — measured, with controls

| suspect | verdict | evidence |
|---|---|---|
| KV cache quantization | **exonerated** | f16/q8/q4 KV all loop at identical 4/8 |
| Context depth | **not the cause** | loop rate at 1.5K ≈ 32K ≈ 128K per bpw level |
| Model scale | **no protection** | 744B GLM loops 4/8 = 27B at same bpw |
| Sampling randomness | **not the cause** | loops reproduce at temperature 0 |
| **Weight bit-width** | **THE CAUSE** | 1.99 bpw = 4/8 → 2.30 bpw = 0/8, controls pass |

The mechanism is attention collapse (heads locking onto the recent window). The
threshold is sharp: on the 27B ladder, looping at ≤2.24 file-level bpw, clean at
≥2.30. `ffn_up` is load-bearing — raising only `ffn_down`+`ffn_gate` changes nothing.

Two hazards found on the way, both worth remembering:

- **Below-knee reasoning models can loop invisibly inside `<think>`** — the negative
  control produced a 4-word cycle repeated 62 times entirely in the reasoning field,
  zero visible output. Any loop check that reads only visible text misses this class.
- **Perplexity gives no threshold.** PPL degrades smoothly across the same range where
  usability falls off a cliff. Loop-rate and generation-success are separate axes and
  both belong next to PPL on any card.

---

## OPTION 1 — RECOMMENDED: keep the current model, fix the sampler

**Zero requantization. Zero VRAM cost. Validated on the production GLM itself.**

```
--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
--temp 0.7 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 1
-fa on
```

Measured chain, matched pairs on the real model:

| config | loop rate |
|---|---|
| plain sampling @ 0.7 | 3/8 |
| DRY (allowed_length 2) | 1/8 — long-narrative holdout survives |
| **DRY (allowed_length 1)** | **0/3 on the holdout, 8/8 clean regression** |

The single-parameter change from the standard DRY config (`dry_allowed_length` 2 → 1)
beat the one prompt that survived everything else, and the full-prompt regression shows
it does not mangle legitimate output. The aggressive alternative (multiplier 1.5) also
worked but is the sledgehammer; this is the gentlest sufficient setting.

Caveats, stated honestly:
- DRY-at-depth was validated at 32K/128K on the 2.30-bpw ladder model (0 loops in all
  20 cells, both reasoning and visible fields). On the *production* GLM, the tuned
  setting was validated at standard depth; a 128K confirmation run on GLM itself is
  cheap (~30 min) whenever wanted.
- Note that plain repetition penalty is NOT a substitute and can *cause* loops.

## OPTION 2 — requantize above the knee: does not fit

The knee demands ≥2.30 file-level bpw. Applied to GLM:

| recipe | projected size | fits 188 GB usable? |
|---|---|---|
| all-FFN → IQ2_KT (cheapest loop-free found) | ~195-204 GB | **NO** |
| uniform IQ2_KT-class | ~210-224 GB | NO |

(Projections by file-size ratio from the 27B ladder; MoE expert share makes GLM's
number worse, not better, since experts ARE the FFN and are 90%+ of weights.)

Sub-options that could revive this path, all unmeasured:
- **Per-expert targeting** finer than all-FFN (e.g. only `ffn_up`+`ffn_down` of experts,
  or only some layers). The 27B data says down+gate alone fails, so optimism should be
  limited, but MoE structure differs.
- **Partial expert offload** — accept ~10-15 GB of experts on CPU. MoE activation
  sparsity makes this less painful than it sounds, but decode speed takes a hit.

## OPTION 3 — hybrid: option 1 now, revisit option 2 after the calibration work

The corpus redesign (docs/GLM-CALIBRATION-CORPUS-DESIGN.md) is happening anyway for
quality reasons. If a future better-calibrated requant is planned, the knee experiment
should be re-run on it — better calibration lowered other cliffs dramatically
(IQ3_KT PPL 65.7 → 14.3), and it is plausible, though unproven, that it moves the
loop knee down too. If it moves below ~2.0, a loop-free all-VRAM GLM becomes possible.

---

## Bottom line

**Deploy option 1 today.** It is validated end-to-end on the production model, costs
nothing, and the failure it addresses is fully characterized. Keep option 3 open: the
calibration corpus may move the knee, and the knee experiment is cheap to re-run.

*All measurements in this repo: tools/loop_rate.py (harness), docs/QUANT-AND-INFERENCE-
REFERENCE.md (full findings), docs/charts/quant-usability-cliff.html (the curve).*
