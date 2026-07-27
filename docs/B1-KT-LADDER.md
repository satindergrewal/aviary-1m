# B1: the KT ladder, and why the loop knee is not the usability threshold

Run 2026-07-28. Qwen3.6-27B (dense, `qwen35`, FFN 17408), CUDA, greedy, 1500
tokens, prompt set v1, f16 KV, `llama.cpp-kt/build-cuda`.

Depths 0 / 8K / 32K, all three metrics, with the IQ1_KT arm serving as the
positive control (see `N3-DEPTH-CONTROL-ARM.md`).

## Result

| quant | bpw | loops d0 | loops 8K | loops 32K | gen success d0 | 8K | 32K |
|-------|-----|----------|----------|-----------|----------------|-----|-----|
| IQ1_KT | 1.75  | **4/8** | 1/6 | 0/4 | 100% | 75%  | 50% |
| IQ2_KT | 2.145 | 0/8     | 0/5 | 0/2 | 100% | 62%  | **25%** |
| IQ3_KT | 3.15  | 0/8     | 0/8 | 0/6 | 100% | **100%** | **75%** |

## The finding

**The loop knee and the usability threshold are different numbers.**

Loops are gone by 2.145 bpw, which is what the earlier work concluded and it
holds. But generation success at depth keeps climbing well past that point:
at 8K, IQ2_KT answers 62% of prompts while IQ3_KT answers 100%. At 32K the
gap is 25% against 75%.

Choosing a quant on loop rate alone says "IQ2_KT is fine, the cliff is behind
us". Adding the success axis says IQ3_KT is materially better in exactly the
regime that matters for long-context work. A model that never loops but answers
one prompt in four at 32K is not usable; it is quiet.

## The lossless floor, added after the fact

Published without it, which was wrong. Measured 2026-07-28 on the same model,
prompt set and harness:

| Qwen3.6-27B at Q8_0 (8.5 bpw, essentially lossless) | **0/20 loops = 0%** |
|---|---|

That floor is what makes the table above interpretable: the 27B does not loop at
all when lossless, so **every loop in the ladder is quantization damage**, cleanly
attributed. The ladder stands as published.

This matters because it does not generalise. The same check on Qwen3-4B returned a
**42% floor**, which invalidated a separate conclusion drawn from it (see
`A1-SCALE-HYPOTHESIS-REOPENED.md`). Run the lossless rung as cell zero of any bpw
ladder before comparing anything.

One honest wrinkle in the floor run: generation success at Q8_0 was 20/24 (83%),
slightly *below* IQ1_KT's 22/24. At n=24 that is noise, and it is a reminder that
the generation-success axis is noisier at shallow depth than the loop axis.

## Caveats

- n = 8 per cell. IQ2_KT's 25% at 32K against IQ1_KT's 50% is p ~ 0.6, i.e.
  noise. Do not read IQ2_KT as *worse* than IQ1_KT at depth. What the row
  supports is only that IQ2_KT is not clearly better on this axis, while
  IQ3_KT is. IQ3_KT restoring 75% is what makes the IQ2_KT dip look like an
  n=8 artefact rather than a real inversion.
- The loop advantage of IQ2_KT over IQ1_KT at depth 0 (0/8 against 4/8) is
  itself only p = 0.077 at this sample size. Prompt set v2 (n=24) exists to
  settle it.
- Dense model. Whether the same ladder shape holds for an MoE, where the
  quantized unit is an expert rather than a full FFN, is untested. See
  `A1-SCALE-HYPOTHESIS-REOPENED.md` for why that distinction matters more than
  it looks.

## Reproduce

```bash
<BOX>/ktdev/depth_sweep.sh <BOX>/ktdev/Qwen27B-IQ2_KT.gguf 'Qwen27B-IQ2_KT|2.145'
```

Raw rows in `<BOX>/ktdev/depth_results/`.
