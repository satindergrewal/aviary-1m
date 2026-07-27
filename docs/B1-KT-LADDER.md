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
/mnt/nvme0/ktdev/depth_sweep.sh /mnt/nvme0/ktdev/Qwen27B-IQ2_KT.gguf 'Qwen27B-IQ2_KT|2.145'
```

Raw rows in `/mnt/nvme0/ktdev/depth_results/`.
