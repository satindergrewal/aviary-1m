# A2 re-run: which component causes the looping

Run 2026-07-28. Qwen3.6-27B (dense, `qwen35`), CUDA, greedy, 1500 tokens,
prompt set **v2** (n=24), f16 KV, `llama.cpp-kt/build-cuda`. All variants already
existed on the box; no requantization.

## Cell zero: the lossless floor

| Qwen3.6-27B at Q8_0 (8.5 bpw) | **0/20 loops = 0%** |
|---|---|

The 27B does not loop at all when lossless, so every number below is quantization
damage rather than model capability. Stated first because omitting it is what
invalidated the A1 conclusion the same night.

## Result

| variant | loops | rate | Fisher vs baseline |
|---------|-------|------|--------------------|
| baseline IQ1_KT (1.75 bpw) | 10/22 | 45% | (reference) |
| **FFN tensors to q8** | **1/22** | **5%** | **p = 0.0039** |
| attention tensors to q8 | 4/23 | 17% | p = 0.057 |
| knee-allffn | 6/24 | 25% | p = 0.22 |
| knee-down | 5/21 | 24% | p = 0.20 |
| knee-downgate | 6/23 | 26% | p = 0.22 |

## What this overturns

The original A2 concluded "FFN is the loop-critical component, **not** attention"
and recorded that as killing the mixed-precision-on-attention route for GLM. It
drew that from attention-q8 scoring 3/8 against a 4/8 baseline on n=8.

**That comparison is Fisher p = 1.000.** Not weak evidence, no evidence.

At n=24, three separate statements, kept separate on purpose:

1. **FFN to q8 works.** 45% to 5%, p = 0.0039. It recovers 40 of the 45 points of
   quantization damage, landing within 5 points of the lossless floor.
2. **Attention to q8 probably works too.** 45% to 17%, p = 0.057. Borderline, not
   proven.
3. **Attention and FFN are NOT distinguishable from each other.** Head to head,
   4/23 against 1/22, **p = 0.346**. There is no basis for saying FFN matters more.

So "kills the mixed-precision-on-attention route" is unsupported on both halves.
Attention-q8 is a live route.

## Why that matters for a MoE, and the caveat that stops it being a plan

In a 256-expert MoE the attention tensors are a tiny fraction of total size, so
upgrading them to q8 costs very little on disk. If the 17% carried across, it
would be the cheapest loop mitigation available.

**It has not been shown to carry across.** This is a dense 27B, where "FFN" is a
full 5120 x 17408 feed-forward. In GLM the equivalent unit is a 6144 x 2048
expert. A1 is the standing lesson on assuming a result crosses the dense/MoE
boundary: it did not, and reporting it as though it had is what needed retracting.

The knee variants (allffn / down / downgate), which upgrade only part of the FFN,
all sit at 24-26% and none reach significance against the baseline. Partial FFN
upgrades do not reproduce the full FFN-q8 result.

## Reproduce

```bash
<BOX>/ktdev/run_a2.sh          # variants
<BOX>/ktdev/run_q8_control.sh  # cell zero
```

Raw rows in `<BOX>/ktdev/depth_results/`.
