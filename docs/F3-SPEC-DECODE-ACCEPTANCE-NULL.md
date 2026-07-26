# F3: speculative-decode acceptance collapse (#23658) does not reproduce

A null result, written up so nobody spends another evening chasing it.

**Date:** 2026-07-26. **Hardware:** single RTX PRO 6000 96 GB (GPU1; GPU0 was busy
training, untouched). **Binary:** llama.cpp-dspark fork build.

---

## What the issue claimed

llama.cpp issue #23658 reported that draft acceptance collapses depending on the
server's `-c` value: `-c 12032` giving AR ~16% versus `-c 12288` giving ~71%, on
Qwen3.6-35B-A3B IQ2_M with `--spec-type draft-mtp`, temp-0 256-token completions.
Broken values were said to be non-2048-aligned; working values mostly 2048 multiples.

**Status of the issue itself: CLOSED.** The maintainer dismissed it as AI-written, so
the reported numbers were treated here as an unverified hypothesis, not a known bug.

## What was measured

Three tests. Every one holds the prompt set, temperature (0), token budget (256), and
binary fixed, varying only the thing under test.

| # | axis tested | spec path | model | result |
|---|---|---|---|---|
| 1 | generation crossing token 2048 | `draft-dspark` | Qwen3-4B-Q8_0 + head4 | **no collapse**, AR *rises* 0.80 to 0.98 |
| 2 | `-c 12032` vs `-c 12288` | `draft-dspark` | Qwen3-4B-Q8_0 + head4 | **bit-identical**, 0.5976 both |
| 3 | `-c 12032` vs `-c 12288` | `draft-mtp` | Qwen3.6-27B-Q4_K_M | **bit-identical**, 0.6691 both |

### Test 1: position within a generation

128-token windows, high-entropy prose, fixed temp 0.7, 2048 an exact window edge:

```
1792-1920  AR 0.7049  pre
1920-2048  AR 0.9208  crosses 2048
2048-2176  AR 0.9896  post
2176-2304  AR 0.9792  post
2304-2432  AR 0.9691  post
pooled pre 0.8027 (n=223) | pooled post 0.9792 (n=289) | delta +0.1765
```

Acceptance *improves* across the boundary. KV reuse was proven per window
(`timings.prompt_n == 1`, i.e. only the new token processed), and the cells were
non-saturated, so the measurement had room to show a drop and did not.

### Tests 2 and 3: the `-c` value

Six fixed prompts, 256 tokens each, temp 0, `cache_prompt` off, only `-c` differing:

```
dspark  -c 12032:  980/1640 = 0.5976     dspark  -c 12288:  980/1640 = 0.5976
MTP     -c 12032: 1019/1523 = 0.6691     MTP     -c 12288: 1019/1523 = 0.6691
```

Not merely close. **Identical prompt by prompt** (e.g. 286 drafted / 159 accepted in
both MTP runs for prompt 0). Same drafts, same accepts, token for token. The `-c` value
does not perturb the compute path at all in either configuration.

Test 3 used the issue's own spec path. Speculation was verified active rather than
silently disabled: acceptance lines were emitted with mean accepted length 2.66-3.31,
and since no `-md` was passed, the only possible source of drafts is the model's
in-model nextn head.

## Verdict

The claimed 16%-vs-71% swing has no support in anything runnable here. F3 closes.

## Caveat, stated precisely

The issue's exact model (Qwen3.6-35B-A3B IQ2_M) could not be tested: the local
35B-A3B GGUF carries **zero** nextn/MTP tensors (733 tensors, 40 blocks), so
`--spec-type draft-mtp` is not runnable against it. This is **not** evidence that the
architecture lacks MTP. Converters routinely drop nextn tensors unless explicitly
asked for them, so a local conversion's silence says nothing about the upstream model.
It does mean the literal stated configuration was not reproducible on this hardware,
and test 3 substitutes a different MTP-bearing model (27B dense, Q4_K_M) to exercise
the same code path.

## Method notes worth keeping

- **`tokens_evaluated` is not a cache-hit indicator.** It reports the *total* prompt
  length. The tokens actually processed are `timings.prompt_n`, and that is the field that
  proves KV reuse (1 means pure reuse). An early conclusion here that "cache reuse is
  broken" was wrong for exactly this reason and had to be retracted.
- **Per-request acceptance is in the response JSON** (`timings.draft_n` and
  `timings.draft_n_accepted`), so no log scraping is needed.
- **Saturating prompts destroy acceptance measurements.** A temp-0 numbered list becomes
  formulaic at depth, the draft head predicts every token, and AR pins at exactly 1.0000
  in every cell, hiding any effect. High-entropy prose at fixed temperature keeps the
  measurement informative.
- **`nohup ... &` inside a returning ssh command does not survive.** The child dies when
  the connection closes. `tmux new-session -d` is the reliable detach.

*Scripts: `f3_fine2.py` (window sweep), `f3_cvalue.py` (prompt set), `f3_drive.sh` /
`f3_drive_mtp.sh` (two-serve drivers), `gguf_names.py` (dependency-free GGUF header
reader for the MTP-tensor check).*
