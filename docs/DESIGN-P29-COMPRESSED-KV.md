# DESIGN — P2-9 compressed-KV native tier: RECONCILED for our tree (verdict changes)

**Status: DESIGN ANALYSIS (2026-08-04 ~01:00). The plan's "KERNELS MAP PENDING" question is
answered, and the answer reframes the item.**
Author: Fable-DS4 | Plan source: PLAN-DS4-PORTS.md §P2-9 (ds4's e4m3 tier, KV÷2.76,
"bit-lossless").

## Finding 1: our V4 tree ALREADY stores the compressed latent
deepseek4.cpp has `build_hca_compressed_kv_from_state` (:381) and
`build_overlap_compressed_kv_from_state` (:439), consumed at :914/:966/:1010 — the MLA/HCA
cache stores the COMPRESSED KV latent, not expanded K/V. The plan's conditional ("if our
MLA cache already stores compressed, the win shrinks") resolves TRUE. The ds4 headline
(÷2.76 vs 2048 B F32 rows) does NOT apply; our baseline is already the latent at F16-class
precision. Remaining win = latent F16 → fp8(e4m3) ≈ **÷2 on KV**, valuable at 240K+ depth
but half the advertised ratio.

## Finding 2: "bit-lossless" does NOT transfer to our serving
ds4's exactness claim rides on serving the ORIGINAL fp8 checkpoint: the model itself emits
e4m3 codes, so storing them is lossless BY IDENTITY. Our fleet serves REQUANTIZED GGUFs
(Q4/IQ2-class weights); the latent is computed in F16/F32 activations and never exists as
e4m3 — storing it as e4m3 is a LOSSY round-trip. Consequences:
- the plan's kill gate "byte-exactness counter 0" is UNREACHABLE for us by construction;
- the real gate becomes a QUALITY gate (loop-rate + task probes per the loop-rate memory —
  noting that memory measured WEIGHT bits drive looping while KV quant was irrelevant, but
  an MLA LATENT is not a standard KV: it feeds every downstream projection, closer to an
  activation than a cache row — quality risk is REAL and unmeasured).

## Verdict: PARKED as M5 with a measurement plan (marginal-ledger doctrine)
Build trigger: a measured need for KV÷2 at deep context that the paged tier (3b) + disk
banks (P1-5) don't already satisfy. Measurement plan when triggered (one box window):
1. latent-fp8 A/B on the synthetic-scale V4 first (llama.cpp already has fp8 storage types
   available on CUDA for KV via type_k on non-FA paths; the MLA state path would need the
   cast pair added — small kernel surface, NOT the full ds4 encode/decode port);
2. quality axes: loop-rate 8-probe + NIAH-lite + task probes vs F16-latent arm;
3. decode curve 64K-256K vs the kv-paged baseline (short 63.6 | 64K 52.3 | 256K 38.5).
If quality holds within noise → schedule the real port; else the item dies with data.

## Credit
Technique source: ds4 (ds4_cuda.cu encode/decode kernels, frexpf/ldexpf pow2-scale
exactness discipline, pinned __fmaf_rn chains) — credit on any future port, per the
credit-ported-code rule.
