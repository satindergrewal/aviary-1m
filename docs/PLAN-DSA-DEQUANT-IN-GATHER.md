# PLAN: dequantize during the DSA gather — ~253K → ~572K context, keeping flat decode

**Status:** designed, API-verified against `llama.cpp-idxfilter` (= `fleet`), **not implemented**.
Every source line below was read, not recalled. Blocked only on a GPU card.

**Why this is the top item in the context-ceiling lane:** it moves the ceiling rather than the
clock. Today we must *choose* between max context and flat decode. This removes the choice.

| KV type | path today | KV/token | ceiling | decode |
|---|---|---|---|---|
| F16 | gather (fast) | 94.1 KiB | ~253K | **flat ~50 t/s** |
| q8_0 | mask only | 52.5 KiB | ~398K | degrades (29 t/s @100K) |
| q4_0 | mask only | 30.2 KiB | ~572K | degrades |

**After this change: q4_0 in the gather path → ~572K AND flat decode = 2.26× context.**

Ceilings are extrapolated from measured inputs, not measured ceilings: 28.2 GiB headroom after
weights, the measured compute-buffer fit `540 MiB + 20.0 MiB/1K ctx`, main MLA `576×2×79`, and
the indexer at **21 layers** (post the merged filter fix, `c23459b0b`).

## Why the dequant is near-free — the reason this is worth doing at all

`k_dsa_prepare_one_batch_kv` (`ggml/src/ggml-cuda/fattn-dsa.cu:67`) **already materialises the
selected rows into a contiguous F16 buffer** for cuBLAS:

```cpp
static __global__ void k_dsa_prepare_one_batch_kv(int nk, int ncol, const int * idx, const char * k_in,
        half * k_out, size_t stride_k, size_t stride_idx) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int i = idx[row*stride_idx + col];
    const half * k_row = (const half *)(k_in + stride_k * i);   // <- only F16 assumption
    k_out += (row*ncol + col)*nk;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        k_out[j] = k_row[j];                                     // <- plain half copy
    }
}
```

Each selected element is touched **exactly once**, and the output is F16 regardless. So decoding
that element from a quantized block instead of copying a `half` changes the source decode and
**nothing downstream** — the GEMM, the softmax, and `k_out`'s layout are untouched. The gather is
already paying the materialisation cost; the dequant rides along inside it.

Note `k_in` is **already `const char *`** with a **byte** `stride_k`, so `k_in + stride_k*i`
addresses a quantized row correctly with no arithmetic change. Only the cast is F16-specific.

## Step 1 — relax the gate

`fattn-dsa.cu:322` currently:

```cpp
if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32) return false;
```

Replace the K/V clauses with a dequantizable-type test plus a block-alignment test. Keep
`mask == F16` and `Q == F32` unchanged.

```cpp
// [TAG_DSA_DEQUANT_GATHER] the gather materialises into F16 anyway, so any type with a
// dequantize_V_* helper can be decoded in-pass. mask/Q constraints are unchanged.
static inline bool dsa_k_type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: return true;
        default: return false;
    }
}
...
if (!dsa_k_type_supported(K->type) || !dsa_k_type_supported(V->type)) return false;
if (mask->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32) return false;
// per-row element count must be a whole number of quant blocks
if (ggml_is_quantized(K->type) && (K->ne[0] % ggml_blck_size(K->type)) != 0) return false;
```

GLM-5.2 satisfies the alignment: `n_embd_head_k = 576 = 18 × 32` for both `QK4_0` and `QK8_0`.
For MLA, V is a view of K (`dsa_v_is_k_view`, `:41`), so V's type follows K's automatically.

## Step 2 — template the gather on the source type

The helpers already exist in `ggml/src/ggml-cuda/fattn-common.cuh`, built for FA kernels reading
a quantized cache. **Verified signatures:**

```cpp
typedef void (*dequantize_V_t)(const void *, void *, const int64_t);   // :375
template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V();                // :642
```

`dequantize_V_q8_0<T, ne>(vx, dst, i0)` takes the **row base pointer**, an output pointer, and the
**element index within the row**; it resolves `ib = i0/QK8_0`, `iqs = i0%QK8_0` itself, and writes
`ne` elements. `ne` must be even (`static_assert(ne % 2 == 0)`).

```cpp
template <ggml_type type_K, int ne>
static __global__ void k_dsa_prepare_one_batch_kv_q(int nk, int ncol, const int * idx,
        const char * k_in, half * k_out, size_t stride_k, size_t stride_idx) {
    constexpr dequantize_V_t dequant = get_dequantize_V<type_K, half, ne>();

    const int row = blockIdx.y;
    const int col = blockIdx.x;
    const int i   = idx[row*stride_idx + col];

    const char * k_row = k_in + stride_k * i;      // byte stride: already correct for any type
    k_out += (row*ncol + col)*nk;

    for (int j = threadIdx.x*ne; j < nk; j += blockDim.x*ne) {
        dequant(k_row, &k_out[j], j);              // decode ne elements straight into the F16 buffer
    }
}
```

With `ne = 2` and `nk = 576`, that is 288 element-pairs across a 256-thread block — one stride
iteration plus a partial. Keep the existing F16 kernel as the `GGML_TYPE_F16` path so the
current, measured-good behaviour is bit-identical and untouched.

## Step 3 — dispatch at the call site

Switch on `K->type` where `k_dsa_prepare_one_batch_kv` is launched (inside
`ggml_cuda_flash_attn_ext_dsa`, `:343`), instantiating the template per type. Only the two types
we actually serve (`Q4_0`, `Q8_0`) need instantiating first; the rest can `return false` at the
gate until wanted, to keep compile time down.

## Step 4 — rotation: nothing to do (verified, not assumed)

The Hadamard/QuaRot rotation is a **graph** operation, not kernel-internal:

- `llama_kv_cache::build_input_k_rot` (`src/llama-kv-cache.cpp:1609`) creates it as a graph
  **input tensor** named `attn_inp_k_rot`.
- `// note: assumes k_rot^2 == I` (`:2133`) — self-inverse, so one matrix serves both directions.
- **`grep -riE "hadamard|rot_k|k_rot|unrotate" ggml/src/ggml-cuda/fattn*.cu fattn*.cuh` returns
  nothing.** There is no rotation logic inside any CUDA FA kernel.

So whatever the rotation arrangement is, the gather path inherits it exactly as the mask path
does. This was the main correctness risk and it is retired by that empty grep.

Related settled fact: `attn_rot_k = 1` on both DSA caches because `576 % 64 == 0`
([[fa-on-disables-the-guard]] records the retraction of my earlier "q4_0 is unrotated" error).

## Falsifiable acceptance test

1. **Correctness first.** `-ctk q8_0 -ctv q8_0` under **`-fa auto`** (not `-fa on` — see
   `docs/RESEARCH-KV-QUANT.md`, `auto` will refuse a bad combo and `on` will not). Confirm the DSA
   gather path is selected, then NIAH at 32K. **Gate: matches the F16 arm.**
2. **Ceiling.** Load at 398K with q8_0 and 572K with q4_0. **Gate: fits in VRAM.** A miss here
   falsifies the extrapolation, not the kernel.
3. **Flat decode — the whole point.** Decode t/s at depths 1K/32K/100K/200K. **Gate: flat, like
   the F16 gather arm (~50 t/s), NOT the mask path's decay to 29 t/s @100K.** If decode degrades,
   the dequant is not free and the change is not worth taking.
4. **Control.** F16 arm must be **bit-identical** to today, proving the new template did not
   perturb the existing path.

## Predicted next ceiling after this lands

Per 1K of context, the two consumers are:

| KV type | KV | compute buffer | ratio |
|---|---|---|---|
| F16 (today) | 91.9 MiB/1K | 20.0 MiB/1K | 4.6 : 1 |
| **q4_0** | **29.5 MiB/1K** | 20.0 MiB/1K | **1.5 : 1** |

KV still leads at q4_0, but the margin collapses from 4.6:1 to 1.5:1, so the compute buffer stops
being a rounding error and becomes the **next** thing worth attacking — the StreamIndex lane in
[[context-ceiling-program]]. Squeezing KV further after this yields progressively less until that
20.0 MiB/1K term is addressed.

## Risks

- **Occupancy/latency.** Dequant adds ALU work inside a memory-bound gather. Expected to hide,
  but test 3 is the arbiter. If it does not hide, try `ne = 4` or stage through shared memory.
- **`ne` and `nk` divisibility.** Guarded at the gate. Non-multiples of `ne` need a tail branch.
- **BF16.** `get_dequantize_V` routes BF16 through `dequantize_V_bf16<float, ne>` — note the
  hard-coded `float`, not `T`. Do not instantiate BF16 with `half` without reading `:657`.
