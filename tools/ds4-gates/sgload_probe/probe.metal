// DS4 simdgroup_load pitch probe -- standalone, no ggml dependency.
//
// Purpose: settle the ONE unexplained quantity in the paged-attention MMA arms -- why only
// FOUR of eight source rows reached the fragment. Hypothesis (ornith 93e8afb): the load
// pitch and the store pitch are INDEPENDENT arguments (as the champion's Metal FA kernel
// uses them: DK to load Q, NS10 to load K, SH to store), and using one shared pitch of 2*D
// makes frag row f read source row 2f, reaching rows 0,2,4,6 only.
//
// Contract under test: simdgroup_load(frag, src, pitch) fills frag[r][c] = src[r*pitch + c].
// If true, LOAD_PITCH == source row stride reaches ALL 8 rows regardless of STORE_PITCH.
//
// Credit: the pitch contract is read off ggml's Metal flash-attention kernel
// (ggml/src/ggml-metal/ggml-metal.metal, Georgi Gerganov and the ggml contributors).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// src is an 8-row x SRC_W-col matrix of floats; we read the leading 8x8 tile.
// Each element is encoded r*100 + c so the dump names its own origin unambiguously.
kernel void sgload_probe(
        device const float * src        [[buffer(0)]],
        device       float * dst        [[buffer(1)]],
        constant     uint  & load_pitch [[buffer(2)]],
        constant     uint  & store_pitch[[buffer(3)]],
        uint tpitg [[thread_position_in_threadgroup]]) {
    simdgroup_float8x8 f = make_filled_simdgroup_matrix<float, 8>(-1.0f);

    simdgroup_barrier(mem_flags::mem_none);
    simdgroup_load(f, src, load_pitch);
    simdgroup_barrier(mem_flags::mem_none);

    simdgroup_store(f, dst, store_pitch);
}
