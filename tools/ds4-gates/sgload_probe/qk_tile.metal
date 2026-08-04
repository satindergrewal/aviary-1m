// DS4 Q@K^T single-tile contract probe -- the exact shapes the paged MMA path will use.
//
// Q is HEAD-INTERLEAVED in the paged kernel: q[tok*n_heads*D + head*D + d]. So for 8
// consecutive query tokens the row stride is n_heads*D, NOT D. Passing D here is the exact
// bug that cost fifteen arms (ornith b340689) -- this probe exists to prove the right one.
// K comes from the staged threadgroup tile tk[t*D + d]: pitch D, transpose=true, because
// mqk[qr][kt] = sum_d Q[qr][d] * K[kt][d] needs mk laid out d x kt.
//
// Credit: shapes read off ggml's Metal flash-attention kernel (ggml-metal.metal:6961-7000),
// Georgi Gerganov and the ggml contributors.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void qk_tile(
        device const float * q      [[buffer(0)]],   // [n_tok][n_heads][D]
        device const float * k      [[buffer(1)]],   // [n_key][D]  (the staged tile shape)
        device       float * dst    [[buffer(2)]],   // [8][ss_pitch]
        constant     uint  & D          [[buffer(3)]],
        constant     uint  & n_heads    [[buffer(4)]],
        constant     uint  & head_idx   [[buffer(5)]],
        constant     uint  & ss_pitch   [[buffer(6)]],
        constant     uint  & use_bad_q_pitch [[buffer(7)]],
        uint tpitg [[thread_position_in_threadgroup]]) {

    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 mq, mk;

    // Q row stride: the WHOLE point. n_heads*D is correct; D is the fifteen-arm bug.
    const uint q_pitch = use_bad_q_pitch ? D : (n_heads * D);
    device const float * pq = q + head_idx * D;
    device const float * pk = k;

    for (uint i = 0; i < D/8; ++i) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(mq, pq + 8*i, q_pitch);
        simdgroup_load(mk, pk + 8*i, D, 0, true);        // transpose -> d x kt
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
    }

    simdgroup_store(mqk, dst, ss_pitch);                 // its OWN stride, independent
}
