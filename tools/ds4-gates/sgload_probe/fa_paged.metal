// DS4 paged flash-attention MMA prototype -- the complete inner loop, standalone.
//
// This is the kernel body destined for kernel_paged_attn_f32, developed here first because
// a standalone runtime-compiled probe iterates in SECONDS while an in-tree arm costs a full
// llama.cpp rebuild. Fifteen in-tree arms cost more than the probe that ended them.
//
// Architecture (credit: ggml's Metal flash-attention kernel, kernel_flash_attn_ext ~6900-7200,
// Georgi Gerganov and the ggml contributors):
//   - the O accumulator lives in THREADGROUP memory, rescaled by scalar threads (`so *= ms`);
//     fragments are transient per iteration. There is no per-row scale on a simdgroup_matrix.
//   - the score tile is stored to `ss`, softmaxed there (masking and bias are trivial in
//     scalar code), then reloaded as the P fragment for P@V.
//
// Measured pitch contract (tools/ds4-gates/sgload_probe, gates 6/6 and 9/9):
//   Q is head-interleaved -> pitch n_heads*D.  K/V from the staged tile -> pitch D.
//   Store pitches are INDEPENDENT of load pitches.  Passing D for Q is the 15-arm bug.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define NSG_MAX 8

kernel void fa_paged(
        device const float   * q           [[buffer(0)]],  // [n_tok][n_heads][D]
        device const half    * kv          [[buffer(1)]],  // [block][2*n_heads_kv][bs][D]
        device const int32_t * block_table [[buffer(2)]],  // [max_blocks]
        device       float   * dst         [[buffer(3)]],  // [n_tok][n_heads][D]
        constant     uint    & D           [[buffer(4)]],
        constant     uint    & n_heads     [[buffer(5)]],
        constant     uint    & head_idx    [[buffer(6)]],
        constant     uint    & n_tok       [[buffer(7)]],  // queries, each attends [0, its pos]
        constant     uint    & bs          [[buffer(8)]],  // paged block size
        constant     uint    & max_blocks  [[buffer(9)]],
        constant     float   & scale       [[buffer(10)]],
        threadgroup  float   * shmem       [[threadgroup(0)]],
        uint3 tgpig  [[threadgroup_position_in_grid]],
        uint3 tpitg3 [[thread_position_in_threadgroup]],
        uint3 ntg3   [[threads_per_threadgroup]]) {

    const uint tid  = tpitg3[0];
    const uint ntg  = ntg3[0];
    const uint nsg  = ntg / 32;
    const uint sg   = tid / 32;
    const uint lane = tid % 32;

    const uint QR = 8 * nsg;              // query rows per threadgroup
    const uint SH = bs;                   // score-tile row stride (its OWN stride)
    const uint PV = D;                    // O accumulator row stride (its OWN stride)

    // ELEMENT TYPES ARE NOT FREE, IN TWO DIFFERENT WAYS:
    //  1. simdgroup_load requires the fragment element type to MATCH the source pointer type
    //     (a float8x8 cannot read half*). So the STAGED operands -- K, V, Q, P -- are half,
    //     which also keeps the K/V staging bandwidth win that made the Q-block increment pay.
    //  2. The ACCUMULATORS must be float. simdgroup_multiply_accumulate takes four
    //     independent floating-point type parameters (dst, a, b, acc), so a float
    //     accumulator with half inputs is legal -- and necessary: accumulating the Q.K dot
    //     in half cost 2.4e-02 relative error at D=64 (measured, and it scaled with D:
    //     4.6e-03 at D=8), which is a real precision loss and not a rounding artifact.

    // ---- threadgroup layout. Sized identically on the host; layout and allocation are ONE
    //      change, because a mismatch here already cost a crash and a false "ALL PASSED".
    threadgroup half  * tk = (threadgroup half *) shmem;
    threadgroup half  * tv = tk + bs*D;
    threadgroup half  * sq = tv + bs*D;                       // staged Q (packed, pitch D)
    // P is FLOAT. Half P was measured to cost ~2e-03 at 3+ blocks while 1-2 block cases sat
    // at 1.5e-04: the kernel quantizes P against the RUNNING max, so the error compounds
    // once the online rescale chain gets long. multiply_accumulate takes independent types,
    // so float P against the half V tile is legal and keeps the K/V staging win intact.
    threadgroup float * ss = (threadgroup float *) (sq + QR*D);   // scores, float
    threadgroup float * sp = ss + QR*SH;                          // P, float
    threadgroup float * so = sp + QR*SH;                      // O accumulator, float
    threadgroup float * M  = so + QR*PV;
    threadgroup float * S  = M + QR;

    const uint qbase = tgpig[0] * QR;     // first query row of this threadgroup
    const uint qrow0 = qbase + 8*sg;      // first query row of THIS simdgroup's tile

    // init accumulator / running stats
    for (uint i = tid; i < QR*PV; i += ntg) so[i] = 0.0f;
    for (uint i = tid; i < QR;    i += ntg) { M[i] = -INFINITY; S[i] = 0.0f; }

    // Stage Q once. Q is HEAD-INTERLEAVED in device memory (row stride n_heads*D); sq is
    // packed (row stride D). The gathering happens here, in scalar code, so the fragment
    // load downstream uses the packed pitch. Rows past n_tok are zeroed rather than skipped
    // -- they must not read out of bounds and must not contribute.
    for (uint i = tid; i < QR*D; i += ntg) {
        const uint r = i / D, d = i % D;
        const uint qg = qbase + r;
        sq[i] = (qg < n_tok) ? (half) q[(qg*n_heads + head_idx)*D + d] : (half) 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Highest query position in this threadgroup decides how many blocks to walk. Uniform by
    // construction (derives from tgpig alone), so every barrier below is threadgroup-uniform.
    const uint q_hi   = min(qbase + QR, n_tok) - 1;
    const uint nblk   = (q_hi + 1 + bs - 1) / bs;

    for (uint bi = 0; bi < nblk; ++bi) {
        const uint pb = (uint) block_table[bi];
        const uint64_t kb = ((uint64_t) pb * 2 * bs + (uint64_t) 0        * bs) * D;
        const uint64_t vb = ((uint64_t) pb * 2 * bs + (uint64_t) 1        * bs) * D;

        threadgroup_barrier(mem_flags::mem_threadgroup);        // prev iter's readers
        for (uint idx = tid; idx < bs*D; idx += ntg) {
            tk[idx] = kv[kb + idx];
            tv[idx] = kv[vb + idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Q@K^T for this simdgroup's 8 query rows against all bs keys of the block
        for (uint cc = 0; cc < bs/8; ++cc) {
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);  // FLOAT acc
            simdgroup_half8x8 mq, mk;                                               // half in
            for (uint i = 0; i < D/8; ++i) {
                simdgroup_barrier(mem_flags::mem_none);
                // Q from the packed staged tile -> pitch D. (In device memory Q is
                // head-interleaved at n_heads*D; the gather above is what makes D correct
                // here. Passing the wrong one of these two is the fifteen-arm bug.)
                simdgroup_load(mq, sq + (8*sg)*D + 8*i, D);
                // K pitch = D from the staged tile, transposed to give d x kt.
                simdgroup_load(mk, tk + cc*8*D + 8*i, D, 0, true);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }
            // store with the score tile's OWN stride
            simdgroup_store(mqk, ss + (8*sg)*SH + cc*8, SH);
        }
        simdgroup_barrier(mem_flags::mem_none);

        // ---- online softmax, scalar, on ss. Masking is trivial here.
        for (uint r = 0; r < 8; ++r) {
            const uint jl  = 8*sg + r;                 // row within the threadgroup tile
            const uint qg  = qrow0 + r;                // global query row
            const bool row_ok = qg < n_tok;
            const uint q_pos  = row_ok ? qg : (n_tok - 1);   // clamp, never early-return

            float m_prev = M[jl];
            float mx = m_prev;
            for (uint c = lane; c < bs; c += 32) {
                const uint kpos = bi*bs + c;
                float s = ss[jl*SH + c] * scale;
                if (kpos > q_pos) s = -INFINITY;       // causal
                ss[jl*SH + c] = s;
                mx = max(mx, s);
            }
            mx = simd_max(mx);

            const float ms = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - mx);
            float sum = 0.0f;
            for (uint c = lane; c < bs; c += 32) {
                const float s  = ss[jl*SH + c];
                const float vs = (s == -INFINITY) ? 0.0f : exp(s - mx);
                sp[jl*SH + c] = vs;                    // P, float, for the fragment load below
                sum += vs;
            }
            sum = simd_sum(sum);

            if (lane == 0) { M[jl] = mx; S[jl] = S[jl]*ms + sum; }
            // rescale the O accumulator in THREADGROUP memory -- no per-row fragment scale
            for (uint i = lane; i < D; i += 32) so[jl*PV + i] *= ms;
        }
        simdgroup_barrier(mem_flags::mem_none);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P @ V
        for (uint dd = 0; dd < D/8; ++dd) {
            simdgroup_float8x8 lo;
            simdgroup_load(lo, so + (8*sg)*PV + dd*8, PV);
            for (uint cc = 0; cc < bs/8; ++cc) {
                simdgroup_float8x8 mp;                 // float P
                simdgroup_half8x8  mv;                 // half V (staged)
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(mp, sp + (8*sg)*SH + cc*8, SH);
                simdgroup_load(mv, tv + cc*8*D + dd*8, D);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(lo, mp, mv, lo);
            }
            simdgroup_store(lo, so + (8*sg)*PV + dd*8, PV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- write out
    for (uint r = 0; r < 8; ++r) {
        const uint jl = 8*sg + r;
        const uint qg = qrow0 + r;
        if (qg >= n_tok) continue;                     // safe: no barrier below this point
        for (uint i = lane; i < D; i += 32) {
            dst[(qg*n_heads + head_idx)*D + i] = so[jl*PV + i] / (S[jl] + 1e-6f);
        }
    }
}
