// Host for the DS4 paged flash-attention MMA prototype. CPU reference in double precision,
// plus an fp32-order reference so accumulation order cannot be mistaken for a contract break.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static float h2f(uint16_t h) {  // reference-side half decode, matches what the GPU staged
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0)       { if (m == 0) f = s << 31; else { e = 127 - 15 + 1;
                        while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; f = (s<<31)|(e<<23)|(m<<13); } }
    else if (e == 31) { f = (s<<31)|0x7f800000|(m<<13); }
    else              { f = (s<<31)|((e - 15 + 127)<<23)|(m<<13); }
    float o; __builtin_memcpy(&o, &f, 4); return o;
}
static uint16_t f2h(float f) { __fp16 h = (__fp16) f; uint16_t o; __builtin_memcpy(&o, &h, 2); return o; }

int main(int argc, char ** argv) {
    const uint32_t D          = argc > 1 ? (uint32_t) atoi(argv[1]) : 64;
    const uint32_t n_heads    = argc > 2 ? (uint32_t) atoi(argv[2]) : 8;
    const uint32_t head_idx   = argc > 3 ? (uint32_t) atoi(argv[3]) : 3;
    const uint32_t n_tok      = argc > 4 ? (uint32_t) atoi(argv[4]) : 40;
    const uint32_t bs         = argc > 5 ? (uint32_t) atoi(argv[5]) : 32;
    const uint32_t nsg        = argc > 6 ? (uint32_t) atoi(argv[6]) : 4;
    const float    scale      = 1.0f / sqrtf((float) D);

    const uint32_t QR         = 8 * nsg;
    const uint32_t max_blocks = (n_tok + bs - 1) / bs;
    const uint32_t n_tg       = (n_tok + QR - 1) / QR;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:@"fa_paged.metal"
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fa_paged"] error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        // Q
        std::vector<float> qh((size_t) n_tok * n_heads * D);
        for (uint32_t t = 0; t < n_tok; ++t)
          for (uint32_t h = 0; h < n_heads; ++h)
            for (uint32_t d = 0; d < D; ++d)
              qh[((size_t) t*n_heads + h)*D + d] = 0.5f*sinf(0.31f*t + 0.7f*h + 0.11f*d);

        // Block table: DELIBERATELY SHUFFLED so a kernel that ignores it and walks linearly
        // produces a different answer. A paged path that passes with an identity table proves
        // nothing about paging.
        // REVERSE the mapping. The earlier (i*7+3) % max_blocks sent logical block 0 to
        // PHYSICAL block 0 whenever max_blocks was 1 or 3, so the early tokens -- which are
        // exactly the ones the causal mask leaves in block 0 -- never exercised remapping at
        // all. Three configs then reported an identical worst case and looked like a broken
        // sweep. Reversal guarantees block 0 -> physical max_blocks-1 for every size > 1.
        std::vector<int32_t> bt(max_blocks);
        for (uint32_t i = 0; i < max_blocks; ++i) bt[i] = (int32_t) (max_blocks - 1 - i);

        // KV: [block][2][bs][D] as half
        std::vector<uint16_t> kvh((size_t) max_blocks * 2 * bs * D);
        for (uint32_t pb = 0; pb < max_blocks; ++pb)
          for (uint32_t t = 0; t < bs; ++t)
            for (uint32_t d = 0; d < D; ++d) {
              kvh[(((size_t) pb*2 + 0)*bs + t)*D + d] = f2h(0.5f*cosf(0.5f*(pb*bs+t) - 0.13f*d));
              kvh[(((size_t) pb*2 + 1)*bs + t)*D + d] = f2h(0.5f*sinf(0.23f*(pb*bs+t) + 0.17f*d));
            }

        id<MTLBuffer> bq  = [dev newBufferWithBytes:qh.data()  length:qh.size()*4  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bkv = [dev newBufferWithBytes:kvh.data() length:kvh.size()*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bbt = [dev newBufferWithBytes:bt.data()  length:bt.size()*4  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd  = [dev newBufferWithLength:qh.size()*4 options:MTLResourceStorageModeShared];
        float * dh = (float *) bd.contents;
        for (size_t i = 0; i < qh.size(); ++i) dh[i] = -999.0f;

        // Threadgroup memory: MUST match fa_paged.metal's layout exactly. Layout and
        // allocation are ONE change -- a mismatch already cost a crash and a false ALL PASSED.
        const uint32_t SH = bs, PV = D;
        // MUST match fa_paged.metal exactly: tk,tv,sq are half; ss,sp,so,M,S are float.
        const size_t smem = ((size_t) 2*bs*D + (size_t) QR*D)*sizeof(uint16_t)
                          + (2*(size_t) QR*SH + (size_t) QR*PV + 2*QR)*sizeof(float);
        if (smem > 32768) { printf("SKIP: smem %zu > 32768\n", smem); return 77; }

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bq offset:0 atIndex:0];  [e setBuffer:bkv offset:0 atIndex:1];
        [e setBuffer:bbt offset:0 atIndex:2]; [e setBuffer:bd  offset:0 atIndex:3];
        [e setBytes:&D length:4 atIndex:4];         [e setBytes:&n_heads length:4 atIndex:5];
        [e setBytes:&head_idx length:4 atIndex:6];  [e setBytes:&n_tok length:4 atIndex:7];
        [e setBytes:&bs length:4 atIndex:8];        [e setBytes:&max_blocks length:4 atIndex:9];
        [e setBytes:&scale length:4 atIndex:10];
        [e setThreadgroupMemoryLength:smem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(n_tg,1,1) threadsPerThreadgroup:MTLSizeMake(32*nsg,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 2; }

        // ---- CPU reference: plain causal attention through the SAME shuffled block table.
        double worst = 0.0; int wr = -1, wd = -1; uint32_t untouched = 0;
        double worst_h = 0.0;
        for (uint32_t t = 0; t < n_tok; ++t) {
            // TWO references, kept genuinely distinct:
            //  exact   -- full-precision Q. Measures the TOTAL error budget including the
            //             kernel's deliberate half staging of Q.
            //  matched -- Q quantized to half exactly as the kernel stages it (sq). Measures
            //             the ALGORITHM. V needs no such treatment: it is half in the source
            //             buffer, so h2f(kvh) is exact truth for it.
            // Keeping one variable for both is how the "exact" column silently stopped being
            // exact when Q quantization was first added -- the two columns then agreed
            // trivially and the agreement looked like evidence.
            std::vector<double> logits(t + 1), logits_m(t + 1);
            double mx = -INFINITY, mx_m = -INFINITY;
            for (uint32_t p = 0; p <= t; ++p) {
                const uint32_t blk = p / bs, off = p % bs, pb = (uint32_t) bt[blk];
                double dot = 0.0, dot_m = 0.0;
                for (uint32_t d = 0; d < D; ++d) {
                    const double kv_d = (double) h2f(kvh[(((size_t) pb*2 + 0)*bs + off)*D + d]);
                    const float  qf   = qh[((size_t) t*n_heads + head_idx)*D + d];
                    dot   += (double) qf            * kv_d;
                    dot_m += (double) h2f(f2h(qf))  * kv_d;
                }
                logits[p]   = dot   * scale;
                logits_m[p] = dot_m * scale;
                if (logits[p]   > mx)   mx   = logits[p];
                if (logits_m[p] > mx_m) mx_m = logits_m[p];
            }
            double sum = 0.0;
            for (uint32_t p = 0; p <= t; ++p) { logits[p] = exp(logits[p] - mx); sum += logits[p]; }
            // SECOND reference: P quantized to half exactly as the kernel stores it. The
            // kernel keeps P in half deliberately (the champion's arrangement -- it is what
            // lets the P@V fragment load match the half V tile). If the GPU tracks THIS
            // reference but not the exact one, the residual is that documented design
            // choice, not a defect. Same discipline that settled the D=256 question: add a
            // reference at the implementation's own precision instead of widening a
            // tolerance until the test goes quiet.
            std::vector<double> logits_h(t + 1);
            double sum_h = 0.0;
            for (uint32_t p = 0; p <= t; ++p) {
                logits_h[p] = (double) (float) exp(logits_m[p] - mx_m);  // P is float in the kernel
                sum_h += logits_h[p];
            }
            for (uint32_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (uint32_t p = 0; p <= t; ++p) {
                    const uint32_t blk = p / bs, off = p % bs, pb = (uint32_t) bt[blk];
                    acc += logits[p] * (double) h2f(kvh[(((size_t) pb*2 + 1)*bs + off)*D + d]);
                }
                double acc_h = 0.0;
                for (uint32_t p = 0; p <= t; ++p) {
                    const uint32_t blk = p / bs, off = p % bs, pb = (uint32_t) bt[blk];
                    acc_h += logits_h[p] * (double) h2f(kvh[(((size_t) pb*2 + 1)*bs + off)*D + d]);
                }
                const double ref   = acc / sum;
                const double ref_h = acc_h / sum_h;
                const double got = dh[((size_t) t*n_heads + head_idx)*D + d];
                if (got == -999.0) { untouched++; continue; }
                const double e2 = fabs(got - ref) / fmax(1e-2, fabs(ref));
                if (e2 > worst) { worst = e2; wr = (int) t; wd = (int) d; }
                const double eh = fabs(got - ref_h) / fmax(1e-2, fabs(ref_h));
                if (eh > worst_h) worst_h = eh;
            }
        }
        printf("D=%u n_heads=%u head=%u n_tok=%u bs=%u nsg=%u QR=%u blocks=%u smem=%zu\n",
               D, n_heads, head_idx, n_tok, bs, nsg, QR, max_blocks, smem);
        printf("  vs exact %.3e at tok=%d d=%d | vs precision-matched %.3e | untouched: %u\n",
               worst, wr, wd, worst_h, untouched);
        // Gate on the precision-matched reference -- it holds the kernel to its OWN
        // documented staging choice, so a logic defect fails while half staging of Q does not.
        // The exact column stays visible so that choice's real cost is never hidden.
        const bool ok = (untouched == 0) && (worst_h < 5e-4);
        printf("  VERDICT: %s\n", ok ? "FA == REF" : "MISMATCH");
        return ok ? 0 : 1;
    }
}
