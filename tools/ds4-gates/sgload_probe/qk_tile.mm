#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char ** argv) {
    const uint32_t D        = argc > 1 ? (uint32_t) atoi(argv[1]) : 64;
    const uint32_t n_heads  = argc > 2 ? (uint32_t) atoi(argv[2]) : 8;
    const uint32_t head_idx = argc > 3 ? (uint32_t) atoi(argv[3]) : 3;
    const uint32_t ss_pitch = argc > 4 ? (uint32_t) atoi(argv[4]) : 8;
    const uint32_t bad      = argc > 5 ? (uint32_t) atoi(argv[5]) : 0;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:@"qk_tile.metal"
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qk_tile"] error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        // 8 query tokens, head-interleaved; 8 key tokens, contiguous rows of D.
        const uint32_t qn = 8 * n_heads * D, kn = 8 * D;
        id<MTLBuffer> bq = [dev newBufferWithLength:qn*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [dev newBufferWithLength:kn*sizeof(float) options:MTLResourceStorageModeShared];
        float * qh = (float *) bq.contents; float * kh = (float *) bk.contents;
        // Deterministic, head-dependent, non-symmetric: a transpose bug cannot hide.
        for (uint32_t t = 0; t < 8; ++t)
          for (uint32_t h = 0; h < n_heads; ++h)
            for (uint32_t d = 0; d < D; ++d)
              qh[(t*n_heads + h)*D + d] = sinf(0.3f*t + 0.7f*h + 0.11f*d);
        for (uint32_t t = 0; t < 8; ++t)
          for (uint32_t d = 0; d < D; ++d)
            kh[t*D + d] = cosf(0.5f*t - 0.13f*d);

        const uint32_t dn = 8*ss_pitch + 64;
        id<MTLBuffer> bd = [dev newBufferWithLength:dn*sizeof(float) options:MTLResourceStorageModeShared];
        float * dh = (float *) bd.contents;
        for (uint32_t i = 0; i < dn; ++i) dh[i] = -999.0f;

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
        [e setBuffer:bd offset:0 atIndex:2];
        [e setBytes:&D length:4 atIndex:3];        [e setBytes:&n_heads length:4 atIndex:4];
        [e setBytes:&head_idx length:4 atIndex:5]; [e setBytes:&ss_pitch length:4 atIndex:6];
        [e setBytes:&bad length:4 atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 2; }

        // TWO references. The double one is exact-ish truth; the fp32 one accumulates in the
        // SAME precision the GPU does. If a mismatch shrinks to noise against the fp32
        // reference, it is accumulation order over D terms -- not a broken load/store
        // contract. Reporting both keeps a large-D failure from being waved away by simply
        // loosening a threshold.
        double worst = 0.0; int worst_r = -1, worst_c = -1;
        double worst32 = 0.0;
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t c = 0; c < 8; ++c) {
                double ref = 0.0;
                float  ref32 = 0.0f;
                for (uint32_t d = 0; d < D; ++d) {
                    ref   += (double) qh[(r*n_heads + head_idx)*D + d] * (double) kh[c*D + d];
                    ref32 +=          qh[(r*n_heads + head_idx)*D + d] *          kh[c*D + d];
                }
                double got = dh[r*ss_pitch + c];
                double e2 = fabs(got - ref) / fmax(1.0, fabs(ref));
                if (e2 > worst) { worst = e2; worst_r = r; worst_c = c; }
                double e32 = fabs(got - (double) ref32) / fmax(1.0, fabs((double) ref32));
                if (e32 > worst32) worst32 = e32;
            }
        }
        printf("D=%u n_heads=%u head=%u ss_pitch=%u q_pitch=%s\n",
               D, n_heads, head_idx, ss_pitch, bad ? "D (THE BUG)" : "n_heads*D (correct)");
        printf("  worst rel err vs f64 %.3e at [%d][%d] | vs f32-order %.3e\n",
               worst, worst_r, worst_c, worst32);
        // Gate on the fp32 reference: it isolates the CONTRACT (which elements were read and
        // where they landed) from fp32 summation order, which no load pitch can fix.
        bool ok = worst32 < 1e-5;
        printf("  VERDICT: %s\n", ok ? "TILE == REF" : "MISMATCH");
        return ok ? 0 : 1;
    }
}
